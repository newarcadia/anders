"""Map Blender vertex weights to native rigid and skinned bindings."""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
if TYPE_CHECKING:
    import bpy
from ..errors import UgxError
from .data import Section, Vertex, is_no_rigid_bone_index

__all__ = ('source_skin_map', 'section_uses_v6_global_bone_indices', 'vertex_group_from_source_bone', 'vertex_group_from_section_bone', 'encoded_bone_from_vertex_group', 'importable_source_skin_map', 'skin_changed', 'encode_skin', 'mesh_has_v4_skin', 'new_vertex_skin', 'new_v4_vertex_skin', 'new_v4_uniform_rigid_bone', 'new_v6_vertex_skin', 'new_v6_rigid_fallback_bone')

def source_skin_map(vertex: Vertex) -> Dict[int, float]:
    result: Dict[int, float] = {}
    for local_bone, weight in zip(vertex.bone_indices, vertex.bone_weights):
        if weight > 0.00001:
            result[local_bone] = result.get(local_bone, 0.0) + weight
    return result


def section_uses_v6_global_bone_indices(section: Section) -> bool:
    return section.cache_version == 6 and not section.local_to_global_bones


def vertex_group_from_source_bone(section: Section, local_bone: int, group_count: int) -> Optional[int]:
    if local_bone < 0:
        return None
    if section.local_to_global_bones:
        if local_bone >= len(section.local_to_global_bones):
            return None
        global_bone = section.local_to_global_bones[local_bone]
    elif section_uses_v6_global_bone_indices(section):
        # Retail v6 skeletons reserve cached bone zero for GrannyRootBone.
        # The stored byte then addresses the cached/global table directly.
        global_bone = local_bone
    elif local_bone > 0 and local_bone <= group_count:
        global_bone = local_bone - 1
    else:
        global_bone = local_bone
    return global_bone if 0 <= global_bone < group_count else None


def vertex_group_from_section_bone(section: Section, bone_index: int, group_count: int) -> Optional[int]:
    if is_no_rigid_bone_index(bone_index):
        return None
    if section.local_to_global_bones:
        if bone_index >= len(section.local_to_global_bones):
            return None
        bone_index = section.local_to_global_bones[bone_index]
    # RigidBoneIndex is global in both PC games when no explicit palette
    # exists. Accessory spans do not add a second local index space.
    return bone_index if 0 <= bone_index < group_count else None


def encoded_bone_from_vertex_group(section: Section, group_index: int, group_count: int) -> Optional[int]:
    if group_index < 0:
        return None
    if section.local_to_global_bones:
        local_for_global = {global_bone: local for local, global_bone in enumerate(section.local_to_global_bones)}
        return local_for_global.get(group_index)
    if section_uses_v6_global_bone_indices(section):
        if not (0 <= group_index < group_count and group_index <= 255):
            return None
        return group_index
    one_based = group_index + 1
    if 0 < one_based <= group_count:
        return one_based
    return group_index if 0 <= group_index <= 255 else None


def importable_source_skin_map(vertex: Vertex, section: Section, group_count: int) -> Dict[int, float]:
    result: Dict[int, float] = {}
    for local_bone, weight in zip(vertex.bone_indices, vertex.bone_weights):
        if weight <= 0.00001:
            continue
        global_bone = vertex_group_from_source_bone(section, local_bone, group_count)
        if global_bone is None:
            continue
        encoded_bone = encoded_bone_from_vertex_group(section, global_bone, group_count)
        if encoded_bone is not None and 0 <= encoded_bone <= 255:
            result[encoded_bone] = result.get(encoded_bone, 0.0) + weight
    return result


def skin_changed(current: Dict[int, float], source: Dict[int, float], epsilon: float = 0.003) -> bool:
    keys = set(current.keys()) | set(source.keys())
    return any(abs(current.get(key, 0.0) - source.get(key, 0.0)) > epsilon for key in keys)


def encode_skin(current: Dict[int, float]) -> Tuple[bytes, bytes]:
    influences = [(bone, weight) for bone, weight in current.items() if weight > 0.00001]
    if not influences:
        raise UgxError("Skinned UGX vertex has no Blender vertex-group weights.")
    influences.sort(key=lambda item: item[1], reverse=True)
    influences = influences[:4]
    total = sum(weight for _, weight in influences)
    if total <= 0.0:
        raise UgxError("Skinned UGX vertex weights sum to zero.")
    scaled = [weight / total * 255.0 for _, weight in influences]
    raw_weights = [int(value) for value in scaled]
    remaining = 255 - sum(raw_weights)
    order = sorted(range(len(scaled)), key=lambda index: scaled[index] - raw_weights[index], reverse=True)
    for index in order[:remaining]:
        raw_weights[index] += 1
    raw_indices = [bone for bone, _ in influences]
    while len(raw_indices) < 4:
        raw_indices.append(0)
        raw_weights.append(0)
    return bytes(raw_indices[:4]), bytes(raw_weights[:4])


def mesh_has_v4_skin(obj: bpy.types.Object, bone_name_to_index: Dict[str, int]) -> bool:
    if not bone_name_to_index or not obj.vertex_groups:
        return False
    for vertex in obj.data.vertices:
        for group in vertex.groups:
            if group.weight <= 0.00001 or group.group >= len(obj.vertex_groups):
                continue
            if obj.vertex_groups[group.group].name in bone_name_to_index:
                return True
    return False


def new_vertex_skin(
    obj: bpy.types.Object,
    vertex_index: int,
    bone_name_to_index: Dict[str, int],
    *,
    one_based: bool,
    label: str,
    fallback_bone_index: Optional[int] = None,
) -> Tuple[bytes, bytes, int]:
    influences: List[Tuple[int, float]] = []
    vertex = obj.data.vertices[vertex_index]
    for group in vertex.groups:
        if group.weight <= 0.00001 or group.group >= len(obj.vertex_groups):
            continue
        bone_name = obj.vertex_groups[group.group].name
        bone_index = bone_name_to_index.get(bone_name)
        if bone_index is None:
            continue
        encoded_index = bone_index + 1 if one_based else bone_index
        if encoded_index > 255:
            raise UgxError(f"Bone {bone_name} cannot be encoded in {label} skin bytes.")
        influences.append((encoded_index, float(group.weight)))
    if not influences and fallback_bone_index is not None:
        encoded_index = int(fallback_bone_index) + (1 if one_based else 0)
        if 0 <= encoded_index <= 255:
            influences.append((encoded_index, 1.0))
    if not influences:
        raise UgxError(f"Skinned {label} UGX export found vertex {vertex_index} with no weights for the export armature.")
    raw_indices, raw_weights = encode_skin(dict(influences))
    return raw_indices, raw_weights, min(4, len(influences))


def new_v4_vertex_skin(
    obj: bpy.types.Object,
    vertex_index: int,
    bone_name_to_index: Dict[str, int],
    fallback_bone_index: Optional[int] = None,
) -> Tuple[bytes, bytes, int]:
    return new_vertex_skin(
        obj,
        vertex_index,
        bone_name_to_index,
        one_based=True,
        label="v4",
        fallback_bone_index=fallback_bone_index,
    )


def new_v4_uniform_rigid_bone(
    obj: bpy.types.Object,
    vertex_indices,
    bone_name_to_index: Dict[str, int],
) -> Optional[int]:
    """Return Ensemble's rigid-only bone when every rendered vertex uses it exclusively."""
    rigid_bone: Optional[int] = None
    for vertex_index in sorted(set(int(index) for index in vertex_indices)):
        influences: Dict[int, float] = {}
        vertex = obj.data.vertices[vertex_index]
        for group in vertex.groups:
            if group.weight <= 0.00001 or group.group >= len(obj.vertex_groups):
                continue
            bone_index = bone_name_to_index.get(obj.vertex_groups[group.group].name)
            if bone_index is not None:
                influences[bone_index] = influences.get(bone_index, 0.0) + float(group.weight)
        if len(influences) != 1:
            return None
        vertex_bone = next(iter(influences))
        if rigid_bone is None:
            rigid_bone = vertex_bone
        elif rigid_bone != vertex_bone:
            return None
    return rigid_bone


def new_v6_vertex_skin(
    obj: bpy.types.Object,
    vertex_index: int,
    bone_name_to_index: Dict[str, int],
    fallback_bone_index: Optional[int] = None,
) -> Tuple[bytes, bytes, int]:
    return new_vertex_skin(
        obj,
        vertex_index,
        bone_name_to_index,
        one_based=False,
        label="v6",
        fallback_bone_index=fallback_bone_index,
    )


def new_v6_rigid_fallback_bone(obj: bpy.types.Object, bone_name_to_index: Dict[str, int],
                               *, default_bone_index: Optional[int] = None) -> Optional[int]:
    requested = str(obj.get("hwek_ugx_rigid_bone_name", "")).strip()
    if requested in bone_name_to_index:
        return bone_name_to_index[requested]
    parent_bone = str(getattr(obj, "parent_bone", "")).strip()
    if parent_bone in bone_name_to_index:
        return bone_name_to_index[parent_bone]
    if default_bone_index is not None:
        return default_bone_index
    matching_groups = [
        bone_name_to_index[group.name]
        for group in obj.vertex_groups
        if group.name in bone_name_to_index
    ]
    if len(matching_groups) == 1:
        return matching_groups[0]
    lowercase_names = {name.lower(): index for name, index in bone_name_to_index.items()}
    for preferred in ("bone_root", "root", "grannyrootbone"):
        if preferred in lowercase_names:
            return lowercase_names[preferred]
    roots = [
        index
        for name, index in bone_name_to_index.items()
        if name.lower().startswith("grannyrootbone")
    ]
    return roots[0] if roots else (min(bone_name_to_index.values()) if bone_name_to_index else None)
