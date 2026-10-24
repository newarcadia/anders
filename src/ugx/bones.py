"""Prepare authored skeletons and preserve imported bone metadata."""

from typing import Any, Dict, List, Optional, Tuple
import bpy
import copy
import json
import struct
from ..errors import UgxError
from ..formats.matrix import mat4_inverse, mat4_mul, uax_matrix4_identity
from .coordinates import blender_to_game_vec3
from .granny import estimate_granny_lod_errors
from .mesh import mesh_int_attribute
from .metadata import lod_section_indices, section_entry_source_index, ugx_source_bytes_for_export
from .read import parse_ugx_data
from .scene import ugx_lod_sibling_objects_for, ugx_ordered_armature_bones
from .transforms import (
    apply_game_delta_rows,
    bone_head_export_position,
    bone_rest_display_rows,
    game_orientation_delta_rows,
    native_bone_export_matrix,
    parse_matrix_rows,
    pose_basis_delta_rows,
)
from .vertices import changed

__all__ = ('new_v4_armature_for', 'ugx_uses_complete_imported_armature_metadata', 'ugx_should_export_as_new', 'new_export_bone_name', 'new_export_bone_matrix_rows', 'new_v4_export_bones', 'normalize_unused_v4_damage_root', 'granny_lod_ordered_bones', 'imported_v6_granny_metadata', 'new_v6_export_bones')

def new_v4_armature_for(obj: bpy.types.Object) -> Optional[bpy.types.Object]:
    for modifier in getattr(obj, "modifiers", []):
        if getattr(modifier, "type", "") == "ARMATURE" and getattr(modifier, "object", None) is not None:
            return modifier.object
    parent = getattr(obj, "parent", None)
    if parent is not None and getattr(parent, "type", "") == "ARMATURE":
        return parent
    return None


def ugx_uses_complete_imported_armature_metadata(obj: bpy.types.Object) -> bool:
    armature = new_v4_armature_for(obj)
    if armature is None:
        return True
    bones = list(getattr(getattr(armature, "data", None), "bones", []))
    if not bones:
        return False
    required = (
        "hwek_ugx_bone_index",
        "hwek_ugx_source_matrix",
        "hwek_ugx_source_display_matrix",
        "hwek_ugx_import_display_matrix",
    )
    return all(all(key in bone for key in required) for bone in bones)


def ugx_should_export_as_new(obj: bpy.types.Object, cache_version: Optional[int]) -> bool:
    if cache_version not in (4, 6) or obj is None or obj.type != "MESH" or not obj.get("hwek_ugx_source_path"):
        return False
    armature = new_v4_armature_for(obj)
    same_import = (armature is not None and obj.get("hwek_ugx_import_id")
                   and armature.get("hwek_ugx_import_id") == obj.get("hwek_ugx_import_id"))
    if not same_import and not ugx_uses_complete_imported_armature_metadata(obj):
        return True
    if cache_version == 4:
        # Preserve the native layout when all imported sections remain present.
        # Retained or joined parts still need a new section table.
        objects = ugx_lod_sibling_objects_for(obj)
        section_count = int(obj.get("hwek_ugx_section_count", 0))
        owned_sections = {index for item in objects for index in lod_section_indices(item)}
        if section_count > 0 and owned_sections != set(range(section_count)):
            return True
        present_sections = set()
        for item in objects:
            section_attr = mesh_int_attribute(item.data, "UGX_Section")
            if section_attr is not None:
                present_sections.update(section_attr[poly.index] for poly in item.data.polygons)
        for item in objects:
            entries = json.loads(item.get("hwek_ugx_sections", "[]"))
            if any(
                int(entry.get("face_count", 0)) > 0
                and section_entry_source_index(index, entry) not in present_sections
                for index, entry in enumerate(entries)
            ):
                return True
    return False


def new_export_bone_name(bone: bpy.types.Bone, index: int) -> str:
    source_name = str(bone.get("hwek_ugx_source_name", bone.name))
    blender_name = str(bone.get("hwek_ugx_blender_name", bone.name))
    value = str(bone.get("hwek_ugx_export_name", source_name))
    if value == source_name and bone.name != blender_name:
        value = bone.name
    return value


def new_export_bone_matrix_rows(
    armature: bpy.types.Object,
    bone: bpy.types.Bone,
    index: int,
    ordered_bones: List[bpy.types.Bone],
) -> List[List[float]]:
    source_matrix = bone.get("hwek_ugx_source_matrix")
    source_display = bone.get("hwek_ugx_source_display_matrix")
    import_display = bone.get("hwek_ugx_import_display_matrix", source_display)
    if not source_matrix or not source_display or not import_display:
        return native_bone_export_matrix(armature,bone)

    source_rows = parse_matrix_rows(source_matrix, f"Bone {index} source matrix")
    if bone.get('hwek_ugx_skeleton_transform_mode') == 'inverse_world':
        return native_bone_export_matrix(armature,bone,source_rows)
    source_display_rows = parse_matrix_rows(source_display, f"Bone {index} display matrix")
    import_display_rows = parse_matrix_rows(import_display, f"Bone {index} import display matrix")
    display_rows = bone_rest_display_rows(armature, bone)
    target_rows = None
    export_matrix = bone.get("hwek_ugx_export_matrix")
    if export_matrix:
        explicit_rows = parse_matrix_rows(export_matrix, f"Bone {index} export matrix")
        if explicit_rows != source_rows:
            target_rows = explicit_rows

    rest_changed = target_rows is None and any(
        changed(display_rows[row][:3], import_display_rows[row][:3])
        for row in range(3)
    )
    current_head = bone_head_export_position(armature, bone)
    original_head_value = bone.get("hwek_ugx_source_head_local", current_head)
    original_head = tuple(float(value) for value in original_head_value[:3]) if len(original_head_value) >= 3 else current_head
    head_changed = changed(current_head, original_head)
    pose_bone = armature.pose.bones.get(bone.name) if armature.pose else None
    pose_delta = None if target_rows is not None else pose_basis_delta_rows(pose_bone)
    if target_rows is None and not (rest_changed or head_changed or pose_delta):
        return source_rows
    if target_rows is not None:
        return target_rows

    rest_delta = game_orientation_delta_rows(display_rows, import_display_rows) if rest_changed else None
    target_display_rows = apply_game_delta_rows(source_display_rows, rest_delta) if rest_delta else [row[:] for row in source_display_rows]
    if pose_delta:
        target_display_rows = apply_game_delta_rows(target_display_rows, pose_delta)
    if head_changed:
        target_display_rows[3][0], target_display_rows[3][1], target_display_rows[3][2] = blender_to_game_vec3(current_head)
    if bone.get("hwek_ugx_skeleton_transform_mode") == "inverse_world":
        return mat4_inverse(target_display_rows)
    parent = bone.parent
    if parent is not None and parent in ordered_bones:
        return mat4_mul(target_display_rows, mat4_inverse(bone_rest_display_rows(armature, parent)))
    return target_display_rows


def new_v4_export_bones(
    obj: bpy.types.Object,
    axis_rows: Optional[List[List[float]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    armature = new_v4_armature_for(obj)
    if armature is None:
        return [{
            "name": "GrannyRootBone",
            "parent": -1,
            "matrix": uax_matrix4_identity(),
        }], {"GrannyRootBone": 0}
    data_bones = ugx_ordered_armature_bones(armature)
    if not data_bones:
        raise UgxError("Skinned UGX export found an armature with no bones.")
    if len(data_bones) > 255:
        raise UgxError("Skinned UGX export supports up to 255 bones because skin indices are bytes.")
    index_by_name = {bone.name: index for index, bone in enumerate(data_bones)}
    bones: List[Dict[str, Any]] = []
    for index, bone in enumerate(data_bones):
        parent_index = index_by_name.get(bone.parent.name, -1) if bone.parent else -1
        inverse_world = new_export_bone_matrix_rows(armature, bone, index, data_bones)
        if axis_rows is not None:
            # Retail UAX tracks contain absolute local transforms and no source
            # model for Granny rebasing. Pre-multiply every inverse bind by the
            # same model-space correction. This rotates the skinned model as a
            # whole while leaving every child bone's local basis unchanged.
            inverse_world = mat4_mul(axis_rows, inverse_world)
        bones.append({
            "name": new_export_bone_name(bone, index),
            "parent": parent_index,
            "matrix": inverse_world,
        })
    return bones, index_by_name


def normalize_unused_v4_damage_root(
    objects: List[bpy.types.Object],
    bones: List[Dict[str, Any]],
    bone_name_to_index: Dict[str, int],
):
    """Keep a multipart damage model's reserved root at the model origin."""
    if len(objects) <= 1 or not bones:
        return
    root = bones[0]
    if int(root.get("parent", -1)) >= 0:
        raise UgxError("Multipart HWDE exports need bone 0 to be the Granny root bone.")
    root_world = mat4_inverse(root.get("matrix", uax_matrix4_identity()))
    root_position = tuple(float(value) for value in root_world[3][:3])
    if max(abs(value) for value in root_position) <= 0.0001:
        return

    root_group_names = {
        name for name, index in bone_name_to_index.items()
        if int(index) == 0
    }
    root_is_weighted = any(
        group.weight > 0.00001
        and 0 <= group.group < len(obj.vertex_groups)
        and obj.vertex_groups[group.group].name in root_group_names
        for obj in objects
        for vertex in obj.data.vertices
        for group in vertex.groups
    )
    if root_is_weighted:
        raise UgxError(
            "Multipart HWDE export cannot move the weighted Granny root bone to the origin. "
            "Move bone 0 to the armature origin before exporting."
        )

    root_world[3][0] = 0.0
    root_world[3][1] = 0.0
    root_world[3][2] = 0.0
    root["matrix"] = mat4_inverse(root_world)


def granny_lod_ordered_bones(
    bones: List[Dict[str, Any]],
    source_indices: Dict[str, int],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    errors = estimate_granny_lod_errors(bones)
    if len(bones) <= 1:
        if bones:
            bones[0]["lod_error"] = float(errors[0])
        return bones, source_indices

    # Granny first writes a parent-before-child traversal, then performs a
    # stable descending LOD sort. The estimator's heap property guarantees
    # parents stay before children when their errors tie.
    children: Dict[int, List[int]] = {index: [] for index in range(len(bones))}
    roots: List[int] = []
    for index, bone in enumerate(bones):
        parent = int(bone.get("parent", -1))
        if 0 <= parent < len(bones) and parent != index:
            children[parent].append(index)
        else:
            roots.append(index)

    parent_first: List[int] = []
    seen = set()

    def append_subtree(index: int):
        if index in seen:
            return
        seen.add(index)
        parent_first.append(index)
        for child in children[index]:
            append_subtree(child)

    for root in roots:
        append_subtree(root)
    for index in range(len(bones)):
        append_subtree(index)
    order = sorted(parent_first, key=lambda index: -errors[index])
    old_to_new = {old_index: new_index for new_index, old_index in enumerate(order)}
    ordered: List[Dict[str, Any]] = []
    for old_index in order:
        bone = copy.deepcopy(bones[old_index])
        old_parent = int(bone.get("parent", -1))
        bone["parent"] = old_to_new.get(old_parent, -1)
        bone["lod_error"] = float(errors[old_index])
        ordered.append(bone)
    remapped_indices = {
        name: old_to_new[old_index]
        for name, old_index in source_indices.items()
    }
    return ordered, remapped_indices


def imported_v6_granny_metadata(
    obj: bpy.types.Object,
    bones: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Recover animation-compatible Granny order/LOD data for an imported rig."""
    armature = new_v4_armature_for(obj)
    carriers = [candidate for candidate in (armature, obj) if candidate is not None]
    for carrier in carriers:
        if not carrier.get("hwek_ugx_source_path"):
            continue
        try:
            source_bytes, _ = ugx_source_bytes_for_export(carrier)
            source_model = parse_ugx_data(str(carrier.get("hwek_ugx_source_path", "")), source_bytes)
        except (OSError, UgxError, ValueError, struct.error):
            continue
        if source_model.cache_version != 6 or not source_model.granny_info.parsed:
            continue
        if len(source_model.bones) != len(bones) or len(source_model.granny_info.bones) != len(bones):
            continue
        compatible = all(
            str(bone.get("name", "")) == source_model.bones[index].name
            and int(bone.get("parent", -1)) == int(source_model.bones[index].parent)
            and source_model.granny_info.bones[index].name == source_model.bones[index].name
            and int(source_model.granny_info.bones[index].parent) == int(source_model.bones[index].parent)
            for index, bone in enumerate(bones)
        )
        if compatible:
            model_name = source_model.granny_info.models[0].name if source_model.granny_info.models else source_model.granny_info.skeleton_name
            if armature is not None:
                model_name = str(armature.get('hwek_ugx_granny_model_name', model_name))
            return {
                "from_file_name": source_model.granny_info.from_file_name,
                "model_name": model_name,
                "skeleton_name": source_model.granny_info.skeleton_name,
                "skeleton_lod_type": int(source_model.granny_info.skeleton_lod_type),
                "lod_errors": [float(bone.lod_error) for bone in source_model.granny_info.bones],
                "preserved_source": True,
            }

    # New imports persist the same data directly on the armature so a moved or
    # unavailable source file does not break the authoring workflow.
    if (armature is not None and int(armature.get('hwek_ugx_cache_version', 0)) == 6
            and "hwek_ugx_granny_skeleton_name" in armature and "hwek_ugx_granny_lod_type" in armature):
        data_bones = ugx_ordered_armature_bones(armature)
        if len(data_bones) == len(bones) and all("hwek_ugx_granny_lod_error" in bone for bone in data_bones):
            compatible = all(
                new_export_bone_name(data_bone, index) == str(bones[index].get("name", ""))
                for index, data_bone in enumerate(data_bones)
            )
            if compatible:
                return {
                    "from_file_name": str(armature.get("hwek_ugx_granny_from_file_name", "gr2ugx")),
                    "model_name": str(armature.get("hwek_ugx_granny_model_name", armature.get("hwek_ugx_granny_skeleton_name", bones[0]["name"]))),
                    "skeleton_name": str(armature.get("hwek_ugx_granny_skeleton_name", bones[0]["name"])),
                    "skeleton_lod_type": int(armature.get("hwek_ugx_granny_lod_type", 0)),
                    "lod_errors": [float(bone["hwek_ugx_granny_lod_error"]) for bone in data_bones],
                    "preserved_source": True,
                }
    return None


def new_v6_export_bones(
    obj: bpy.types.Object,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, Any]]:
    bones, source_indices = new_v4_export_bones(obj)
    metadata = imported_v6_granny_metadata(obj, bones)
    if metadata is not None:
        for bone, lod_error in zip(bones, metadata["lod_errors"]):
            bone["lod_error"] = float(lod_error)
        metadata['bone_indices'] = source_indices
        return bones, source_indices, metadata

    # Animation track groups address the model name, independently of the
    # Blender object label and the reserved root added to custom HW2 rigs.
    model_name = bones[0]["name"] if bones else "GrannyRootBone"
    armature = new_v4_armature_for(obj)
    if armature is not None:
        model_name = str(armature.get("hwek_ugx_granny_model_name", model_name))
    if bones and not str(bones[0].get("name", "")).lower().startswith("grannyrootbone"):
        # Shipped HW2 skinned UGXs reserve cached index zero for an identity
        # Granny root. gr2ugx's +1 conversion is relative to the authored
        # skeleton below that root; retail vertex bytes address the resulting
        # cached table directly. Synthesize it for a completely custom rig.
        # Native bone searches also include a case-insensitive lookup.
        names = {bone["name"].lower() for bone in bones}
        root_name = "GrannyRootBone"
        suffix = 1
        while root_name.lower() in names:
            root_name = f"GrannyRootBone_{suffix}"
            suffix += 1
        rooted_bones: List[Dict[str, Any]] = [{
            "name": root_name,
            "parent": -1,
            "matrix": uax_matrix4_identity(),
        }]
        for bone in bones:
            rooted = copy.deepcopy(bone)
            parent = int(rooted.get("parent", -1))
            rooted["parent"] = parent + 1 if parent >= 0 else 0
            rooted_bones.append(rooted)
        bones = rooted_bones
        source_indices = {name: index + 1 for name, index in source_indices.items()}

    bones, source_indices = granny_lod_ordered_bones(bones, source_indices)
    metadata = {
        "from_file_name": "gr2ugx",
        "model_name": model_name,
        "skeleton_name": bones[0]["name"] if bones else "GrannyRootBone",
        "skeleton_lod_type": 1,
        "lod_errors": [float(bone.get("lod_error", -1.0)) for bone in bones],
        "preserved_source": False,
        "bone_indices": source_indices,
    }
    return bones, source_indices, metadata
