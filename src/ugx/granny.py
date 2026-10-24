"""Build Granny skeletons, mesh bindings and bone-local bounds."""

from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple
import copy
import struct
from ..formats.granny import (
    GRX_BONE_BINDING_SIZE,
    GRX_BONE_SIZE,
    GRX_MESH_SIZE,
    GRX_MODEL_SIZE,
    GRX_ROOT_SIZE,
    GrannyBoneBindingInfo,
    GrannyBoneInfo,
    GrannyMeshInfo,
    granny_mesh_for_model_index,
    GrxWriter,
    identity_grx_transform,
    safe_grx_name,
    write_grx_array_header,
    write_grx_transform,
    write_grx_type_schema,
)
from ..formats.matrix import mat4_inverse, mat4_mul
from .data import Accessory, Section, UgxModel, is_no_rigid_bone_index
from .skeleton_lod import estimate_granny_lod_errors
from .skin import vertex_group_from_section_bone, vertex_group_from_source_bone
from .transforms import granny_transform_from_local_rows

__all__ = ('selected_section_bounds', 'selected_section_required_bone_indices', 'authored_granny_meshes', 'transform_model_point_to_bone', 'accessory_bone_local_bounds', 'granny_meshes_for_model', 'estimate_granny_lod_errors', 'granny_bones_for_model', 'build_granny_info_chunk')

def selected_section_bounds(
    sections: List[Section],
    section_indices: Iterable[int],
) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    selected = {int(index) for index in section_indices}
    positions = [vertex.pos for index, section in enumerate(sections) if index in selected for vertex in section.vertices]
    if not positions:
        return None
    return (
        tuple(min(position[axis] for position in positions) for axis in range(3)),
        tuple(max(position[axis] for position in positions) for axis in range(3)),
    )


def selected_section_required_bone_indices(model: UgxModel, section_indices: Iterable[int]) -> List[int]:
    selected = {int(index) for index in section_indices}
    required = set()
    for section_index, section in enumerate(model.sections):
        if section_index not in selected:
            continue
        if not is_no_rigid_bone_index(section.rigid_bone_index):
            required.add(int(section.rigid_bone_index))
        for vertex in section.vertices:
            for local_bone, weight in zip(vertex.bone_indices, vertex.bone_weights):
                if weight <= 0.00001:
                    continue
                bone_index = vertex_group_from_source_bone(section, int(local_bone), len(model.bones))
                if bone_index is not None:
                    required.add(bone_index)
    return sorted(index for index in required if 0 <= index < len(model.bones))


def authored_granny_meshes(
    model: UgxModel,
    mesh_names: List[str],
    mesh_section_indices: List[List[int]],
) -> List[GrannyMeshInfo]:
    meshes: List[GrannyMeshInfo] = []
    for mesh_index, name in enumerate(mesh_names):
        section_indices = mesh_section_indices[mesh_index] if mesh_index < len(mesh_section_indices) else []
        bounds = section_bone_local_bounds(model, section_indices)
        required = selected_section_required_bone_indices(model, section_indices)
        if not required and model.bones:
            required = [0]
        bindings = [
            GrannyBoneBindingInfo(model.bones[bone_index].name,
                                  bounds.get(bone_index, ((0., 0., 0.), (0., 0., 0.)))[0],
                                  bounds.get(bone_index, ((0., 0., 0.), (0., 0., 0.)))[1])
            for bone_index in required
        ]
        meshes.append(GrannyMeshInfo(safe_grx_name(name, ""), bindings))
    return meshes


def transform_model_point_to_bone(
    position: Tuple[float, float, float],
    inverse_world: List[List[float]],
) -> Tuple[float, float, float]:
    value = (float(position[0]), float(position[1]), float(position[2]), 1.0)
    return tuple(
        sum(value[row] * float(inverse_world[row][column]) for row in range(4))
        for column in range(3)
    )


def accessory_bone_local_bounds(
    model: UgxModel,
    accessory_index: int,
) -> Dict[int, Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """Build Granny bone-binding boxes in each binding bone's local space."""
    if 0 <= accessory_index < len(model.accessories):
        section_indices = list(model.accessories[accessory_index].section_indices)
    else:
        section_indices = [
            index for index, section in enumerate(model.sections)
            if section.accessory_index == accessory_index
        ]
    return section_bone_local_bounds(model, section_indices)


def section_bone_local_bounds(model, section_indices):
    lows: Dict[int, List[float]] = {}
    highs: Dict[int, List[float]] = {}
    first_by_name = {}
    for index, bone in enumerate(model.bones):
        first_by_name.setdefault(bone.name, index)

    def expand(bone_index: int, position: Tuple[float, float, float]):
        if not (0 <= bone_index < len(model.bones)):
            return
        # Granny mesh bindings resolve names to their first skeleton match.
        bound = first_by_name[model.bones[bone_index].name]
        local = transform_model_point_to_bone(position, model.bones[bound].matrix)
        if bone_index not in lows:
            lows[bone_index] = list(local)
            highs[bone_index] = list(local)
            return
        for axis in range(3):
            lows[bone_index][axis] = min(lows[bone_index][axis], local[axis])
            highs[bone_index][axis] = max(highs[bone_index][axis], local[axis])

    for section_index in section_indices:
        if not (0 <= section_index < len(model.sections)):
            continue
        section = model.sections[section_index]
        if section.rigid_only:
            bone_index = vertex_group_from_section_bone(section, section.rigid_bone_index, len(model.bones))
            if bone_index is None and model.bones:
                bone_index = 0
            if bone_index is not None:
                for vertex in section.vertices:
                    expand(bone_index, vertex.pos)
            continue
        for vertex in section.vertices:
            for encoded_index, weight in zip(vertex.bone_indices, vertex.bone_weights):
                if float(weight) < (1.0 / 64.0):
                    continue
                bone_index = vertex_group_from_source_bone(section, int(encoded_index), len(model.bones))
                if bone_index is not None:
                    expand(bone_index, vertex.pos)

    return {
        bone_index: (tuple(low), tuple(highs[bone_index]))
        for bone_index, low in lows.items()
    }


def granny_meshes_for_model(model: UgxModel, object_name: str = "") -> List[GrannyMeshInfo]:
    existing = model.granny_info.meshes
    bound_names = model.granny_info.models[0].mesh_names if model.granny_info.models else []
    meshes: List[GrannyMeshInfo] = []
    accessories = model.accessories or [Accessory(0, max(1, len(model.bones)))]
    mesh_count = len(bound_names) if model.granny_info.models else (len(existing) or len(accessories))
    for index in range(mesh_count):
        accessory = accessories[index] if index < len(accessories) else accessories[0]
        name = bound_names[index] if index < len(bound_names) else (existing[index].name if index < len(existing) else (object_name or f"mesh_{index}"))
        existing_mesh = granny_mesh_for_model_index(model.granny_info, index)
        if existing_mesh and existing_mesh.bone_bindings:
            bindings = [copy.deepcopy(binding) for binding in existing_mesh.bone_bindings]
        else:
            local_bounds = accessory_bone_local_bounds(model, index)
            required = sorted(local_bounds)
            if not required:
                required = [accessory.first_bone if 0 <= accessory.first_bone < len(model.bones) else 0]
            bindings = [
                GrannyBoneBindingInfo(
                    model.bones[bone_index].name if model.bones else "GrannyRootBone",
                    local_bounds.get(bone_index, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))[0],
                    local_bounds.get(bone_index, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))[1],
                )
                for bone_index in required
            ]
        meshes.append(GrannyMeshInfo(safe_grx_name(name, ""), bindings))
    return meshes


def granny_bones_for_model(model: UgxModel) -> List[GrannyBoneInfo]:
    source_bones = model.granny_info.bones if model.granny_info.parsed else []
    existing_by_name = {}
    for bone in source_bones:
        existing_by_name.setdefault(bone.name, bone)
    world_rows = [mat4_inverse(bone.matrix) for bone in model.bones]
    sources = []
    for index, bone in enumerate(model.bones):
        existing = existing_by_name.get(bone.name)
        if index < len(source_bones) and source_bones[index].name == bone.name and source_bones[index].parent == bone.parent:
            existing = source_bones[index]
        sources.append(existing)
    use_lod = model.granny_info.skeleton_lod_type != 0 if model.granny_info.parsed else model.cache_version == 6
    errors = [existing.lod_error if existing else (-1.0 if use_lod else 0.0) for existing in sources]
    if any(error < 0.0 for error in errors):
        errors = estimate_granny_lod_errors(model.bones, errors)
    bones = []
    for index, (bone, existing) in enumerate(zip(model.bones, sources)):
        parent_world = world_rows[bone.parent] if 0 <= bone.parent < len(world_rows) and bone.parent != index else None
        local_rows = mat4_mul(world_rows[index], mat4_inverse(parent_world)) if parent_world is not None else world_rows[index]
        bones.append(GrannyBoneInfo(
            safe_grx_name(bone.name, ""),
            bone.parent,
            existing.transform if existing else granny_transform_from_local_rows(local_rows),
            [row[:] for row in bone.matrix],
            errors[index],
        ))
    if not bones:
        bones.append(GrannyBoneInfo("GrannyRootBone", -1))
    return bones


def build_granny_info_chunk(model: UgxModel, object_name: str = "") -> bytes:
    writer = GrxWriter()
    root_offset = writer.reserve(GRX_ROOT_SIZE, 1)
    writer.align(16)

    bones = granny_bones_for_model(model)
    skeleton_name = safe_grx_name(model.granny_info.skeleton_name if model.granny_info.parsed else bones[0].name, "")
    meshes = granny_meshes_for_model(model, object_name)
    model_name = safe_grx_name(model.granny_info.models[0].name if model.granny_info.models else skeleton_name, "")

    skeleton_refs_offset = writer.reserve(8, 16)
    skeleton_offset = writer.reserve(40, 16)
    struct.pack_into("<Q", writer.data, skeleton_refs_offset, skeleton_offset)
    writer.string_ref(skeleton_offset, skeleton_name)
    writer.align(32)
    bones_offset = len(writer.data)
    for bone in bones:
        offset = len(writer.data)
        writer.data.extend(b"\0" * GRX_BONE_SIZE)
        writer.string_ref(offset, bone.name)
        struct.pack_into("<i", writer.data, offset + 8, int(bone.parent))
        write_grx_transform(writer.data, offset + 12, bone.transform)
        for row in range(4):
            struct.pack_into("<4f", writer.data, offset + 80 + row * 16, *bone.inverse_world[row])
        struct.pack_into("<f", writer.data, offset + 144, float(bone.lod_error))
    write_grx_array_header(writer.data, skeleton_offset + 8, len(bones), bones_offset)
    # HW2 v6 uses Granny EstimatedLOD. Fresh v6 bones receive the same
    # hierarchy-based error estimate as Granny's skeleton builder. Fresh HWDE
    # v4 armatures have no calculated LOD table and use NoSkeletonLOD instead.
    skeleton_lod_type = (
        int(model.granny_info.skeleton_lod_type)
        if model.granny_info.parsed
        else (1 if model.cache_version == 6 else 0)
    )
    struct.pack_into("<i", writer.data, skeleton_offset + 20, skeleton_lod_type)

    mesh_refs_offset = writer.reserve(len(meshes) * 8, 16)
    mesh_offsets = []
    for mesh in meshes:
        mesh_offset = writer.reserve(GRX_MESH_SIZE, 16)
        mesh_offsets.append(mesh_offset)
        writer.string_ref(mesh_offset, mesh.name)
        binding_offset = writer.reserve(len(mesh.bone_bindings) * GRX_BONE_BINDING_SIZE, 16)
        for binding_index, binding in enumerate(mesh.bone_bindings):
            offset = binding_offset + binding_index * GRX_BONE_BINDING_SIZE
            writer.string_ref(offset, binding.bone_name)
            struct.pack_into("<3f3f", writer.data, offset + 8, *binding.bounds_min, *binding.bounds_max)
        write_grx_array_header(writer.data, mesh_offset + 48, len(mesh.bone_bindings), binding_offset)
    for index, mesh_offset in enumerate(mesh_offsets):
        struct.pack_into("<Q", writer.data, mesh_refs_offset + index * 8, mesh_offset)

    model_refs_offset = writer.reserve(8, 16)
    model_offset = writer.reserve(GRX_MODEL_SIZE, 16)
    struct.pack_into("<Q", writer.data, model_refs_offset, model_offset)
    writer.string_ref(model_offset, model_name)
    struct.pack_into("<Q", writer.data, model_offset + 8, skeleton_offset)
    placement = model.granny_info.models[0].initial_placement if model.granny_info.models else identity_grx_transform()
    write_grx_transform(writer.data, model_offset + 16, placement)
    # The model includes a 16-byte ExtendedData variant after MeshBindings.
    # Its pointers must stay clear of the following mesh-reference array.
    mesh_binding_offset = writer.reserve(len(mesh_offsets) * 8, 16)
    for index, mesh_offset in enumerate(mesh_offsets):
        struct.pack_into("<Q", writer.data, mesh_binding_offset + index * 8, mesh_offset)
    write_grx_array_header(writer.data, model_offset + 84, len(mesh_offsets), mesh_binding_offset)

    write_grx_type_schema(writer)
    writer.string_ref(root_offset + 16, model.granny_info.from_file_name if model.granny_info.parsed else "gr2ugx")
    write_grx_array_header(writer.data, root_offset + 48, 1, skeleton_refs_offset)
    write_grx_array_header(writer.data, root_offset + 84, len(mesh_offsets), mesh_refs_offset)
    write_grx_array_header(writer.data, root_offset + 96, 1, model_refs_offset)
    writer.finish_strings()
    return bytes(writer.data)
