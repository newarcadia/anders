"""Validate native model, skeleton and instanced-index contracts."""

from __future__ import annotations
from typing import List, Optional, Tuple
import math
import struct
from ..errors import UgxError
from ..formats.binary import PACKED_NULL_OFFSET_64, Reader, read_packed_array_header
from ..formats.ecf import CACHED_DATA_CHUNK, EcfFile
from ..formats.granny import granny_mesh_for_model_index, grx_array_header, grx_cstring, grx_u64, validate_granny_model_layout
from ..formats.granny_layout import validate_granny_layout
from ..formats.matrix import mat4_inverse, mat4_mul
from .bone_data import matrices_close
from .data import (
    CACHED_SIGNATURE_HW1_PC_BYTES,
    UGX_MAX_SECTION_TRIANGLES,
    UgxModel,
    V4_MAX_SKELETON_BONES,
    V6_MAX_SKELETON_BONES,
    ugx_game_name,
)
from .granny import selected_section_required_bone_indices
from .read import triplet_valid
from .skin import vertex_group_from_source_bone
from .transforms import granny_transform_matches_local_rows, parse_matrix_rows
from .vertices import f32le, layout_for, tangent_layout_for

__all__ = ('validate_granny_model_contract', 'validate_game_ready_ugx_model', 'validate_v4_instanced_index_buffer', 'validate_v6_instanced_index_buffer')

def validate_granny_model_contract(
    model: UgxModel,
    game_label: str,
    *,
    require_v4_generated_transforms: bool = False,
    source_model: Optional[UgxModel] = None,
):
    granny = model.granny_info
    if not granny.parsed or not granny.models:
        raise UgxError(f"{game_label} UGX export produced no readable Granny model bindings.")
    if len(granny.bones) != len(model.bones):
        raise UgxError(f"{game_label} UGX cached bones and Granny skeleton have different bone counts.")
    world_rows = [mat4_inverse(bone.matrix) for bone in model.bones]
    for index, (cached_bone, granny_bone) in enumerate(zip(model.bones, granny.bones)):
        if cached_bone.name != granny_bone.name:
            raise UgxError(f"{game_label} UGX cached bone {index} does not match the Granny skeleton.")
        if cached_bone.parent != granny_bone.parent:
            raise UgxError(f"{game_label} UGX cached bone {index} has a different Granny parent.")
        if model.cache_version == 4 and not matrices_close(cached_bone.matrix, granny_bone.inverse_world):
            raise UgxError(f"HWDE v4 cached bone {index} does not match Granny.bt InverseWorld4x4.")
        if model.cache_version == 4 and require_v4_generated_transforms:
            parent_world = world_rows[cached_bone.parent] if 0 <= cached_bone.parent < len(world_rows) and cached_bone.parent != index else None
            local_rows = mat4_mul(world_rows[index], mat4_inverse(parent_world)) if parent_world is not None else world_rows[index]
            if not granny_transform_matches_local_rows(granny_bone.transform, local_rows):
                raise UgxError(f"HWDE v4 Granny.bt bone {index} LocalTransform does not match the cached skeleton.")

    mesh_names = granny.models[0].mesh_names
    if not mesh_names and model.sections:
        raise UgxError(f"{game_label} UGX Granny model has no mesh bindings.")
    cached_bone_names = {bone.name for bone in model.bones}
    all_binding_names = set()
    source_meshes = []
    if source_model is not None and model.cache_version == 6:
        source_info = source_model.granny_info
        source_count = len(source_info.models[0].mesh_names) if source_info.models else len(source_info.meshes)
        source_meshes = [granny_mesh_for_model_index(source_info, index) for index in range(source_count)]
    for mesh_index, mesh_name in enumerate(mesh_names):
        granny_mesh = granny_mesh_for_model_index(granny, mesh_index)
        if granny_mesh is None:
            raise UgxError(f"{game_label} UGX Granny model mesh {mesh_name!r} has no mesh record.")
        binding_names = {binding.bone_name for binding in granny_mesh.bone_bindings}
        all_binding_names.update(binding_names)
        source_unknown = set()
        if source_model is not None and model.cache_version == 6:
            source_mesh = granny_mesh_for_model_index(source_model.granny_info, mesh_index)
            if source_mesh is None or source_mesh.name != mesh_name:
                matches = [mesh for mesh in source_meshes if mesh and mesh.name == mesh_name]
                source_mesh = matches[0] if len(matches) == 1 else None
            if source_mesh:
                source_bone_names = {bone.name for bone in source_model.bones}
                source_unknown = {binding.bone_name for binding in source_mesh.bone_bindings} - source_bone_names
        unknown = sorted(binding_names - cached_bone_names - source_unknown)
        if unknown:
            raise UgxError(f"{game_label} UGX Granny mesh {mesh_name!r} binds unknown bone(s): {unknown[:8]}")
    required_names = {
        model.bones[index].name
        for index in selected_section_required_bone_indices(model, range(len(model.sections)))
    }
    missing = sorted(required_names - all_binding_names)
    if missing and source_model is not None and model.cache_version == 6:
        source_bindings = {binding.bone_name for mesh in source_meshes if mesh for binding in mesh.bone_bindings}
        source_required = {source_model.bones[index].name
                           for index in selected_section_required_bone_indices(source_model, range(len(source_model.sections)))}
        missing = sorted(set(missing) - (source_required - source_bindings))
    if missing:
        raise UgxError(f"{game_label} UGX Granny meshes are missing skin binding(s): {missing[:8]}")
    for accessory_index, accessory in enumerate(model.accessories):
        if model.cache_version == 6 and accessory.bone_count == 0:
            # HW2 mesh accessories can use the model's shared skeleton.
            continue
        skin_sections = [index for index, section in enumerate(model.sections)
                         if section.accessory_index == accessory_index and not section.rigid_only
                         and layout_for(section.layout, "I") is not None and layout_for(section.layout, "W") is not None]
        required_indices = selected_section_required_bone_indices(model, skin_sections)
        outside_range = [
            index for index in required_indices
            if not (accessory.first_bone <= index < accessory.first_bone + accessory.bone_count)
        ]
        if outside_range:
            raise UgxError(
                f"{game_label} UGX accessory {accessory_index} excludes used bone index(es): {outside_range[:8]}"
            )


def validate_game_ready_ugx_model(model: UgxModel, source_model: Optional[UgxModel] = None):
    empty_source = source_model is not None and not source_model.sections and not model.sections
    if not model.sections and not empty_source:
        raise UgxError("UGX export produced no sections.")
    if not model.bones:
        raise UgxError("UGX export produced no bones; Halo Wars unit UGX files need at least GrannyRootBone.")
    if not model.materials and not (empty_source and not source_model.materials):
        raise UgxError("UGX export produced no materials.")
    bone_limit = V4_MAX_SKELETON_BONES if model.cache_version == 4 else V6_MAX_SKELETON_BONES
    if len(model.bones) > bone_limit:
        raise UgxError(f"{ugx_game_name(model.cache_version)} UGX exceeds its {bone_limit}-bone skeleton limit.")
    for index, bone in enumerate(model.bones):
        if bone.parent < -1 or bone.parent >= len(model.bones) or bone.parent == index:
            raise UgxError(f"UGX bone {index} has an invalid parent index {bone.parent}.")
        parse_matrix_rows(bone.matrix, f"UGX bone {index} matrix")
    ecf = EcfFile.parse(model.original_bytes)
    grx_chunk = ecf.chunks.get(0x703)
    if grx_chunk is None:
        raise UgxError("UGX export is missing Granny model chunk 0x703.")
    validate_granny_model_layout(grx_chunk.data)
    validate_granny_layout(grx_chunk.data)
    if grx_cstring(grx_chunk.data, grx_u64(grx_chunk.data, 16)).lower() != 'gr2ugx':
        raise UgxError("UGX Granny source marker must be gr2ugx.")
    model_count, _ = grx_array_header(grx_chunk.data, 96)
    if model_count != 1:
        raise UgxError("UGX Granny file-info must contain exactly one model.")
    cached_chunk = ecf.chunks.get(CACHED_DATA_CHUNK)
    if model.cache_version == 4:
        if cached_chunk is None or cached_chunk.data[:4] != CACHED_SIGNATURE_HW1_PC_BYTES:
            raise UgxError(
                "HWDE v4 cached data has the Xbox/big-endian signature; "
                "the PC loader requires bytes 04 00 34 C2."
            )
    if model.cache_version == 6:
        if cached_chunk is None or len(cached_chunk.data) < 160:
            raise UgxError("HW2 UGX cached-data chunk is too small for the v6 header table.")
        cached_reader = Reader(cached_chunk.data, False)
        cached_reader.seek(64)
        cached_headers = [read_packed_array_header(cached_reader, True) for _ in range(6)]
        sections_header, bones_header, accessories_header, valid_accessories_header, bounds_low_header, bounds_high_header = cached_headers
        empty_sections_valid = empty_source and sections_header.length == 0 and sections_header.offset == PACKED_NULL_OFFSET_64
        if not empty_sections_valid and not triplet_valid(cached_chunk.data, sections_header, bones_header, accessories_header):
            raise UgxError("HW2 UGX v6 cached-array headers are not at the 010 Editor offset 0x40.")
        expected_lengths = (
            len(model.sections),
            len(model.bones),
            len(model.accessories),
            len(model.bones),
            len(model.bones),
        )
        table_headers = (sections_header, bones_header, accessories_header, bounds_low_header, bounds_high_header)
        if tuple(header.length for header in table_headers) != expected_lengths:
            raise UgxError("HW2 UGX v6 cached-array header counts do not match the parsed model tables.")
        if valid_accessories_header.length > len(model.accessories):
            raise UgxError("HW2 UGX has more valid accessory indices than accessory records.")
        if valid_accessories_header.length and valid_accessories_header.offset + valid_accessories_header.length * 4 > len(cached_chunk.data):
            raise UgxError("HW2 UGX valid accessory indices point outside the cached-data chunk.")
        valid_indices = struct.unpack_from(f"<{valid_accessories_header.length}i", cached_chunk.data, valid_accessories_header.offset) if valid_accessories_header.length else ()
        if len(set(valid_indices)) != len(valid_indices) or any(index < 0 or index >= len(model.accessories) for index in valid_indices):
            raise UgxError("HW2 UGX valid accessory indices contain duplicates or point outside the accessory table.")
        for label, header, alignment in (
            ("sections", sections_header, 8),
            ("bones", bones_header, 8),
            ("accessories", accessories_header, 8),
            ("valid accessories", valid_accessories_header, 4),
            ("bone low bounds", bounds_low_header, 4),
            ("bone high bounds", bounds_high_header, 4),
        ):
            if header.length and header.offset % alignment:
                raise UgxError(f"HW2 UGX v6 {label} array is not {alignment}-byte aligned.")
        for index in range(bones_header.length):
            name_offset = struct.unpack_from("<Q", cached_chunk.data, bones_header.offset + index * 80)[0]
            if name_offset & 1:
                raise UgxError(f"HW2 UGX v6 bone {index} name has an invalid odd packed-string offset.")
        for index in range(sections_header.length):
            remap_length, _, remap_offset = struct.unpack_from(
                "<IIQ", cached_chunk.data, sections_header.offset + index * 72 + 56
            )
            if remap_length == 0 and remap_offset != PACKED_NULL_OFFSET_64:
                raise UgxError(f"HW2 UGX v6 section {index} has a noncanonical empty bone-remap pointer.")
        validate_granny_model_contract(model, "HW2", source_model=source_model)
    for index, section in enumerate(model.sections):
        label = f"UGX section {index}"
        if section.vertex_stride <= 0 or section.vertex_buffer_size <= 0:
            raise UgxError(f"{label} has no vertex buffer data.")
        if section.vertex_count <= 0 or not section.vertices:
            raise UgxError(f"{label} has no vertices; Halo Wars DE still submits every section to the renderer.")
        if section.vertex_count != len(section.vertices):
            raise UgxError(f"{label} vertex count points past the vertex buffer.")
        if section.vertex_count * section.vertex_stride > section.vertex_buffer_size:
            raise UgxError(f"{label} vertex buffer is smaller than its declared vertices.")
        if len(section.indices) < 3 or len(section.indices) % 3:
            raise UgxError(f"{label} has no complete triangles; Halo Wars DE still submits every section to the renderer.")
        if model.cache_version == 4 and len(section.indices) // 3 > UGX_MAX_SECTION_TRIANGLES:
            raise UgxError(
                f"{label} exceeds HWDE's {UGX_MAX_SECTION_TRIANGLES:,}-triangle section limit."
            )
        if any(vertex_index < 0 or vertex_index >= len(section.vertices) for vertex_index in section.indices):
            raise UgxError(f"{label} has indices outside its vertex range.")
        if layout_for(section.layout, "P") is None:
            raise UgxError(f"{label} has no position vertex element.")
        if model.cache_version == 6 and tangent_layout_for(section.layout) is not None and layout_for(section.layout, "N") is not None:
            tangent_layout = tangent_layout_for(section.layout)
            normal_layout = layout_for(section.layout, "N")
            raw_vertices = section.source_vertex_data
            if len(raw_vertices) < section.vertex_count * section.vertex_stride:
                raise UgxError(f"{label} has incomplete source vertex bytes for basis validation.")
            for vertex_index in range(section.vertex_count):
                start = vertex_index * section.vertex_stride
                packed_tangent = f32le(raw_vertices, start + tangent_layout.offset)
                packed_normal = f32le(raw_vertices, start + normal_layout.offset)
                if not math.isfinite(packed_tangent) or not math.isfinite(packed_normal):
                    raise UgxError(f"{label} vertex {vertex_index} has a non-finite packed tangent basis.")
                if abs(packed_tangent) >= 1.0 or abs(packed_normal) >= 1.0:
                    raise UgxError(f"{label} vertex {vertex_index} has a packed tangent basis outside HW2's native range.")
        if section.max_bones < 0 or section.max_bones > 4:
            raise UgxError(f"{label} exceeds the four-bone vertex influence limit.")
        for key in ("lod_min_distance", "lod_max_distance", "sorting_bias"):
            if not math.isfinite(float(getattr(section, key))):
                raise UgxError(f"{label} {key} is not finite.")
        skinned = layout_for(section.layout, "I") is not None and layout_for(section.layout, "W") is not None
        for vertex_index, vertex in enumerate(section.vertices):
            if not all(math.isfinite(value) for value in (*vertex.pos, *vertex.uv0, *vertex.uv1, *vertex.uv2)):
                raise UgxError(f"{label} vertex {vertex_index} has non-finite position or UV data.")
            if not skinned:
                continue
            if not any(weight > 0.0 for weight in vertex.bone_weights):
                raise UgxError(f"{label} vertex {vertex_index} has no bone weights.")
            for bone_index, weight in zip(vertex.bone_indices, vertex.bone_weights):
                if not math.isfinite(weight) or weight < 0.0:
                    raise UgxError(f"{label} vertex {vertex_index} has an invalid bone weight.")
                if weight > 0.0 and vertex_group_from_source_bone(section, bone_index, len(model.bones)) is None:
                    raise UgxError(f"{label} vertex {vertex_index} references missing bone {bone_index}.")
        if section.material_index < 0 or section.material_index >= len(model.materials):
            raise UgxError(f"{label} material index points outside the material table.")
        if model.accessories and (section.accessory_index < 0 or section.accessory_index >= len(model.accessories)):
            raise UgxError(f"{label} accessory index points outside the accessory table.")


def validate_v4_instanced_index_buffer(model: UgxModel, cached: bytes, index_buffer: bytes):
    if len(cached) < 58:
        raise UgxError("HWDE v4 cached data is too short for its instancing header.")
    max_instances, instance_index_multiplier = struct.unpack_from("<hh", cached, 48)
    if max_instances <= 0 or instance_index_multiplier <= 0:
        raise UgxError("HWDE v4 cached data advertises invalid instancing values.")
    if max_instances * instance_index_multiplier >= 0xFFFF:
        raise UgxError("HWDE v4 cached data advertises instance indices outside the 16-bit range.")

    occupied_ranges: List[Tuple[int, int]] = []
    for section_index, section in enumerate(model.sections):
        base_indices = section.indices
        if not base_indices:
            continue
        if any(index >= instance_index_multiplier for index in base_indices):
            raise UgxError(
                f"HWDE v4 section {section_index} has a base index outside its instance multiplier."
            )
        instance_bytes = len(base_indices) * 2
        required_end = section.index_offset + instance_bytes * max_instances
        if section.index_offset < 0 or required_end > len(index_buffer):
            raise UgxError(
                f"HWDE v4 section {section_index} advertises {max_instances} instances but its "
                f"index buffer ends at {len(index_buffer)} bytes; at least {required_end} bytes are required."
            )
        for previous_start, previous_end in occupied_ranges:
            if section.index_offset < previous_end and required_end > previous_start:
                raise UgxError(f"HWDE v4 section {section_index} overlaps another instanced index range.")
        occupied_ranges.append((section.index_offset, required_end))

        for instance_index in range(max_instances):
            block_offset = section.index_offset + instance_index * instance_bytes
            actual = struct.unpack_from(f"<{len(base_indices)}H", index_buffer, block_offset)
            instance_offset = instance_index_multiplier * instance_index
            expected = tuple(index + instance_offset for index in base_indices)
            if actual != expected:
                raise UgxError(
                    f"HWDE v4 section {section_index} instance {instance_index} index block "
                    "does not match Ensemble's instanced-index encoding."
                )


def validate_v6_instanced_index_buffer(model: UgxModel, cached: bytes, index_buffer: bytes):
    if len(cached) < 64:
        raise UgxError("HW2 v6 cached data is too short for its instancing header.")
    max_instances, instance_index_multiplier = struct.unpack_from("<hH", cached, 48)
    if max_instances <= 0 or instance_index_multiplier <= 0:
        raise UgxError("HW2 v6 cached data advertises invalid instancing values.")
    if max_instances * instance_index_multiplier >= 0xFFFF:
        raise UgxError("HW2 v6 cached data advertises instance indices outside the 16-bit range.")

    occupied_ranges: List[Tuple[int, int]] = []
    for section_index, section in enumerate(model.sections):
        base_indices = section.indices
        if not base_indices:
            continue
        if any(index >= instance_index_multiplier for index in base_indices):
            raise UgxError(f"HW2 v6 section {section_index} has a base index outside its instance multiplier.")
        instance_bytes = len(base_indices) * 2
        required_end = section.index_offset + instance_bytes * max_instances
        if section.index_offset < 0 or required_end > len(index_buffer):
            raise UgxError(
                f"HW2 v6 section {section_index} advertises {max_instances} instances but its "
                f"index buffer ends at {len(index_buffer)} bytes; at least {required_end} bytes are required."
            )
        for previous_start, previous_end in occupied_ranges:
            if section.index_offset < previous_end and required_end > previous_start:
                raise UgxError(f"HW2 v6 section {section_index} overlaps another instanced index range.")
        occupied_ranges.append((section.index_offset, required_end))

        for instance_index in range(max_instances):
            block_offset = section.index_offset + instance_index * instance_bytes
            actual = struct.unpack_from(f"<{len(base_indices)}H", index_buffer, block_offset)
            instance_offset = instance_index_multiplier * instance_index
            expected = tuple(index + instance_offset for index in base_indices)
            if actual != expected:
                raise UgxError(
                    f"HW2 v6 section {section_index} instance {instance_index} index block "
                    "does not match Ensemble's instanced-index encoding."
                )
