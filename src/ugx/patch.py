"""Patch retained UGX geometry, sections, bone transforms and bounds."""

from typing import Any, Dict, List, Optional, Tuple
import bpy
import copy
import json
import math
import struct
import numpy as np
from ..errors import UgxError
from ..formats.binary import (
    PACKED_NULL_OFFSET_64,
    align4,
    align_data,
    packed_array_header,
    write_packed_offset,
)
from ..formats.ecf import (
    CACHED_DATA_CHUNK,
    INDEX_BUFFER_CHUNK,
    VERTEX_BUFFER_CHUNK,
    ecf_chunk_records,
    replace_ecf_chunk,
)
from ..formats.granny import safe_grx_name
from ..formats.matrix import mat4_inverse, mat4_mul
from ..materials.context import ugx_material_index_for_slot
from ..materials.data import parse_bool_value
from .coordinates import blender_to_game_vec3, reverse_triangle_winding
from .data import (
    CachedBounds,
    Section,
    UV_SEMANTICS,
    UgxModel,
    V4_MAX_SECTION_VERTICES,
    V6_PACKED_BASIS_TYPE,
    source_vertex_section_ranges,
    v4_instance_settings,
    v6_instance_settings,
)
from .granny import build_granny_info_chunk
from .mesh import (
    export_vector_for_edited_position,
    mesh_bitangent_signs_by_vertex,
    mesh_colors_by_vertex,
    mesh_colors_by_loop,
    mesh_calculated_basis_by_loop_world,
    mesh_calculated_normals_by_vertex_world,
    mesh_calculated_tangents_by_vertex_world,
    mesh_connected_vertices,
    mesh_has_split_uv,
    mesh_int_attribute,
    mesh_normals_by_vertex_world,
    mesh_position_changed_vertices,
    mesh_skin_map,
    vertex_group_bone_indices,
    mesh_tangents_by_vertex_world,
    mesh_uv_by_loop,
    mesh_uv_by_vertex,
    mesh_uv_data,
    object_vertex_game_position,
)
from .metadata import bone_bound_property_values, section_metadata_value
from .scene import ugx_armature_for, ugx_ordered_armature_bones
from .snapshots import mesh_buffer, mesh_game_positions
from .model_space import ugx_model_matrix
from .skin import encode_skin, importable_source_skin_map, skin_changed, source_skin_map
from .transforms import (
    apply_game_delta_rows,
    blender_matrix_to_game_rows,
    bone_head_export_position,
    bone_rest_display_rows,
    game_orientation_delta_rows,
    matrix_changed,
    parse_matrix_rows,
    parse_vec3,
    pose_basis_delta_rows,
    write_matrix_rows,
)
from .vertex_authoring import orthonormal_tangent_tuple, tangent_for_game_normal, v4_tangent_with_handedness
from .vertices import (
    changed,
    encode_vertex_color,
    layout_for,
    source_tangent_bits,
    section_position_changed,
    tangent_layout_for,
    write_f32_endian,
    write_i32,
    write_normal_vec3,
    write_vec2,
    write_vec3,
    write_vec3_preserve_dec3n_bits,
)

__all__ = ('patch_cached_bone_names', 'same_topology', 'validate_same_topology', 'mesh_positions_changed', 'patch_mesh_vertices', 'patch_position_edits', 'encode_rebuilt_vertex', 'source_template_for_section', 'rebuild_topology_buffers', 'rebuild_v6_structural_chunks', 'patch_bone_matrices', 'patch_bone_bounds', 'ugx_mesh_splits_material_sections', 'patch_section_materials', 'patch_section_cached_metadata', 'exported_mesh_positions', 'mesh_cached_bounds', 'patch_cached_bounds')


def patch_position_edits(data, model, obj):
    """Patch moved vertices and their incident shading while other buffers match import."""
    mesh = obj.data
    source = [vertex for section in model.sections for vertex in section.vertices]
    positions = mesh_game_positions(obj)
    if not np.all(np.isfinite(positions)):
        raise UgxError(f'Mesh {obj.name!r} contains a non-finite position.')
    original = np.asarray([vertex.pos for vertex in source])
    moved = np.any(np.abs(positions - original) > 0.00001, axis=1)
    vertex_base = 0
    for section in model.sections:
        for local_index in np.flatnonzero(moved[vertex_base:vertex_base + len(section.vertices)]):
            index = vertex_base + int(local_index)
            moved[index] = section_position_changed(section, int(local_index), positions[index])
        vertex_base += len(section.vertices)
    if not np.any(moved):
        return positions
    triangles = mesh_buffer(mesh.loops, 'vertex_index', dtype=np.int32).reshape((-1, 3))
    refresh = np.zeros(len(positions), dtype=np.bool_)
    refresh[triangles[np.any(moved[triangles], axis=1)].ravel()] = True
    refresh |= moved
    incident = triangles[np.any(refresh[triangles], axis=1)]
    local = mesh_buffer(mesh.vertices, 'co', 3).astype(np.float64)
    edge1 = local[incident[:, 1]] - local[incident[:, 0]]
    edge2 = local[incident[:, 2]] - local[incident[:, 0]]
    face_normals = np.cross(edge1, edge2)
    valid = np.sum(face_normals * face_normals, axis=1) > 0.0000001
    normal_sums = np.zeros_like(local)
    for corner in range(3):
        np.add.at(normal_sums, incident[valid, corner], face_normals[valid])
    normal_lengths = np.linalg.norm(normal_sums, axis=1)
    valid_normals = normal_lengths * normal_lengths > 0.0000001
    normal_sums[valid_normals] /= normal_lengths[valid_normals, None]
    normal_sums[~valid_normals] = mesh_buffer(mesh.vertices, 'normal', 3)[~valid_normals]
    tangent_sums = np.zeros_like(local)
    layer = mesh.uv_layers.get('_HW1_Source_Tangent_UV') or mesh.uv_layers.get('UV0')
    valid_tangents = np.zeros(len(local), dtype=np.bool_)
    if layer is not None:
        uv_data, uv_property = mesh_uv_data(mesh, layer)
        uv = mesh_buffer(uv_data, uv_property, 2).astype(np.float64).reshape((-1, 3, 2))
        uv = uv[np.any(refresh[triangles], axis=1)]
        duv1, duv2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
        denominator = duv1[:, 0] * duv2[:, 1] - duv2[:, 0] * duv1[:, 1]
        valid = np.abs(denominator) > 0.0000001
        tangents = (edge1[valid] * duv2[valid, 1, None] - edge2[valid] * duv1[valid, 1, None]) / denominator[valid, None]
        nonzero = np.sum(tangents * tangents, axis=1) > 0.0000001
        for corner in range(3):
            np.add.at(tangent_sums, incident[valid][nonzero, corner], tangents[nonzero])
        lengths = np.linalg.norm(tangent_sums, axis=1)
        valid_tangents = lengths * lengths > 0.0000001
        tangent_sums[valid_tangents] /= lengths[valid_tangents, None]
    matrix = np.asarray(ugx_model_matrix(obj), dtype=np.float64)[:3, :3]
    normal_matrix = np.linalg.inv(matrix).T
    identity = np.all(np.abs(matrix - np.eye(3)) <= 0.0000001)

    def game_vector(value, transform):
        value = transform @ value
        if not identity:
            length = np.linalg.norm(value)
            if length > 0:
                value = value / length
        return (-float(value[0]), float(value[2]), -float(value[1]))

    normal_attr = mesh.attributes.get('UGX_Normal')
    tangent_attr = mesh.attributes.get('UGX_Tangent')
    sign_attr = mesh.attributes.get('UGX_BitangentSign')
    vertex_base = 0
    for section in model.sections:
        pos_layout = layout_for(section.layout, 'P')
        normal_layout = layout_for(section.layout, 'N')
        tangent_layout = tangent_layout_for(section.layout)
        selected = np.flatnonzero(refresh[vertex_base:vertex_base + len(section.vertices)])
        for local_index in selected:
            index = vertex_base + int(local_index)
            native = section.vertices[local_index]
            offset = section.vertex_file_offset + int(local_index) * section.vertex_stride
            if moved[index] and pos_layout is not None:
                if not write_vec3(data, offset + pos_layout.offset, pos_layout.type_id, positions[index]):
                    raise UgxError(f'Position element type {pos_layout.type_id} is not writable yet.')
            calculated_normal = game_vector(normal_sums[index], normal_matrix)
            normal = game_vector(normal_attr.data[index].vector, normal_matrix) if normal_attr else native.normal
            if not changed(normal, native.normal):
                normal = calculated_normal
            if normal_layout and changed(normal, native.normal):
                sign = float(sign_attr.data[index].value) if sign_attr else native.bitangent_sign
                if not write_normal_vec3(data, offset + normal_layout.offset, normal_layout.type_id, normal, sign):
                    raise UgxError(f'Normal element type {normal_layout.type_id} is not writable yet.')
            tangent = game_vector(tangent_attr.data[index].vector, matrix) if tangent_attr else native.tangent
            if not changed(tangent, native.tangent):
                tangent = game_vector(tangent_sums[index], matrix) if valid_tangents[index] else tangent
            if tangent_layout and changed(tangent, native.tangent):
                if tangent_layout.type_id == V6_PACKED_BASIS_TYPE:
                    tangent = orthonormal_tangent_tuple(calculated_normal, tangent)
                if not write_vec3_preserve_dec3n_bits(data, offset + tangent_layout.offset, tangent_layout.type_id, tangent, source_tangent_bits(native, tangent_layout)):
                    raise UgxError(f'Tangent element type {tangent_layout.type_id} is not writable yet.')
        vertex_base += len(section.vertices)
    return positions

def patch_cached_bone_names(data: bytearray, model: UgxModel, armature_bones: List[bpy.types.Bone]) -> bytearray:
    records = ecf_chunk_records(bytes(data))
    cached_record = next((record for record in records if record.id == CACHED_DATA_CHUNK), None)
    cached_payload = bytearray(cached_record.data) if cached_record else bytearray()
    grew = False
    for index, source_bone in enumerate(model.bones):
        if index >= len(armature_bones):
            continue
        bone = armature_bones[index]
        value = bone.get("hwek_ugx_export_name", source_bone.name)
        if value == source_bone.name and bone.name != bone.get("hwek_ugx_blender_name", bone.name):
            value = bone.name
        value = str(value)
        if value == source_bone.name:
            continue
        raw = value.encode("utf-8")
        if b"\0" in raw:
            raise UgxError(f"Bone {index} name contains a NUL byte.")
        if not cached_record or not source_bone.name_offset_field_offset or not source_bone.name_offset_field_size:
            raise UgxError(f"Bone {index} name has no writable cached string reference.")
        field_offset = source_bone.name_offset_field_offset - cached_record.offset
        if field_offset < 0 or field_offset + source_bone.name_offset_field_size > len(cached_payload):
            raise UgxError(f"Bone {index} name offset field points outside cached data.")
        align_data(cached_payload, 2)
        new_offset = len(cached_payload)
        cached_payload.extend(raw + b"\0")
        write_packed_offset(cached_payload, field_offset, source_bone.name_offset_field_size, new_offset, source_bone.big_endian)
        grew = True
    return replace_ecf_chunk(data, CACHED_DATA_CHUNK, bytes(cached_payload)) if grew else data


def same_topology(model: UgxModel, mesh: bpy.types.Mesh) -> bool:
    expected_vertices = sum(len(section.vertices) for section in model.sections)
    if len(mesh.vertices) != expected_vertices:
        return False
    expected_faces: List[Tuple[int, int, int]] = []
    vertex_base = 0
    for section in model.sections:
        for i in range(0, len(section.indices) - 2, 3):
            a, b, c = section.indices[i], section.indices[i + 1], section.indices[i + 2]
            if a != b and b != c and c != a:
                expected_faces.append(reverse_triangle_winding((vertex_base + a, vertex_base + b, vertex_base + c)))
        vertex_base += len(section.vertices)
    actual_faces = [tuple(poly.vertices) for poly in mesh.polygons]
    return actual_faces == expected_faces


def validate_same_topology(model: UgxModel, mesh: bpy.types.Mesh):
    if not same_topology(model, mesh):
        raise UgxError("Face/index changes need a full UGX index-buffer rebuild.")


def mesh_positions_changed(model: UgxModel, obj: bpy.types.Object) -> bool:
    mesh = obj.data
    expected_vertices = sum(len(section.vertices) for section in model.sections)
    if len(mesh.vertices) != expected_vertices:
        return True
    vertex_base = 0
    for section in model.sections:
        for local_index, source_vertex in enumerate(section.vertices):
            position = object_vertex_game_position(obj, vertex_base + local_index)
            if section_position_changed(section, local_index, position):
                return True
        vertex_base += len(section.vertices)
    return False


def patch_mesh_vertices(data: bytearray, model: UgxModel, obj: bpy.types.Object):
    mesh = obj.data
    skin_bones = vertex_group_bone_indices(obj, ugx_armature_for(obj))
    validate_same_topology(model, mesh)
    uvs_by_layer = {layer_name: mesh_uv_by_vertex(mesh, layer_name, model.cache_version) for _, layer_name, _ in UV_SEMANTICS}
    normals = mesh_normals_by_vertex_world(obj)
    tangents = mesh_tangents_by_vertex_world(obj)
    bitangent_signs = mesh_bitangent_signs_by_vertex(mesh)
    colors = mesh_colors_by_vertex(mesh)
    calculated_normals = mesh_calculated_normals_by_vertex_world(obj)
    calculated_tangents = mesh_calculated_tangents_by_vertex_world(obj)
    shading_refresh_vertices = mesh_connected_vertices(mesh, mesh_position_changed_vertices(model, obj))
    vertex_base = 0
    for section in model.sections:
        pos_layout = layout_for(section.layout, "P")
        normal_layout = layout_for(section.layout, "N")
        tangent_layout = tangent_layout_for(section.layout)
        index_layout = layout_for(section.layout, "I")
        weight_layout = layout_for(section.layout, "W")
        color_layout = layout_for(section.layout, 'D')
        for local_index, source_vertex in enumerate(section.vertices):
            vertex_index = vertex_base + local_index
            vertex_file_offset = section.vertex_file_offset + local_index * section.vertex_stride
            color = colors.get(vertex_index)
            if color_layout is not None and color is not None and changed(color, source_vertex.color):
                offset = vertex_file_offset + color_layout.offset
                encoded_color = encode_vertex_color(color, color_layout.type_id)
                data[offset:offset + len(encoded_color)] = encoded_color
            if pos_layout is not None:
                position = object_vertex_game_position(obj, vertex_index)
                position_changed = section_position_changed(section, local_index, position)
                if position_changed:
                    if vertex_file_offset + pos_layout.offset + pos_layout.size > len(data):
                        raise UgxError("Position patch points outside the UGX file.")
                    if not write_vec3(data, vertex_file_offset + pos_layout.offset, pos_layout.type_id, position):
                        raise UgxError(f"Position element type {pos_layout.type_id} is not writable yet.")
            else:
                position_changed = False
            for semantic, layer_name, attr_name in UV_SEMANTICS:
                uv_layout = layout_for(section.layout, semantic)
                layer_uvs = uvs_by_layer[layer_name]
                if uv_layout is None or vertex_index not in layer_uvs:
                    continue
                uv = layer_uvs[vertex_index]
                if changed(uv, getattr(source_vertex, attr_name)):
                    if vertex_file_offset + uv_layout.offset + uv_layout.size > len(data):
                        raise UgxError(f"{layer_name} patch points outside the UGX file.")
                    if not write_vec2(data, vertex_file_offset + uv_layout.offset, uv_layout.type_id, uv):
                        raise UgxError(f"{layer_name} element type {uv_layout.type_id} is not writable yet.")
            if normal_layout is not None:
                normal = export_vector_for_edited_position(vertex_index, source_vertex.normal, normals, calculated_normals, vertex_index in shading_refresh_vertices)
                if normal is not None and changed(normal, source_vertex.normal):
                    if vertex_file_offset + normal_layout.offset + normal_layout.size > len(data):
                        raise UgxError("Normal patch points outside the UGX file.")
                    bitangent_sign = bitangent_signs.get(vertex_index, source_vertex.bitangent_sign)
                    if not write_normal_vec3(
                        data,
                        vertex_file_offset + normal_layout.offset,
                        normal_layout.type_id,
                        normal,
                        bitangent_sign,
                    ):
                        raise UgxError(f"Normal element type {normal_layout.type_id} is not writable yet.")
            if tangent_layout is not None:
                tangent = export_vector_for_edited_position(vertex_index, source_vertex.tangent, tangents, calculated_tangents, vertex_index in shading_refresh_vertices)
                if tangent is None and vertex_index in shading_refresh_vertices:
                    normal = calculated_normals.get(vertex_index, source_vertex.normal)
                    tangent = tangent_for_game_normal(normal)
                if tangent is not None and changed(tangent, source_vertex.tangent):
                    if tangent_layout.type_id == V6_PACKED_BASIS_TYPE:
                        tangent_normal = calculated_normals.get(vertex_index, source_vertex.normal)
                        tangent = orthonormal_tangent_tuple(tangent_normal, tangent)
                    if vertex_file_offset + tangent_layout.offset + tangent_layout.size > len(data):
                        raise UgxError("Tangent patch points outside the UGX file.")
                    if not write_vec3_preserve_dec3n_bits(data, vertex_file_offset + tangent_layout.offset, tangent_layout.type_id, tangent, source_tangent_bits(source_vertex, tangent_layout)):
                        raise UgxError(f"Tangent element type {tangent_layout.type_id} is not writable yet.")
            if index_layout is not None and weight_layout is not None:
                current_skin = mesh_skin_map(obj, section, vertex_index, skin_bones, len(model.bones))
                source_skin = source_skin_map(source_vertex)
                imported_source_skin = importable_source_skin_map(source_vertex, section, len(model.bones))
                # Raw weights that did not survive as Blender groups stay byte-preserved until they are editable in Blender.
                if skin_changed(imported_source_skin, source_skin):
                    if skin_changed(current_skin, imported_source_skin):
                        raise UgxError("Skin edit would drop source bone weights that are not editable in Blender.")
                    continue
                if skin_changed(current_skin, source_skin):
                    if index_layout.type_id != 6 or weight_layout.type_id not in (6, 9):
                        raise UgxError(f"Skin element types I={index_layout.type_id} W={weight_layout.type_id} are not writable yet.")
                    index_offset = vertex_file_offset + index_layout.offset
                    weight_offset = vertex_file_offset + weight_layout.offset
                    if index_offset + index_layout.size > len(data) or weight_offset + weight_layout.size > len(data):
                        raise UgxError("Skin patch points outside the UGX file.")
                    raw_indices, raw_weights = encode_skin(current_skin)
                    data[index_offset:index_offset + 4] = raw_indices
                    data[weight_offset:weight_offset + 4] = raw_weights
        vertex_base += len(section.vertices)


def encode_rebuilt_vertex(
    data: bytearray,
    model: UgxModel,
    obj: bpy.types.Object,
    section: Section,
    section_start: int,
    vertex_index: int,
    source_global: int,
    uvs_by_layer,
    normals,
    tangents,
    calculated_normals=None,
    calculated_tangents=None,
    refresh_shading_vertices=None,
    uv_overrides=None,
    normal_override=None,
    tangent_override=None,
    bitangent_sign_override=None,
    skin_bones=None,
    color_override=None,
) -> bytes:
    calculated_normals = calculated_normals or {}
    calculated_tangents = calculated_tangents or {}
    refresh_shading_vertices = refresh_shading_vertices or set()
    source_local = source_global - section_start
    if source_local < 0 or source_local >= len(section.vertices):
        raise UgxError("Edited face uses a vertex from a different UGX section; moving faces between sections needs section rebuild support.")
    if section.source_vertex_data:
        raw_offset = source_local * section.vertex_stride
        if raw_offset + section.vertex_stride > len(section.source_vertex_data):
            raise UgxError("Section-local source vertex bytes are incomplete.")
        encoded = bytearray(section.source_vertex_data[raw_offset:raw_offset + section.vertex_stride])
    else:
        raw_offset = section.vertex_file_offset + source_local * section.vertex_stride
        if raw_offset + section.vertex_stride > len(model.original_bytes):
            raise UgxError("Source vertex bytes point outside the original UGX file.")
        encoded = bytearray(model.original_bytes[raw_offset:raw_offset + section.vertex_stride])
    source_vertex = section.vertices[source_local]
    color_layout = layout_for(section.layout, 'D')
    if color_layout is not None and color_override is not None and changed(color_override, source_vertex.color):
        color = encode_vertex_color(color_override, color_layout.type_id)
        encoded[color_layout.offset:color_layout.offset + len(color)] = color
    pos_layout = layout_for(section.layout, "P")
    normal_layout = layout_for(section.layout, "N")
    tangent_layout = tangent_layout_for(section.layout)
    index_layout = layout_for(section.layout, "I")
    weight_layout = layout_for(section.layout, "W")

    if pos_layout is not None:
        position = object_vertex_game_position(obj, vertex_index)
        position_changed = changed(position, source_vertex.pos)
        if position_changed and not write_vec3(encoded, pos_layout.offset, pos_layout.type_id, position):
            raise UgxError(f"Position element type {pos_layout.type_id} is not writable yet.")
    else:
        position_changed = False
    for semantic, layer_name, attr_name in UV_SEMANTICS:
        uv_layout = layout_for(section.layout, semantic)
        if uv_layout is None:
            continue
        uv = (uv_overrides or {}).get(layer_name)
        if uv is None and vertex_index in uvs_by_layer.get(layer_name, {}):
            uv = uvs_by_layer[layer_name][vertex_index]
        if uv is not None and changed(uv, getattr(source_vertex, attr_name)) and not write_vec2(encoded, uv_layout.offset, uv_layout.type_id, uv):
            raise UgxError(f"{layer_name} element type {uv_layout.type_id} is not writable yet.")
    if normal_layout is not None:
        normal = normal_override
        if normal is None:
            normal = export_vector_for_edited_position(vertex_index, source_vertex.normal, normals, calculated_normals, vertex_index in refresh_shading_vertices)
        if normal is not None and (normal_override is not None or changed(normal, source_vertex.normal)):
            bitangent_sign = source_vertex.bitangent_sign if bitangent_sign_override is None else float(bitangent_sign_override)
            if not write_normal_vec3(encoded, normal_layout.offset, normal_layout.type_id, normal, bitangent_sign):
                raise UgxError(f"Normal element type {normal_layout.type_id} is not writable yet.")
    if tangent_layout is not None:
        tangent = tangent_override
        if tangent is None:
            tangent = export_vector_for_edited_position(vertex_index, source_vertex.tangent, tangents, calculated_tangents, vertex_index in refresh_shading_vertices)
        if tangent is None and vertex_index in refresh_shading_vertices:
            normal = calculated_normals.get(vertex_index, source_vertex.normal)
            tangent = tangent_for_game_normal(normal)
        if tangent is not None and (tangent_override is not None or changed(tangent, source_vertex.tangent)):
            if model.cache_version == 4 and bitangent_sign_override is not None:
                tangent_normal = normal_override or calculated_normals.get(vertex_index) or source_vertex.normal
                tangent = v4_tangent_with_handedness(tangent_normal, tangent, bitangent_sign_override)
            elif tangent_layout.type_id == V6_PACKED_BASIS_TYPE:
                tangent_normal = normal_override or calculated_normals.get(vertex_index) or source_vertex.normal
                tangent = orthonormal_tangent_tuple(tangent_normal, tangent)
            if not write_vec3_preserve_dec3n_bits(encoded, tangent_layout.offset, tangent_layout.type_id, tangent, source_tangent_bits(source_vertex, tangent_layout)):
                raise UgxError(f"Tangent element type {tangent_layout.type_id} is not writable yet.")
    if index_layout is not None and weight_layout is not None:
        current_skin = mesh_skin_map(obj, section, vertex_index, skin_bones, len(model.bones))
        source_skin = source_skin_map(source_vertex)
        imported_source_skin = importable_source_skin_map(source_vertex, section, len(model.bones))
        if skin_changed(imported_source_skin, source_skin):
            if skin_changed(current_skin, imported_source_skin):
                raise UgxError("Skin edit would drop source bone weights that are not editable in Blender.")
        elif skin_changed(current_skin, source_skin):
            if index_layout.type_id != 6 or weight_layout.type_id not in (6, 9):
                raise UgxError(f"Skin element types I={index_layout.type_id} W={weight_layout.type_id} are not writable yet.")
            raw_indices, raw_weights = encode_skin(current_skin)
            encoded[index_layout.offset:index_layout.offset + 4] = raw_indices
            encoded[weight_layout.offset:weight_layout.offset + 4] = raw_weights
    return bytes(encoded)


def source_template_for_section(section: Section, section_start: int, section_end: int, obj: bpy.types.Object, vertex_index: int, source_global: int) -> int:
    if section_start <= source_global < section_end:
        return source_global
    if not section.vertices:
        raise UgxError("Cannot create vertices in an empty UGX section.")
    position = object_vertex_game_position(obj, vertex_index)
    best_local = min(
        range(len(section.vertices)),
        key=lambda index: (
            (section.vertices[index].pos[0] - position[0]) ** 2
            + (section.vertices[index].pos[1] - position[1]) ** 2
            + (section.vertices[index].pos[2] - position[2]) ** 2
        ),
    )
    # New Blender vertices borrow the nearest section-local vertex bytes for fields the UI does not expose.
    return section_start + best_local


def rebuild_topology_buffers(data: bytearray, model: UgxModel, obj: bpy.types.Object, rebuild_cached_structure: bool = False, preserve_source_basis: bool = False) -> bytearray:
    skin_bones = vertex_group_bone_indices(obj, ugx_armature_for(obj))
    mesh = obj.data
    section_attr = mesh_int_attribute(mesh, "UGX_Section")
    source_attr = mesh_int_attribute(mesh, "UGX_SourceVertex")
    if section_attr is None or len(section_attr) < len(mesh.polygons):
        raise UgxError("Topology edits need the UGX_Section face attribute imported by this addon.")
    if source_attr is None or len(source_attr) < len(mesh.vertices):
        raise UgxError("Topology edits need the UGX_SourceVertex point attribute imported by this addon.")

    records = ecf_chunk_records(bytes(data))
    cached_record = next((record for record in records if record.id == CACHED_DATA_CHUNK), None)
    vertex_record = next((record for record in records if record.id == VERTEX_BUFFER_CHUNK), None)
    index_record = next((record for record in records if record.id == INDEX_BUFFER_CHUNK), None)
    if not cached_record:
        raise UgxError("UGX is missing cached data.")
    if not vertex_record or not index_record:
        raise UgxError("UGX is missing vertex or index buffers.")
    cached_payload = bytearray(cached_record.data)
    rebuilt_sections = []
    uvs_by_layer = {
        layer_name: (mesh_uv_by_vertex(mesh, layer_name, model.cache_version) if not mesh_has_split_uv(mesh, layer_name, model.cache_version) else {})
        for _, layer_name, _ in UV_SEMANTICS
    }
    uv_loops_by_layer = {layer_name: mesh_uv_by_loop(mesh, layer_name, model.cache_version) for _, layer_name, _ in UV_SEMANTICS}
    color_loops = mesh_colors_by_loop(mesh)
    normals = mesh_normals_by_vertex_world(obj)
    tangents = mesh_tangents_by_vertex_world(obj)
    bitangent_signs = mesh_bitangent_signs_by_vertex(mesh)
    calculated_normals = mesh_calculated_normals_by_vertex_world(obj)
    calculated_tangents = mesh_calculated_tangents_by_vertex_world(obj)
    loop_basis = mesh_calculated_basis_by_loop_world(obj)
    shading_refresh_vertices = set(range(len(mesh.vertices)))
    preserve_source_basis = preserve_source_basis and same_topology(model, mesh)
    section_ranges = source_vertex_section_ranges(model)
    if preserve_source_basis:
        changed_basis_vertices = mesh_position_changed_vertices(model, obj)
        uv_changed_sections = set()
        for poly in mesh.polygons:
            section_index = section_attr[poly.index]
            if section_index in uv_changed_sections:
                continue
            section = model.sections[section_index]
            section_start, section_end = section_ranges[section_index]
            for loop_index, vertex_index in zip(poly.loop_indices, poly.vertices):
                source_global = source_template_for_section(section, section_start, section_end, obj, vertex_index, source_attr[vertex_index])
                source_vertex = section.vertices[source_global - section_start]
                if any(changed(uv_loops_by_layer[layer_name][loop_index], getattr(source_vertex, attr_name))
                       for semantic, layer_name, attr_name in UV_SEMANTICS
                       if layout_for(section.layout, semantic) is not None and loop_index in uv_loops_by_layer[layer_name]):
                    # UV edits can split or merge tangent groups within the section.
                    changed_basis_vertices.update(range(section_start, section_end))
                    uv_changed_sections.add(section_index)
                    break
        shading_refresh_vertices = mesh_connected_vertices(mesh, changed_basis_vertices)

    section_groups = []
    for section_index, section in enumerate(model.sections):
        polygons = [poly for poly in mesh.polygons if section_attr[poly.index] == section_index]
        by_material = {}
        for poly in polygons:
            material_index = ugx_material_index_for_slot(obj, int(poly.material_index), model)
            by_material.setdefault(material_index, []).append(poly)
        if model.cache_version != 6 and len(by_material) > 1:
            raise UgxError("HWDE section material edits need a fresh authoring export.")
        if not by_material:
            by_material[section.material_index] = []
        for material_index, material_polygons in sorted(by_material.items()):
            section_groups.append((section_index, copy.deepcopy(section), material_polygons, material_index))

    for section_index, section, polygons, material_index in section_groups:
        section_start, section_end = section_ranges[section_index]
        if material_index < 0 or material_index >= len(model.materials):
            raise UgxError("Section material index points outside the UGX material table.")

        local_for_mesh_vertex: Dict[int, int] = {}
        section_vertex_bytes = bytearray()
        section_indices: List[int] = []
        for poly in polygons:
            if len(poly.vertices) != 3:
                raise UgxError("UGX topology rebuild currently supports triangles only.")
            blender_corners = list(zip(poly.loop_indices, poly.vertices))
            for loop_index, vertex_index in reverse_triangle_winding(blender_corners):
                loop_uvs = {
                    layer_name: uv_loops_by_layer[layer_name].get(loop_index)
                    for semantic, layer_name, _ in UV_SEMANTICS
                    if layout_for(section.layout, semantic) is not None
                }
                source_global = None
                keep_source_basis = False
                if preserve_source_basis and vertex_index not in shading_refresh_vertices:
                    source_global = source_template_for_section(section, section_start, section_end, obj, vertex_index, source_attr[vertex_index])
                    source_vertex = section.vertices[source_global - section_start]
                    keep_source_basis = (
                        not changed(normals.get(vertex_index, source_vertex.normal), source_vertex.normal)
                        and not changed(tangents.get(vertex_index, source_vertex.tangent), source_vertex.tangent)
                        and bitangent_signs.get(vertex_index, source_vertex.bitangent_sign) == source_vertex.bitangent_sign
                    )
                # Repacking an unchanged native basis can lose precision in packed formats.
                basis = None if keep_source_basis else loop_basis.get(loop_index)
                if basis is None and not keep_source_basis:
                    normal = calculated_normals.get(vertex_index, (0.0, 1.0, 0.0))
                    tangent = calculated_tangents.get(vertex_index, tangent_for_game_normal(normal))
                    basis = (normal, orthonormal_tangent_tuple(normal, tangent), bitangent_signs.get(vertex_index, 1.0))
                basis_key = None if basis is None else (
                    tuple(round(float(component), 7) for component in basis[0]),
                    tuple(round(float(component), 7) for component in basis[1]),
                    -1 if float(basis[2]) < 0.0 else 1,
                )
                vertex_key = (
                    vertex_index,
                    tuple(loop_uvs.get(layer_name) for _, layer_name, _ in UV_SEMANTICS),
                    basis_key,
                    color_loops.get(loop_index),
                )
                if vertex_key not in local_for_mesh_vertex:
                    if source_global is None:
                        source_global = source_template_for_section(section, section_start, section_end, obj, vertex_index, source_attr[vertex_index])
                    if len(local_for_mesh_vertex) >= 0x10000:
                        raise UgxError("UGX index buffers are 16-bit; section has too many vertices.")
                    local_for_mesh_vertex[vertex_key] = len(local_for_mesh_vertex)
                    section_vertex_bytes.extend(encode_rebuilt_vertex(
                        data,
                        model,
                        obj,
                        section,
                        section_start,
                        vertex_index,
                        source_global,
                        uvs_by_layer,
                        normals,
                        tangents,
                        calculated_normals,
                        calculated_tangents,
                        shading_refresh_vertices,
                        loop_uvs,
                        basis[0] if basis is not None else None,
                        basis[1] if basis is not None else None,
                        basis[2] if basis is not None else None,
                        skin_bones,
                        color_loops.get(loop_index),
                    ))
                section_indices.append(local_for_mesh_vertex[vertex_key])

        if len(section_indices) % 3:
            raise UgxError("Rebuilt section index count is not divisible by three.")
        section_index_bytes = bytearray()
        for index in section_indices:
            section_index_bytes.extend(struct.pack(">H" if section.big_endian else "<H", index))
        rebuilt_sections.append({
            "section": section,
            "material_index": material_index,
            "vertex_bytes": bytes(section_vertex_bytes),
            "index_bytes": bytes(section_index_bytes),
            "vertex_count": len(local_for_mesh_vertex),
            "tri_count": len(section_indices) // 3,
        })

    if rebuild_cached_structure or (model.cache_version == 6 and len(rebuilt_sections) != len(model.sections)):
        model.sections = [item["section"] for item in rebuilt_sections]
        return rebuild_v6_structural_chunks(data, model, obj, rebuilt_sections)

    if model.cache_version == 4:
        if any(item["vertex_count"] > V4_MAX_SECTION_VERTICES for item in rebuilt_sections):
            raise UgxError("HWDE v4 topology edit exceeds the section vertex limit; split the mesh into more sections.")
        max_instances, instance_index_multiplier = v4_instance_settings(rebuilt_sections)
        struct.pack_into("<hh", cached_payload, 48, max_instances, instance_index_multiplier)
    else:
        max_instances, instance_index_multiplier = v6_instance_settings(rebuilt_sections)
        struct.pack_into("<hH", cached_payload, 48, max_instances, instance_index_multiplier)
    for item in rebuilt_sections:
        base_indices = struct.unpack(f"<{len(item['index_bytes']) // 2}H", item["index_bytes"])
        item["index_bytes"] = b"".join(
            struct.pack(f"<{len(base_indices)}H", *(index + instance_index_multiplier * instance for index in base_indices))
            for instance in range(max_instances)
        )

    def span_end(offsets: List[int], offset: int, fallback: int) -> int:
        return next((candidate for candidate in offsets if candidate > offset), fallback)

    vertex_offsets = sorted(section.vertex_offset for section in model.sections)
    index_offsets = sorted(section.index_offset for section in model.sections)
    fits_original_slots = all(
        item["section"].vertex_offset + len(item["vertex_bytes"]) <= span_end(vertex_offsets, item["section"].vertex_offset, len(vertex_record.data))
        and item["section"].index_offset + len(item["index_bytes"]) <= span_end(index_offsets, item["section"].index_offset, len(index_record.data))
        for item in rebuilt_sections
    )

    if fits_original_slots:
        vertex_payload = bytearray(vertex_record.data)
        index_payload = bytearray(index_record.data)
        for item in rebuilt_sections:
            section = item["section"]
            vertex_start = section.vertex_offset
            index_start = section.index_offset // 2
            vertex_payload[section.vertex_offset:section.vertex_offset + len(item["vertex_bytes"])] = item["vertex_bytes"]
            index_payload[section.index_offset:section.index_offset + len(item["index_bytes"])] = item["index_bytes"]

            def cached_local(file_offset: int) -> int:
                local = file_offset - cached_record.offset
                if local < 0 or local + 4 > len(cached_payload):
                    raise UgxError("Cached section field points outside the cached-data chunk.")
                return local

            write_i32(cached_payload, cached_local(section.material_index_offset), item["material_index"], section.big_endian)
            write_i32(cached_payload, cached_local(section.index_offset_field_offset), index_start, section.big_endian)
            write_i32(cached_payload, cached_local(section.tri_count_field_offset), item["tri_count"], section.big_endian)
            write_i32(cached_payload, cached_local(section.vertex_offset_field_offset), vertex_start, section.big_endian)
            write_i32(cached_payload, cached_local(section.vertex_size_field_offset), len(item["vertex_bytes"]), section.big_endian)
            write_i32(cached_payload, cached_local(section.vertex_count_field_offset), item["vertex_count"], section.big_endian)
    else:
        index_payload = bytearray()
        vertex_payload = bytearray()
        for item in rebuilt_sections:
            section = item["section"]
            vertex_start = len(vertex_payload)
            index_start = len(index_payload) // 2
            vertex_payload.extend(item["vertex_bytes"])
            index_payload.extend(item["index_bytes"])

            def cached_local(file_offset: int) -> int:
                local = file_offset - cached_record.offset
                if local < 0 or local + 4 > len(cached_payload):
                    raise UgxError("Cached section field points outside the cached-data chunk.")
                return local

            write_i32(cached_payload, cached_local(section.material_index_offset), item["material_index"], section.big_endian)
            write_i32(cached_payload, cached_local(section.index_offset_field_offset), index_start, section.big_endian)
            write_i32(cached_payload, cached_local(section.tri_count_field_offset), item["tri_count"], section.big_endian)
            write_i32(cached_payload, cached_local(section.vertex_offset_field_offset), vertex_start, section.big_endian)
            write_i32(cached_payload, cached_local(section.vertex_size_field_offset), len(item["vertex_bytes"]), section.big_endian)
            write_i32(cached_payload, cached_local(section.vertex_count_field_offset), item["vertex_count"], section.big_endian)

        old_vertex_used_end = max([section.vertex_offset + section.vertex_buffer_size for section in model.sections] + [0])
        source_instances = max(1,struct.unpack_from('<h',cached_record.data,48)[0])
        old_index_used_end = max([section.index_offset + len(section.indices) * 2 * source_instances for section in model.sections] + [0])
        if old_vertex_used_end < len(vertex_record.data):
            vertex_payload.extend(vertex_record.data[old_vertex_used_end:])
        if old_index_used_end < len(index_record.data):
            index_payload.extend(index_record.data[old_index_used_end:])

    data = replace_ecf_chunk(data, CACHED_DATA_CHUNK, bytes(cached_payload))
    data = replace_ecf_chunk(data, VERTEX_BUFFER_CHUNK, bytes(vertex_payload))
    data = replace_ecf_chunk(data, INDEX_BUFFER_CHUNK, bytes(index_payload))
    return data


def rebuild_v6_structural_chunks(data: bytearray, model: UgxModel, obj: bpy.types.Object, rebuilt_sections: List[Dict[str, Any]]) -> bytearray:
    if model.cache_version != 6 or not rebuilt_sections:
        raise UgxError("Structural mesh changes require a non-empty Halo Wars 2 v6 UGX.")
    records = ecf_chunk_records(bytes(data))
    cached_record = next((record for record in records if record.id == CACHED_DATA_CHUNK), None)
    if not cached_record or len(cached_record.data) < 160:
        raise UgxError("HW2 cached data is too small to rebuild structurally.")

    vertex_payload = bytearray()
    section_infos: List[Dict[str, Any]] = []
    for item in rebuilt_sections:
        section = item["section"]
        section.vertex_offset = len(vertex_payload)
        section.vertex_buffer_size = len(item["vertex_bytes"])
        section.vertex_count = item["vertex_count"]
        section.indices = list(struct.unpack(f"<{len(item['index_bytes']) // 2}H", item["index_bytes"])) if item["index_bytes"] else []
        section.material_index = item["material_index"]
        vertex_payload.extend(item["vertex_bytes"])
        section_infos.append({
            "vertex_count": section.vertex_count,
            "tri_count": int(item["tri_count"]),
        })

    max_instances, instance_index_multiplier = v6_instance_settings(section_infos)
    index_payload = bytearray()
    for item in rebuilt_sections:
        section = item["section"]
        section.index_offset = len(index_payload)
        for instance_index in range(max_instances):
            instance_offset = instance_index_multiplier * instance_index
            for index in section.indices:
                index_payload.extend(struct.pack("<H", int(index) + instance_offset))

    positions = [object_vertex_game_position(obj, index) for index in range(len(obj.data.vertices))]
    if not positions:
        raise UgxError("Structural HW2 export has no vertices.")
    mins = tuple(min(value[axis] for value in positions) for axis in range(3))
    maxs = tuple(max(value[axis] for value in positions) for axis in range(3))
    center = tuple((mins[axis] + maxs[axis]) * 0.5 for axis in range(3))
    radius = math.sqrt(sum((maxs[axis] - center[axis]) ** 2 for axis in range(3)))

    # The v6 signature plus UGX_CachedDataHeaderV6 occupy exactly 64 bytes.
    # The six PackedArrayHeader records begin at 0x40 in the 010 Editor contract.
    cached = bytearray(cached_record.data[:64])
    struct.pack_into("<3f", cached, 8, *center)
    struct.pack_into("<f", cached, 20, radius)
    struct.pack_into("<3f", cached, 24, *mins)
    struct.pack_into("<3f", cached, 36, *maxs)
    struct.pack_into("<hH", cached, 48, max_instances, instance_index_multiplier)
    headers_offset = len(cached)
    cached.extend(b"\0" * 96)
    bone_name_offsets = []
    for index, bone in enumerate(model.bones):
        align_data(cached, 2)
        bone_name_offsets.append(len(cached))
        cached.extend(safe_grx_name(bone.name, "").encode("utf-8") + b"\0")
    align_data(cached, 4)

    local_remap_offsets = []
    for section in model.sections:
        if section.local_to_global_bones:
            align_data(cached, 4)
            local_remap_offsets.append(len(cached))
            for value in section.local_to_global_bones:
                cached.extend(struct.pack("<i", int(value)))
        else:
            local_remap_offsets.append(PACKED_NULL_OFFSET_64)
    align_data(cached, 8)

    sections_offset = len(cached)
    for index, item in enumerate(rebuilt_sections):
        section = item["section"]
        record_offset = len(cached)
        cached.extend(struct.pack(
            "<iiiiiiiiiiBxxx3f",
            item["material_index"], section.accessory_index, section.max_bones, section.rigid_bone_index,
            section.index_offset // 2, item["tri_count"], section.vertex_offset, len(item["vertex_bytes"]),
            section.vertex_stride, item["vertex_count"], int(bool(section.rigid_only)),
            section.lod_min_distance, section.lod_max_distance, section.sorting_bias,
        ))
        cached.extend(packed_array_header(len(section.local_to_global_bones), local_remap_offsets[index]))
        if len(cached) - record_offset != 72:
            raise UgxError("Internal structural v6 section size mismatch.")
    align_data(cached, 8)

    bones_offset = len(cached)
    for index, bone in enumerate(model.bones):
        flat = [float(bone.matrix[row][column]) for row in range(4) for column in range(4)]
        cached.extend(struct.pack("<Q16fiI", bone_name_offsets[index], *flat, bone.parent, 0))
    align_data(cached, 4)

    accessory_section_offsets = []
    for accessory_index in range(len(model.accessories)):
        accessory_section_offsets.append(len(cached))
        for section_index, section in enumerate(model.sections):
            if section.accessory_index == accessory_index:
                cached.extend(struct.pack("<i", section_index))
    valid_accessories_offset = len(cached)
    for index in range(len(model.accessories)):
        cached.extend(struct.pack("<i", index))
    align_data(cached, 8)

    accessories_offset = len(cached)
    for index, accessory in enumerate(model.accessories):
        section_count = sum(section.accessory_index == index for section in model.sections)
        cached.extend(struct.pack("<ii", accessory.first_bone, accessory.bone_count))
        cached.extend(packed_array_header(section_count, accessory_section_offsets[index]))
    align4(cached)

    bone_bounds_low_offset = len(cached)
    for bone in model.bones:
        cached.extend(struct.pack("<3f", *bone.bounds_low))
    bone_bounds_high_offset = len(cached)
    for bone in model.bones:
        cached.extend(struct.pack("<3f", *bone.bounds_high))
    headers = b"".join((
        packed_array_header(len(model.sections), sections_offset),
        packed_array_header(len(model.bones), bones_offset),
        packed_array_header(len(model.accessories), accessories_offset),
        packed_array_header(len(model.accessories), valid_accessories_offset),
        packed_array_header(len(model.bones), bone_bounds_low_offset),
        packed_array_header(len(model.bones), bone_bounds_high_offset),
    ))
    cached[headers_offset:headers_offset + len(headers)] = headers
    data = replace_ecf_chunk(data, CACHED_DATA_CHUNK, bytes(cached))
    data = replace_ecf_chunk(data, VERTEX_BUFFER_CHUNK, bytes(vertex_payload))
    data = replace_ecf_chunk(data, INDEX_BUFFER_CHUNK, bytes(index_payload))
    data = replace_ecf_chunk(data, 0x703, build_granny_info_chunk(model, obj.name))
    return data


def patch_bone_matrices(data: bytearray, model: UgxModel, armature: Optional[bpy.types.Object]):
    armature_bones = ugx_ordered_armature_bones(armature)
    parent_indices = {bone.name: index for index, bone in enumerate(armature_bones)}
    rest_display_cache: Dict[int, List[List[float]]] = {}

    def current_rest_display(index: int) -> List[List[float]]:
        cached = rest_display_cache.get(index)
        if cached is not None:
            return cached
        rows = bone_rest_display_rows(armature, armature_bones[index])
        rest_display_cache[index] = rows
        return rows

    for index, source_bone in enumerate(model.bones):
        if index >= len(armature_bones) or not source_bone.matrix_offset:
            continue
        bone = armature_bones[index]
        parent = parent_indices[bone.parent.name] if bone.parent is not None else -1
        parent_changed = parent != source_bone.parent
        if parent_changed:
            write_i32(data, source_bone.matrix_offset + 64, parent, source_bone.big_endian)
        source_rows = source_bone.matrix
        target_rows = None
        pose_bone = armature.pose.bones.get(bone.name) if armature and armature.pose else None

        export_matrix = bone.get("hwek_ugx_export_matrix")
        if export_matrix:
            target_rows = parse_matrix_rows(export_matrix, f"Bone {index} export matrix")
            if target_rows == source_rows:
                target_rows = None

        source_display_rows = parse_matrix_rows(bone.get("hwek_ugx_source_display_matrix"), f"Bone {index} display matrix")
        import_display_rows = parse_matrix_rows(bone.get("hwek_ugx_import_display_matrix", bone.get("hwek_ugx_source_display_matrix")), f"Bone {index} import display matrix")
        display_rows = current_rest_display(index)
        rest_changed = target_rows is None and any(changed(display_rows[row][:3], import_display_rows[row][:3]) for row in range(3))
        current = bone_head_export_position(armature, bone) if armature else (float(bone.head_local.x), float(bone.head_local.y), float(bone.head_local.z))
        original_head_value = bone.get("hwek_ugx_source_head_local", current)
        original_head = tuple(float(value) for value in original_head_value[:3]) if len(original_head_value) >= 3 else current
        head_changed = changed(current, original_head)
        pose_delta = None if target_rows is not None else pose_basis_delta_rows(pose_bone)
        animation = armature.animation_data if armature else None
        native_pose = (bone.get('hwek_ugx_skeleton_transform_mode') == 'inverse_world' and pose_bone is not None
                       and not (animation is not None and (animation.action is not None or animation.nla_tracks)))
        if target_rows is None and native_pose:
            reference_pose = parse_matrix_rows(bone.get('hwek_ugx_import_pose_display_matrix',
                                               bone.get('hwek_ugx_import_display_matrix', bone.get('hwek_ugx_source_display_matrix'))),
                                               f'Bone {index} import pose matrix')
            current_pose = blender_matrix_to_game_rows(ugx_model_matrix(armature) @ pose_bone.matrix)
            if matrix_changed(current_pose, reference_pose):
                # Full model-space frames include inherited parent motion.
                # Retain native scale/shear missing from Blender edit bones.
                world = mat4_mul(mat4_mul(mat4_inverse(source_rows), mat4_inverse(reference_pose)), current_pose)
                target_rows = mat4_inverse(world)
        elif target_rows is None and (rest_changed or head_changed or pose_delta or parent_changed):
            rest_delta = game_orientation_delta_rows(display_rows, import_display_rows) if rest_changed else None
            display_rows = apply_game_delta_rows(source_display_rows, rest_delta) if rest_delta else [row[:] for row in source_display_rows]
            if pose_delta:
                display_rows = apply_game_delta_rows(display_rows, pose_delta)
            if head_changed:
                display_rows[3][0], display_rows[3][1], display_rows[3][2] = blender_to_game_vec3(current)
            if bone.get("hwek_ugx_skeleton_transform_mode") == "inverse_world":
                target_rows = mat4_inverse(display_rows)
            else:
                if 0 <= parent < len(model.bones) and parent < len(armature_bones) and parent != index:
                    target_rows = mat4_mul(display_rows, mat4_inverse(current_rest_display(parent)))
                else:
                    target_rows = display_rows

        if target_rows is None:
            continue
        write_matrix_rows(data, source_bone.matrix_offset, target_rows, source_bone.big_endian)


def patch_bone_bounds(data: bytearray, model: UgxModel, obj: bpy.types.Object):
    armature_bones = ugx_ordered_armature_bones(ugx_armature_for(obj))
    entries = {
        int(entry.get("index", index)): entry
        for index, entry in enumerate(json.loads(obj.get("hwek_ugx_bone_bounds", "[]")))
    }
    for index, source_bone in enumerate(model.bones):
        entry = entries.get(index, {})
        data_bone = armature_bones[index] if index < len(armature_bones) else None
        big = bool(entry.get("big_endian", source_bone.big_endian))
        for key, offset_key, source in (
            ("low", "low_offset", source_bone.bounds_low),
            ("high", "high_offset", source_bone.bounds_high),
        ):
            offset = int(entry.get(offset_key, getattr(source_bone, f"bounds_{key}_offset", 0)))
            if not offset:
                continue
            values = bone_bound_property_values(data_bone, key, source)
            if values is None:
                values = parse_vec3(entry.get(key, source), f"Bone {index} bounds {key}")
            if not changed(values, source):
                continue
            if offset + 12 > len(data):
                raise UgxError(f"Bone {index} bounds {key} patch points outside the UGX file.")
            for axis, value in enumerate(values):
                write_f32_endian(data, offset + axis * 4, value, big)


def ugx_mesh_splits_material_sections(obj: bpy.types.Object) -> bool:
    sections = mesh_int_attribute(obj.data, "UGX_Section")
    if sections is None:
        return False
    assigned = {}
    for polygon in obj.data.polygons:
        section = int(sections[polygon.index])
        material = int(polygon.material_index)
        previous = assigned.setdefault(section, material)
        if previous != material:
            return True
    return False


def patch_section_materials(data: bytearray, model: UgxModel, obj: bpy.types.Object):
    mesh = obj.data
    face_cursor = 0
    for section in model.sections:
        face_count = 0
        for i in range(0, len(section.indices) - 2, 3):
            a, b, c = section.indices[i], section.indices[i + 1], section.indices[i + 2]
            if a != b and b != c and c != a:
                face_count += 1
        if not face_count:
            continue
        materials = {ugx_material_index_for_slot(obj, int(poly.material_index), model) for poly in mesh.polygons[face_cursor:face_cursor + face_count]}
        face_cursor += face_count
        if len(materials) > 1:
            raise UgxError("UGX sections store one material; split material edits need a section/index rebuild.")
        material_index = next(iter(materials))
        if material_index < 0 or material_index >= len(model.materials):
            raise UgxError("Section material index points outside the UGX material table.")
        if material_index == section.material_index:
            continue
        if section.material_index_offset + 4 > len(data):
            raise UgxError("Section material patch points outside the UGX file.")
        write_i32(data, section.material_index_offset, material_index, section.big_endian)


def patch_section_cached_metadata(data: bytearray, model: UgxModel, obj: bpy.types.Object):
    entries = json.loads(obj.get("hwek_ugx_sections", "[]"))
    for index, section in enumerate(model.sections):
        if index >= len(entries):
            break
        entry = entries[index]
        accessory_index = int(section_metadata_value(obj, index, entry, "accessory_index", section.accessory_index))
        if accessory_index != section.accessory_index and section.accessory_index_offset:
            if section.accessory_index_offset + 4 > len(data):
                raise UgxError("Section accessory patch points outside the UGX file.")
            write_i32(data, section.accessory_index_offset, accessory_index, section.big_endian)
        for key, source, offset in (
            ("max_bones", section.max_bones, section.max_bones_offset),
            ("rigid_bone_index", section.rigid_bone_index, section.rigid_bone_index_offset),
        ):
            if not offset:
                continue
            try:
                value = int(section_metadata_value(obj, index, entry, key, source))
            except (TypeError, ValueError):
                raise UgxError(f"Section {index} {key} is not an integer.")
            if value == int(source):
                continue
            if offset + 4 > len(data):
                raise UgxError(f"Section {index} {key} patch points outside the UGX file.")
            write_i32(data, offset, value, section.big_endian)
        if section.rigid_only_offset:
            value = 1 if parse_bool_value(section_metadata_value(obj, index, entry, "rigid_only", section.rigid_only), bool(section.rigid_only)) else 0
            if value != int(bool(section.rigid_only)):
                if section.rigid_only_offset + 1 > len(data):
                    raise UgxError(f"Section {index} rigid_only patch points outside the UGX file.")
                data[section.rigid_only_offset] = value
        for key, source, offset in (
            ("lod_min_distance", section.lod_min_distance, section.lod_min_distance_offset),
            ("lod_max_distance", section.lod_max_distance, section.lod_max_distance_offset),
            ("sorting_bias", section.sorting_bias, section.sorting_bias_offset),
        ):
            if not offset:
                continue
            value = float(section_metadata_value(obj, index, entry, key, source))
            if not math.isfinite(value):
                raise UgxError(f"Section {index} {key} is not finite.")
            if abs(value - source) <= 0.00001:
                continue
            if offset + 4 > len(data):
                raise UgxError(f"Section {index} {key} patch points outside the UGX file.")
            write_f32_endian(data, offset, value, section.big_endian)


def exported_mesh_positions(obj: bpy.types.Object, topology_unchanged: bool) -> List[Tuple[float, float, float]]:
    mesh = obj.data
    if topology_unchanged or not mesh.polygons:
        return [object_vertex_game_position(obj, index) for index in range(len(mesh.vertices))]
    used = sorted({vertex_index for poly in mesh.polygons for vertex_index in poly.vertices})
    return [object_vertex_game_position(obj, index) for index in used]


def mesh_cached_bounds(obj: bpy.types.Object, topology_unchanged: bool, positions=None) -> Optional[CachedBounds]:
    if positions is None:
        positions = exported_mesh_positions(obj, topology_unchanged)
    if len(positions) == 0:
        return None
    if isinstance(positions, np.ndarray):
        mins, maxs = tuple(positions.min(axis=0)), tuple(positions.max(axis=0))
    else:
        mins = tuple(min(position[axis] for position in positions) for axis in range(3))
        maxs = tuple(max(position[axis] for position in positions) for axis in range(3))
    center = tuple((mins[axis] + maxs[axis]) * 0.5 for axis in range(3))
    radius = math.sqrt(sum((maxs[axis] - center[axis]) ** 2 for axis in range(3)))
    return CachedBounds(center, radius, mins, maxs)


def patch_cached_bounds(data: bytearray, model: UgxModel, obj: bpy.types.Object, topology_unchanged: bool, geometry_changed: bool, positions=None):
    if not geometry_changed:
        return
    source = model.cached_bounds
    if not source.center_offset:
        return
    target = mesh_cached_bounds(obj, topology_unchanged, positions)
    if target is None:
        return
    if (
        not changed(target.center, source.center)
        and abs(target.radius - source.radius) <= 0.00001
        and not changed(target.bounds_min, source.bounds_min)
        and not changed(target.bounds_max, source.bounds_max)
    ):
        return
    for offset, values in (
        (source.center_offset, target.center),
        (source.bounds_min_offset, target.bounds_min),
        (source.bounds_max_offset, target.bounds_max),
    ):
        if offset + 12 > len(data):
            raise UgxError("Cached bounds patch points outside the UGX file.")
        for axis, value in enumerate(values):
            write_f32_endian(data, offset + axis * 4, value, source.big_endian)
    if source.radius_offset + 4 > len(data):
        raise UgxError("Cached bounds radius patch points outside the UGX file.")
    write_f32_endian(data, source.radius_offset, target.radius, source.big_endian)
