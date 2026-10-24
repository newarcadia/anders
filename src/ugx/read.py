"""Read UGX sections, vertex streams, cached metadata and materials."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import math
import os
import struct
try:
    import numpy as np
except ImportError:
    np = None
from ..errors import UgxError
from ..formats.bdt import BdtValue, bdt_float_pair, bdt_int, child_named, read_bdt
from ..formats.binary import (
    PackedArrayHeader,
    Reader,
    cstring_capacity_at,
    read_cstring_at,
    read_packed_array_header,
    read_packed_string_offset,
)
from ..formats.ecf import (
    CACHED_DATA_CHUNK,
    Chunk,
    EcfFile,
    INDEX_BUFFER_CHUNK,
    MATERIAL_CHUNK,
    UGX_FILE_ID,
    VERTEX_BUFFER_CHUNK,
)
from ..formats.granny import parse_granny_info, repair_granny_model_binding_overlap
from ..formats.granny_layout import validate_granny_layout
from ..materials.data import (
    Material,
    V4_MATERIAL_MAP_USAGES,
    bdt_value_json,
    hogan_material_name,
    hogan_texture_usages,
    ufx_constant_buffer_json,
    v4_material_slot_flags,
    v4_material_usage,
    v4_material_uvw_values,
    parse_ufx_shader_hash,
)
from ..materials.skin_pairs import HW2_SKINNED_SHADER_INPUTS
from .data import (
    Accessory,
    Bone,
    CACHED_SIGNATURE_HW1_BE,
    CACHED_SIGNATURE_HW1_LE,
    CACHED_SIGNATURE_HW2,
    CachedBounds,
    Section,
    UV_SEMANTICS,
    UgxModel,
    V6_PACKED_BASIS_TYPE,
    Vertex,
    VertexLayout,
)
from .vertices import (
    build_vertex_layout,
    decode_hw2_packed_normal,
    decode_hw2_packed_tangent,
    decode_tangent,
    decode_vec2,
    decode_vec3,
    decode_vertex_color,
    layout_for,
    tangent_layout_for,
)

__all__ = ('valid_array', 'triplet_valid', 'try_read_array_headers', 'i32_at', 'u32_at', 'u64_at', 'short_cached_string_score', 'v4_section_record_score', 'v4_section_record_size', 'read_indices', 'read_vertices_v4', 'v6_uv_offset', 'v6_skin_offset', 'build_v6_layout', 'read_vertices_v6', 'read_matrix', 'read_cached_data', 'read_materials', 'parse_ugx_data', 'parse_ugx')

def valid_array(data: bytes, header: PackedArrayHeader, max_length: int, allow_zero: bool) -> bool:
    if allow_zero and header.length == 0 and header.offset in (0, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF):
        return True
    return (
        (allow_zero or header.length > 0)
        and header.length < max_length
        and header.offset >= 64
        and header.offset < len(data)
        and header.offset % 4 == 0
    )


def triplet_valid(data: bytes, sections: PackedArrayHeader, bones: PackedArrayHeader, accessories: PackedArrayHeader) -> bool:
    return (
        valid_array(data, sections, 1024, bones.length > 0)
        and valid_array(data, bones, 1024, True)
        and valid_array(data, accessories, 4096, True)
    )


def try_read_array_headers(data: bytes, big: bool, is64: bool, offset: int):
    try:
        reader = Reader(data, big)
        reader.seek(offset)
        sections = read_packed_array_header(reader, is64)
        bones = read_packed_array_header(reader, is64)
        accessories = read_packed_array_header(reader, is64)
        valid = read_packed_array_header(reader, is64)
        bone_bounds_low = read_packed_array_header(reader, is64)
        bone_bounds_high = read_packed_array_header(reader, is64)
        if triplet_valid(data, sections, bones, accessories):
            return sections, bones, accessories, valid, bone_bounds_low, bone_bounds_high
    except Exception:
        return None
    return None


def i32_at(data: bytes, offset: int, big: bool) -> int:
    if offset < 0 or offset + 4 > len(data):
        return 0
    return struct.unpack_from(">i" if big else "<i", data, offset)[0]


def u32_at(data: bytes, offset: int, big: bool) -> int:
    if offset < 0 or offset + 4 > len(data):
        return 0
    return struct.unpack_from(">I" if big else "<I", data, offset)[0]


def u64_at(data: bytes, offset: int, big: bool) -> int:
    if offset < 0 or offset + 8 > len(data):
        return 0
    return struct.unpack_from(">Q" if big else "<Q", data, offset)[0]


def short_cached_string_score(data: bytes, offset: int) -> int:
    if offset == 0xFFFFFFFF or offset == 0xFFFFFFFFFFFFFFFF:
        return 1
    if offset < 0 or offset >= len(data):
        return 0
    text = read_cstring_at(data, offset)
    if len(text) > 64:
        return 0
    return 2 if text and all(32 <= ord(char) < 127 for char in text) else 1


def v4_section_record_score(cached: bytes, vertex_size: int, index_size: int, header: PackedArrayHeader, big: bool, is64: bool, record_size: int) -> int:
    score = 0
    samples = min(header.length, 8)
    if samples <= 0 or header.offset + record_size > len(cached):
        return -1
    for index in range(samples):
        pos = header.offset + index * record_size
        if pos + min(record_size, 144) > len(cached):
            break
        material_index = i32_at(cached, pos, big)
        index_offset = i32_at(cached, pos + 16, big) * 2
        tri_count = i32_at(cached, pos + 20, big)
        vertex_offset = i32_at(cached, pos + 24, big)
        vertex_buffer_size = i32_at(cached, pos + 28, big)
        vertex_stride = i32_at(cached, pos + 32, big)
        vertex_count = i32_at(cached, pos + 36, big)
        if -1 <= material_index < 4096:
            score += 1
        if 0 <= index_offset <= index_size and 0 <= tri_count < 1000000 and index_offset + tri_count * 6 <= index_size:
            score += 3
        if 0 < vertex_stride <= 256 and 0 <= vertex_count < 1000000 and 0 <= vertex_offset <= vertex_size and vertex_offset + vertex_count * vertex_stride <= vertex_size:
            score += 4
        if vertex_buffer_size == vertex_count * vertex_stride:
            score += 1
        pack_offset = u64_at(cached, pos + 56, big) if is64 else u32_at(cached, pos + 48, big)
        decl_offset = u64_at(cached, pos + 64, big) if is64 else u32_at(cached, pos + 52, big)
        score += short_cached_string_score(cached, pack_offset)
        score += short_cached_string_score(cached, decl_offset)
        types_offset = pos + (72 if is64 else 56)
        if types_offset + 17 * 4 <= len(cached):
            types = [u32_at(cached, types_offset + i * 4, big) for i in range(17)]
            score += sum(1 for value in types if value <= 20 or value == 0xFFFFFFFF)
        if is64 and record_size == 152 and pos + 152 <= len(cached):
            # Native DE x64 padding makes the two candidate sizes
            # distinguishable even for a one-section file.
            native_tail = cached[pos + 140:pos + 152]
            score += 12 if (
                native_tail[:4] == b"\0" * 4
                and native_tail[4] in (0, 1)
                and native_tail[5:] == b"\0" * 7
            ) else -12
        elif is64 and record_size == 144 and pos + 144 <= len(cached):
            legacy_tail = cached[pos + 140:pos + 144]
            score += 3 if legacy_tail[0] in (0, 1) and legacy_tail[1:] == b"\0" * 3 else -3
    return score


def v4_section_record_size(cached: bytes, vertex_size: int, index_size: int, header: PackedArrayHeader, big: bool, is64: bool) -> int:
    # HaloWarsDocs UGX.bt documents the original 144-byte layout and flags the
    # HWDE change as a TODO. The native x64 BSection in Ensemble's source is
    # 152 bytes: UnivertPacker has four bytes of tail alignment before
    # mRigidOnly, and BSection has another four bytes of tail alignment.
    # Every shipped DE PC v4 section uses 152. Keep 144 only so files emitted
    # by older addon builds remain recoverable in Blender.
    if not is64:
        return 144
    scores = {
        144: v4_section_record_score(cached, vertex_size, index_size, header, big, is64, 144),
        152: v4_section_record_score(cached, vertex_size, index_size, header, big, is64, 152),
    }
    return 152 if scores[152] >= scores[144] else 144


def read_indices(data: bytes, offset: int, count: int, vertex_count: int, big: bool) -> List[int]:
    result: List[int] = []
    if offset < 0 or count <= 0 or offset >= len(data):
        return result
    count = min(count, (len(data) - offset) // 2)
    return [index for (index,) in struct.iter_unpack('>H' if big else '<H', memoryview(data)[offset:offset + count * 2]) if index < vertex_count]


def read_vertices_v4(data: bytes, offset: int, count: int, stride: int, layout: List[VertexLayout]) -> List[Vertex]:
    if offset < 0 or count <= 0 or stride <= 0 or offset >= len(data):
        return []
    count = min(count, (len(data) - offset) // stride)
    pos_layout = layout_for(layout, "P")
    normal_layout = layout_for(layout, "N")
    tangent_layout = tangent_layout_for(layout)
    uv_layouts = [(attr_name, layout_for(layout, semantic)) for semantic, _, attr_name in UV_SEMANTICS]
    index_layout = layout_for(layout, "I")
    weight_layout = layout_for(layout, "W")
    color_layout = layout_for(layout, "D")
    vertices: List[Vertex] = []
    for i in range(count):
        start = offset + i * stride
        end = min(len(data), start + stride)
        vertex = Vertex()
        if color_layout and start + color_layout.offset + color_layout.size <= end:
            vertex.color = decode_vertex_color(data, start + color_layout.offset, color_layout.type_id)
        if pos_layout and start + pos_layout.offset + pos_layout.size <= end:
            vertex.pos = decode_vec3(data, start + pos_layout.offset, pos_layout.type_id, vertex.pos)
        if normal_layout and start + normal_layout.offset + normal_layout.size <= end:
            vertex.normal = decode_vec3(data, start + normal_layout.offset, normal_layout.type_id, vertex.normal)
        if tangent_layout and start + tangent_layout.offset + tangent_layout.size <= end:
            vertex.tangent, vertex.tangent_bits = decode_tangent(data, start + tangent_layout.offset, tangent_layout.type_id, vertex.tangent)
            # Retail HW1 stores tangent-frame handedness in the tangent length.
            # Its vertex shader flips cross(T, N) when dot(T, T) is below 0.5.
            tangent_length_squared = sum(component * component for component in vertex.tangent)
            vertex.bitangent_sign = -1.0 if tangent_length_squared < 0.5 else 1.0
        for attr_name, uv_layout in uv_layouts:
            if uv_layout and start + uv_layout.offset + uv_layout.size <= end:
                setattr(vertex, attr_name, decode_vec2(data, start + uv_layout.offset, uv_layout.type_id, getattr(vertex, attr_name)))
        if index_layout and weight_layout and start + index_layout.offset + index_layout.size <= end and start + weight_layout.offset + weight_layout.size <= end:
            raw_indices = tuple(data[start + index_layout.offset:start + index_layout.offset + 4].ljust(4, b"\0"))
            raw_weights = tuple(data[start + weight_layout.offset:start + weight_layout.offset + 4].ljust(4, b"\0"))
            total = sum(raw_weights)
            vertex.bone_indices = raw_indices
            vertex.bone_weights = tuple((w / total) if total else 0.0 for w in raw_weights)
        vertices.append(vertex)
    return vertices


def v6_uv_offset(stride: int) -> int:
    return 8 if stride >= 12 else -1


def v6_skin_offset(data: bytes, offset: int, count: int, stride: int, max_influences: int, bone_count: int, shader_offset=None) -> int:
    if stride < 32 or count <= 0 or offset < 0 or offset + count * stride > len(data):
        return 20
    influences = max_influences if 1 <= max_influences <= 4 else 4
    base_valid = True
    alternate_valid = True
    # Some HW2 streams insert COLOR0 before the quantized skin fields.
    # Require the complete alternate stream to satisfy its bone and weight contract.
    for index in range(count):
        start = offset + index * stride
        base_weights = data[start + 24:start + 28]
        base_valid &= sum(base_weights) == 255 or sum(base_weights[:influences]) == 255
        alternate_indices = data[start + 24:start + 28]
        alternate_weights = data[start + 28:start + 32]
        alternate_valid &= sum(alternate_weights) == 255 and all(not weight or bone < bone_count for bone, weight in zip(alternate_indices, alternate_weights))
        if not alternate_valid and shader_offset != 24:
            return 20
    if base_valid != alternate_valid:
        return 20 if base_valid else 24
    return shader_offset if shader_offset in (20, 24) else 20


def build_v6_layout(data: bytes, offset: int, count: int, stride: int, skinned: bool, skin_offset: int = 20) -> Tuple[List[VertexLayout], str]:
    layout = [VertexLayout("P", 17, 0, 6)]
    uv_offset = v6_uv_offset(stride)
    if uv_offset >= 0 and uv_offset + 4 <= stride:
        layout.append(VertexLayout("T0", 16, uv_offset, 4))
    tangent_offset = 12
    if tangent_offset + 4 <= stride:
        layout.append(VertexLayout("A0", V6_PACKED_BASIS_TYPE, tangent_offset, 4))
        normal_offset = tangent_offset + 4
        if normal_offset + 4 <= stride:
            layout.append(VertexLayout("N", V6_PACKED_BASIS_TYPE, normal_offset, 4))
    if skinned and skin_offset + 8 <= stride:
        layout.append(VertexLayout("I", 6, skin_offset, 4))
        layout.append(VertexLayout("W", 9, skin_offset + 4, 4))
    color_offset = 28 if skinned and skin_offset == 20 else 20
    if color_offset + 4 <= stride:
        layout.append(VertexLayout('D', 9, color_offset, 4))
    return layout, f"v6_pos_half4_uv_half2@{uv_offset}" if uv_offset >= 0 else "v6_pos_half4"


def read_vertices_v6(data: bytes, offset: int, count: int, stride: int, skinned: bool, max_influences: int = 4, skin_offset: int = 20, bone_count: int = 256) -> List[Vertex]:
    if offset < 0 or count <= 0 or stride < 8 or offset >= len(data):
        return []
    count = min(count, (len(data) - offset) // stride)
    if np is not None and stride >= 20 and count:
        return read_vertices_v6_bulk(data, offset, count, stride, skinned, max_influences, skin_offset, bone_count)
    uv_offset = v6_uv_offset(stride)
    vertices: List[Vertex] = []
    for i in range(count):
        start = offset + i * stride
        vertex = Vertex()
        position = struct.unpack_from('<3e', data, start)
        vertex.pos = tuple(component if math.isfinite(component) else 0.0 for component in position)
        if uv_offset >= 0 and start + uv_offset + 4 <= len(data):
            uv = decode_vec2(data, start + uv_offset, 16, vertex.uv0)
            if all(math.isfinite(component) for component in uv):
                vertex.uv0 = uv
        tangent_offset = uv_offset + 4 if uv_offset >= 0 else 12
        if tangent_offset + 4 <= stride:
            vertex.tangent = decode_hw2_packed_tangent(data, start + tangent_offset)
            vertex.tangent_bits = 1
        normal_offset = tangent_offset + 4
        if normal_offset + 4 <= stride:
            vertex.normal, vertex.bitangent_sign = decode_hw2_packed_normal(data, start + normal_offset)
        color_offset = 28 if skinned and skin_offset == 20 else 20
        if color_offset + 4 <= stride:
            vertex.color = decode_vertex_color(data, start + color_offset, 9)
        if skinned and skin_offset + 8 <= stride:
            influences = max_influences if 1 <= max_influences <= 4 else 4
            indices = tuple(data[start + skin_offset:start + skin_offset + 4])
            weights = tuple(data[start + skin_offset + 4:start + skin_offset + 8])
            # Retail streams can understate MaxBones while carrying a complete four-byte weight vector.
            if sum(weights) == 255 and all(not weight or bone < bone_count for bone, weight in zip(indices, weights)):
                influences = 4
            raw_weights = weights[:influences] + (0,) * (4 - influences)
            raw_indices = tuple(bone if weight else 0 for bone, weight in zip(indices, raw_weights))
            total = sum(raw_weights)
            vertex.bone_indices = raw_indices
            vertex.bone_weights = tuple((w / total) if total else 0.0 for w in raw_weights)
        vertices.append(vertex)
    return vertices


def read_vertices_v6_bulk(data, offset, count, stride, skinned, max_influences, skin_offset, bone_count):
    def field(relative, dtype, width):
        return np.ndarray((count, width), dtype=dtype, buffer=data, offset=offset + relative,
                          strides=(stride, np.dtype(dtype).itemsize))

    def basis(relative, signed, fallback):
        # Signaling NaNs follow the scalar decoder's finite-value fallback.
        with np.errstate(invalid='ignore'):
            packed = field(relative, '<f4', 1).astype(np.float64)
        finite = np.isfinite(packed[:, 0])
        value = np.where(np.isfinite(packed), packed, 0)
        if not signed:
            value = np.abs(value)
        scaled = value * (1.0, 256.0, 65536.0)
        decoded = (scaled - np.floor(scaled)) * 2.0 - 1.0
        lengths = np.sqrt(np.sum(decoded * decoded, axis=1))
        valid = finite & np.isfinite(lengths) & (lengths > 0.0000001)
        decoded[valid] /= lengths[valid, None]
        decoded[~valid] = fallback
        return decoded.tolist(), packed[:, 0]

    positions = field(0, '<f2', 3).astype(np.float64)
    positions[~np.isfinite(positions)] = 0
    uvs = field(8, '<f2', 2).astype(np.float64)
    uvs[~np.all(np.isfinite(uvs), axis=1)] = 0
    tangents, _ = basis(12, True, (1, 0, 0))
    normals, packed_normal = basis(16, False, (0, 1, 0))
    signs = np.where(packed_normal > 0, -1.0, 1.0)
    color_offset = 28 if skinned and skin_offset == 20 else 20
    colors = (field(color_offset, 'u1', 4).astype(np.float64)/255).tolist() if color_offset + 4 <= stride else [(1., 1., 1., 1.)]*count
    if skinned and skin_offset + 8 <= stride:
        indices = field(skin_offset, 'u1', 4).copy()
        weights = field(skin_offset + 4, 'u1', 4).copy()
        influences = max_influences if 1 <= max_influences <= 4 else 4
        complete = (weights.sum(axis=1) == 255) & np.all((weights == 0) | (indices < bone_count), axis=1)
        weights[~complete, influences:] = 0
        indices[weights == 0] = 0
        totals = weights.sum(axis=1)
        normalized = weights / np.where(totals, totals, 1)[:, None]
        return [Vertex(tuple(pos), tuple(normal), tuple(tangent), 1, float(sign), tuple(uv),
                       bone_indices=tuple(bones), bone_weights=tuple(values), color=tuple(color))
                for pos, normal, tangent, sign, uv, bones, values, color in
                zip(positions.tolist(), normals, tangents, signs, uvs.tolist(), indices.tolist(), normalized.tolist(), colors)]
    return [Vertex(tuple(pos), tuple(normal), tuple(tangent), 1, float(sign), tuple(uv), color=tuple(color))
            for pos, normal, tangent, sign, uv, color in zip(positions.tolist(), normals, tangents, signs, uvs.tolist(), colors)]


def read_matrix(reader: Reader) -> List[List[float]]:
    return [[reader.f32() for _ in range(4)] for _ in range(4)]


def read_cached_data(cached: Chunk, vertex_buffer: Chunk, index_buffer: Chunk, reuse_geometry_from=None, skin_offsets=None) -> Tuple[int, List[Section], List[Bone], List[Accessory], CachedBounds]:
    sig_be = Reader(cached.data, True).u32()
    sig_le = Reader(cached.data, False).u32()
    can_hw1_be = sig_be == CACHED_SIGNATURE_HW1_BE
    can_hw1_le = sig_be == CACHED_SIGNATURE_HW1_LE or sig_le == CACHED_SIGNATURE_HW1_LE
    can_hw2 = sig_be == CACHED_SIGNATURE_HW2 or sig_le == CACHED_SIGNATURE_HW2
    if not (can_hw1_be or can_hw1_le or can_hw2):
        raise UgxError("Unsupported UGX cached-data signature.")

    cache_version = 6 if can_hw2 else 4
    big = can_hw1_be
    is64 = not big
    if can_hw1_le or can_hw2:
        big = False
        is64 = True

    bounds = CachedBounds(big_endian=big)
    if len(cached.data) >= 48:
        bounds_reader = Reader(cached.data, big)
        bounds_reader.seek(8)
        bounds.center = (bounds_reader.f32(), bounds_reader.f32(), bounds_reader.f32())
        bounds.radius = bounds_reader.f32()
        bounds.bounds_min = (bounds_reader.f32(), bounds_reader.f32(), bounds_reader.f32())
        bounds.bounds_max = (bounds_reader.f32(), bounds_reader.f32(), bounds_reader.f32())
        bounds.center_offset = cached.offset + 8
        bounds.radius_offset = cached.offset + 20
        bounds.bounds_min_offset = cached.offset + 24
        bounds.bounds_max_offset = cached.offset + 36

    reader = Reader(cached.data, big)
    reader.skip(4)
    reader.skip(60 if cache_version == 6 or (cache_version == 4 and is64) else 56)
    sections_header = read_packed_array_header(reader, is64)
    bones_header = read_packed_array_header(reader, is64)
    accessories_header = read_packed_array_header(reader, is64)
    valid_accessories_header = read_packed_array_header(reader, is64)
    bone_bounds_low_header = read_packed_array_header(reader, is64) if reader.can_read(16 if is64 else 8) else PackedArrayHeader(0, 0)
    bone_bounds_high_header = read_packed_array_header(reader, is64) if reader.can_read(16 if is64 else 8) else PackedArrayHeader(0, 0)
    if not triplet_valid(cached.data, sections_header, bones_header, accessories_header):
        for offset in range(32, 128, 4):
            found = try_read_array_headers(cached.data, big, is64, offset)
            if found:
                sections_header, bones_header, accessories_header, valid_accessories_header, bone_bounds_low_header, bone_bounds_high_header = found
                break
    _ = valid_accessories_header

    sections: List[Section] = []
    if sections_header.offset < len(cached.data):
        reader.seek(sections_header.offset)
        record_size = 72 if cache_version == 6 else v4_section_record_size(cached.data, len(vertex_buffer.data), len(index_buffer.data), sections_header, big, is64)
        for _ in range(sections_header.length):
            if not reader.can_read(record_size):
                break
            section_start = reader.pos
            section = Section()
            section.cache_version = cache_version
            section.section_record_file_offset = cached.offset + section_start
            section.material_index_offset = cached.offset + reader.pos
            section.big_endian = big
            section.material_index = reader.i32()
            section.accessory_index_offset = cached.offset + reader.pos
            section.accessory_index = reader.i32()
            section.max_bones_offset = cached.offset + reader.pos
            section.max_bones = reader.i32()
            section.rigid_bone_index_offset = cached.offset + reader.pos
            section.rigid_bone_index = reader.i32()
            section.index_offset_field_offset = cached.offset + reader.pos
            index_offset = reader.i32() * 2
            section.tri_count_field_offset = cached.offset + reader.pos
            index_count = reader.i32() * 3
            section.vertex_offset_field_offset = cached.offset + reader.pos
            vertex_offset = reader.i32()
            section.vertex_size_field_offset = cached.offset + reader.pos
            vertex_buffer_size = reader.i32()
            section.vertex_stride = reader.i32()
            section.vertex_count_field_offset = cached.offset + reader.pos
            vertex_count = reader.i32()
            section.index_offset = max(index_offset, 0)
            section.vertex_offset = max(vertex_offset, 0)
            section.vertex_buffer_size = max(vertex_buffer_size, 0)
            section.vertex_file_offset = vertex_buffer.offset + section.vertex_offset
            section.vertex_count = max(vertex_count, 0)
            if cache_version == 4:
                remap = read_packed_array_header(reader, is64)
                pack_order_offset = read_packed_string_offset(reader, is64)
                decl_order_offset = read_packed_string_offset(reader, is64)
                types = [reader.u32() for _ in range(17)]
                if record_size == 152:
                    reader.skip(4)
                section.rigid_only_offset = cached.offset + reader.pos
                section.rigid_only = reader.u8() != 0
                reader.skip(3)
                section.pack_order = read_cstring_at(cached.data, pack_order_offset)
                section.decl_order = read_cstring_at(cached.data, decl_order_offset)
                layout = build_vertex_layout(section.pack_order, section.decl_order, types)
                section.layout = layout
                if remap.offset < len(cached.data):
                    back = reader.pos
                    reader.seek(remap.offset)
                    for _ in range(remap.length):
                        if reader.can_read(4):
                            section.local_to_global_bones.append(reader.i32())
                    reader.seek(back)
            else:
                section.rigid_only_offset = cached.offset + reader.pos
                section.rigid_only = reader.u8() != 0
                reader.skip(3)
                section.lod_min_distance_offset = cached.offset + reader.pos
                section.lod_min_distance = reader.f32()
                section.lod_max_distance_offset = cached.offset + reader.pos
                section.lod_max_distance = reader.f32()
                section.sorting_bias_offset = cached.offset + reader.pos
                section.sorting_bias = reader.f32()
                remap = read_packed_array_header(reader, is64)
                if remap.offset < len(cached.data):
                    back = reader.pos
                    reader.seek(remap.offset)
                    for _ in range(remap.length):
                        if reader.can_read(4):
                            section.local_to_global_bones.append(reader.i32())
                    reader.seek(back)
                skin_offset = v6_skin_offset(vertex_buffer.data, vertex_offset, vertex_count, section.vertex_stride, section.max_bones, len(section.local_to_global_bones) or bones_header.length, (skin_offsets or {}).get(section.material_index)) if not section.rigid_only else 20
                section.layout, section.pack_order = build_v6_layout(vertex_buffer.data, vertex_offset, vertex_count, section.vertex_stride, not section.rigid_only, skin_offset)
            source_vertex_end = section.vertex_offset + section.vertex_count * section.vertex_stride
            if 0 <= section.vertex_offset <= source_vertex_end <= len(vertex_buffer.data):
                section.source_vertex_data = bytes(vertex_buffer.data[section.vertex_offset:source_vertex_end])
            previous = reuse_geometry_from.sections[len(sections)] if reuse_geometry_from is not None and len(sections) < len(reuse_geometry_from.sections) else None
            if (previous is not None and previous.cache_version == cache_version
                    and previous.vertex_count == section.vertex_count and previous.vertex_stride == section.vertex_stride
                    and previous.layout == section.layout and previous.rigid_only == section.rigid_only
                    and previous.max_bones == section.max_bones
                    and previous.local_to_global_bones == section.local_to_global_bones
                    and len(reuse_geometry_from.bones) == bones_header.length
                    and previous.source_vertex_data == section.source_vertex_data):
                # Return independent records. Native buffers and layout must
                # match exactly before reusing their decoded scalar values.
                section.vertices = [Vertex(v.pos, v.normal, v.tangent, v.tangent_bits, v.bitangent_sign,
                                           v.uv0, v.uv1, v.uv2, v.bone_indices, v.bone_weights, v.color) for v in previous.vertices]
            elif cache_version == 4:
                section.vertices = read_vertices_v4(vertex_buffer.data, vertex_offset, vertex_count, section.vertex_stride, section.layout)
            else:
                section.vertices = read_vertices_v6(vertex_buffer.data, vertex_offset, vertex_count, section.vertex_stride, not section.rigid_only, section.max_bones, skin_offset, len(section.local_to_global_bones) or bones_header.length)
            section.indices = read_indices(index_buffer.data, index_offset, index_count, len(section.vertices), big)
            sections.append(section)
            reader.seek(section_start + record_size)

    bones: List[Bone] = []
    if bones_header.offset < len(cached.data):
        reader.seek(bones_header.offset)
        record_size = (8 if is64 else 4) + 64 + 4 + (4 if is64 else 0)
        for _ in range(bones_header.length):
            if not reader.can_read(record_size):
                break
            name_offset_field_offset = cached.offset + reader.pos
            name_offset = read_packed_string_offset(reader, is64)
            name = read_cstring_at(cached.data, name_offset)
            name_capacity = cstring_capacity_at(cached.data, name_offset)
            matrix_offset = reader.pos
            matrix = read_matrix(reader)
            parent = reader.i32()
            if is64 and reader.can_read(4):
                reader.skip(4)
            bones.append(Bone(name, matrix, parent, cached.offset + name_offset if name_capacity else 0, name_capacity, name_offset_field_offset, 8 if is64 else 4, cached.offset + matrix_offset, big))

    accessories: List[Accessory] = []
    if accessories_header.offset < len(cached.data):
        reader.seek(accessories_header.offset)
        record_size = 8 + (16 if is64 else 8)
        for _ in range(accessories_header.length):
            if not reader.can_read(record_size):
                break
            accessory_start = reader.pos
            first_bone = reader.i32()
            bone_count = reader.i32()
            section_indices_header = read_packed_array_header(reader, is64)
            section_indices: List[int] = []
            if section_indices_header.offset < len(cached.data):
                back = reader.pos
                reader.seek(section_indices_header.offset)
                for _ in range(section_indices_header.length):
                    if not reader.can_read(4):
                        break
                    section_indices.append(reader.i32())
                reader.seek(back)
            accessories.append(Accessory(first_bone, bone_count, section_indices))
            reader.seek(accessory_start + record_size)

    for section in sections:
        if 0 <= section.accessory_index < len(accessories):
            accessory = accessories[section.accessory_index]
            section.accessory_first_bone = accessory.first_bone
            section.accessory_bone_count = accessory.bone_count

    for header, attr, offset_attr in (
        (bone_bounds_low_header, "bounds_low", "bounds_low_offset"),
        (bone_bounds_high_header, "bounds_high", "bounds_high_offset"),
    ):
        if header.offset >= len(cached.data):
            continue
        reader.seek(header.offset)
        for index in range(min(header.length, len(bones))):
            if not reader.can_read(12):
                break
            setattr(bones[index], offset_attr, cached.offset + reader.pos)
            setattr(bones[index], attr, (reader.f32(), reader.f32(), reader.f32()))

    return cache_version, sections, bones, accessories, bounds


def read_materials(data: Optional[bytes], required_count: int, file_base_offset: int = 0) -> List[Material]:
    materials: List[Material] = []
    if data:
        for root in read_bdt(data, file_base_offset):
            if root.name != "Materials":
                continue
            for material_node in root.children:
                if material_node.name != "Material":
                    continue
                name_value = material_node.attrs.get("Name", BdtValue("", ""))
                material = Material(name_value.text or f"material_{len(materials)}", [], name_value.file_offset, name_value.string_capacity, name_value.entry_offset,
                                    bdt_data=bytes(data),bdt_index=len(materials))
                material.native_name=name_value.text if 'Name' in material_node.attrs else None
                material.legacy_containers=tuple(name for name in ('NameValues','Maps') if child_named(material_node,name) is not None)
                name_values = child_named(material_node, "NameValues")
                if name_values:
                    for value_node in name_values.children:
                        if value_node.value_info and value_node.value_info.name:
                            material.v4_attrs[value_node.value_info.name] = bdt_value_json(value_node.value_info)
                        for name, value in value_node.attrs.items():
                            material.v4_attrs[name] = bdt_value_json(value)
                hogan = child_named(material_node, "HoganMaterial")
                material.has_hogan_material=hogan is not None
                if hogan:
                    material.ufx_attrs = {name: bdt_value_json(value) for name, value in hogan.attrs.items()}
                    for child in hogan.children:
                        if child.name in ("VSCBData", "PSCBData"):
                            buffer = ufx_constant_buffer_json(child)
                            if buffer is not None:
                                material.ufx_constant_buffers.append(buffer)
                textures = child_named(hogan, "textures") if hogan else None
                if textures is not None:
                    material.has_texture_array=True
                    material.source_texture_paths=tuple(textures.strings)
                if textures and textures.strings:
                    material.name = hogan_material_name(len(materials), name_value, hogan, textures)
                    texture_usages = hogan_texture_usages(textures.strings)
                    for index, texture_path in enumerate(textures.strings):
                        usage = texture_usages[index]
                        if not usage:
                            continue
                        slot = textures.string_slots[index] if index < len(textures.string_slots) else {"offset": 0, "capacity": 0}
                        material.slots.append({
                            "usage": usage,
                            "path": texture_path,
                            "offset": slot.get("offset", 0),
                            "capacity": slot.get("capacity", 0),
                            "hogan_index": index,
                        })
                maps = child_named(material_node, "Maps")
                if maps:
                    map_slots: List[Dict[str, Any]] = []
                    seen_map_usages = set()
                    for map_node in maps.children:
                        usage = v4_material_usage(map_node.name)
                        if usage:
                            seen_map_usages.add(usage)
                        map_data = child_named(map_node, "Map")
                        slot = {
                            "usage": usage or map_node.name,
                            "path": "",
                            "offset": 0,
                            "capacity": 0,
                            "channel": 0,
                            "flags": v4_material_slot_flags(usage, {}),
                            "bdt_category": True,
                            "empty_bdt_category": True,
                        }
                        if "UVWVel" in map_node.attrs:
                            slot["uvwvel"] = map_node.attrs["UVWVel"].text
                            slot["uvwvel_info"] = bdt_value_json(map_node.attrs["UVWVel"])
                            pair = bdt_float_pair(map_node.attrs.get("UVWVel"))
                            if pair is not None:
                                slot["uv_scroll"] = list(v4_material_uvw_values(usage, slot))
                        if map_data and "Name" in map_data.attrs:
                            path = map_data.attrs["Name"]
                            slot.update({
                                "usage": usage if usage in V4_MATERIAL_MAP_USAGES else map_node.name,
                                "path": path.text,
                                "offset": path.file_offset,
                                "capacity": path.string_capacity,
                                "channel": bdt_int(map_data.attrs.get("Channel"), 0),
                                "flags": bdt_int(map_data.attrs.get("Flags"), 0),
                                "empty_bdt_category": False,
                            })
                        if map_data:
                            for field in ('Channel', 'Flags'):
                                if field in map_data.attrs:
                                    slot[field.lower() + '_info'] = bdt_value_json(map_data.attrs[field])
                        if usage in V4_MATERIAL_MAP_USAGES or slot.get("path"):
                            map_slots.append(slot)
                    for usage in V4_MATERIAL_MAP_USAGES:
                        if usage in seen_map_usages:
                            continue
                        map_slots.append({
                            "usage": usage,
                            "path": "",
                            "offset": 0,
                            "capacity": 0,
                            "channel": 0,
                            "flags": v4_material_slot_flags(usage, {}),
                            "bdt_category": False,
                            "empty_bdt_category": True,
                            "generated_category": True,
                        })
                    path_prefix = "\\" if any(str(slot.get("path", "")).startswith("\\") for slot in map_slots) else ""
                    for slot in map_slots:
                        if bool(slot.get("empty_bdt_category", False)) and path_prefix:
                            slot["path_prefix"] = path_prefix
                    material.slots.extend(map_slots)
                materials.append(material)
    while len(materials) < required_count:
        materials.append(Material(f"material_{len(materials)}"))
    return materials


def parse_ugx_data(path: str, data: bytes, reuse_geometry_from=None) -> UgxModel:
    ecf = EcfFile.parse(data)
    if ecf.file_id != UGX_FILE_ID:
        raise UgxError("ECF file is not a UGX model.")
    cached = ecf.chunks.get(CACHED_DATA_CHUNK)
    vertex_buffer = ecf.chunks.get(VERTEX_BUFFER_CHUNK)
    index_buffer = ecf.chunks.get(INDEX_BUFFER_CHUNK)
    if not cached or not vertex_buffer or not index_buffer:
        raise UgxError("UGX is missing cached, vertex, or index buffer chunks.")
    material_chunk = ecf.chunks.get(MATERIAL_CHUNK)
    materials = read_materials(material_chunk.data if material_chunk else None, 0, material_chunk.offset if material_chunk else 0)
    skin_offsets = {}
    for index, material in enumerate(materials):
        name = str(material.ufx_attrs.get('shaderPermutationName0', {}).get('value', '')).upper()
        contract = HW2_SKINNED_SHADER_INPUTS.get(name)
        if contract is None:
            continue
        try:
            header_hash = parse_ufx_shader_hash(material.ufx_attrs.get('shaderPermutationHash0', {}).get('value', 0), 'Shader pass 0 UFX header hash')
        except (UgxError, ValueError):
            continue
        if header_hash == contract[0]:
            skin_offsets[index] = 24 if contract[1] else 20
    cache_version, sections, bones, accessories, cached_bounds = read_cached_data(cached, vertex_buffer, index_buffer, reuse_geometry_from, skin_offsets)
    required_materials = max([s.material_index for s in sections if s.material_index >= 0] + [-1]) + 1
    while len(materials) < required_materials:
        materials.append(Material(f'material_{len(materials)}'))
    granny_chunk = ecf.chunks.get(0x703)
    granny_data = repair_granny_model_binding_overlap(granny_chunk.data) if granny_chunk else None
    if granny_data is not None:
        validate_granny_layout(granny_data)
    granny_info = parse_granny_info(granny_data)
    return UgxModel(os.path.abspath(path), data, cache_version, sections, bones, materials, cached_bounds, accessories, granny_info)


def parse_ugx(path: str) -> UgxModel:
    with open(path, "rb") as handle:
        return parse_ugx_data(path, handle.read())
