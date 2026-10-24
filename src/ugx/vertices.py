"""Decode and encode packed native vertex attributes."""

from __future__ import annotations
from typing import Iterable, List, Optional, Tuple
import math
import struct
from ..formats.binary import u32le
from ..errors import UgxError
from .data import V6_PACKED_BASIS_TYPE, Vertex, VertexLayout

__all__ = ('half_to_float', 'u16le', 'f32le', 'sign_extend_10', 'decode_dec3n_bits', 'decode_dec3n', 'normalized_tuple', 'f32_value', 'decode_hw2_packed_basis_value', 'decode_hw2_packed_tangent', 'decode_hw2_packed_normal', 'encode_hw2_packed_magnitude', 'hw2_basis_angular_error', 'encode_hw2_packed_tangent_value', 'encode_hw2_packed_normal_value', 'write_hw2_packed_tangent', 'write_hw2_packed_normal', 'tangent_layout_for', 'pack_dec3n', 'write_dec3n', 'write_vec3_preserve_dec3n_bits', 'decode_tangent', 'source_tangent_bits', 'vertex_type_size', 'section_element_type', 'parse_vertex_tokens', 'build_vertex_layout', 'layout_for', 'decode_vec3', 'decode_vec2', 'write_half', 'write_f32', 'write_f32_endian', 'write_i32', 'write_vec3', 'write_normal_vec3', 'write_vec2', 'changed', 'decode_vertex_color', 'encode_vertex_color', 'expand_hw2_vertex_base')

def half_to_float(value: int) -> float:
    return struct.unpack("<e", struct.pack("<H", value & 0xFFFF))[0]


def decode_vertex_color(data, offset, type_id):
    if type_id == 4:
        return struct.unpack_from('<4f', data, offset)
    if type_id in (5, 9):
        channels = struct.unpack_from('<4B', data, offset)
        if type_id == 5:
            channels = (channels[2], channels[1], channels[0], channels[3])
        return tuple(value/255. for value in channels)
    return (1., 1., 1., 1.)


def encode_vertex_color(value, type_id):
    if len(value) != 4 or not all(math.isfinite(component) for component in value):
        raise UgxError('Vertex colours need four finite components.')
    if type_id == 4:
        if any(abs(component) > 3.4028234663852886e38 for component in value):
            raise UgxError('Float vertex colours exceed the 32-bit storage range.')
        return struct.pack('<4f', *value)
    if type_id not in (5, 9):
        raise UgxError(f'Vertex colour element type {type_id} is not writable yet.')
    if any(component < 0 or component > 1 for component in value):
        raise UgxError('Byte vertex colours need components between zero and one.')
    channels = [round(component*255) for component in value]
    if type_id == 5:
        channels[0], channels[2] = channels[2], channels[0]
    return bytes(channels)


def u16le(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def f32le(data: bytes, offset: int) -> float:
    return struct.unpack_from("<f", data, offset)[0]


def sign_extend_10(value: int) -> int:
    value &= 0x3FF
    return value | ~0x3FF if value & 0x200 else value


def decode_dec3n_bits(data: bytes, offset: int) -> Tuple[Tuple[float, float, float], int]:
    packed = u32le(data, offset)
    return (
        (
            sign_extend_10(packed) / 511.0,
            sign_extend_10(packed >> 10) / 511.0,
            sign_extend_10(packed >> 20) / 511.0,
        ),
        (packed >> 30) & 0x3,
    )


def decode_dec3n(data: bytes, offset: int) -> Tuple[float, float, float]:
    return decode_dec3n_bits(data, offset)[0]


def normalized_tuple(value: Tuple[float, float, float], fallback: Tuple[float, float, float]) -> Tuple[float, float, float]:
    length = math.sqrt(sum(float(component) * float(component) for component in value))
    if not math.isfinite(length) or length <= 0.0000001:
        return fallback
    return tuple(float(component) / length for component in value)


def f32_value(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def decode_hw2_packed_basis_value(
    packed: float,
    preserve_sign: bool,
    fallback: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Match the HW2 UFX shader's frac(p * {1,256,65536}) decoder."""
    if not math.isfinite(packed):
        return fallback
    value = float(packed) if preserve_sign else abs(float(packed))

    def component(scale: float) -> float:
        scaled = value * scale
        return (scaled - math.floor(scaled)) * 2.0 - 1.0

    return normalized_tuple(
        (component(1.0), component(256.0), component(65536.0)),
        fallback,
    )


def decode_hw2_packed_tangent(data: bytes, offset: int) -> Tuple[float, float, float]:
    return decode_hw2_packed_basis_value(f32le(data, offset), True, (1.0, 0.0, 0.0))


def decode_hw2_packed_normal(data: bytes, offset: int) -> Tuple[Tuple[float, float, float], float]:
    packed = f32le(data, offset)
    normal = decode_hw2_packed_basis_value(packed, False, (0.0, 1.0, 0.0))
    # Native Cutter/UFX convention: positive packed normal means a negative
    # bitangent sign; zero/negative means positive.
    return normal, (-1.0 if packed > 0.0 else 1.0)


def encode_hw2_packed_magnitude(value: Tuple[float, float, float]) -> float:
    vector = normalized_tuple(value, (0.0, 1.0, 0.0))
    quantized = [max(0, min(255, round((component + 1.0) * 127.5))) for component in vector]
    packed_bits = (quantized[0] << 16) | (quantized[1] << 8) | quantized[2]
    return f32_value(packed_bits / 16777216.0)


def hw2_basis_angular_error(left: Tuple[float, float, float], right: Tuple[float, float, float]) -> float:
    a = normalized_tuple(left, (1.0, 0.0, 0.0))
    b = normalized_tuple(right, (1.0, 0.0, 0.0))
    return 1.0 - max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))


def encode_hw2_packed_tangent_value(value: Tuple[float, float, float]) -> float:
    target = normalized_tuple(value, (1.0, 0.0, 0.0))
    positive = encode_hw2_packed_magnitude(target)
    negative = f32_value(-encode_hw2_packed_magnitude(tuple(-component for component in target)))
    positive_error = hw2_basis_angular_error(
        decode_hw2_packed_basis_value(positive, True, (1.0, 0.0, 0.0)), target
    )
    negative_error = hw2_basis_angular_error(
        decode_hw2_packed_basis_value(negative, True, (1.0, 0.0, 0.0)), target
    )
    return negative if negative_error < positive_error else positive


def encode_hw2_packed_normal_value(value: Tuple[float, float, float], bitangent_sign: float) -> float:
    magnitude = encode_hw2_packed_magnitude(normalized_tuple(value, (0.0, 1.0, 0.0)))
    if magnitude == 0.0:
        magnitude = f32_value(1.0 / 16777216.0)
    return magnitude if float(bitangent_sign) < 0.0 else f32_value(-magnitude)


def write_hw2_packed_tangent(data: bytearray, offset: int, value: Tuple[float, float, float]):
    write_f32(data, offset, encode_hw2_packed_tangent_value(value))


def write_hw2_packed_normal(
    data: bytearray,
    offset: int,
    value: Tuple[float, float, float],
    bitangent_sign: float,
):
    write_f32(data, offset, encode_hw2_packed_normal_value(value, bitangent_sign))


def expand_hw2_vertex_base(raw: bytearray, source: Vertex) -> bytearray:
    if len(raw) >= 20:
        return raw
    if len(raw) < 8:
        raise UgxError('HW2 vertex stream is shorter than its position field.')
    known = 12 if len(raw) >= 12 else 8
    tail = raw[known:]
    raw = raw[:known] + bytes(20-known)
    if known == 8:
        struct.pack_into('<2e', raw, 8, *source.uv0)
    write_hw2_packed_tangent(raw, 12, source.tangent)
    write_hw2_packed_normal(raw, 16, source.normal, source.bitangent_sign)
    raw.extend(tail)
    return raw


def tangent_layout_for(layout: Iterable[VertexLayout]) -> Optional[VertexLayout]:
    return layout_for(layout, "A0") or layout_for(layout, "A")


def pack_dec3n(value: Tuple[float, float, float], bits: int = 1) -> int:
    packed = 0
    for i, component in enumerate(value):
        quantized = max(-511, min(511, round(component * 511.0))) & 0x3FF
        packed |= quantized << (i * 10)
    return packed | ((bits & 0x3) << 30)


def write_dec3n(data: bytearray, offset: int, value: Tuple[float, float, float], bits: int = 1):
    struct.pack_into("<I", data, offset, pack_dec3n(value, bits))


def write_vec3_preserve_dec3n_bits(data: bytearray, offset: int, type_id: int, value: Tuple[float, float, float], bits: int) -> bool:
    if type_id == 15:
        write_dec3n(data, offset, value, bits)
        return True
    return write_vec3(data, offset, type_id, value)


def decode_tangent(data: bytes, offset: int, type_id: int, fallback: Tuple[float, float, float]) -> Tuple[Tuple[float, float, float], int]:
    if type_id == 15 and offset + 4 <= len(data):
        return decode_dec3n_bits(data, offset)
    return decode_vec3(data, offset, type_id, fallback), 1


def source_tangent_bits(source_vertex: Vertex, tangent_layout: Optional[VertexLayout]) -> int:
    if tangent_layout is not None and tangent_layout.type_id == 15:
        return source_vertex.tangent_bits
    return 1


def vertex_type_size(type_id: int) -> int:
    return {
        0: 0,
        1: 4,
        2: 8,
        3: 12,
        4: 16,
        5: 4,
        6: 4,
        7: 4,
        8: 8,
        9: 4,
        10: 4,
        11: 8,
        12: 4,
        13: 8,
        14: 4,
        15: 4,
        16: 4,
        17: 8,
        18: 2,
        19: 4,
        20: 4,
        V6_PACKED_BASIS_TYPE: 4,
    }.get(type_id, 0)


def section_element_type(types: List[int], semantic: str) -> int:
    if semantic == "P":
        return types[0]
    if semantic == "B" or (len(semantic) == 2 and semantic[0] == "B" and semantic[1].isdigit()):
        return types[1]
    if semantic in ("G", "X") or (len(semantic) == 2 and semantic[0] in ("G", "X") and semantic[1].isdigit()):
        return types[2]
    if semantic == "A" or (len(semantic) == 2 and semantic[0] == "A" and semantic[1].isdigit()):
        return types[3]
    if semantic == "N":
        return types[4]
    if len(semantic) == 2 and semantic[0] == "T" and semantic[1].isdigit():
        index = 5 + int(semantic[1])
        return types[index] if index < len(types) else 0
    if semantic == "I":
        return types[13]
    if semantic == "W":
        return types[14]
    if semantic == "D":
        return types[15]
    if semantic == "X":
        return types[16]
    return 0


def parse_vertex_tokens(pack_order: str, decl_order: str) -> List[str]:
    order = decl_order or pack_order
    tokens: List[str] = []
    i = 0
    while i < len(order):
        c = order[i]
        if c in ("T", "A", "B", "G", "X") and i + 1 < len(order) and order[i + 1].isdigit():
            tokens.append(order[i:i + 2])
            i += 2
            continue
        if c == "S":
            tokens.extend(["I", "W"])
        elif not c.isspace():
            tokens.append(c)
        i += 1
    return tokens


def build_vertex_layout(pack_order: str, decl_order: str, types: List[int]) -> List[VertexLayout]:
    layout: List[VertexLayout] = []
    offset = 0
    for token in parse_vertex_tokens(pack_order, decl_order):
        type_id = section_element_type(types, token)
        size = vertex_type_size(type_id)
        if size:
            layout.append(VertexLayout(token, type_id, offset, size))
            offset += size
    return layout


def layout_for(layout: Iterable[VertexLayout], semantic: str) -> Optional[VertexLayout]:
    return next((item for item in layout if item.semantic == semantic), None)


def decode_vec3(data: bytes, offset: int, type_id: int, fallback: Tuple[float, float, float]) -> Tuple[float, float, float]:
    size = vertex_type_size(type_id)
    if offset + size > len(data):
        return fallback
    if type_id == 3:
        return f32le(data, offset), f32le(data, offset + 4), f32le(data, offset + 8)
    if type_id == 4:
        return f32le(data, offset), f32le(data, offset + 4), f32le(data, offset + 8)
    if type_id == 17:
        return half_to_float(u16le(data, offset)), half_to_float(u16le(data, offset + 2)), half_to_float(u16le(data, offset + 4))
    if type_id == 11:
        values = [max(-1.0, struct.unpack_from("<h", data, offset + i * 2)[0] / 32767.0) for i in range(3)]
        return values[0], values[1], values[2]
    if type_id == 13:
        return u16le(data, offset) / 65535.0, u16le(data, offset + 2) / 65535.0, u16le(data, offset + 4) / 65535.0
    if type_id == 9:
        return data[offset] / 255.0 * 2.0 - 1.0, data[offset + 1] / 255.0 * 2.0 - 1.0, data[offset + 2] / 255.0 * 2.0 - 1.0
    if type_id == 15:
        return decode_dec3n(data, offset)
    if type_id == 19:
        packed = u32le(data, offset)
        return (packed & 0x3FF) / 1023.0, ((packed >> 10) & 0x3FF) / 1023.0, ((packed >> 20) & 0x3FF) / 1023.0
    if type_id == V6_PACKED_BASIS_TYPE:
        return decode_hw2_packed_tangent(data, offset)
    return fallback


def decode_vec2(data: bytes, offset: int, type_id: int, fallback: Tuple[float, float]) -> Tuple[float, float]:
    size = vertex_type_size(type_id)
    if offset + size > len(data):
        return fallback
    if type_id == 2:
        return f32le(data, offset), f32le(data, offset + 4)
    if type_id == 16:
        return half_to_float(u16le(data, offset)), half_to_float(u16le(data, offset + 2))
    if type_id == 10:
        return max(-1.0, struct.unpack_from("<h", data, offset)[0] / 32767.0), max(-1.0, struct.unpack_from("<h", data, offset + 2)[0] / 32767.0)
    if type_id == 12:
        return u16le(data, offset) / 65535.0, u16le(data, offset + 2) / 65535.0
    return fallback


def write_half(data: bytearray, offset: int, value: float):
    struct.pack_into("<e", data, offset, float(value))


def write_f32(data: bytearray, offset: int, value: float):
    struct.pack_into("<f", data, offset, float(value))


def write_f32_endian(data: bytearray, offset: int, value: float, big: bool):
    struct.pack_into(">f" if big else "<f", data, offset, float(value))


def write_i32(data: bytearray, offset: int, value: int, big: bool):
    struct.pack_into(">i" if big else "<i", data, offset, int(value))


def write_vec3(data: bytearray, offset: int, type_id: int, value: Tuple[float, float, float]) -> bool:
    if type_id == 3:
        for i, component in enumerate(value):
            write_f32(data, offset + i * 4, component)
        return True
    if type_id == 4:
        for i, component in enumerate(value):
            write_f32(data, offset + i * 4, component)
        return True
    if type_id == 17:
        for i, component in enumerate(value):
            write_half(data, offset + i * 2, component)
        return True
    if type_id == 11:
        for i, component in enumerate(value):
            struct.pack_into("<h", data, offset + i * 2, max(-32767, min(32767, round(component * 32767.0))))
        return True
    if type_id == 13:
        for i, component in enumerate(value):
            struct.pack_into("<H", data, offset + i * 2, max(0, min(65535, round(component * 65535.0))))
        return True
    if type_id == 9:
        for i, component in enumerate(value):
            data[offset + i] = max(0, min(255, round((component + 1.0) * 127.5)))
        return True
    if type_id == 15:
        packed = 0
        for i, component in enumerate(value):
            quantized = max(-511, min(511, round(component * 511.0))) & 0x3FF
            packed |= quantized << (i * 10)
        packed |= 0x1 << 30
        struct.pack_into("<I", data, offset, packed)
        return True
    if type_id == 19:
        packed = 0
        for i, component in enumerate(value):
            packed |= max(0, min(1023, round(component * 1023.0))) << (i * 10)
        struct.pack_into("<I", data, offset, packed)
        return True
    if type_id == V6_PACKED_BASIS_TYPE:
        write_hw2_packed_tangent(data, offset, value)
        return True
    return False


def write_normal_vec3(
    data: bytearray,
    offset: int,
    type_id: int,
    value: Tuple[float, float, float],
    bitangent_sign: float = 1.0,
) -> bool:
    if type_id == V6_PACKED_BASIS_TYPE:
        write_hw2_packed_normal(data, offset, value, bitangent_sign)
        return True
    return write_vec3(data, offset, type_id, value)


def section_position_changed(section, index, position):
    if not changed(position, section.vertices[index].pos):
        return False
    element = layout_for(section.layout, 'P')
    if element is None or section.big_endian:
        return True
    offset = index * section.vertex_stride + element.offset
    original = section.source_vertex_data[offset:offset + element.size]
    if len(original) != element.size:
        return True
    encoded = bytearray(original)
    return not write_vec3(encoded, 0, element.type_id, position) or encoded != original


def write_vec2(data: bytearray, offset: int, type_id: int, value: Tuple[float, float]) -> bool:
    if type_id == 2:
        write_f32(data, offset, value[0])
        write_f32(data, offset + 4, value[1])
        return True
    if type_id == 16:
        write_half(data, offset, value[0])
        write_half(data, offset + 2, value[1])
        return True
    if type_id == 10:
        struct.pack_into("<h", data, offset, max(-32767, min(32767, round(value[0] * 32767.0))))
        struct.pack_into("<h", data, offset + 2, max(-32767, min(32767, round(value[1] * 32767.0))))
        return True
    if type_id == 12:
        struct.pack_into("<H", data, offset, max(0, min(65535, round(value[0] * 65535.0))))
        struct.pack_into("<H", data, offset + 2, max(0, min(65535, round(value[1] * 65535.0))))
        return True
    return False


def changed(values: Iterable[float], source: Iterable[float], epsilon: float = 0.00001) -> bool:
    return any(abs(float(a) - float(b)) > epsilon for a, b in zip(values, source))
