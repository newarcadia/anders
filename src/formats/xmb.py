"""Read and write HW2 binary XML and compressed damage documents."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import struct
import zlib
from ..errors import UgxError
from .ecf import EcfChunkRecord, ecf_chunk_records, replace_ecf_chunk

__all__ = ('XmbVariant', 'XmbElement', 'decompress_hw2_binary_xml', 'xmb_variant', 'parse_hw2_xmb', 'xmb_string_variant', 'build_hw2_xmb', 'build_compressed_xml_stream', 'wrap_hw2_damage_xmb')

@dataclass
class XmbVariant:
    raw: int
    type_id: int
    flags: int
    length: int
    value: str
    indirect_data: bytes = b""


@dataclass
class XmbElement:
    parent: int
    name: XmbVariant
    text: XmbVariant
    attributes: List[Tuple[XmbVariant, XmbVariant]] = field(default_factory=list)
    children: List[int] = field(default_factory=list)


def decompress_hw2_binary_xml(data: bytes) -> Tuple[bytes, bytes, EcfChunkRecord]:
    records = ecf_chunk_records(data)
    if len(records) != 1:
        raise UgxError("Paired HW2 damage XMB must contain one ECF payload chunk.")
    record = records[0]
    stream = record.data
    if len(stream) < 40 or struct.unpack_from(">I", stream, 0)[0] != 0xCC34EEAD:
        raise UgxError("Paired damage XMB has an unsupported compressed stream.")
    mode = struct.unpack_from(">I", stream, 8)[0]
    unpacked_size, packed_size = struct.unpack_from(">QQ", stream, 12)
    if mode != 0 or 36 + packed_size + 4 > len(stream):
        raise UgxError("Paired damage XMB uses an unsupported compressed stream mode.")
    try:
        payload = zlib.decompress(stream[36:36 + packed_size], -15)
    except zlib.error as exc:
        raise UgxError(f"Could not decompress paired damage XMB: {exc}") from exc
    if len(payload) != unpacked_size or struct.unpack_from(">I", stream, 36 + packed_size)[0] != 0xA5D91776:
        raise UgxError("Paired damage XMB compressed stream is truncated.")
    return payload, stream[:36], record


def xmb_variant(data: bytes, offset: int, pool: bytes) -> XmbVariant:
    raw = struct.unpack_from("<I", data, offset)[0]
    payload = raw & 0x00FFFFFF
    type_id = (raw >> 24) & 0x0F
    length = ((raw >> 28) & 0x03) + 1
    flags = (raw >> 30) & 0x03
    indirect = bool(flags & 2)
    value = ""
    indirect_data = b""
    if type_id == 7:
        value = "true" if payload else "false"
    elif type_id in (3, 4):
        number = struct.unpack_from("<I", pool, payload)[0] if indirect else payload
        if not (flags & 1):
            sign_bit = 0x800000 if type_id == 3 else 0x80000000
            if number & sign_bit:
                number -= sign_bit << 1
        value = str(number)
        if indirect:
            indirect_data = pool[payload:payload + 4]
    elif type_id in (1, 2, 5):
        if type_id == 1:
            # Phoenix Float24: sign, six exponent bits (bias 31), and
            # seventeen mantissa bits. Preserve signed zero as well.
            exponent = (payload >> 17) & 0x3F
            bits = (payload & 0x800000) << 8
            if exponent:
                bits |= ((exponent - 31 + 127) << 23) | ((payload & 0x1FFFF) << 6)
            number = struct.unpack("<f", struct.pack("<I", bits))[0]
        elif type_id == 5:
            number = (payload & 0x7FFFFF) / 10000.0 * (-1.0 if not (flags & 1) and payload & 0x800000 else 1.0)
        else:
            number = struct.unpack_from("<f", pool, payload)[0]
            indirect_data = pool[payload:payload + 4]
        value = format(number, ".9g")
    elif type_id == 6:
        value = format(struct.unpack_from("<d", pool, payload)[0], ".17g")
        indirect_data = pool[payload:payload + 8]
    elif type_id in (8, 9):
        if indirect:
            if type_id == 9:
                end = payload
                while end + 1 < len(pool) and pool[end:end + 2] != b"\0\0":
                    end += 2
                indirect_data = pool[payload:end + 2]
                value = indirect_data[:-2].decode("utf-16-le", "replace")
            else:
                end = pool.find(b"\0", payload)
                end = len(pool) if end < 0 else end
                indirect_data = pool[payload:end + 1]
                value = pool[payload:end].decode("utf-8", "replace")
        else:
            value = bytes((payload >> shift) & 0xFF for shift in (0, 8, 16)).split(b"\0", 1)[0].decode("utf-8", "replace")
    elif type_id == 10:
        indirect_data = pool[payload:payload + length * 4]
        value = ",".join(format(item, ".9g") for item in struct.unpack_from(f"<{length}f", pool, payload))
    return XmbVariant(raw, type_id, flags, length, value, indirect_data)


def parse_hw2_xmb(payload: bytes) -> List[XmbElement]:
    if len(payload) < 40 or struct.unpack_from("<I", payload, 0)[0] != 0x71439800:
        raise UgxError("Paired damage document is not a little-endian 64-bit HW2 XMB.")
    count = struct.unpack_from("<I", payload, 8)[0]
    elements_offset = struct.unpack_from("<Q", payload, 16)[0]
    pool_size = struct.unpack_from("<I", payload, 24)[0]
    pool_offset = struct.unpack_from("<Q", payload, 32)[0]
    if not count or elements_offset + count * 48 > len(payload) or pool_offset + pool_size > len(payload):
        raise UgxError("Paired damage XMB tables are invalid.")
    pool = payload[pool_offset:pool_offset + pool_size]
    elements = []
    for index in range(count):
        offset = elements_offset + index * 48
        parent = struct.unpack_from("<i", payload, offset)[0]
        name = xmb_variant(payload, offset + 4, pool)
        text = xmb_variant(payload, offset + 8, pool)
        attr_count, attr_offset = struct.unpack_from("<I4xQ", payload, offset + 16)
        child_count, child_offset = struct.unpack_from("<I4xQ", payload, offset + 32)
        attributes = [
            (xmb_variant(payload, attr_offset + item * 8, pool), xmb_variant(payload, attr_offset + item * 8 + 4, pool))
            for item in range(attr_count)
        ]
        children = list(struct.unpack_from(f"<{child_count}i", payload, child_offset)) if child_count else []
        elements.append(XmbElement(parent, name, text, attributes, children))
    return elements


def xmb_string_variant(value: str, template: Optional[XmbVariant] = None) -> XmbVariant:
    type_id = template.type_id if template and template.type_id in (8, 9) else 8
    return XmbVariant(0, type_id, 2, 1, value, b"")


def build_hw2_xmb(elements: List[XmbElement]) -> bytes:
    pool = bytearray()
    def encode_variant(value: XmbVariant) -> int:
        if value.type_id not in (8, 9) and value.indirect_data:
            alignment = 8 if value.type_id == 6 else 4
            while len(pool) % alignment:
                pool.append(0)
            position = len(pool)
            pool.extend(value.indirect_data)
            return (value.raw & 0xFF000000) | position
        if value.type_id in (8, 9):
            raw = value.value.encode("utf-16-le") + b"\0\0" if value.type_id == 9 else value.value.encode("utf-8") + b"\0"
            alignment = 2 if value.type_id == 9 else 1
            while len(pool) % alignment:
                pool.append(0)
            position = len(pool)
            pool.extend(raw)
            return (2 << 30) | (value.type_id << 24) | position
        return value.raw

    element_offset = 48
    out = bytearray(element_offset + len(elements) * 48)
    cursor = len(out)
    attr_offsets = []
    child_offsets = []
    for element in elements:
        while cursor % 8:
            out.append(0); cursor += 1
        attr_offsets.append(cursor if element.attributes else 0)
        for key, value in element.attributes:
            out.extend(b"\0" * 8); cursor += 8
        while cursor % 4:
            out.append(0); cursor += 1
        child_offsets.append(cursor if element.children else 0)
        for child in element.children:
            out.extend(struct.pack("<i", child)); cursor += 4
    while len(out) % 16:
        out.append(0)
    pool_offset = len(out)
    struct.pack_into("<I", out, 0, 0x71439800)
    struct.pack_into("<I", out, 8, len(elements))
    struct.pack_into("<Q", out, 16, element_offset)
    struct.pack_into("<I", out, 24, 0)
    struct.pack_into("<Q", out, 32, pool_offset)
    for index, element in enumerate(elements):
        offset = element_offset + index * 48
        struct.pack_into("<iII4xI4xQI4xQ", out, offset, element.parent, encode_variant(element.name), encode_variant(element.text), len(element.attributes), attr_offsets[index], len(element.children), child_offsets[index])
        for item, (key, value) in enumerate(element.attributes):
            struct.pack_into("<II", out, attr_offsets[index] + item * 8, encode_variant(key), encode_variant(value))
    out.extend(pool)
    struct.pack_into("<I", out, 24, len(pool))
    return bytes(out)


def build_compressed_xml_stream(payload: bytes) -> bytes:
    compressor = zlib.compressobj(9, zlib.DEFLATED, -15)
    packed = compressor.compress(payload) + compressor.flush()
    source_adler, packed_adler = zlib.adler32(payload), zlib.adler32(packed)
    # The header checksum uses the native structure order, which differs from
    # the stream's serialized field order.
    checksum = zlib.adler32(struct.pack('>QIQII', len(payload), source_adler, len(packed), packed_adler, 0))
    header = struct.pack('>IIIQQII', 0xCC34EEAD, checksum, 0, len(payload), len(packed), source_adler, packed_adler)
    return header + packed + struct.pack('>I', 0xA5D91776)


def wrap_hw2_damage_xmb(source: bytes, payload: bytes, stream_header: bytes, record: EcfChunkRecord) -> bytes:
    stream = build_compressed_xml_stream(payload)
    return bytes(replace_ecf_chunk(bytearray(source), record.id, stream))
