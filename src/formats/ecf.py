"""Read, assemble and patch ECF resource containers."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import struct
from ..errors import UgxError
from .binary import Reader, adler32, write_u32_be

__all__ = ('UGX_FILE_ID', 'UAX_FILE_ID', 'CACHED_DATA_CHUNK', 'INDEX_BUFFER_CHUNK', 'VERTEX_BUFFER_CHUNK', 'MATERIAL_CHUNK', 'Chunk', 'EcfFile', 'EcfChunkRecord', 'refresh_changed_ecf_adlers', 'UGX_ECF_CHUNK_LAYOUT', 'build_ecf_file', 'validate_game_ready_ugx_ecf_layout', 'ecf_chunk_records', 'replace_ecf_chunk')

UGX_FILE_ID = 0xAAC93746


UAX_FILE_ID = 0xAAC93747


CACHED_DATA_CHUNK = 0x700


INDEX_BUFFER_CHUNK = 0x701


VERTEX_BUFFER_CHUNK = 0x702


MATERIAL_CHUNK = 0x704


@dataclass
class Chunk:
    id: int
    offset: int
    size: int
    data: bytes


@dataclass
class EcfFile:
    file_id: int
    chunks: Dict[int, Chunk]

    @staticmethod
    def parse(data: bytes) -> "EcfFile":
        reader = Reader(data, True)
        signature = reader.u32()
        if signature != 0xDABA7737:
            raise UgxError("Not an ECF file.")
        header_size = reader.u32()
        reader.skip(8)
        chunk_count = reader.u16()
        reader.skip(2)
        file_id = reader.u32()
        extra_size = reader.u16()
        reader.skip(6)
        if header_size < 32 or header_size > len(data):
            raise UgxError("Invalid ECF header size.")
        reader.seek(header_size)
        chunks: Dict[int, Chunk] = {}
        for _ in range(chunk_count):
            chunk_id = reader.u64()
            offset = reader.u32()
            size = reader.u32()
            reader.skip(8)
            if extra_size:
                reader.skip(extra_size)
            if offset + size > len(data):
                raise UgxError("ECF chunk points outside the file.")
            chunks[chunk_id] = Chunk(chunk_id, offset, size, data[offset:offset + size])
        return EcfFile(file_id, chunks)


@dataclass
class EcfChunkRecord:
    index: int
    record: int
    id: int
    offset: int
    size: int
    flags: int
    alignment_log2: int
    resource_flags: int
    extra: bytes
    data: bytes


def refresh_changed_ecf_adlers(data: bytearray, original: bytes):
    records = ecf_chunk_records(bytes(data))
    source = {record.id: record for record in ecf_chunk_records(original)}
    if not records:
        return
    for record in records:
        before = source.get(record.id)
        if before is None or record.data != before.data:
            write_u32_be(data, record.record+16, adler32(record.data))
    header_size = struct.unpack_from('>I', data, 4)[0]
    if data[12:header_size] != original[12:header_size]:
        write_u32_be(data, 8, adler32(bytes(data[12:header_size])))


UGX_ECF_CHUNK_LAYOUT = {
    0x703: (4, 0),                    # Granny data: 16-byte aligned.
    CACHED_DATA_CHUNK: (4, 0),        # Cached metadata / UAX Granny data: 16-byte aligned.
    VERTEX_BUFFER_CHUNK: (5, 1),      # GPU resource: 32-byte aligned.
    INDEX_BUFFER_CHUNK: (5, 1),       # GPU resource: 32-byte aligned.
    MATERIAL_CHUNK: (2, 0),           # BDT material data: 4-byte aligned.
}


def build_ecf_file(chunks: List[Tuple[int, bytes]], file_id: int = UGX_FILE_ID) -> bytes:
    header_size = 32
    record_size = 24
    payload_offset = header_size + len(chunks) * record_size
    records = []
    for chunk_id, payload in chunks:
        alignment_log2, resource_flags = UGX_ECF_CHUNK_LAYOUT.get(chunk_id, (2, 0))
        alignment = 1 << alignment_log2
        payload_offset = (payload_offset + alignment - 1) & ~(alignment - 1)
        records.append((chunk_id, payload_offset, payload, alignment_log2, resource_flags))
        payload_offset += len(payload)
    out = bytearray(struct.pack(">IIIIHHIHHI", 0xDABA7737, header_size, 0, payload_offset, len(chunks), 0, int(file_id), 0, 0, 0))
    for chunk_id, offset, payload, alignment_log2, resource_flags in records:
        out.extend(struct.pack(
            ">QIII BBH",
            chunk_id,
            offset,
            len(payload),
            adler32(payload),
            0,
            alignment_log2,
            resource_flags,
        ))
    write_u32_be(out, 8, adler32(bytes(out[12:header_size])))
    for _, offset, payload, _, _ in records:
        while len(out) < offset:
            out.append(0)
        out.extend(payload)
    return bytes(out)


def validate_game_ready_ugx_ecf_layout(data: bytes):
    records = {record.id: record for record in ecf_chunk_records(data)}
    for chunk_id, (alignment_log2, resource_flags) in UGX_ECF_CHUNK_LAYOUT.items():
        record = records.get(chunk_id)
        if record is None:
            raise UgxError(f"UGX ECF is missing required chunk {chunk_id:#x}.")
        alignment = 1 << alignment_log2
        if record.offset % alignment:
            raise UgxError(
                f"UGX ECF chunk {chunk_id:#x} starts at {record.offset:#x}, "
                f"not a {alignment}-byte boundary."
            )
        if record.alignment_log2 != alignment_log2:
            raise UgxError(
                f"UGX ECF chunk {chunk_id:#x} advertises {1 << record.alignment_log2}-byte "
                f"alignment instead of {alignment}-byte alignment."
            )
        if record.resource_flags != resource_flags:
            raise UgxError(
                f"UGX ECF chunk {chunk_id:#x} has resource flags {record.resource_flags:#x}; "
                f"expected {resource_flags:#x}."
            )


def ecf_chunk_records(data: bytes) -> List[EcfChunkRecord]:
    if len(data) < 32 or struct.unpack_from(">I", data, 0)[0] != 0xDABA7737:
        return []
    header_size = struct.unpack_from(">I", data, 4)[0]
    chunk_count = struct.unpack_from(">H", data, 16)[0]
    extra_size = struct.unpack_from(">H", data, 24)[0]
    record_size = 24 + extra_size
    if header_size < 32 or header_size + chunk_count * record_size > len(data):
        return []
    records = []
    for index in range(chunk_count):
        record = header_size + index * record_size
        offset = struct.unpack_from(">I", data, record + 8)[0]
        size = struct.unpack_from(">I", data, record + 12)[0]
        if offset + size > len(data):
            return []
        records.append(EcfChunkRecord(
            index,
            record,
            struct.unpack_from(">Q", data, record)[0],
            offset,
            size,
            data[record + 20],
            data[record + 21],
            struct.unpack_from(">H", data, record + 22)[0],
            data[record + 24:record + record_size],
            data[offset:offset + size],
        ))
    return records


def replace_ecf_chunk(data: bytearray, chunk_id: int, payload: bytes) -> bytearray:
    records = ecf_chunk_records(bytes(data))
    if not records:
        raise UgxError("Invalid ECF chunk table.")
    target = next((item for item in records if item.id == chunk_id), None)
    if not target:
        raise UgxError("UGX is missing the requested ECF chunk.")
    if target.size == len(payload):
        data[target.offset:target.offset + target.size] = payload
        write_u32_be(data, target.record + 16, adler32(payload))
        header_size = struct.unpack_from(">I", data, 4)[0]
        write_u32_be(data, 8, adler32(bytes(data[12:header_size])))
        return data

    header_size = struct.unpack_from(">I", data, 4)[0]
    extra_size = struct.unpack_from(">H", data, 24)[0]
    record_size = 24 + extra_size
    out = bytearray(data[:header_size + len(records) * record_size])
    source_end = max(header_size + len(records) * record_size, *(record.offset+record.size for record in records))
    footer = bytes(data[source_end:])
    declared_footer = max(0, min(len(footer), struct.unpack_from('>I', data, 12)[0]-source_end))
    source_cursor = len(out)
    rewritten: List[EcfChunkRecord] = []
    # Chunk-table order can differ from payload order. Retain each source gap
    # beside the same payload, including bytes outside the declared chunks.
    for record in sorted(records, key=lambda item: (item.offset, item.size, item.index)):
        placeholder = record.offset == 0 and record.size == 0
        if placeholder and (record.id != chunk_id or not payload):
            rewritten.append(record)
            continue
        if not placeholder:
            if record.offset < source_cursor:
                raise UgxError("Cannot resize overlapping ECF chunks.")
            out.extend(data[source_cursor:record.offset])
        alignment = 1 << record.alignment_log2 if record.alignment_log2 < 31 else 1
        out.extend(bytes((-len(out)) % alignment))
        chunk_data = payload if record.id == chunk_id else record.data
        rewritten.append(EcfChunkRecord(
            record.index, record.record, record.id, len(out), len(chunk_data),
            record.flags, record.alignment_log2, record.resource_flags, record.extra, chunk_data,
        ))
        out.extend(chunk_data)
        if not placeholder:
            source_cursor = record.offset+record.size

    if len(out)+declared_footer > 0xFFFFFFFF:
        raise UgxError("Repacked ECF is too large.")
    write_u32_be(out, 12, len(out)+declared_footer)
    for record in rewritten:
        write_u32_be(out, record.record + 8, record.offset)
        write_u32_be(out, record.record + 12, record.size)
        write_u32_be(out, record.record + 16, adler32(record.data))
    write_u32_be(out, 8, adler32(bytes(out[12:header_size])))
    out.extend(footer)
    return out
