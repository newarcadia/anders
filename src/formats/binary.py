"""Bounded binary readers, packed offsets and field writers."""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
import struct
import zlib
from ..errors import UgxError

__all__ = ('Reader', 'PackedArrayHeader', 'read_cstring_at', 'cstring_capacity_at', 'read_packed_array_header', 'read_packed_string_offset', 'u32le', 'patch_cstring', 'write_u32_be', 'write_u32_le', 'write_u16_le', 'adler32', 'write_packed_offset', 'sha256_bytes', 'packed_array_header', 'PACKED_NULL_OFFSET_64', 'align_data', 'align4')

class Reader:
    def __init__(self, data: bytes, big: bool = False):
        self.data = data
        self.big = big
        self.pos = 0

    def can_read(self, size: int) -> bool:
        return 0 <= size <= len(self.data) - self.pos

    def require(self, size: int, what: str):
        if not self.can_read(size):
            raise UgxError(f"Unexpected end of file while reading {what}.")

    def seek(self, pos: int):
        if pos < 0 or pos > len(self.data):
            raise UgxError("Seek outside binary buffer.")
        self.pos = pos

    def skip(self, size: int):
        self.seek(self.pos + size)

    def _read(self, fmt: str, size: int):
        self.require(size, fmt)
        prefix = ">" if self.big else "<"
        value = struct.unpack_from(prefix + fmt, self.data, self.pos)[0]
        self.pos += size
        return value

    def u8(self) -> int:
        return self._read("B", 1)

    def u16(self) -> int:
        return self._read("H", 2)

    def u32(self) -> int:
        return self._read("I", 4)

    def i32(self) -> int:
        return self._read("i", 4)

    def u64(self) -> int:
        return self._read("Q", 8)

    def f32(self) -> float:
        return self._read("f", 4)


@dataclass
class PackedArrayHeader:
    length: int
    offset: int


def read_cstring_at(data: bytes, offset: int) -> str:
    if offset < 0 or offset >= len(data):
        return ""
    end = data.find(b"\0", offset)
    if end < 0:
        end = len(data)
    return data[offset:end].decode("utf-8", "replace")


def cstring_capacity_at(data: bytes, offset: int) -> int:
    if offset < 0 or offset >= len(data):
        return 0
    end = data.find(b"\0", offset)
    return 0 if end < 0 else end - offset


def read_packed_array_header(reader: Reader, is64: bool) -> PackedArrayHeader:
    if is64:
        length = reader.u32()
        reader.u32()
        return PackedArrayHeader(length, reader.u64())
    return PackedArrayHeader(reader.u32(), reader.u32())


def read_packed_string_offset(reader: Reader, is64: bool) -> int:
    return reader.u64() if is64 else reader.u32()


def u32le(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def patch_cstring(data: bytearray, offset: int, capacity: int, value: str, label: str):
    if not offset or not capacity:
        return
    raw = value.encode("utf-8")
    if b"\0" in raw:
        raise UgxError(f"{label} contains a NUL byte.")
    if offset + capacity >= len(data):
        raise UgxError(f"{label} patch offset is outside the file.")
    if len(raw) > capacity:
        raise UgxError(f"{label} is longer than its original storage ({len(raw)} > {capacity}).")
    data[offset:offset + capacity + 1] = b"\0" * (capacity + 1)
    data[offset:offset + len(raw)] = raw


def write_u32_be(data: bytearray, offset: int, value: int):
    struct.pack_into(">I", data, offset, value & 0xFFFFFFFF)


def write_u32_le(data: bytearray, offset: int, value: int):
    struct.pack_into("<I", data, offset, value & 0xFFFFFFFF)


def write_u16_le(data: bytearray, offset: int, value: int):
    struct.pack_into("<H", data, offset, value & 0xFFFF)


def adler32(data: bytes) -> int:
    return zlib.adler32(data) & 0xFFFFFFFF


def write_packed_offset(data: bytearray, offset: int, size: int, value: int, big: bool):
    if size == 8:
        struct.pack_into(">Q" if big else "<Q", data, offset, value)
    elif size == 4:
        struct.pack_into(">I" if big else "<I", data, offset, value)
    else:
        raise UgxError("Unsupported packed offset size.")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def packed_array_header(length: int, offset: int) -> bytes:
    return struct.pack("<IIQ", int(length), 0, int(offset))


PACKED_NULL_OFFSET_64 = 0x00000000FFFFFFFF


def align_data(data: bytearray, alignment: int):
    while len(data) % alignment:
        data.append(0)


def align4(data: bytearray):
    align_data(data, 4)
