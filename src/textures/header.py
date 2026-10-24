"""Read native texture headers, byte order and format identifiers."""

from __future__ import annotations
import math

__all__ = ('fourcc', 'byteswap32', 'le16', 'le32', 'be32', 'plausible_dimension', 'normalize_texture_format', 'infer_compressed_format', 'float_to_u8')

def fourcc(text: str) -> int:
    return ord(text[0]) | (ord(text[1]) << 8) | (ord(text[2]) << 16) | (ord(text[3]) << 24)


def byteswap32(value: int) -> int:
    return ((value & 0xFF) << 24) | ((value & 0xFF00) << 8) | ((value >> 8) & 0xFF00) | ((value >> 24) & 0xFF)


def le16(data: bytes, offset: int) -> int:
    return data[offset] | (data[offset + 1] << 8)


def le32(data: bytes, offset: int) -> int:
    return data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16) | (data[offset + 3] << 24)


def be32(data: bytes, offset: int) -> int:
    return (data[offset] << 24) | (data[offset + 1] << 16) | (data[offset + 2] << 8) | data[offset + 3]


def plausible_dimension(value: int) -> bool:
    return 0 < value <= 16384


def normalize_texture_format(data_format: int) -> int:
    swapped = byteswap32(data_format)
    for text in ("DXT1", "DXT3", "DXT5", "ATI2", "BC5U", "BC5S"):
        value = fourcc(text)
        if data_format == value or swapped == value:
            return value
    if data_format in (4, 14, 71, 72):
        return fourcc("DXT1")
    if data_format in (5, 6, 7, 8, 12, 15, 16, 18, 77, 78):
        return fourcc("DXT5")
    if data_format in (82, 83, 84):
        return fourcc("BC5U")
    return data_format


def infer_compressed_format(width: int, height: int, payload_size: int) -> int:
    if not plausible_dimension(width) or not plausible_dimension(height):
        return 0
    dxt1_size = ((width + 3) // 4) * ((height + 3) // 4) * 8
    dxt5_size = ((width + 3) // 4) * ((height + 3) // 4) * 16
    if dxt1_size <= payload_size < dxt5_size:
        return fourcc("DXT1")
    if payload_size >= dxt5_size:
        return fourcc("DXT5")
    return 0


def float_to_u8(value: float) -> int:
    if not math.isfinite(value):
        return 0
    return max(0, min(255, int(value * 255.0 + 0.5)))
