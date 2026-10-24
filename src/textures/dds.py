"""Decode DDS surfaces and preserve native pixel formats during encoding."""

from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple
import math
import struct
from ..errors import UgxError
from .compression import decode_compressed_to_rgba, encode_compressed_from_rgba
from .header import (
    be32,
    byteswap32,
    float_to_u8,
    fourcc,
    infer_compressed_format,
    le32,
    normalize_texture_format,
    plausible_dimension,
)

try:
    import numpy as np
except ImportError:
    np = None





__all__ = ('tolerant_dds_dimension', 'tolerant_dds_mip_count', 'find_tolerant_fourcc', 'read_tolerant_dx10_format', 'downsample_rgba2x', 'half_float_value', 'encode_rgba16f_from_floats', 'downsample_rgba_floats2x', 'dds_is_luminance8', 'mask_shift_bits', 'dds_rgb_masks', 'dds_rgba8_masks', 'masked_to_u8', 'u8_to_masked', 'decode_rgb_dds_payload', 'encode_rgb_dds_payload', 'encode_luminance8_from_rgba', 'encode_dds_rgba_copy', 'flip_rgba_rows_bytes', 'flip_rgba_rows_floats', 'decode_dds_to_rgba', 'decode_texture_payload', 'float_dds_layout', 'unsigned_dds_float', 'decode_float_dds_surface', 'encode_float_dds_surface')

def tolerant_dds_dimension(data: bytes, offset: int) -> int:
    value = le32(data, offset)
    if plausible_dimension(value):
        return value
    swapped = be32(data, offset)
    return swapped if plausible_dimension(swapped) else value


def tolerant_dds_mip_count(data: bytes, offset: int = 28) -> int:
    value = le32(data, offset)
    if 0 < value <= 32:
        return value
    if data[offset] == 0 and 0 < data[offset + 1] <= 32 and data[offset + 2] == 0 and data[offset + 3] == 0:
        return data[offset + 1]
    swapped = be32(data, offset)
    return swapped if 0 < swapped <= 32 else 1


def find_tolerant_fourcc(data: bytes, offset: int = 84) -> int:
    known = ("DXT1", "DXT3", "DXT5", "BC5U", "BC5S", "ATI2", "DX10")
    for index in range(offset, min(offset + 5, len(data) - 3)):
        for text in known:
            if data[index:index + 4] == text.encode("ascii"):
                return fourcc(text)
    return le32(data, offset)


def read_tolerant_dx10_format(data: bytes, offset: int = 128) -> int:
    if len(data) < offset + 4:
        return 0
    value = le32(data, offset)
    if 0 < value < 200:
        return value
    if data[offset] == 0 and 0 < data[offset + 1] < 200 and data[offset + 2] == 0 and data[offset + 3] == 0:
        return data[offset + 1]
    return value


def downsample_rgba2x(src: bytes, width: int, height: int) -> Tuple[int, int, bytes]:
    new_width, new_height = max(1, width // 2), max(1, height // 2)
    if np is not None:
        pixels = np.frombuffer(src, dtype=np.uint8).reshape(height, width, 4)
        y = np.minimum(np.arange(new_height) * 2, height - 1)
        x = np.minimum(np.arange(new_width) * 2, width - 1)
        total = pixels[y[:, None], x[None, :]].astype(np.uint16)
        total += pixels[y[:, None], np.minimum(x + 1, width - 1)[None, :]]
        total += pixels[np.minimum(y + 1, height - 1)[:, None], x[None, :]]
        total += pixels[np.minimum(y + 1, height - 1)[:, None], np.minimum(x + 1, width - 1)[None, :]]
        return new_width, new_height, (total // 4).astype(np.uint8).tobytes()
    out = bytearray(new_width * new_height * 4)
    for y in range(new_height):
        for x in range(new_width):
            samples = []
            for sy in (min(height - 1, y * 2), min(height - 1, y * 2 + 1)):
                for sx in (min(width - 1, x * 2), min(width - 1, x * 2 + 1)):
                    samples.append((sy * width + sx) * 4)
            dst = (y * new_width + x) * 4
            for channel in range(4):
                out[dst + channel] = sum(src[index + channel] for index in samples) // len(samples)
    return new_width, new_height, bytes(out)


def half_float_value(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(-65504.0, min(65504.0, float(value)))


def encode_rgba16f_from_floats(rgba: List[float], width: int, height: int) -> bytes:
    if len(rgba) != width * height * 4:
        raise UgxError("RGBA buffer does not match DDS dimensions.")
    out = bytearray()
    for pixel in range(width * height):
        index = pixel * 4
        out.extend(struct.pack("<eeee", *(half_float_value(rgba[index + channel]) for channel in range(4))))
    return bytes(out)


def downsample_rgba_floats2x(src: List[float], width: int, height: int) -> Tuple[int, int, List[float]]:
    new_width, new_height = max(1, width // 2), max(1, height // 2)
    out = [0.0] * (new_width * new_height * 4)
    for y in range(new_height):
        for x in range(new_width):
            samples = []
            for sy in (min(height - 1, y * 2), min(height - 1, y * 2 + 1)):
                for sx in (min(width - 1, x * 2), min(width - 1, x * 2 + 1)):
                    samples.append((sy * width + sx) * 4)
            dst = (y * new_width + x) * 4
            for channel in range(4):
                out[dst + channel] = sum(src[index + channel] for index in samples) / len(samples)
    return new_width, new_height, out


def dds_is_luminance8(data: bytes, offset: int = 0) -> bool:
    return (
        len(data) >= offset + 128
        and le32(data, offset + 80) & 0x20000
        and le32(data, offset + 84) == 0
        and le32(data, offset + 88) == 8
        and le32(data, offset + 92) == 0xFF
    )


def mask_shift_bits(mask: int) -> Optional[Tuple[int, int]]:
    if not mask:
        return None
    shift = 0
    while shift < 32 and not (mask & (1 << shift)):
        shift += 1
    bits = 0
    while shift + bits < 32 and (mask & (1 << (shift + bits))):
        bits += 1
    if mask != ((1 << bits) - 1) << shift:
        return None
    return shift, bits


def dds_rgb_masks(data: bytes, offset: int = 0) -> Optional[Tuple[Tuple[int, int, int, int], int]]:
    if len(data) < offset + 128 or data[offset:offset + 4] != b"DDS ":
        return None
    bit_count = le32(data, offset + 88)
    if le32(data, offset + 84) != 0 or bit_count not in (24, 32):
        return None
    if not (le32(data, offset + 80) & 0x40):
        return None
    masks = tuple(le32(data, offset + item) for item in (92, 96, 100, 104))
    if not all(mask_shift_bits(mask) for mask in masks[:3]):
        return None
    if bit_count == 32 and masks[3] and not mask_shift_bits(masks[3]):
        return None
    return masks, bit_count // 8


def dds_rgba8_masks(data: bytes, offset: int = 0) -> Optional[Tuple[int, int, int, int]]:
    entry = dds_rgb_masks(data, offset)
    return entry[0] if entry and entry[1] == 4 else None


def masked_to_u8(value: int, mask: int, fallback: int) -> int:
    entry = mask_shift_bits(mask)
    if not entry:
        return fallback
    shift, bits = entry
    raw = (value & mask) >> shift
    return raw if bits == 8 else (raw * 255 + ((1 << bits) - 1) // 2) // ((1 << bits) - 1)


def u8_to_masked(value: int, mask: int) -> int:
    entry = mask_shift_bits(mask)
    if not entry:
        return 0
    shift, bits = entry
    raw = value if bits == 8 else (value * ((1 << bits) - 1) + 127) // 255
    return (raw << shift) & mask


def decode_rgb_dds_payload(data: bytes, width: int, height: int, payload_offset: int, masks: Tuple[int, int, int, int], bytes_per_pixel: int) -> Optional[bytes]:
    pixel_count = width * height
    if len(data) < payload_offset + pixel_count * bytes_per_pixel:
        return None
    rgba = bytearray(pixel_count * 4)
    for pixel in range(pixel_count):
        src = payload_offset + pixel * bytes_per_pixel
        value = int.from_bytes(data[src:src + bytes_per_pixel], "little")
        dst = pixel * 4
        rgba[dst] = masked_to_u8(value, masks[0], 0)
        rgba[dst + 1] = masked_to_u8(value, masks[1], 0)
        rgba[dst + 2] = masked_to_u8(value, masks[2], 0)
        rgba[dst + 3] = masked_to_u8(value, masks[3], 255)
    return bytes(rgba)


def encode_rgb_dds_payload(rgba: bytes, width: int, height: int, masks: Tuple[int, int, int, int], bytes_per_pixel: int) -> bytes:
    if len(rgba) != width * height * 4:
        raise UgxError("RGBA buffer does not match DDS dimensions.")
    if np is not None:
        pixels = np.frombuffer(rgba, dtype=np.uint8).reshape(-1, 4).astype(np.uint32)
        packed = np.zeros(width * height, dtype=np.uint32)
        for channel, mask in enumerate(masks):
            entry = mask_shift_bits(mask)
            if not entry:
                continue
            shift, bits = entry
            raw = pixels[:, channel] if bits == 8 else (pixels[:, channel] * ((1 << bits) - 1) + 127) // 255
            packed |= (raw << shift) & mask
        return packed.astype("<u4").view(np.uint8).reshape(-1, 4)[:, :bytes_per_pixel].tobytes()
    out = bytearray()
    for pixel in range(width * height):
        src = pixel * 4
        value = (
            u8_to_masked(rgba[src], masks[0])
            | u8_to_masked(rgba[src + 1], masks[1])
            | u8_to_masked(rgba[src + 2], masks[2])
            | u8_to_masked(rgba[src + 3], masks[3])
        )
        out.extend(int(value).to_bytes(bytes_per_pixel, "little"))
    return bytes(out)


def encode_luminance8_from_rgba(rgba: bytes, width: int, height: int) -> bytes:
    if len(rgba) != width * height * 4:
        raise UgxError("RGBA buffer does not match DDS dimensions.")
    return rgba[3::4]


def encode_dds_rgba_copy(original: bytes, rgba: bytes, rgba_floats: Optional[List[float]] = None) -> bytes:
    if len(original) < 128 or original[:4] != b"DDS ":
        raise UgxError("DDS write needs a DDS texture.")
    height = tolerant_dds_dimension(original, 12)
    width = tolerant_dds_dimension(original, 16)
    mip_count = tolerant_dds_mip_count(original)
    texture_format = find_tolerant_fourcc(original)
    payload_offset = 128
    if texture_format == fourcc("DX10"):
        if len(original) < 148:
            raise UgxError("DDS DX10 header is truncated.")
        texture_format = read_tolerant_dx10_format(original)
        payload_offset = 148
    texture_format = normalize_texture_format(texture_format)
    payload = bytearray()
    rgb_masks = dds_rgb_masks(original) if payload_offset == 128 else None
    if rgb_masks:
        masks, bytes_per_pixel = rgb_masks
        level_width, level_height, level_rgba = width, height, rgba
        for _ in range(mip_count):
            payload.extend(encode_rgb_dds_payload(level_rgba, level_width, level_height, masks, bytes_per_pixel))
            if level_width == 1 and level_height == 1:
                continue
            level_width, level_height, level_rgba = downsample_rgba2x(level_rgba, level_width, level_height)
    elif texture_format == 113:
        level_width, level_height = width, height
        level_rgba_floats = rgba_floats if rgba_floats is not None else [value / 255.0 for value in rgba]
        for _ in range(mip_count):
            payload.extend(encode_rgba16f_from_floats(level_rgba_floats, level_width, level_height))
            if level_width == 1 and level_height == 1:
                continue
            level_width, level_height, level_rgba_floats = downsample_rgba_floats2x(level_rgba_floats, level_width, level_height)
    else:
        level_width, level_height, level_rgba = width, height, rgba
        for _ in range(mip_count):
            if payload_offset == 128 and dds_is_luminance8(original):
                payload.extend(encode_luminance8_from_rgba(level_rgba, level_width, level_height))
            else:
                payload.extend(encode_compressed_from_rgba(level_rgba, level_width, level_height, texture_format))
            if level_width == 1 and level_height == 1:
                continue
            level_width, level_height, level_rgba = downsample_rgba2x(level_rgba, level_width, level_height)
    if len(payload) > len(original) - payload_offset:
        raise UgxError("Encoded DDS payload size does not match the original texture.")
    # Blender only edits the first DDS surface we import; preserve array/cubemap tail bytes until those surfaces are exposed.
    return original[:payload_offset] + bytes(payload) + original[payload_offset + len(payload):]


def flip_rgba_rows_bytes(data: bytes, width: int, height: int) -> bytes:
    row_size = int(width) * 4
    if row_size <= 0 or int(height) <= 1:
        return bytes(data)
    return b"".join(
        data[y * row_size:(y + 1) * row_size]
        for y in range(int(height) - 1, -1, -1)
    )


def flip_rgba_rows_floats(data: Iterable[float], width: int, height: int) -> List[float]:
    values = list(data)
    row_size = int(width) * 4
    if row_size <= 0 or int(height) <= 1:
        return values
    return [
        value
        for y in range(int(height) - 1, -1, -1)
        for value in values[y * row_size:(y + 1) * row_size]
    ]


def decode_dds_to_rgba(data: bytes) -> Optional[Tuple[int, int, bytes]]:
    if len(data) < 128 or data[:4] != b"DDS ":
        return None
    height = tolerant_dds_dimension(data, 12)
    width = tolerant_dds_dimension(data, 16)
    texture_format = find_tolerant_fourcc(data)
    payload_offset = 128
    if texture_format == fourcc("DX10") and len(data) >= 148:
        texture_format = normalize_texture_format(read_tolerant_dx10_format(data))
        payload_offset = 148
    if payload_offset == 128 and dds_is_luminance8(data):
        pixel_count = width * height
        if len(data) < payload_offset + pixel_count:
            return None
        rgba = bytearray(pixel_count * 4)
        for pixel, value in enumerate(data[payload_offset:payload_offset + pixel_count]):
            dst = pixel * 4
            rgba[dst:dst + 4] = bytes((value, value, value, value))
        return width, height, bytes(rgba)
    rgb_masks = dds_rgb_masks(data) if payload_offset == 128 else None
    if rgb_masks:
        masks, bytes_per_pixel = rgb_masks
        rgba = decode_rgb_dds_payload(data, width, height, payload_offset, masks, bytes_per_pixel)
        return (width, height, rgba) if rgba else None
    if texture_format == 113:
        pixel_count = width * height
        if len(data) < payload_offset + pixel_count * 8:
            return None
        rgba = bytearray(pixel_count * 4)
        for pixel in range(pixel_count):
            r, g, b, a = struct.unpack_from("<eeee", data, payload_offset + pixel * 8)
            dst = pixel * 4
            rgba[dst:dst + 4] = bytes((float_to_u8(r), float_to_u8(g), float_to_u8(b), float_to_u8(a)))
        return width, height, bytes(rgba)
    rgba = decode_compressed_to_rgba(data[payload_offset:], width, height, texture_format)
    return (width, height, rgba) if rgba else None


def decode_texture_payload(payload: bytes, data_format: int, width: int, height: int) -> Optional[bytes]:
    dds = decode_dds_to_rgba(payload)
    if dds:
        return dds[2]
    numeric_format = data_format if data_format < 200 else byteswap32(data_format)
    if numeric_format in (1, 21):
        if len(payload) < width * height * 4:
            return None
        rgba = bytearray(width * height * 4)
        for pixel in range(width * height):
            src = pixel * 4
            rgba[src:src + 4] = bytes((payload[src + 2], payload[src + 1], payload[src], payload[src + 3]))
        return bytes(rgba)
    if numeric_format in (3, 28, 50):
        if len(payload) < width * height:
            return None
        rgba = bytearray(width * height * 4)
        for pixel, value in enumerate(payload[:width * height]):
            dst = pixel * 4
            rgba[dst:dst + 4] = bytes((value, value, value, 255))
        return bytes(rgba)
    texture_format = normalize_texture_format(data_format)
    if texture_format not in (fourcc("DXT1"), fourcc("DXT5")):
        texture_format = infer_compressed_format(width, height, len(payload))
    return decode_compressed_to_rgba(payload, width, height, texture_format)


def float_dds_layout(data: bytes) -> Optional[Dict[str, int]]:
    if len(data) < 128 or data[:4] != b"DDS ":
        return None
    width, height = tolerant_dds_dimension(data, 16), tolerant_dds_dimension(data, 12)
    if not plausible_dimension(width) or not plausible_dimension(height):
        return None
    format_id = find_tolerant_fourcc(data)
    payload_offset, layers = 128, 1
    if format_id == fourcc("DX10"):
        if len(data) < 148:
            return None
        dxgi = read_tolerant_dx10_format(data)
        format_id = {2: 116, 10: 113, 16: 115, 26: 26, 34: 112, 41: 114, 54: 111}.get(dxgi, 0)
        if le32(data, 132) != 3:
            return None
        payload_offset = 148
        layers = max(1, le32(data, 140)) * (6 if le32(data, 136) & 4 else 1)
    elif le32(data, 112) & 0x200:
        layers = sum(bool(le32(data, 112) & (0x400 << index)) for index in range(6))
    channels, component_size = {111: (1, 2), 112: (2, 2), 113: (4, 2),
                                114: (1, 4), 115: (2, 4), 116: (4, 4), 26: (3, 0)}.get(format_id, (0, 0))
    if not channels or not layers:
        return None
    pixel_size = channels * component_size if component_size else 4
    mip_count = max(1, tolerant_dds_mip_count(data))
    surface_size = sum(max(1, width >> level) * max(1, height >> level) * pixel_size for level in range(mip_count))
    if payload_offset + surface_size * layers > len(data):
        raise UgxError("Floating-point DDS surface data is truncated.")
    return {"width": width, "height": height, "layers": layers, "format": format_id,
            "channels": channels, "component_size": component_size, "pixel_size": pixel_size,
            "payload_offset": payload_offset, "mip_count": mip_count, "surface_size": surface_size}


def unsigned_dds_float(value: int, mantissa_bits: int) -> float:
    exponent, mantissa = value >> mantissa_bits, value & ((1 << mantissa_bits) - 1)
    if exponent == 31:
        return math.inf if mantissa == 0 else math.nan
    if exponent == 0:
        return math.ldexp(float(mantissa), -14 - mantissa_bits)
    return math.ldexp(1.0 + mantissa / float(1 << mantissa_bits), exponent - 15)


def decode_float_dds_surface(data: bytes, offset: int, count: int, layout: Dict[str, int]):
    channels = layout["channels"]
    if np is not None:
        pixels = np.zeros((count, 4), dtype=np.float32)
        pixels[:, 3] = 1.0
        if layout["format"] == 26:
            packed = np.frombuffer(data, dtype="<u4", count=count, offset=offset)
            for channel, bits, shift in ((0, 6, 0), (1, 6, 11), (2, 5, 22)):
                values = np.array([unsigned_dds_float(value, bits) for value in range(1 << (bits + 5))], dtype=np.float32)
                pixels[:, channel] = values[(packed >> shift) & ((1 << (bits + 5)) - 1)]
        else:
            dtype = "<f2" if layout["component_size"] == 2 else "<f4"
            pixels[:, :channels] = np.frombuffer(data, dtype=dtype, count=count * channels, offset=offset).reshape(count, channels)
        return pixels.reshape(-1)
    pixels = []
    for index in range(count):
        if layout["format"] == 26:
            value = le32(data, offset + index * 4)
            values = [unsigned_dds_float(value & 2047, 6), unsigned_dds_float((value >> 11) & 2047, 6), unsigned_dds_float(value >> 22, 5)]
        else:
            code = "e" if layout["component_size"] == 2 else "f"
            values = list(struct.unpack_from("<" + code * channels, data, offset + index * layout["pixel_size"]))
        pixels.extend((values + [0.0] * (3 - len(values)) + [1.0])[:4])
    return pixels


def encode_float_dds_surface(values, layout: Dict[str, int]) -> bytes:
    if np is not None:
        pixels = np.asarray(values, dtype=np.float32).reshape(-1, 4)
        if layout["format"] == 26:
            packed = np.zeros(len(pixels), dtype=np.uint32)
            for channel, mantissa_bits, shift in ((0, 6, 0), (1, 6, 11), (2, 5, 22)):
                source = np.maximum(pixels[:, channel], 0)
                half = np.minimum(source, 65504).astype("<f2").view("<u2").astype(np.uint32)
                dropped = 10 - mantissa_bits
                rounded = (half + (1 << (dropped - 1)) - 1 + ((half >> dropped) & 1)) >> dropped
                rounded = np.minimum(rounded, (31 << mantissa_bits) - 1)
                rounded[np.isposinf(source)] = 31 << mantissa_bits
                rounded[np.isnan(source)] = (31 << mantissa_bits) | 1
                packed |= rounded << shift
            return packed.astype("<u4").tobytes()
        source = pixels[:, :layout["channels"]]
        if layout["component_size"] == 2:
            source = np.clip(source, -65504, 65504)
        return source.astype("<f2" if layout["component_size"] == 2 else "<f4").tobytes()
    out = bytearray()
    for offset in range(0, len(values), 4):
        if layout["format"] == 26:
            packed = 0
            for channel, bits, shift in ((0, 6, 0), (1, 6, 11), (2, 5, 22)):
                value = values[offset + channel]
                if math.isnan(value):
                    raw = (31 << bits) | 1
                elif value == math.inf:
                    raw = 31 << bits
                else:
                    half = struct.unpack("<H", struct.pack("<e", min(65504, max(0, value))))[0]
                    drop = 10 - bits
                    raw = min((31 << bits) - 1, (half + (1 << (drop - 1)) - 1 + ((half >> drop) & 1)) >> drop)
                packed |= raw << shift
            out.extend(struct.pack("<I", packed))
        else:
            code = "e" if layout["component_size"] == 2 else "f"
            source = values[offset:offset + layout["channels"]]
            if code == "e":
                source = [half_float_value(value) for value in source]
            out.extend(struct.pack("<" + code * layout["channels"], *source))
    return bytes(out)
