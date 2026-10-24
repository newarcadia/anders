"""Encode and decode block-compressed texture surfaces."""

from __future__ import annotations
from typing import List, Optional, Tuple
import struct
from ..errors import UgxError
from .header import fourcc, le16, le32, plausible_dimension

try:
    import numpy as np
except ImportError:
    np = None




__all__ = ('color565', 'decode_dxt1_block', 'decode_dxt3_block', 'decode_dxt5_block', 'rgb_to_565', 'color_error', 'dxt1_palette', 'block_pixels', 'encode_dxt1_block_rgba', 'bc4_alpha_palette', 'encode_bc4_alpha_block', 'encode_dxt3_alpha_block', 'BitWriter', 'BC7_WEIGHTS_4BIT', 'bc7_palette', 'bc7_endpoint_bits', 'encode_bc7_mode6_block', 'encode_texture_blocks_numpy', 'encode_compressed_from_rgba', 'decode_bc4_values', 'decode_bc5_block', 'decode_compressed_to_rgba')

def color565(value: int) -> Tuple[int, int, int, int]:
    return ((value >> 11) & 31) * 255 // 31, ((value >> 5) & 63) * 255 // 63, (value & 31) * 255 // 31, 255


def decode_dxt1_block(data: bytes, offset: int, width: int, height: int, bx: int, by: int, rgba: bytearray, allow_alpha: bool = True):
    c0 = le16(data, offset)
    c1 = le16(data, offset + 2)
    colors = [list(color565(c0)), list(color565(c1)), [0, 0, 0, 255], [0, 0, 0, 255]]
    if c0 > c1 or not allow_alpha:
        for i in range(3):
            colors[2][i] = (2 * colors[0][i] + colors[1][i]) // 3
            colors[3][i] = (colors[0][i] + 2 * colors[1][i]) // 3
    else:
        for i in range(3):
            colors[2][i] = (colors[0][i] + colors[1][i]) // 2
        colors[3] = [0, 0, 0, 0]
    lookup = le32(data, offset + 4)
    for y in range(4):
        for x in range(4):
            px = bx * 4 + x
            py = by * 4 + y
            if px >= width or py >= height:
                continue
            index = (lookup >> (2 * (y * 4 + x))) & 3
            dst = (py * width + px) * 4
            rgba[dst:dst + 4] = bytes(colors[index])


def decode_dxt3_block(data: bytes, offset: int, width: int, height: int, bx: int, by: int, rgba: bytearray):
    decode_dxt1_block(data, offset + 8, width, height, bx, by, rgba, False)
    alpha_bits = int.from_bytes(data[offset:offset + 8], "little")
    for y in range(4):
        for x in range(4):
            px = bx * 4 + x
            py = by * 4 + y
            if px >= width or py >= height:
                continue
            alpha = (alpha_bits >> (4 * (y * 4 + x))) & 0xF
            rgba[(py * width + px) * 4 + 3] = (alpha << 4) | alpha


def decode_dxt5_block(data: bytes, offset: int, width: int, height: int, bx: int, by: int, rgba: bytearray):
    alpha = decode_bc4_values(data, offset)
    decode_dxt1_block(data, offset + 8, width, height, bx, by, rgba, False)
    for y in range(4):
        for x in range(4):
            px = bx * 4 + x
            py = by * 4 + y
            if px >= width or py >= height:
                continue
            index = alpha[1] >> (3 * (y * 4 + x)) & 7
            rgba[(py * width + px) * 4 + 3] = alpha[0][index]


def rgb_to_565(pixel: Tuple[int, int, int, int]) -> int:
    return ((pixel[0] * 31 + 127) // 255 << 11) | ((pixel[1] * 63 + 127) // 255 << 5) | ((pixel[2] * 31 + 127) // 255)


def color_error(left, right) -> int:
    return sum((int(left[i]) - int(right[i])) ** 2 for i in range(3))


def dxt1_palette(c0: int, c1: int) -> List[Tuple[int, int, int, int]]:
    colors = [list(color565(c0)), list(color565(c1)), [0, 0, 0, 255], [0, 0, 0, 255]]
    if c0 > c1:
        for i in range(3):
            colors[2][i] = (2 * colors[0][i] + colors[1][i]) // 3
            colors[3][i] = (colors[0][i] + 2 * colors[1][i]) // 3
    else:
        for i in range(3):
            colors[2][i] = (colors[0][i] + colors[1][i]) // 2
        colors[3] = [0, 0, 0, 0]
    return [tuple(color) for color in colors]


def block_pixels(rgba: bytes, width: int, height: int, bx: int, by: int) -> List[Tuple[int, int, int, int]]:
    pixels = []
    for y in range(4):
        for x in range(4):
            px = min(width - 1, bx * 4 + x)
            py = min(height - 1, by * 4 + y)
            src = (py * width + px) * 4
            pixels.append(tuple(rgba[src:src + 4]))
    return pixels


def encode_dxt1_block_rgba(pixels: List[Tuple[int, int, int, int]], allow_alpha: bool) -> bytes:
    transparent = allow_alpha and any(pixel[3] < 128 for pixel in pixels)
    opaque = [pixel for pixel in pixels if not transparent or pixel[3] >= 128] or [(0, 0, 0, 255)]
    cmin, cmax = opaque[0], opaque[0]
    best = -1
    for left in opaque:
        for right in opaque:
            error = color_error(left, right)
            if error > best:
                best, cmin, cmax = error, left, right
    c0, c1 = rgb_to_565(cmax), rgb_to_565(cmin)
    if c0 == c1:
        c1 = c0 - 1 if c0 else 1
    if transparent and c0 > c1:
        c0, c1 = c1, c0
    elif not transparent and c0 <= c1:
        c0, c1 = c1, c0
    palette = dxt1_palette(c0, c1)
    lookup = 0
    limit = 3 if transparent else 4
    for index, pixel in enumerate(pixels):
        if transparent and pixel[3] < 128:
            choice = 3
        else:
            choice = min(range(limit), key=lambda item: color_error(pixel, palette[item]))
        lookup |= choice << (2 * index)
    return struct.pack("<HHI", c0, c1, lookup)


def bc4_alpha_palette(a0: int, a1: int) -> List[int]:
    alpha = [a0, a1, 0, 0, 0, 0, 0, 0]
    if a0 > a1:
        for i in range(1, 7):
            alpha[i + 1] = ((7 - i) * a0 + i * a1) // 7
    else:
        for i in range(1, 5):
            alpha[i + 1] = ((5 - i) * a0 + i * a1) // 5
        alpha[6] = 0
        alpha[7] = 255
    return alpha


def encode_bc4_alpha_block(alphas: List[int]) -> bytes:
    a0, a1 = max(alphas), min(alphas)
    palette = bc4_alpha_palette(a0, a1)
    bits = 0
    for index, alpha in enumerate(alphas):
        choice = min(range(8), key=lambda item: abs(alpha - palette[item]))
        bits |= choice << (3 * index)
    return bytes((a0, a1)) + bytes((bits >> (8 * i)) & 0xFF for i in range(6))


def encode_dxt3_alpha_block(alphas: List[int]) -> bytes:
    bits = 0
    for index, alpha in enumerate(alphas):
        bits |= max(0, min(15, round(alpha / 17.0))) << (4 * index)
    return bits.to_bytes(8, "little")


class BitWriter:
    def __init__(self):
        self.value = 0
        self.pos = 0

    def put(self, value: int, bits: int):
        self.value |= (int(value) & ((1 << bits) - 1)) << self.pos
        self.pos += bits

    def bytes(self, size: int) -> bytes:
        return self.value.to_bytes(size, "little")


BC7_WEIGHTS_4BIT = (0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64)


def bc7_palette(c0: Tuple[int, int, int, int], c1: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
    return [
        tuple(((64 - weight) * c0[channel] + weight * c1[channel] + 32) >> 6 for channel in range(4))
        for weight in BC7_WEIGHTS_4BIT
    ]


def bc7_endpoint_bits(color: Tuple[int, int, int, int]) -> Tuple[Tuple[int, int, int, int], int, Tuple[int, int, int, int]]:
    best_error, best_q, best_p, best_color = None, (0, 0, 0, 0), 0, (0, 0, 0, 0)
    for pbit in (0, 1):
        q = tuple(max(0, min(127, int(round((color[channel] - pbit) / 2.0)))) for channel in range(4))
        decoded = tuple((q[channel] << 1) | pbit for channel in range(4))
        error = sum((decoded[channel] - color[channel]) ** 2 for channel in range(4))
        if best_error is None or error < best_error:
            best_error, best_q, best_p, best_color = error, q, pbit, decoded
    return best_q, best_p, best_color


def encode_bc7_mode6_block(pixels: List[Tuple[int, int, int, int]]) -> bytes:
    c0, c1 = pixels[0], pixels[0]
    best = -1
    for left in pixels:
        for right in pixels:
            error = color_error(left, right) + (left[3] - right[3]) ** 2
            if error > best:
                best, c0, c1 = error, left, right
    q0, p0, decoded0 = bc7_endpoint_bits(c0)
    q1, p1, decoded1 = bc7_endpoint_bits(c1)
    palette = bc7_palette(decoded0, decoded1)
    choices = [min(range(16), key=lambda item: color_error(pixel, palette[item]) + (pixel[3] - palette[item][3]) ** 2) for pixel in pixels]
    if choices[0] > 7:
        q0, q1 = q1, q0
        p0, p1 = p1, p0
        choices = [15 - choice for choice in choices]
    writer = BitWriter()
    writer.put(0x40, 7)
    for channel in range(4):
        writer.put(q0[channel], 7)
        writer.put(q1[channel], 7)
    writer.put(p0, 1)
    writer.put(p1, 1)
    writer.put(choices[0], 3)
    for choice in choices[1:]:
        writer.put(choice, 4)
    if writer.pos != 128:
        raise UgxError("BC7 encoder wrote an invalid block.")
    return writer.bytes(16)


def encode_texture_blocks_numpy(pixels, texture_format: int) -> bytes:
    # Bound scratch memory by processing a few block rows at a time.
    count = len(pixels)
    rows = np.arange(count)
    bc7 = texture_format in (98, 99)
    bc5 = texture_format in (fourcc("ATI2"), fourcc("BC5U"), fourcc("BC5S"))

    def alpha_blocks(values):
        high, low = values.max(axis=1), values.min(axis=1)
        palette = np.stack([high, low] + [((7 - i) * high + i * low) // 7 for i in range(1, 7)], axis=1)
        # Equal endpoints use BC4's six-entry palette and explicit 0/255.
        equal = high == low
        palette[equal, 6] = 0
        palette[equal, 7] = 255
        choices = np.abs(values[:, :, None] - palette[:, None, :]).argmin(axis=2).astype(np.uint64)
        bits = np.sum(choices << (np.arange(16, dtype=np.uint64) * 3), axis=1, dtype=np.uint64)
        return np.concatenate([high[:, None], low[:, None], bits.astype("<u8").view(np.uint8).reshape(count, 8)[:, :6]], axis=1).astype(np.uint8)

    if bc5:
        return np.concatenate([alpha_blocks(pixels[:, :, 0]), alpha_blocks(pixels[:, :, 1])], axis=1).tobytes()

    channels = 4 if bc7 else 3
    colors = pixels[:, :, :channels]
    distances = np.sum((colors[:, :, None, :] - colors[:, None, :, :]) ** 2, axis=3)
    transparent = (pixels[:, :, 3] < 128).any(axis=1) if texture_format == fourcc("DXT1") else np.zeros(count, dtype=bool)
    if transparent.any():
        opaque = pixels[:, :, 3] >= 128
        distances[transparent[:, None, None] & ~(opaque[:, :, None] & opaque[:, None, :])] = -1
    farthest = distances.reshape(count, 256).argmax(axis=1)
    endpoints = np.stack([pixels[rows, farthest // 16], pixels[rows, farthest % 16]], axis=1)
    endpoints[transparent & (pixels[:, :, 3] < 128).all(axis=1)] = (0, 0, 0, 255)

    if bc7:
        quantized = np.stack([np.clip(np.rint((endpoints - bit) / 2.0), 0, 127).astype(np.int32) for bit in (0, 1)], axis=2)
        decoded = quantized * 2 + np.arange(2)[None, None, :, None]
        errors = np.sum((decoded - endpoints[:, :, None, :]) ** 2, axis=3)
        pbits = errors.argmin(axis=2)
        q = np.take_along_axis(quantized, pbits[:, :, None, None], axis=2)[:, :, 0, :]
        decoded = q * 2 + pbits[:, :, None]
        weights = np.array(BC7_WEIGHTS_4BIT)[None, :, None]
        palette = ((64 - weights) * decoded[:, 0, None, :] + weights * decoded[:, 1, None, :] + 32) >> 6
        choices = np.sum((pixels[:, :, None, :] - palette[:, None, :, :]) ** 2, axis=3).argmin(axis=2)
        swap = choices[:, 0] > 7
        q[swap] = q[swap, ::-1]
        pbits[swap] = pbits[swap, ::-1]
        choices[swap] = 15 - choices[swap]
        packed = np.zeros((count, 16), dtype=np.uint8)
        offset = 0

        def put(values, bits):
            nonlocal offset
            values = np.asarray(values, dtype=np.uint32)
            for bit in range(bits):
                packed[:, offset // 8] |= (((values >> bit) & 1) << (offset % 8)).astype(np.uint8)
                offset += 1

        put(0x40, 7)
        for channel in range(4):
            for endpoint in range(2):
                put(q[:, endpoint, channel], 7)
        put(pbits[:, 0], 1)
        put(pbits[:, 1], 1)
        put(choices[:, 0], 3)
        for pixel in range(1, 16):
            put(choices[:, pixel], 4)
        return packed.tobytes()

    codes = ((endpoints[:, :, 0] * 31 + 127) // 255 << 11) | ((endpoints[:, :, 1] * 63 + 127) // 255 << 5) | ((endpoints[:, :, 2] * 31 + 127) // 255)
    equal = codes[:, 0] == codes[:, 1]
    codes[equal, 1] = np.where(codes[equal, 0] != 0, codes[equal, 0] - 1, 1)
    swap = np.where(transparent, codes[:, 0] > codes[:, 1], codes[:, 0] <= codes[:, 1])
    codes[swap] = codes[swap, ::-1]
    rgb = np.stack([((codes >> 11) & 31) * 255 // 31, ((codes >> 5) & 63) * 255 // 63, (codes & 31) * 255 // 31], axis=2)
    palette = np.stack([rgb[:, 0], rgb[:, 1], (2 * rgb[:, 0] + rgb[:, 1]) // 3, (rgb[:, 0] + 2 * rgb[:, 1]) // 3], axis=1)
    palette[transparent, 2] = (rgb[transparent, 0] + rgb[transparent, 1]) // 2
    errors = np.sum((pixels[:, :, None, :3] - palette[:, None, :, :]) ** 2, axis=3)
    errors[transparent, :, 3] = 1 << 30
    choices = errors.argmin(axis=2).astype(np.uint32)
    choices[transparent[:, None] & (pixels[:, :, 3] < 128)] = 3
    indices = np.sum(choices << (np.arange(16, dtype=np.uint32) * 2), axis=1, dtype=np.uint32)
    color_blocks = np.concatenate([codes.astype("<u2").view(np.uint8).reshape(count, 4), indices.astype("<u4").view(np.uint8).reshape(count, 4)], axis=1)
    if texture_format == fourcc("DXT5"):
        return np.concatenate([alpha_blocks(pixels[:, :, 3]), color_blocks], axis=1).tobytes()
    if texture_format == fourcc("DXT3"):
        alpha = np.rint(pixels[:, :, 3] / 17.0).astype(np.uint64)
        bits = np.sum(alpha << (np.arange(16, dtype=np.uint64) * 4), axis=1, dtype=np.uint64)
        return np.concatenate([bits.astype("<u8").view(np.uint8).reshape(count, 8), color_blocks], axis=1).tobytes()
    return color_blocks.tobytes()


def encode_compressed_from_rgba(rgba: bytes, width: int, height: int, texture_format: int) -> bytes:
    if len(rgba) != width * height * 4:
        raise UgxError("RGBA buffer does not match DDS dimensions.")
    bc5_formats = (fourcc("ATI2"), fourcc("BC5U"), fourcc("BC5S"))
    bc7_formats = (98, 99)
    if texture_format not in (fourcc("DXT1"), fourcc("DXT3"), fourcc("DXT5"), *bc5_formats, *bc7_formats):
        raise UgxError("DDS write currently supports DXT1/DXT3/DXT5/BC5/BC7/R16G16B16A16_FLOAT only.")
    out = bytearray()
    if np is not None:
        source = np.frombuffer(rgba, dtype=np.uint8).reshape(height, width, 4)
        block_width = (width + 3) // 4
        x = np.minimum(np.arange(block_width * 4), width - 1)
        rows_per_batch = max(1, 1024 // block_width)
        for by in range(0, (height + 3) // 4, rows_per_batch):
            end = min((height + 3) // 4, by + rows_per_batch)
            y = np.minimum(np.arange(by * 4, end * 4), height - 1)
            pixels = source[y[:, None], x[None, :]].reshape(end - by, 4, block_width, 4, 4)
            pixels = pixels.transpose(0, 2, 1, 3, 4).reshape(-1, 16, 4).astype(np.int32)
            out.extend(encode_texture_blocks_numpy(pixels, texture_format))
        return bytes(out)
    for by in range((height + 3) // 4):
        for bx in range((width + 3) // 4):
            pixels = block_pixels(rgba, width, height, bx, by)
            if texture_format == fourcc("DXT3"):
                out.extend(encode_dxt3_alpha_block([pixel[3] for pixel in pixels]))
                out.extend(encode_dxt1_block_rgba(pixels, False))
            elif texture_format == fourcc("DXT5"):
                out.extend(encode_bc4_alpha_block([pixel[3] for pixel in pixels]))
                out.extend(encode_dxt1_block_rgba(pixels, False))
            elif texture_format in bc5_formats:
                out.extend(encode_bc4_alpha_block([pixel[0] for pixel in pixels]))
                out.extend(encode_bc4_alpha_block([pixel[1] for pixel in pixels]))
            elif texture_format in bc7_formats:
                out.extend(encode_bc7_mode6_block(pixels))
            else:
                out.extend(encode_dxt1_block_rgba(pixels, True))
    return bytes(out)


def decode_bc4_values(data: bytes, offset: int) -> Tuple[List[int], int]:
    alpha = [0] * 8
    alpha[0] = data[offset]
    alpha[1] = data[offset + 1]
    if alpha[0] > alpha[1]:
        for i in range(1, 7):
            alpha[i + 1] = ((7 - i) * alpha[0] + i * alpha[1]) // 7
    else:
        for i in range(1, 5):
            alpha[i + 1] = ((5 - i) * alpha[0] + i * alpha[1]) // 5
        alpha[6] = 0
        alpha[7] = 255
    alpha_bits = 0
    for i in range(6):
        alpha_bits |= data[offset + 2 + i] << (8 * i)
    return alpha, alpha_bits


def decode_bc5_block(data: bytes, offset: int, width: int, height: int, bx: int, by: int, rgba: bytearray):
    red, red_bits = decode_bc4_values(data, offset)
    green, green_bits = decode_bc4_values(data, offset + 8)
    for y in range(4):
        for x in range(4):
            px = bx * 4 + x
            py = by * 4 + y
            if px >= width or py >= height:
                continue
            shift = 3 * (y * 4 + x)
            dst = (py * width + px) * 4
            rgba[dst] = red[(red_bits >> shift) & 7]
            rgba[dst + 1] = green[(green_bits >> shift) & 7]
            rgba[dst + 2] = 255
            rgba[dst + 3] = 255


def decode_compressed_to_rgba(data: bytes, width: int, height: int, texture_format: int) -> Optional[bytes]:
    if not plausible_dimension(width) or not plausible_dimension(height):
        return None
    bc5_formats = (fourcc("ATI2"), fourcc("BC5U"), fourcc("BC5S"))
    if texture_format not in (fourcc("DXT1"), fourcc("DXT3"), fourcc("DXT5"), *bc5_formats):
        return None
    rgba = bytearray([255]) * (width * height * 4)
    block_bytes = 8 if texture_format == fourcc("DXT1") else 16
    offset = 0
    for by in range((height + 3) // 4):
        for bx in range((width + 3) // 4):
            if offset + block_bytes > len(data):
                return None
            if texture_format == fourcc("DXT1"):
                decode_dxt1_block(data, offset, width, height, bx, by, rgba)
            elif texture_format == fourcc("DXT3"):
                decode_dxt3_block(data, offset, width, height, bx, by, rgba)
            elif texture_format == fourcc("DXT5"):
                decode_dxt5_block(data, offset, width, height, bx, by, rgba)
            else:
                decode_bc5_block(data, offset, width, height, bx, by, rgba)
            offset += block_bytes
    return bytes(rgba)
