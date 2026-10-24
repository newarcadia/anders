"""Read DDX containers and decode their native texture payloads."""

from __future__ import annotations
from dataclasses import dataclass
from ..errors import UgxError
from ..formats.binary import Reader
from ..formats.ecf import EcfFile
from .dds import decode_dds_to_rgba, decode_texture_payload

__all__ = ('DDX_FILE_ID', 'DDX_HEADER_CHUNK', 'DDX_MIP0_CHUNK', 'DDX_MIPCHAIN_CHUNK', 'DDX_HEADER_SIGNATURE', 'DdxTexture', 'decode_ddx')

DDX_FILE_ID = 0x13CF5D01


DDX_HEADER_CHUNK = 0x1D8828C6ECAF45F2


DDX_MIP0_CHUNK = 0x3F74B8E87D2B44BF


DDX_MIPCHAIN_CHUNK = 0x46F1FD3F394348B8


DDX_HEADER_SIGNATURE = 0xDDBB7738


@dataclass
class DdxTexture:
    path: str
    width: int
    height: int
    rgba: bytes


def decode_ddx(path: str) -> DdxTexture:
    with open(path, "rb") as handle:
        data = handle.read()
    dds = decode_dds_to_rgba(data)
    if dds:
        width, height, rgba = dds
        return DdxTexture(path, width, height, rgba)
    ecf = EcfFile.parse(data)
    if ecf.file_id != DDX_FILE_ID:
        raise UgxError("Texture is not a DDX file.")
    header = ecf.chunks.get(DDX_HEADER_CHUNK)
    mip = ecf.chunks.get(DDX_MIP0_CHUNK) or ecf.chunks.get(DDX_MIPCHAIN_CHUNK)
    if not header or not mip:
        raise UgxError("DDX is missing header or mip payload.")
    reader = Reader(header.data, True)
    if reader.u32() != DDX_HEADER_SIGNATURE:
        raise UgxError("Invalid DDX header signature.")
    reader.skip(8)
    reader.skip(4)
    width_pow = reader.u8()
    height_pow = reader.u8()
    reader.skip(2)
    data_format = reader.u32()
    if width_pow >= 31 or height_pow >= 31:
        raise UgxError("DDX dimensions are invalid.")
    width = 1 << width_pow
    height = 1 << height_pow
    rgba = decode_texture_payload(mip.data, data_format, width, height)
    if rgba is None and width != height:
        swapped = decode_texture_payload(mip.data, data_format, height, width)
        if swapped is not None:
            width, height, rgba = height, width, swapped
    if rgba is None:
        raise UgxError("Unsupported DDX payload format.")
    return DdxTexture(path, width, height, rgba)
