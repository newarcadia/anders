"""Read UAX animation containers and write native float curves."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import os
import struct
from ..errors import UgxError
from ..formats.ecf import EcfFile, UAX_FILE_ID
from ..formats.granny_layout import validate_granny_layout
from ..formats.granny import (
    GRX_MEMBER_INLINE,
    GRX_MEMBER_INT16,
    GRX_MEMBER_REFERENCE_TO_ARRAY,
    GRX_MEMBER_UINT8,
    GrannyInfo,
    GrxWriter,
    grx_array_header,
    grx_cstring,
    grx_ref_array_from_header,
    grx_u64,
    parse_granny_info,
    write_grx_array_header,
)

__all__ = ('UAX_GRANNY_CHUNK', 'UaxFile', 'UAX_CURVE_TYPE_NAME', 'UAX_CURVE_TYPE_NAMES', 'UAX_CURVE_SCHEMA_DEFS', 'write_uax_float_curve', 'parse_uax_data', 'parse_uax')

UAX_GRANNY_CHUNK = 0x700


@dataclass
class UaxFile:
    path: str
    original_bytes: bytes
    granny_info: GrannyInfo


UAX_CURVE_TYPE_NAME = "CurveDataHeader_DaK32fC32f"
UAX_CURVE_TYPE_NAMES = {1: UAX_CURVE_TYPE_NAME, 3: "CurveDataHeader_DaConstant32f"}


UAX_CURVE_SCHEMA_DEFS = {
    "CurveDataHeader": [
        (GRX_MEMBER_UINT8, "Format", "", 0),
        (GRX_MEMBER_UINT8, "Degree", "", 0),
    ],
    UAX_CURVE_TYPE_NAMES[3]: [
        (GRX_MEMBER_INLINE, UAX_CURVE_TYPE_NAMES[3], "CurveDataHeader", 0),
        (GRX_MEMBER_INT16, "Padding", "", 0),
        (GRX_MEMBER_REFERENCE_TO_ARRAY, "Controls", "Real32", 0),
    ],
    UAX_CURVE_TYPE_NAME: [
        (GRX_MEMBER_INLINE, UAX_CURVE_TYPE_NAME, "CurveDataHeader", 0),
        (GRX_MEMBER_INT16, "Padding", "", 0),
        (GRX_MEMBER_REFERENCE_TO_ARRAY, "Knots", "Real32", 0),
        (GRX_MEMBER_REFERENCE_TO_ARRAY, "Controls", "Real32", 0),
    ],
}


def write_uax_float_curve(writer: GrxWriter, knots: List[float], controls: List[float], degree: int = 1) -> int:
    if not knots:
        knots = [0.0]
    packed_controls = struct.pack(f"<{len(controls)}f", *controls)
    if controls and len(controls) % len(knots) == 0:
        dimension = len(controls)//len(knots)
        first = packed_controls[:dimension*4]
        # Compare serialized controls so reducing a constant channel keeps
        # every native float32 component, including signed zero, unchanged.
        if packed_controls == first*len(knots):
            curve_offset = writer.reserve(16, 16)
            struct.pack_into('<BBh', writer.data, curve_offset, 3, 0, 0)
            controls_offset = writer.reserve(len(first), 16)
            writer.data[controls_offset:controls_offset+len(first)] = first
            write_grx_array_header(writer.data, curve_offset+4, dimension, controls_offset)
            return curve_offset
    curve_offset = writer.reserve(28, 16)
    # Generic float curves use table index 1; the authored degree controls
    # interpolation independently of the serialized curve family.
    struct.pack_into("<BBh", writer.data, curve_offset, 1, degree, 0)
    knots_offset = writer.reserve(len(knots) * 4, 16)
    struct.pack_into(f"<{len(knots)}f", writer.data, knots_offset, *knots)
    controls_offset = writer.reserve(len(controls) * 4, 16)
    writer.data[controls_offset:controls_offset+len(packed_controls)] = packed_controls
    write_grx_array_header(writer.data, curve_offset + 4, len(knots), knots_offset)
    write_grx_array_header(writer.data, curve_offset + 16, len(controls), controls_offset)
    return curve_offset


def parse_uax_data(path: str, data: bytes, decode_animation_curves=True) -> UaxFile:
    ecf = EcfFile.parse(data)
    if ecf.file_id != UAX_FILE_ID:
        raise UgxError("ECF file is not a UAX animation.")
    granny = ecf.chunks.get(UAX_GRANNY_CHUNK)
    if not granny:
        raise UgxError("UAX is missing Granny animation chunk 0x700.")
    if len(granny.data) < 148:
        raise UgxError("UAX Granny file-info record is truncated.")
    if grx_cstring(granny.data, grx_u64(granny.data, 16)).lower() != 'gr2ugx':
        raise UgxError("UAX Granny source marker must be gr2ugx.")
    count, offset = grx_array_header(granny.data, 120)
    if not count or offset <= 0 or offset + count * 8 > len(granny.data):
        raise UgxError("UAX animation reference array is empty or outside its Granny chunk.")
    for animation in grx_ref_array_from_header(granny.data, 120):
        if animation <= 0 or animation + 56 > len(granny.data):
            raise UgxError("UAX animation record is truncated.")
        group_count, groups = uax_array_range(granny.data, animation + 20, 8, 'track-group reference')
        for index in range(group_count):
            group = grx_u64(granny.data, groups + index * 8)
            if group <= 0 or group + 164 > len(granny.data):
                raise UgxError("UAX track-group record is truncated.")
            for field, stride, name in ((8, 32, 'vector-track'), (20, 60, 'transform-track'),
                                         (32, 4, 'transform LOD'), (44, 20, 'text-track')):
                uax_array_range(granny.data, group + field, stride, name)
    validate_granny_layout(granny.data)
    info = parse_granny_info(granny.data, decode_animation_curves)
    return UaxFile(os.path.abspath(path), data, info)


def uax_array_range(data, field, stride, name):
    count, offset = grx_array_header(data, field)
    if count and (offset <= 0 or offset + count * stride > len(data)):
        raise UgxError(f'UAX {name} array is outside its Granny chunk.')
    return count, offset


def parse_uax(path: str, decode_animation_curves=True) -> UaxFile:
    with open(path, "rb") as handle:
        return parse_uax_data(path, handle.read(), decode_animation_curves)
