"""Edit the selected animation group's native placement independently of its pose."""
import math
import struct

from ..errors import UgxError
from ..formats.ecf import EcfFile
from ..formats.granny import GRX_TRANSFORM_SIZE, identity_grx_transform, write_grx_transform
from .uax import UAX_GRANNY_CHUNK

__all__ = ('native_uax_group_placement', 'uax_placement_bytes', 'uax_placement_component',
           'set_uax_placement_component', 'write_uax_group_placement')

PLACEMENT_PROPERTY = 'hwek_uax_initial_placement'


def identity_placement_bytes():
    record = bytearray(GRX_TRANSFORM_SIZE)
    write_grx_transform(record, 0, identity_grx_transform())
    return bytes(record)


def native_uax_group_placement(uax, animation, group_index):
    groups = animation.track_groups or uax.granny_info.track_groups
    if group_index == -1:
        return identity_placement_bytes()
    if not 0 <= group_index < len(groups):
        raise UgxError(f'Animation {animation.name}: its native placement group is unavailable.')
    data = EcfFile.parse(uax.original_bytes).chunks[UAX_GRANNY_CHUNK].data
    offset = groups[group_index].offset + 56
    return bytes(data[offset:offset+GRX_TRANSFORM_SIZE])


def uax_placement_bytes(action):
    if PLACEMENT_PROPERTY not in action:
        return identity_placement_bytes()
    try:
        record = bytes.fromhex(str(action[PLACEMENT_PROPERTY]))
    except ValueError as exc:
        raise UgxError(f'Animation {action.name}: initial placement is not a native transform.') from exc
    if len(record) != GRX_TRANSFORM_SIZE:
        raise UgxError(f'Animation {action.name}: initial placement must contain 68 native bytes.')
    return record


def uax_placement_component(action, field):
    offset, count, _flag = {'position': (4, 3, 1), 'rotation': (16, 4, 2), 'scale_shear': (32, 9, 4)}[field]
    values = struct.unpack_from(f'<{count}f', uax_placement_bytes(action), offset)
    return (values[3], *values[:3]) if field == 'rotation' else values


def set_uax_placement_component(action, field, values):
    offset, count, flag = {'position': (4, 3, 1), 'rotation': (16, 4, 2), 'scale_shear': (32, 9, 4)}[field]
    values = tuple(float(value) for value in values)
    if len(values) != count or not all(math.isfinite(value) for value in values):
        raise UgxError(f'Animation {action.name}: initial placement {field} requires {count} finite values.')
    if field == 'rotation':
        values = (*values[1:], values[0])
    record = bytearray(uax_placement_bytes(action))
    try:
        struct.pack_into(f'<{count}f', record, offset, *values)
    except (OverflowError, struct.error) as exc:
        raise UgxError(f'Animation {action.name}: initial placement {field} exceeds native float32 range.') from exc
    struct.pack_into('<I', record, 0, struct.unpack_from('<I', record)[0] | flag)
    action[PLACEMENT_PROPERTY] = record.hex()


def write_uax_group_placement(data, offset, action):
    if PLACEMENT_PROPERTY in action:
        data[offset+56:offset+56+GRX_TRANSFORM_SIZE] = uax_placement_bytes(action)
