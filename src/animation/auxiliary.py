"""Retain native auxiliary track data independently of the editable bone curves."""
import json
import struct
import zlib

from ..errors import UgxError
from ..formats.ecf import EcfFile
from ..formats.granny import (
    GRX_ROOT_SIZE, GRX_FILE_INFO_TRACK_GROUPS,
    grx_array_header, grx_ref_array_from_header, grx_u64, write_grx_array_header,
)
from ..formats.granny_layout import compact_granny_data, granny_pointer_fields
from .uax import UAX_GRANNY_CHUNK

__all__ = ('capture_uax_auxiliary_source', 'write_uax_auxiliary_source')

AUXILIARY_SOURCE = 'hwek_uax_auxiliary_source'


def capture_uax_auxiliary_source(uax, animation, group_index):
    groups = animation.track_groups or uax.granny_info.track_groups
    if not 0 <= group_index < len(groups):
        return ''
    source = EcfFile.parse(uax.original_bytes).chunks[UAX_GRANNY_CHUNK].data
    group = groups[group_index].offset
    if not (grx_array_header(source, group + 8)[0] or grx_array_header(source, group + 44)[0]
            or any(grx_u64(source, group + field) for field in (140, 148, 156))):
        return ''
    data = bytearray(source)
    data[:GRX_ROOT_SIZE] = bytes(GRX_ROOT_SIZE)
    struct.pack_into('<Q', data, 16, grx_u64(source, 16))
    # Bone channels and their LOD table belong to the newly sampled tracks.
    data[group + 20:group + 44] = bytes(24)
    data.extend(b'\0' * (-len(data) % 16))
    references = len(data)
    data.extend(struct.pack('<Q', group))
    write_grx_array_header(data, GRX_FILE_INFO_TRACK_GROUPS, 1, references)
    return zlib.compress(compact_granny_data(data)).hex()


def write_uax_auxiliary_source(writer, group_offset, action, group_index):
    snapshot = action.get(AUXILIARY_SOURCE, '')
    if not snapshot:
        metadata = json.loads(action.get('hwek_uax_track_groups_metadata', '[]'))
        if 0 <= group_index < len(metadata):
            group = metadata[group_index]
            if any(group.get(field) for field in ('vector_tracks', 'text_tracks', 'periodic_loop_offset',
                                                  'extended_data_type_offset', 'extended_data_object_offset')):
                raise UgxError(f'Animation {action.name}: auxiliary native tracks are unavailable. '
                               'Reimport the UAX before creating a source-independent export.')
        return
    try:
        data = bytearray(zlib.decompress(bytes.fromhex(snapshot)))
    except (ValueError, zlib.error) as exc:
        raise UgxError(f'Animation {action.name}: auxiliary native track data is invalid.') from exc
    pointers = granny_pointer_fields(data)
    groups = grx_ref_array_from_header(data, GRX_FILE_INFO_TRACK_GROUPS)
    if len(groups) != 1:
        raise UgxError(f'Animation {action.name}: auxiliary native tracks have no unique group.')
    source_group = groups[0]
    base = writer.reserve(len(data), 16)
    for field in pointers:
        target = grx_u64(data, field)
        if target:
            # References to the auxiliary owner must follow the new bone group.
            struct.pack_into('<Q', data, field, group_offset if target == source_group else base + target)
    writer.data[base:base + len(data)] = data
    for first, last in ((8, 20), (44, 56), (140, 164)):
        writer.data[group_offset + first:group_offset + last] = data[source_group + first:source_group + last]
