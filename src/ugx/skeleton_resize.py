"""Resize native PC skeleton tables without replacing unrelated asset data."""
import struct

from ..errors import UgxError
from ..formats.binary import align_data
from ..formats.ecf import EcfFile, replace_ecf_chunk
from .data import V4_MAX_SKELETON_BONES
from .skeleton_order import bone_headers
from .skeleton_patch import patch_granny_skeleton
from .skin import vertex_group_from_section_bone, vertex_group_from_source_bone
from .vertices import layout_for

__all__ = ('resize_ugx_skeleton', 'refresh_resized_bone_bounds')


def refresh_resized_bone_bounds(data, model, overrides):
    lows = [[1e38]*3 for _ in model.bones]
    highs = [[-1e38]*3 for _ in model.bones]
    for section in model.sections:
        for vertex in section.vertices:
            if section.rigid_only:
                influences = [(vertex_group_from_section_bone(section, section.rigid_bone_index, len(model.bones)), 1.)]
            else:
                influences = [(vertex_group_from_source_bone(section, bone, len(model.bones)), weight)
                              for bone, weight in zip(vertex.bone_indices, vertex.bone_weights)]
            for bone, weight in influences:
                if bone is None or weight < 1./64.:
                    continue
                for axis, value in enumerate(vertex.pos):
                    lows[bone][axis] = min(lows[bone][axis], value)
                    highs[bone][axis] = max(highs[bone][axis], value)
    for index, bone in enumerate(model.bones):
        for key, values in (('low', lows[index]), ('high', highs[index])):
            if (index, key) in overrides:
                continue
            offset = getattr(bone, 'bounds_'+key+'_offset')
            if offset:
                struct.pack_into('<3f', data, offset, *values)
                setattr(bone, 'bounds_'+key, tuple(values))


def resize_ugx_skeleton(data, model, bones, source_indices, lod_errors, *, removed_bone_fallback=None):
    limit = V4_MAX_SKELETON_BONES if model.cache_version == 4 else 255
    if not 0 < len(bones) <= limit:
        raise UgxError(f'UGX v{model.cache_version} skeleton edits require between 1 and {limit} bones.')
    if len(bones) != len(source_indices) or len(bones) != len(lod_errors):
        raise UgxError('Skeleton edit arrays have different lengths.')
    if any(index is not None and not 0 <= index < len(model.bones) for index in source_indices):
        raise UgxError('Skeleton edit references a missing source bone.')
    if any(bone.big_endian or bone.name_offset_field_size != 8 for bone in model.bones):
        raise UgxError('Resizing a skeleton requires a PC v4 or v6 UGX.')
    if removed_bone_fallback is not None and not 0 <= removed_bone_fallback < len(bones):
        raise UgxError('Removed-bone fallback points outside the target skeleton.')
    old_to_new = {}
    for new, old in enumerate(source_indices):
        if old is not None:
            old_to_new.setdefault(old, new)

    def mapped(old):
        if old in old_to_new:
            return old_to_new[old]
        if removed_bone_fallback is not None:
            return removed_bone_fallback
        raise UgxError(f'Removed bone {old} still has geometry weights. Reassign its vertices before export.')

    file = EcfFile.parse(bytes(data))
    cached_chunk = file.chunks[0x700]
    cached = bytearray(cached_chunk.data)
    old_bones, header = bone_headers(cached, model, cached_chunk.offset)
    align_data(cached, 8)
    destination = len(cached)
    cached.extend(b'\0' * (len(bones)*80))
    for index, (bone, source) in enumerate(zip(bones, source_indices)):
        offset = destination+index*80
        if source is not None:
            cached[offset:offset+80] = cached_chunk.data[old_bones+source*80:old_bones+(source+1)*80]
        if source is None or bone.name != model.bones[source].name:
            raw = bone.name.encode('utf-8')
            if b'\0' in raw:
                raise UgxError(f'Bone {index} name contains a NUL byte.')
            align_data(cached, 2)
            struct.pack_into('<Q', cached, offset, len(cached))
            cached.extend(raw+b'\0')
        struct.pack_into('<16fi', cached, offset+8, *(v for row in bone.matrix for v in row), bone.parent)
    struct.pack_into('<I', cached, header, len(bones))
    struct.pack_into('<Q', cached, header+8, destination)
    for bound_header, key in ((header+48, 'low'), (header+64, 'high')):
        count, _, _ = struct.unpack_from('<IIQ', cached_chunk.data, bound_header)
        if count and count != len(model.bones):
            raise UgxError('Cached bone bounds do not match the source skeleton.')
        align_data(cached, 4)
        offset = len(cached)
        for bone, source in zip(bones, source_indices):
            cached.extend(struct.pack('<3f', *getattr(bone, 'bounds_'+key)))
        struct.pack_into('<I', cached, bound_header, len(bones))
        struct.pack_into('<Q', cached, bound_header+8, offset)

    vertices = bytearray(file.chunks[0x702].data)
    for section in model.sections:
        palette = []
        if section.local_to_global_bones:
            for source in section.local_to_global_bones:
                if source in old_to_new and old_to_new[source] not in palette:
                    palette.append(old_to_new[source])
            palette.extend(index for index in range(len(bones)) if index not in palette)
            align_data(cached, 4)
            offset = len(cached)
            cached.extend(struct.pack('<'+'i'*len(palette), *palette))
            local_header = section.section_record_file_offset-cached_chunk.offset+(40 if model.cache_version == 4 else 56)
            struct.pack_into('<I', cached, local_header, len(palette))
            struct.pack_into('<Q', cached, local_header+8, offset)
        old = vertex_group_from_section_bone(section, section.rigid_bone_index, len(model.bones))
        if old is not None:
            target = mapped(old)
            struct.pack_into('<i', cached, section.rigid_bone_index_offset-cached_chunk.offset,
                             palette.index(target) if palette else target)
        if section.rigid_only:
            continue
        indices = layout_for(section.layout, 'I')
        if indices is None:
            continue
        if indices.type_id != 6:
            raise UgxError(f'Skeleton edits cannot write skin index element type {indices.type_id}.')
        for vertex_index, vertex in enumerate(section.vertices):
            offset = section.vertex_file_offset-file.chunks[0x702].offset+vertex_index*section.vertex_stride+indices.offset
            for influence, (encoded, weight) in enumerate(zip(vertex.bone_indices, vertex.bone_weights)):
                if weight <= 0:
                    continue
                old = vertex_group_from_source_bone(section, encoded, len(model.bones))
                if old is None:
                    raise UgxError('Source skin index points outside its skeleton.')
                target = mapped(old)
                vertices[offset+influence] = palette.index(target) if palette else target+(model.cache_version == 4)
    accessory_count, _, accessories = struct.unpack_from('<IIQ', cached_chunk.data, header+16)
    for index in range(accessory_count):
        struct.pack_into('<ii', cached, accessories+index*24, 0, len(bones))
    native = patch_granny_skeleton(file.chunks[0x703].data, model.bones, bones,
                                  source_indices=source_indices, lod_errors=lod_errors)
    result = replace_ecf_chunk(bytearray(data), 0x702, bytes(vertices))
    result = replace_ecf_chunk(result, 0x700, bytes(cached))
    return replace_ecf_chunk(result, 0x703, native)
