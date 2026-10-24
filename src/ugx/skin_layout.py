"""Retain native vertex fields while adding editable bone weights."""
import struct
import math

from ..errors import UgxError
from ..formats.binary import align_data
from ..formats.ecf import EcfFile, replace_ecf_chunk
from .data import UGX_NO_RIGID_BONE
from .skeleton_order import bone_headers
from .skin import encode_skin, vertex_group_from_section_bone
from .vertices import layout_for, expand_hw2_vertex_base, encode_vertex_color

__all__ = ('rewrite_rigid_section_skin',)


def rewrite_rigid_section_skin(data, model, section_weights, *, force_skinned=(), color_sections=(), required_bones=None, required_influences=None):
    """Weights address global target bones, one dictionary per source vertex."""
    if not section_weights:
        return data
    if model.cache_version not in (4, 6) or any(bone.big_endian for bone in model.bones):
        raise UgxError('Editing rigid section weights requires a PC v4 or v6 UGX.')
    required_bones = required_bones or {}
    required_influences = required_influences or {}
    file = EcfFile.parse(bytes(data))
    cached_chunk = file.chunks[0x700]
    cached = bytearray(cached_chunk.data)
    vertices = bytearray(file.chunks[0x702].data)
    rigid = [section.rigid_only for section in model.sections]
    rigid_bones = [vertex_group_from_section_bone(section, section.rigid_bone_index, len(model.bones))
                   for section in model.sections]
    palettes = [list(section.local_to_global_bones) for section in model.sections]
    accessory_bones = {}
    active_skin = set()
    for index, weights in section_weights.items():
        section = model.sections[index]
        if len(weights) != section.vertex_count:
            raise UgxError(f'Section {index}: skin edits do not cover its source vertices.')
        if any(not math.isfinite(weight) or weight < 0 for vertex in weights for weight in vertex.values()):
            raise UgxError(f'Section {index}: skin edits contain an invalid weight.')
        used = {bone for vertex in weights for bone, weight in vertex.items() if weight > 0.00001}
        used.update(required_bones.get(index, ()))
        if any(not 0 <= bone < len(model.bones) for bone in used):
            raise UgxError(f'Section {index}: skin edits reference a missing bone.')
        if not weights or not used:
            raise UgxError(f'Section {index}: skin edits have no bone weights.')
        accessory_bones.setdefault(section.accessory_index, set()).update(used)
        palette = palettes[index]
        record = section.section_record_file_offset-cached_chunk.offset
        for bone in sorted(used):
            if palette and bone not in palette:
                palette.append(bone)
        if len(palette) > 256:
            raise UgxError(f'Section {index}: skin palette exceeds 256 bones.')
        if palette != section.local_to_global_bones:
            align_data(cached, 4)
            pointer = len(cached)
            cached.extend(struct.pack('<'+'i'*len(palette), *palette))
            header = record+(40 if model.cache_version == 4 else 56)
            struct.pack_into('<I', cached, header, len(palette))
            struct.pack_into('<Q', cached, header+8, pointer)
        uniform = next(iter(used)) if len(used) == 1 and all(len(vertex) == 1 for vertex in weights) else None
        if section.rigid_only and uniform is not None and index not in force_skinned:
            struct.pack_into('<i', cached, record+12, palette.index(uniform) if palette else uniform)
            rigid_bones[index] = uniform
            continue
        index_layout = layout_for(section.layout, 'I')
        weight_layout = layout_for(section.layout, 'W')
        stride = section.vertex_stride
        insertion = None
        add_color = False
        if model.cache_version == 4 and index_layout is not None and weight_layout is not None:
            if index_layout.type_id != 6 or weight_layout.type_id not in (6, 9):
                raise UgxError(f'Section {index}: existing skin fields are not byte encoded.')
            indices, influences = index_layout.offset, weight_layout.offset
        elif model.cache_version == 4:
            if index_layout is not None or weight_layout is not None:
                raise UgxError(f'Section {index}: its source packer has an incomplete skin declaration.')
            insertion = max((element.offset+element.size for element in section.layout), default=0)
            if insertion > stride:
                raise UgxError(f'Section {index}: source attributes exceed the vertex stride.')
            indices, influences = insertion, insertion+4
            stride += 8
            for field, order in ((56, section.pack_order), (64, section.decl_order)):
                if not order and field == 64:
                    continue
                align_data(cached, 2)
                pointer = len(cached)
                cached.extend((order+'S').encode('ascii')+b'\0')
                struct.pack_into('<Q', cached, record+field, pointer)
            struct.pack_into('<II', cached, record+72+13*4, 6, 9)
        else:
            insertion = 24 if index in color_sections else 20
            if stride < 20:
                stride += 20-(12 if stride >= 12 else 8)
            short_base = 12 if section.vertex_stride >= 12 else 8
            opaque_color = section.vertex_stride < 20 and section.vertex_stride-short_base >= 4
            add_color = (index in color_sections or opaque_color) and layout_for(section.layout, 'D') is None
            if add_color:
                stride += 4
            indices, influences = insertion, insertion+4
            stride += 8
        start = len(vertices)
        maximum = max(1,required_influences.get(index,1))
        for vertex, current in enumerate(weights):
            offset = section.vertex_offset+vertex*section.vertex_stride
            raw = bytearray(file.chunks[0x702].data[offset:offset+section.vertex_stride])
            if len(raw) != section.vertex_stride:
                raise UgxError(f'Section {index}: its source vertex stream is incomplete.')
            if model.cache_version == 6:
                raw = expand_hw2_vertex_base(raw, section.vertices[vertex])
                if add_color:
                    raw[20:20] = encode_vertex_color(section.vertices[vertex].color, 9)
            if insertion is not None:
                raw[insertion:insertion] = bytes(8)
            encoded = {palette.index(bone) if palette else bone+(model.cache_version == 4): weight
                       for bone, weight in current.items() if weight > 0.00001}
            bone_bytes, weight_bytes = encode_skin(encoded)
            raw[indices:indices+4] = bone_bytes
            raw[influences:influences+4] = weight_bytes
            maximum = max(maximum, sum(weight > 0 for weight in weight_bytes))
            vertices.extend(raw)
        struct.pack_into('<ii', cached, record+8, maximum, UGX_NO_RIGID_BONE)
        struct.pack_into('<iii', cached, record+24, start, section.vertex_count*stride, stride)
        cached[section.rigid_only_offset-cached_chunk.offset] = 0
        rigid[index] = False
        active_skin.add(index)
        rigid_bones[index] = None
    all_rigid = all(rigid)
    only_bones = set(rigid_bones)
    single_rigid = all_rigid and len(only_bones) == 1 and None not in only_bones
    cached[54] = int(all_rigid)
    cached[55] = int(not any(palettes))
    cached[56] = int(all(not rigid[index] and (index in active_skin or
        (layout_for(section.layout,'I') is not None and layout_for(section.layout,'W') is not None))
        for index,section in enumerate(model.sections)))
    cached[57] = int(single_rigid)
    if single_rigid:
        struct.pack_into('<i', cached, 4, next(iter(only_bones)))
    _, bone_header = bone_headers(cached, model, cached_chunk.offset)
    count, _, accessories = struct.unpack_from('<IIQ', cached, bone_header+16)
    for accessory, used in accessory_bones.items():
        if not 0 <= accessory < count:
            raise UgxError(f'Skin edit references missing accessory {accessory}.')
        offset = accessories+accessory*24
        first, length = struct.unpack_from('<ii', cached, offset)
        if length:
            used.update((first, first+length-1))
        struct.pack_into('<ii', cached, offset, min(used), max(used)-min(used)+1)
    result = replace_ecf_chunk(bytearray(data), 0x700, bytes(cached))
    return replace_ecf_chunk(result, 0x702, bytes(vertices))
