"""Add native colour fields without changing other vertex channels."""
import struct
from ..formats.binary import align_data
from ..formats.ecf import EcfFile, replace_ecf_chunk
from ..errors import UgxError
from .vertices import layout_for, encode_vertex_color, expand_hw2_vertex_base

__all__ = ('rewrite_section_colors',)


def rewrite_section_colors(data, model, schemas):
    if not schemas:
        return data
    if model.cache_version not in (4, 6) or any(section.big_endian for section in model.sections):
        raise UgxError('Editing vertex colour layouts requires a PC v4 or v6 UGX.')
    file = EcfFile.parse(data)
    cached_chunk = file.chunks[0x700]
    cached = bytearray(cached_chunk.data)
    vertices = bytearray()
    for index, section in enumerate(model.sections):
        align_data(vertices, 4)
        start = len(vertices)
        record = section.section_record_file_offset-cached_chunk.offset
        old_color = layout_for(section.layout, 'D')
        schema = schemas.get(index)
        stride = section.vertex_stride
        if schema is not None:
            kind, prefix = schema
            old_size = old_color.size if old_color is not None else 0
            if model.cache_version == 6:
                skinned = layout_for(section.layout, 'I') is not None and not section.rigid_only
                insertion = 20 if prefix or not skinned else 28
                if stride < 20:
                    stride += 20-(12 if stride >= 12 else 8)
            else:
                insertion = old_color.offset if old_color is not None else stride
            size = 16 if kind == 4 else 4
            stride += size-old_size
            if model.cache_version == 4:
                for field, order in ((56, section.pack_order), (64, section.decl_order)):
                    if not order and field == 64:
                        continue
                    align_data(cached, 2)
                    pointer = len(cached)
                    cached.extend((order+('D' if old_color is None else '')).encode('ascii')+b'\0')
                    struct.pack_into('<Q', cached, record+field, pointer)
                struct.pack_into('<I', cached, record+72+15*4, kind)
        for vertex, source in enumerate(section.vertices):
            offset = section.vertex_offset+vertex*section.vertex_stride
            raw = bytearray(file.chunks[0x702].data[offset:offset+section.vertex_stride])
            if len(raw) != section.vertex_stride:
                raise UgxError(f'Section {index}: its source vertex stream is incomplete.')
            if schema is not None:
                if model.cache_version == 6:
                    raw = expand_hw2_vertex_base(raw, source)
                if old_color is not None:
                    del raw[old_color.offset:old_color.offset+old_size]
                raw[insertion:insertion] = encode_vertex_color(source.color, kind)
            vertices.extend(raw)
        struct.pack_into('<iii', cached, record+24, start, section.vertex_count*stride, stride)
    result = replace_ecf_chunk(bytearray(data), 0x700, bytes(cached))
    return replace_ecf_chunk(result, 0x702, bytes(vertices))
