"""Add tangent fields when HW1 materials gain normal textures."""
import math
import struct

from ..errors import UgxError
from ..formats.binary import align_data
from ..formats.ecf import EcfFile, replace_ecf_chunk
from ..textures.paths import disabled_texture_path
from .vertices import layout_for, tangent_layout_for

__all__ = ('extend_hw1_tangents',)


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _unit(value, fallback):
    length = math.sqrt(sum(component*component for component in value))
    return tuple(component/length for component in value) if length > 1e-12 else fallback


def _section_tangents(section, channel):
    count = len(section.vertices)
    tangents = [[0.0]*3 for _ in range(count)]
    bitangents = [[0.0]*3 for _ in range(count)]
    for offset in range(0, len(section.indices)-2, 3):
        indices = section.indices[offset:offset+3]
        if any(index < 0 or index >= count for index in indices):
            raise UgxError('HW1 tangent generation found an index outside its vertex section.')
        vertices = [section.vertices[index] for index in indices]
        edges = [tuple(vertex.pos[axis]-vertices[0].pos[axis] for axis in range(3)) for vertex in vertices[1:]]
        uv = [getattr(vertex, 'uv'+str(channel)) for vertex in vertices]
        du = (uv[1][0]-uv[0][0], uv[2][0]-uv[0][0])
        dv = (uv[1][1]-uv[0][1], uv[2][1]-uv[0][1])
        determinant = du[0]*dv[1]-du[1]*dv[0]
        if abs(determinant) < 1e-12:
            continue
        tangent = tuple((edges[0][axis]*dv[1]-edges[1][axis]*dv[0])/determinant for axis in range(3))
        bitangent = tuple((edges[1][axis]*du[0]-edges[0][axis]*du[1])/determinant for axis in range(3))
        for index in indices:
            for axis in range(3):
                tangents[index][axis] += tangent[axis]
                bitangents[index][axis] += bitangent[axis]
    result = []
    for vertex, tangent, bitangent in zip(section.vertices, tangents, bitangents):
        normal = _unit(vertex.normal, (0.0, 1.0, 0.0))
        projection = sum(a*b for a,b in zip(normal, tangent))
        axis = (0.0, 1.0, 0.0) if abs(normal[1]) < 0.9 else (1.0, 0.0, 0.0)
        fallback = _unit(_cross(axis, normal), (1.0, 0.0, 0.0))
        tangent = _unit(tuple(tangent[i]-normal[i]*projection for i in range(3)), fallback)
        # The native shader uses cross(T, N), with half-length T for a flipped frame.
        scale = 0.5 if sum(a*b for a,b in zip(_cross(tangent, normal), bitangent)) < 0 else 1.0
        result.append(tuple(component*scale for component in tangent))
    return result


def extend_hw1_tangents(data, model):
    if model.cache_version != 4:
        return data, set()
    additions = {}
    for index, section in enumerate(model.sections):
        if tangent_layout_for(section.layout) is not None or not 0 <= section.material_index < len(model.materials):
            continue
        slot = next((slot for slot in model.materials[section.material_index].slots
                     if slot.get('usage') == 'normal' and not disabled_texture_path(str(slot.get('path', '')))), None)
        if slot is None:
            continue
        channel = int(slot.get('channel', 0))
        if section.big_endian or channel not in (0, 1, 2) or layout_for(section.layout, 'N') is None or layout_for(section.layout, 'T'+str(channel)) is None:
            raise UgxError(f'Section {index}: adding an HW1 normal map needs PC vertex normals and UV{channel}.')
        additions[index] = _section_tangents(section, channel)
    if not additions:
        return data, set()
    file = EcfFile.parse(bytes(data))
    cached_chunk = file.chunks[0x700]
    cached, vertices = bytearray(cached_chunk.data), bytearray()
    source = file.chunks[0x702].data
    for index, section in enumerate(model.sections):
        align_data(vertices, 4)
        start = len(vertices)
        tangents = additions.get(index)
        stride = section.vertex_stride + (12 if tangents is not None else 0)
        for vertex in range(section.vertex_count):
            offset = section.vertex_offset+vertex*section.vertex_stride
            raw = source[offset:offset+section.vertex_stride]
            if len(raw) != section.vertex_stride:
                raise UgxError(f'Section {index}: its source vertex stream is incomplete.')
            vertices.extend(raw)
            if tangents is not None:
                vertices.extend(struct.pack('<3f', *tangents[vertex]))
        record = section.section_record_file_offset-cached_chunk.offset
        struct.pack_into('<3i', cached, record+24, start, section.vertex_count*stride, stride)
        if tangents is not None:
            is64 = section.rigid_only_offset-section.section_record_file_offset >= 140
            order = (section.decl_order or section.pack_order)+'A0'
            align_data(cached, 2)
            pointer = len(cached)
            cached.extend(order.encode('ascii')+b'\0')
            pointer_field = record+(56 if is64 else 48)
            struct.pack_into('<'+('Q' if is64 else 'I'), cached, pointer_field, pointer)
            if section.decl_order:
                struct.pack_into('<'+('Q' if is64 else 'I'), cached, pointer_field+(8 if is64 else 4), pointer)
            struct.pack_into('<I', cached, record+(72 if is64 else 56)+3*4, 3)
    used = max(section.vertex_offset+section.vertex_buffer_size for section in model.sections)
    vertices.extend(source[used:])
    result = replace_ecf_chunk(bytearray(data), 0x700, bytes(cached))
    result = replace_ecf_chunk(result, 0x702, bytes(vertices))
    return result, set(additions)
