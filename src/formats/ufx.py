"""Read UFX shader programs, vertex inputs and header hashes."""

from __future__ import annotations
from typing import Optional
import struct
from ..errors import UgxError

__all__ = ('ufx_shader_header_hash', 'ufx_vertex_input_signatures', 'ufx_pixel_programs')


def ufx_program_chunks(data: bytes):
    """Yield validated DXBC chunk tables embedded in a UFX resource."""
    if len(data) < 16 or data[:4] != b'UFXS':
        raise UgxError('Shader has no complete UFXS header.')
    cursor = 16
    while True:
        base = data.find(b'DXBC', cursor)
        if base < 0:
            return
        if base+32 > len(data):
            raise UgxError('Shader has an incomplete DXBC header.')
        size, count = struct.unpack_from('<II', data, base+24)
        if not 0 < count <= 64 or size < 32+count*4 or size > len(data)-base:
            raise UgxError('Shader has an invalid DXBC container extent.')
        cursor = base+size
        chunks = {}
        for index in range(count):
            relative = struct.unpack_from('<I', data, base+32+index*4)[0]
            if relative < 32+count*4 or relative+8 > size:
                raise UgxError('Shader DXBC chunk header is outside its container.')
            offset = base+relative
            length = struct.unpack_from('<I', data, offset+4)[0]
            if length > size-relative-8:
                raise UgxError('Shader DXBC chunk data is outside its container.')
            chunks[data[offset:offset+4]] = data[offset+8:offset+8+length]
        yield chunks


def ufx_pixel_programs(data: bytes):
    result = []
    for chunks in ufx_program_chunks(data):
        code = chunks.get(b'SHEX', chunks.get(b'SHDR', b''))
        if len(code) >= 4 and struct.unpack_from('<I', code)[0] >> 16 == 0:
            result.append(code)
    return tuple(result)


def ufx_vertex_input_signatures(data: bytes):
    """Read vertex input contracts from embedded DXBC containers."""
    result = []
    for chunks in ufx_program_chunks(data):
        code = chunks.get(b'SHEX', chunks.get(b'SHDR', b''))
        if len(code) < 4 or struct.unpack_from('<I', code)[0] >> 16 != 1:
            continue
        extended = b'ISG1' in chunks
        signature = chunks.get(b'ISG1' if extended else b'ISGN', b'')
        if len(signature) < 8:
            raise UgxError('Vertex shader has no complete input signature.')
        count, offset = struct.unpack_from('<II', signature)
        stride = 32 if extended else 24
        if offset < 8 or count > (len(signature)-offset)//stride:
            raise UgxError('Vertex shader input records are outside their signature.')
        inputs = []
        for index in range(count):
            record = offset+index*stride+(4 if extended else 0)
            name, semantic_index, system, component, _ = struct.unpack_from('<5I', signature, record)
            if name < offset+count*stride or name >= len(signature):
                raise UgxError('Vertex shader input name overlaps its records or is outside the signature.')
            end = signature.find(b'\0', name)
            if end < 0:
                raise UgxError('Vertex shader input name is unterminated.')
            try:
                semantic = signature[name:end].decode('ascii')
            except UnicodeDecodeError as exc:
                raise UgxError('Vertex shader input name is not ASCII.') from exc
            inputs.append((semantic, semantic_index, system, component, signature[record+20]))
        result.append(tuple(inputs))
    return result


def ufx_shader_header_hash(path: str) -> Optional[int]:
    if not path:
        return None
    try:
        with open(path, "rb") as handle:
            header = handle.read(12)
    except OSError:
        return None
    if len(header) < 12 or header[:4] != b"UFXS":
        return None
    return struct.unpack_from("<I", header, 8)[0]
