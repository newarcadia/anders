"""Select matching native vertex shaders when a rigid mesh gains weights."""
import re
import struct

from ..errors import UgxError
from ..formats.ufx import ufx_vertex_input_signatures, ufx_pixel_programs
from .hw2 import hw2_material_spec_from_blender
from .properties import ufx_attr_property_name
from .skin_pairs import HW2_SHADER_SKIN_PAIRS, HW2_SKINNED_SHADER_INPUTS, HW2_SHADER_SKIN_BITS
from .ufx import ufx_shader_path_for_name

__all__ = ('material_skinned_shader_plan',)


def shader_inputs(path, expected_hash):
    if not path:
        raise UgxError('The matching UFX file is unavailable. Set the Halo Wars 2 shader directory in Anders preferences.')
    with open(path, 'rb') as handle:
        data = handle.read()
    if len(data) < 12 or struct.unpack_from('<I', data, 8)[0] != expected_hash:
        raise UgxError(f'Shader {path}: its UFX header hash does not match the material.')
    inputs = ufx_vertex_input_signatures(data)
    if not inputs:
        raise UgxError(f'Shader {path}: no vertex input contract is available.')
    return data, inputs


def shader_color_prefix(signatures):
    for signature in signatures:
        semantics = [item[0] for item in signature]
        if 'COLOR' in semantics and semantics.index('COLOR') < semantics.index('BLENDINDICES'):
            return True
    return False


def material_skinned_shader_plan(mat, source_path):
    spec = hw2_material_spec_from_blender(mat, False)
    edits = {ufx_attr_property_name('skinned'): True}
    color = False
    if not any(spec['shader_names']):
        raise UgxError(f'Material {mat.name}: assign a compatible skinned material before weighting this part.')
    for index, (name, header_hash) in enumerate(zip(spec['shader_names'], spec['shader_hashes'])):
        if not name:
            continue
        match = re.fullmatch(r'(.+)_([0-9A-Fa-f]{16})', name)
        flags = int(match[2], 16) if match else 0
        family = match[1].upper() if match else ''
        skin_bit = HW2_SHADER_SKIN_BITS.get(family)
        pair = HW2_SHADER_SKIN_PAIRS.get(name.upper())
        existing = HW2_SKINNED_SHADER_INPUTS.get(name.upper())
        if pair is not None and header_hash == pair[0]:
            target_name, target_hash = pair[1:3]
            color |= pair[3]
        elif existing is not None and header_hash == existing[0]:
            target_name, target_hash = name, header_hash
            color |= existing[1]
        elif match and skin_bit is not None and not flags & (1 << skin_bit):
            source_data, before = shader_inputs(ufx_shader_path_for_name(source_path, name), header_hash)
            target_name = f'{match[1]}_{flags | (1 << skin_bit):016X}'
            path = ufx_shader_path_for_name(source_path, target_name)
            if not path:
                raise UgxError(f'Material {mat.name}: {name} has no available skinned UFX permutation. Assign a compatible skinned material before weighting this part.')
            with open(path, 'rb') as handle:
                data = handle.read()
            after = ufx_vertex_input_signatures(data)
            compatible = len(before) == len(after) and bool(after)
            for left, right in zip(before, after):
                compatible &= set(left).issubset(right) and {item[0] for item in right if item not in left} == {'BLENDINDICES', 'BLENDWEIGHT'}
            if not compatible:
                raise UgxError(f'Material {mat.name}: {target_name} changes vertex inputs beyond adding skin weights.')
            if ufx_pixel_programs(source_data) != ufx_pixel_programs(data):
                raise UgxError(f'Material {mat.name}: {target_name} changes pixel shading. Assign a compatible skinned material before weighting this part.')
            target_hash = struct.unpack_from('<I', data, 8)[0]
            color |= shader_color_prefix(after)
        else:
            if not spec['skinned']:
                raise UgxError(f'Material {mat.name}: {name} needs an explicit compatible skinned material before weighting this part.')
            _, inputs = shader_inputs(ufx_shader_path_for_name(source_path, name), header_hash)
            if any(not {'BLENDINDICES', 'BLENDWEIGHT'}.issubset(item[0] for item in signature) for signature in inputs):
                raise UgxError(f'Material {mat.name}: {name} does not read bone indices and weights.')
            target_name, target_hash = name, header_hash
            color |= shader_color_prefix(inputs)
        edits[ufx_attr_property_name(f'shaderPermutationName{index}')] = target_name
        edits[ufx_attr_property_name(f'shaderPermutationHash{index}')] = str(target_hash)
    return edits, color
