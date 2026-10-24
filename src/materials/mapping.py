"""Read static HW1 texture mappings without changing the authored graph."""
import math
from mathutils import Euler, Matrix, Vector
from ..errors import UgxError

__all__ = ('hw1_texture_mapping', 'hw1_mapping_source', 'hw1_sampler_extension')


def hw1_mapping_source(node):
    if node.type == 'REROUTE':
        return node.inputs[0]
    if node.type == 'MAPPING' or node.bl_idname == 'ShaderNodeAndersHW1UV':
        return node.inputs['Vector']
    if node.type == 'VECT_MATH' and node.operation in ('SCALE', 'MULTIPLY', 'ADD', 'SUBTRACT'):
        return node.inputs[0]
    return None


def mapping_constant(socket, mat):
    if socket.is_linked:
        raise UgxError(f'Material {mat.name}: {socket.node.name} {socket.name} must be constant for HW1 UV export.')
    value = socket.default_value
    values = tuple(value) if hasattr(value, '__len__') else (float(value),)
    if not all(math.isfinite(component) for component in values):
        raise UgxError(f'Material {mat.name}: {socket.node.name} has a non-finite UV mapping.')
    return values if len(values) > 1 else values[0]


def hw1_texture_mapping(texture, mat):
    vector = texture.inputs['Vector']
    pending = vector.links[0].from_node if vector.is_linked else None
    chain, seen = [], set()
    while pending is not None and pending.type != 'UVMAP':
        if pending.as_pointer() in seen:
            raise UgxError(f'Material {mat.name}: {texture.name} has a cyclic UV mapping.')
        seen.add(pending.as_pointer())
        source = hw1_mapping_source(pending)
        if source is None or not source.is_linked:
            raise UgxError(f'Material {mat.name}: bake the UV mapping on {texture.name} before export.')
        chain.append(pending)
        pending = source.links[0].from_node
    layer = pending.uv_map if pending is not None else 'UV0'
    if layer not in ('', 'UV0', 'UV1', 'UV2'):
        raise UgxError(f'Material {mat.name}: {texture.name} uses unsupported UV layer {layer}. Use UV0, UV1 or UV2.')
    matrix, velocity, coordinates = Matrix.Identity(4), Vector((0, 0, 0)), None
    for node in reversed(chain):
        transform = Matrix.Identity(4)
        if node.bl_idname == 'ShaderNodeAndersHW1UV':
            if coordinates is not None:
                raise UgxError(f'Material {mat.name}: use one HW1 UV Animation node per texture.')
            coordinates = node
            speed = [mapping_constant(node.inputs[name], mat) for name in ('Speed U', 'Speed V', 'Speed W')]
            velocity += Vector((speed[0], -speed[1], speed[2]))
        elif node.type == 'MAPPING':
            if node.vector_type not in ('POINT', 'VECTOR', 'TEXTURE'):
                raise UgxError(f'Material {mat.name}: bake Normal mapping on {node.name} before export.')
            scale = Vector(mapping_constant(node.inputs['Scale'], mat))
            rotation = Euler(mapping_constant(node.inputs['Rotation'], mat)).to_matrix().to_4x4()
            translation = Vector(mapping_constant(node.inputs['Location'], mat))
            transform = rotation @ Matrix.Diagonal((*scale, 1))
            if node.vector_type != 'VECTOR':
                transform = Matrix.Translation(translation) @ transform
            if node.vector_type == 'TEXTURE':
                if any(abs(value) < 1e-12 for value in scale):
                    raise UgxError(f'Material {mat.name}: {node.name} has a zero Texture mapping scale.')
                transform.invert()
        elif node.type == 'VECT_MATH':
            if node.operation == 'SCALE':
                scale = mapping_constant(node.inputs['Scale'], mat)
                transform = Matrix.Diagonal((scale, scale, scale, 1))
            else:
                value = Vector(mapping_constant(node.inputs[1], mat))
                if node.operation == 'MULTIPLY':
                    transform = Matrix.Diagonal((*value, 1))
                else:
                    transform = Matrix.Translation(-value if node.operation == 'SUBTRACT' else value)
        matrix = transform @ matrix
        velocity = transform.to_3x3() @ velocity
    if abs(matrix[2][0]) + abs(matrix[2][1]) + abs(matrix[2][3]) > 1e-6:
        raise UgxError(f'Material {mat.name}: {texture.name} maps surface UVs into W. Bake this mapping before export.')
    return layer or 'UV0', matrix, (velocity.x, -velocity.y, velocity.z), coordinates


def hw1_sampler_extension(usage, flags, velocity):
    # The native UGX renderer uses one address mode for all axes and forces
    # wrap on scrolling maps. Environment cubemaps always clamp.
    if usage == 'env':
        return 'EXTEND'
    scrolling = sum(float(value) ** 2 for value in velocity) > 0.0000125 ** 2
    return 'REPEAT' if (int(flags) & 3) != 3 or scrolling else 'EXTEND'
