"""Read native UV scales, address modes and animation values."""

from __future__ import annotations
from ..errors import UgxError
from .sockets import ugx_constant_socket_value, ugx_uv_source
from .mapping import hw1_mapping_source

__all__ = ('material_uv_animation', 'hw1_texture_coordinates')

def material_uv_animation(mat, bindings):
    animations = []
    for binding in bindings:
        vector = binding["node"].inputs["Vector"]
        node = ugx_uv_source(vector.links[0].from_node) if vector.is_linked else None
        if node is not None and node.bl_idname == "ShaderNodeAndersUVAnimation":
            if any(socket.is_linked and socket.name not in ("Vector", "Speed U", "Speed V") for socket in node.inputs):
                raise UgxError(f"Material {mat.name}: UV speed must be a constant value for UGX export.")
            animations.append(node)
    if not animations:
        return None
    speed = tuple(ugx_constant_socket_value(animations[0].inputs[name]) for name in ("Speed U", "Speed V"))
    if len(animations) != len(bindings) or any(
        tuple(ugx_constant_socket_value(node.inputs[name]) for name in ("Speed U", "Speed V")) != speed for node in animations
    ):
        raise UgxError(f"Material {mat.name}: connect the same UV animation speed to every texture.")
    return animations[0]


def hw1_texture_coordinates(texture):
    vector = texture.inputs["Vector"]
    node = vector.links[0].from_node if vector.is_linked else None
    seen = set()
    while node is not None and node.as_pointer() not in seen:
        if node.bl_idname == "ShaderNodeAndersHW1UV":
            return node
        seen.add(node.as_pointer())
        source = hw1_mapping_source(node)
        node = source.links[0].from_node if source is not None and source.is_linked else None
    return None
