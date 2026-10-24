"""Resolve native material inputs from Blender shader sockets."""

import bpy
from ..errors import UgxError
from ..ugx.data import ugx_game_name
from .context import material_for_ugx_shader_node

__all__ = ('ugx_parameter_socket_property', 'ugx_constant_socket_value', 'ugx_uv_source', 'ugx_uv_map_name', 'active_ugx_material_shader')

def ugx_parameter_socket_property(socket):
    if socket.is_output:
        return None
    bindings = socket.node.get("hwek_parameter_properties", {})
    return bindings.get(socket.identifier)


def ugx_constant_socket_value(socket):
    prop = ugx_parameter_socket_property(socket)
    if prop:
        mat = material_for_ugx_shader_node(socket.node)
        if mat is not None and prop in mat:
            return float(mat[prop])
    if not socket.is_linked:
        return float(socket.default_value)
    source = socket.links[0].from_socket
    if source.node.bl_idname != "ShaderNodeAndersParameters" or source.name not in source.node.inputs:
        raise UgxError("Material parameters must be constant buffer values for UGX export.")
    return ugx_constant_socket_value(source.node.inputs[source.name])


def ugx_uv_source(node):
    if node.type == "VECT_MATH" and node.operation == "SCALE":
        vector = node.inputs[0]
        if not vector.is_linked:
            raise UgxError("Connect UV coordinates to the UV Scale node.")
        ugx_constant_socket_value(node.inputs["Scale"])
        return vector.links[0].from_node
    return node


def ugx_uv_map_name(node, mat):
    node = ugx_uv_source(node)
    if node.bl_idname in ("ShaderNodeAndersUVAnimation", "ShaderNodeAndersHW1UV"):
        vector = node.inputs.get("Vector")
        if vector is None or not vector.is_linked or vector.links[0].from_node.type != "UVMAP":
            raise UgxError(f"Material {mat.name}: connect a UV Map to UV Animation Vector.")
        node = vector.links[0].from_node
    if node.type != "UVMAP":
        raise UgxError(f"Material {mat.name}: use a UV Map for texture coordinates.")
    return node.uv_map


def active_ugx_material_shader(mat: bpy.types.Material, cache_version: int):
    if not mat.use_nodes or mat.node_tree is None:
        raise UgxError(f"Material {mat.name}: enable Use Nodes and connect an Anders material node.")
    outputs = [node for node in mat.node_tree.nodes if node.type == "OUTPUT_MATERIAL" and node.is_active_output]
    if not outputs or not outputs[0].inputs["Surface"].is_linked:
        raise UgxError(f"Material {mat.name}: connect an Anders material node to Material Output Surface.")
    link = outputs[0].inputs["Surface"].links[0]
    seen = set()
    while link.from_node.type == "REROUTE":
        node = link.from_node
        if node.as_pointer() in seen or not node.inputs[0].is_linked:
            raise UgxError(f"Material {mat.name}: material output has an incomplete reroute.")
        seen.add(node.as_pointer())
        link = node.inputs[0].links[0]
    node = link.from_node
    group = getattr(node, "node_tree", None)
    if node.type not in ("GROUP", "CUSTOM GROUP") or group is None or int(group.get("hwek_ugx_shader_game_version", 0)) != cache_version:
        raise UgxError(f"Material {mat.name}: connect an Anders {ugx_game_name(cache_version)} node directly to Surface. Bake unsupported shaders first.")
    return node
