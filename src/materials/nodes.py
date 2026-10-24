"""Create and upgrade native material shader nodes."""

import bpy
from .context import material_output_node
from .groups import ensure_hw_shader_group
from .node_types import ensure_ugx_material_node_types
from .sockets import active_ugx_material_shader
from .ownership import remember_material_shader_state

__all__ = ('upgrade_ugx_material_node', 'create_hw_shader_node')

def upgrade_ugx_material_node(mat: bpy.types.Material, cache_version: int):
    expected_type = "ShaderNodeAndersHW2" if cache_version == 6 else "ShaderNodeAndersHWDE"
    node = active_ugx_material_shader(mat, cache_version)
    if node.bl_idname == expected_type:
        if not node.get("hwek_material_ui_version"):
            node["hwek_material_ui_version"] = 1
        return node
    incoming = [(link.from_socket, link.to_socket.name) for socket in node.inputs for link in socket.links]
    outgoing = [(link.from_socket.name, link.to_socket) for socket in node.outputs for link in socket.links]
    values = {}
    for socket in node.inputs:
        if hasattr(socket, "default_value"):
            value = socket.default_value
            try:
                values[socket.name] = tuple(value)
            except TypeError:
                values[socket.name] = value
    replacement = mat.node_tree.nodes.new(expected_type)
    if node.node_tree.get("hwek_parameter_interface"):
        replacement.node_tree = node.node_tree
    replacement.location = node.location
    replacement.width = node.width
    replacement.label = node.label
    for name, value in values.items():
        socket = replacement.inputs.get(name)
        if socket is not None and hasattr(socket, "default_value"):
            socket.default_value = value
    for source, name in incoming:
        socket = replacement.inputs.get(name)
        if socket is not None:
            mat.node_tree.links.new(source, socket)
    for name, target in outgoing:
        socket = replacement.outputs.get(name)
        if socket is not None:
            mat.node_tree.links.new(socket, target)
    name = node.name
    mat.node_tree.nodes.remove(node)
    replacement.name = name
    remember_material_shader_state(mat, replacement)
    return replacement


def create_hw_shader_node(mat: bpy.types.Material, cache_version: int):
    group = ensure_hw_shader_group(cache_version)
    output = material_output_node(mat)
    if group is None or output is None or "Surface" not in output.inputs:
        return None
    ensure_ugx_material_node_types()
    node = mat.node_tree.nodes.new("ShaderNodeAndersHW2" if cache_version == 6 else "ShaderNodeAndersHWDE")
    node.name = group.name
    node.label = ""
    node.node_tree = group
    node.location = (-110, 90)
    surface = output.inputs["Surface"]
    for link in list(surface.links):
        mat.node_tree.links.remove(link)
    if "BSDF" in node.outputs:
        mat.node_tree.links.new(node.outputs["BSDF"], surface)
    mat["hwek_ugx_shader_group"] = group.name
    remember_material_shader_state(mat, node)
    return node
