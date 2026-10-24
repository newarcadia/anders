"""Bind native shader parameters to editable node inputs."""

import bpy
import json
from .context import ugx_material_cache_version
from .data import HW1_PARAMETER_SOCKETS, hw_shader_socket_name
from .groups import hw_shader_group_matches_builtin
from .properties import (
    ufx_buffer_float_label,
    ufx_buffer_float_property_name,
    ufx_buffer_float_values_from_hex,
    ufx_buffer_property_name,
    v4_attr_property_name,
)
from .sockets import active_ugx_material_shader, ugx_constant_socket_value, ugx_parameter_socket_property

__all__ = ('cleanup_ugx_texture_nodes', 'bind_ugx_parameter_socket', 'retarget_ugx_parameter_drivers', 'migrate_ugx_parameter_nodes', 'hw2_parameter_property', 'ensure_ugx_parameter_nodes', 'connect_ugx_uv_scales', 'draw_inline_ugx_parameters')

def cleanup_ugx_texture_nodes(mat):
    tree = mat.node_tree
    version = ugx_material_cache_version(mat)
    shader = active_ugx_material_shader(mat, version)
    slots = json.loads(mat.get("hwek_ugx_texture_slots", "[]"))
    for node in list(tree.nodes):
        usage = node.get("hwek_ugx_usage")
        if node.type == "TEX_IMAGE" and usage and not any(socket.is_linked for socket in node.outputs):
            index = int(node.get("hwek_ugx_slot_index", -1))
            if version == 6 and 0 <= index < len(slots):
                continue
            socket = shader.inputs.get(hw_shader_socket_name(usage, ugx_material_cache_version(mat)))
            if socket is not None and socket.is_linked:
                tree.nodes.remove(node)
    for node in list(tree.nodes):
        if node.get("hwek_uv_scale") and not any(socket.is_linked for socket in node.outputs):
            tree.nodes.remove(node)
    maps = {}
    for node in list(tree.nodes):
        if node.type != "UVMAP" or not (node.name.startswith("UGX UV") or node.label.startswith("UGX UV")):
            continue
        key = (node.uv_map, node.from_instancer)
        if key in maps:
            for link in list(node.outputs["UV"].links):
                tree.links.new(maps[key].outputs["UV"], link.to_socket)
            tree.nodes.remove(node)
        elif not node.outputs["UV"].is_linked:
            tree.nodes.remove(node)
        else:
            maps[key] = node


def bind_ugx_parameter_socket(mat, socket, prop):
    if prop not in mat:
        return
    for link in list(socket.links):
        mat.node_tree.links.remove(link)
    # Blender 5 sockets do not support custom properties. Nodes own the bindings.
    bindings = dict(socket.node.get("hwek_parameter_properties", {}))
    bindings[socket.identifier] = prop
    socket.node["hwek_parameter_properties"] = bindings
    value = mat[prop]
    if socket.type == "INT":
        value = int(value)
    socket.default_value = value
    path = socket.path_from_id("default_value")
    animation = mat.node_tree.animation_data
    curve = next((item for item in animation.drivers if item.data_path == path), None) if animation is not None else None
    if curve is not None and any(variable.targets[0].id == mat and variable.targets[0].data_path == f'[{json.dumps(prop)}]' for variable in curve.driver.variables):
        return
    driver = socket.driver_add("default_value").driver
    driver.type = "SCRIPTED"
    driver.expression = "value"
    for variable in list(driver.variables):
        driver.variables.remove(variable)
    variable = driver.variables.new()
    variable.name = "value"
    variable.type = "SINGLE_PROP"
    variable.targets[0].id_type = "MATERIAL"
    variable.targets[0].id = mat
    variable.targets[0].data_path = f'[{json.dumps(prop)}]'


def retarget_ugx_parameter_drivers(mat):
    animation = mat.node_tree.animation_data if mat.node_tree is not None else None
    if animation is None:
        return
    for curve in animation.drivers:
        if curve.driver.expression != "value":
            continue
        for variable in curve.driver.variables:
            target = variable.targets[0]
            if target.id_type == "MATERIAL" and target.id != mat and target.data_path.startswith('["hwek_ugx_'):
                target.id = mat


def migrate_ugx_parameter_nodes(mat, shader):
    tree = mat.node_tree
    legacy = [node for node in tree.nodes if node.bl_idname == "ShaderNodeAndersParameters"]
    for node in legacy:
        hw1 = bool(node.get("hw1_parameters"))
        index = int(node.get("buffer_index", 0))
        name = str(node.get("buffer_name", ""))
        for offset, socket in enumerate(node.inputs):
            prop = v4_attr_property_name(socket.name) if hw1 else ufx_buffer_float_property_name(index, name, offset)
            value = ugx_constant_socket_value(socket)
            mat[prop] = int(value) if socket.type == "INT" else value
            output = node.outputs.get(socket.name)
            if output is None:
                continue
            for link in list(output.links):
                if link.to_socket.type in ("VALUE", "INT"):
                    bind_ugx_parameter_socket(mat, link.to_socket, prop)
        if node.get("raw_buffer"):
            mat[ufx_buffer_property_name(index, name)] = node["raw_buffer"]
        tree.nodes.remove(node)
    if legacy:
        if not shader.node_tree.get("hwek_inline_parameters"):
            shader.node_tree = shader.node_tree.copy()
        for item in list(shader.node_tree.interface.items_tree):
            if item.item_type == "SOCKET" and (item.name == "Material Parameters" or item.name.endswith(" Parameters")):
                shader.node_tree.interface.remove(item)
    shader.node_tree["hwek_inline_parameters"] = True


def hw2_parameter_property(mat, buffer_name, offset):
    for index, buffer in enumerate(json.loads(mat.get("hwek_ugx_ufx_constant_buffers", "[]"))):
        if buffer.get("name") == buffer_name and len(ufx_buffer_float_values_from_hex(str(buffer.get("hex", "")))) > offset:
            return ufx_buffer_float_property_name(index, buffer_name, offset)
    return None


def ensure_ugx_parameter_nodes(mat):
    version = ugx_material_cache_version(mat)
    shader = active_ugx_material_shader(mat, version)
    if not shader.node_tree.get("hwek_inline_parameters") and not hw_shader_group_matches_builtin(shader.node_tree, version):
        shader.node_tree = shader.node_tree.copy()
    migrate_ugx_parameter_nodes(mat, shader)
    if version == 6 and str(json.loads(mat.get("hwek_ugx_ufx_attrs", "{}")).get("shaderPermutationName0", {}).get("value", "")).upper().startswith("HOGAN_STANDARD_"):
        socket = shader.inputs.get("Emission")
        prop = hw2_parameter_property(mat, "PSCBData", 6)
        if socket is not None and prop is not None and not ugx_parameter_socket_property(socket) and not socket.is_linked:
            original = mat.get(prop + "_source", mat[prop])
            if socket.default_value != original and mat[prop] == original:
                mat[prop] = float(socket.default_value)
    tree = mat.node_tree
    if version == 4:
        for name, target in HW1_PARAMETER_SOCKETS.items():
            socket = shader.inputs.get(target)
            prop = v4_attr_property_name(name)
            if socket is not None and prop in mat:
                bind_ugx_parameter_socket(mat, socket, prop)
                socket.hide = True
        # RGB values stay editable separately on the material node.
        color = shader.inputs.get("SpecColor")
        if color is not None:
            color.hide = True
            for component, name in enumerate(("SpecColorR", "SpecColorG", "SpecColorB")):
                prop = v4_attr_property_name(name)
                if prop not in mat:
                    continue
                driver = color.driver_add("default_value", component).driver
                driver.type = "SCRIPTED"
                driver.expression = "value"
                for variable in list(driver.variables):
                    driver.variables.remove(variable)
                variable = driver.variables.new()
                variable.name = "value"
                variable.type = "SINGLE_PROP"
                variable.targets[0].id_type = "MATERIAL"
                variable.targets[0].id = mat
                variable.targets[0].data_path = f'[{json.dumps(prop)}]'
    else:
        attrs = json.loads(mat.get("hwek_ugx_ufx_attrs", "{}"))
        standard = str(attrs.get("shaderPermutationName0", {}).get("value", "")).upper().startswith("HOGAN_STANDARD_")
        prop = hw2_parameter_property(mat, "PSCBData", 6)
        if standard and prop is not None and shader.inputs.get("Emission") is not None:
            bind_ugx_parameter_socket(mat, shader.inputs["Emission"], prop)
            shader.inputs["Emission"].hide = True
        if shader.inputs.get("Shader Passes") is None:
            shader.node_tree.interface.new_socket(name="Shader Passes", in_out="INPUT", socket_type="NodeSocketFloat")
        shader.inputs["Shader Passes"].hide_value = True
        if not shader.inputs["Shader Passes"].is_linked:
            node = tree.nodes.new("ShaderNodeAndersShaderPasses")
            node.width = 360
            node.location = (450, -150)
            node.node_tree = bpy.data.node_groups.new("Anders Shader Passes", "ShaderNodeTree")
            node.node_tree.interface.new_socket(name="Passes", in_out="OUTPUT", socket_type="NodeSocketFloat")
            node.node_tree.nodes.new("NodeGroupOutput")
            tree.links.new(node.outputs["Passes"], shader.inputs["Shader Passes"])
        if standard:
            for animation in [node for node in tree.nodes if node.bl_idname == "ShaderNodeAndersUVAnimation"]:
                for offset, name in enumerate(("Speed U", "Speed V")):
                    prop = hw2_parameter_property(mat, "VSCBData", offset)
                    if prop is not None:
                        bind_ugx_parameter_socket(mat, animation.inputs[name], prop)
                        animation.inputs[name].hide = True
            connect_ugx_uv_scales(mat, shader)
    cleanup_ugx_texture_nodes(mat)


def connect_ugx_uv_scales(mat, shader):
    tree = mat.node_tree
    for usage, offset in (("diffuse", 0), ("normal", 1), ("packedmask", 2), ("specular", 3), ("emissive", 5)):
        prop = hw2_parameter_property(mat, "PSCBData", offset)
        socket = shader.inputs.get(hw_shader_socket_name(usage, 6))
        if prop is None or socket is None or not socket.is_linked or socket.links[0].from_node.type != "TEX_IMAGE":
            continue
        texture = socket.links[0].from_node
        vector = texture.inputs["Vector"]
        source = vector.links[0].from_socket if vector.is_linked else None
        if source is not None and source.node.get("hwek_uv_scale"):
            bind_ugx_parameter_socket(mat, source.node.inputs["Scale"], prop)
            continue
        if source is None:
            uv = next((node for node in tree.nodes if node.type == "UVMAP" and node.uv_map == "UV0"), None)
            if uv is None:
                uv = tree.nodes.new("ShaderNodeUVMap")
                uv.name = "UGX UV0"
                uv.uv_map = "UV0"
            source = uv.outputs["UV"]
        scale = tree.nodes.new("ShaderNodeVectorMath")
        scale.operation = "SCALE"
        scale.label = ufx_buffer_float_label("PSCBData", offset, 8)
        scale["hwek_uv_scale"] = True
        scale.location = (texture.location.x - 240, texture.location.y)
        tree.links.new(source, scale.inputs[0])
        bind_ugx_parameter_socket(mat, scale.inputs["Scale"], prop)
        tree.links.new(scale.outputs["Vector"], vector)


def draw_inline_ugx_parameters(layout, mat, cache_version):
    if cache_version == 4:
        attrs = json.loads(mat.get("hwek_ugx_v4_attrs", "{}"))
        layout.label(text="Material Parameters")
        labels = {
            "SpecPower": "Specular Power", "SpecColorR": "Specular Red", "SpecColorG": "Specular Green", "SpecColorB": "Specular Blue",
            "EnvReflectivity": "Reflectivity", "EnvSharpness": "Reflection Sharpness", "EnvFresnel": "Fresnel",
            "EnvFresnelPower": "Fresnel Power", "AccessoryIndex": "Accessory Index", "BlendType": "Blend Type",
        }
        for name in attrs:
            prop = v4_attr_property_name(name)
            if prop in mat:
                layout.prop(mat, f'[{json.dumps(prop)}]', text=labels.get(name, name))
        return
    standard = str(json.loads(mat.get("hwek_ugx_ufx_attrs", "{}")).get("shaderPermutationName0", {}).get("value", "")).upper().startswith("HOGAN_STANDARD_")
    for index, buffer in enumerate(json.loads(mat.get("hwek_ugx_ufx_constant_buffers", "[]"))):
        name = str(buffer.get("name", "Buffer"))
        values = ufx_buffer_float_values_from_hex(str(buffer.get("hex", "")))
        layout.label(text={"VSCBData": "Vertex Parameters", "PSCBData": "Pixel Parameters"}.get(name, name))
        if not values:
            prop = ufx_buffer_property_name(index, name)
            if prop in mat:
                layout.prop(mat, f'[{json.dumps(prop)}]', text="Bytes")
        for offset in range(len(values)):
            prop = ufx_buffer_float_property_name(index, name, offset)
            if prop in mat:
                label = ufx_buffer_float_label(name, offset, len(values)) if standard else f"Float {offset}"
                label = {
                    "Animated UV speed U": "UV Speed U", "Animated UV speed V": "UV Speed V",
                    "Reserved vertex scalar A": "Vertex Scalar A", "Reserved vertex scalar B": "Vertex Scalar B",
                    "Packed map UV scale": "Mask UV Scale", "Specular team-tint strength": "Team Tint",
                    "Emissive intensity": "Emission", "Reserved pixel scalar": "Pixel Scalar",
                }.get(label, label)
                layout.prop(mat, f'[{json.dumps(prop)}]', text=label)
