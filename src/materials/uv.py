"""Build and synchronize native texture coordinates and UV animation."""

import bpy
from bpy.app.handlers import persistent
import json
from ..errors import UgxError
from .bindings import ugx_image_node_from_socket, ugx_material_node_bindings
from ..textures.paths import disabled_texture_path
from .context import material_output_node, ugx_material_cache_version, v4_shader_node_for_material
from .data import v4_material_slot_flags, v4_material_uvw_values
from .mapping import hw1_sampler_extension, hw1_texture_mapping
from .parameters import cleanup_ugx_texture_nodes, ensure_ugx_parameter_nodes
from .properties import ufx_buffer_float_property_name, ufx_buffer_float_values_from_hex
from .uv_data import hw1_texture_coordinates, material_uv_animation

__all__ = ('ugx_texture_vector', 'migrate_ugx_uv_inputs', 'connect_hw1_texture_coordinates', 'arrange_ugx_material_nodes', 'add_ugx_uv_animation', 'refresh_hw1_texture_sampling', 'sync_hw1_texture_sampling')

def ugx_texture_vector(mat: bpy.types.Material, layer_name: str = "UV0"):
    nodes = mat.node_tree.nodes
    existing = next((node for node in nodes if node.get("hwek_ugx_uv_vector") and node.get("hwek_ugx_uv_layer", "UV0") == layer_name), None)
    if existing is not None and existing.outputs.get("Vector") is not None:
        return existing.outputs["Vector"]
    uv = nodes.new("ShaderNodeUVMap")
    uv.label = "UGX " + layer_name
    uv.uv_map = layer_name
    uv["hwek_ugx_uv_vector"] = True
    uv["hwek_ugx_uv_layer"] = layer_name
    uv.location = (-900, 120)
    return uv.outputs["UV"]


def migrate_ugx_uv_inputs(mat):
    for node in list(mat.node_tree.nodes):
        if node.bl_idname != "ShaderNodeAndersUVAnimation" or node.inputs.get("Vector") is not None:
            continue
        old_uv = node.node_tree.nodes.get("UV Map")
        uv_name = old_uv.uv_map if old_uv is not None else "UV0"
        node.node_tree = node.node_tree.copy()
        group = node.node_tree
        group.interface.new_socket(name="Vector", in_out="INPUT", socket_type="NodeSocketVector")
        old_uv = group.nodes.get("UV Map")
        inputs = next(item for item in group.nodes if item.type == "GROUP_INPUT")
        if old_uv is not None:
            for link in list(old_uv.outputs["UV"].links):
                group.links.new(inputs.outputs["Vector"], link.to_socket)
            group.nodes.remove(old_uv)
        uv = mat.node_tree.nodes.new("ShaderNodeUVMap")
        uv.name = "UGX " + uv_name
        uv.uv_map = uv_name
        uv.location = (node.location.x - 220, node.location.y)
        mat.node_tree.links.new(uv.outputs["UV"], node.inputs["Vector"])
    cleanup_ugx_texture_nodes(mat)


def upgrade_hw1_uv_animation(node):
    group = node.node_tree
    if group.get('hwek_hw1_uv_version', 0) >= 2:
        return
    offset = next((item for item in group.nodes if item.type == 'COMBXYZ'), None)
    if offset is None or not offset.inputs['Y'].is_linked:
        return
    group = group.copy()
    node.node_tree = group
    offset = group.nodes[offset.name]
    source = offset.inputs['Y'].links[0].from_socket
    invert = group.nodes.new('ShaderNodeMath')
    invert.operation = 'MULTIPLY'
    invert.inputs[1].default_value = -1.0
    group.links.new(source, invert.inputs[0])
    group.links.new(invert.outputs[0], offset.inputs['Y'])
    group['hwek_hw1_uv_version'] = 2


def connect_hw1_texture_coordinates(mat):
    bindings = ugx_material_node_bindings(mat, 4, allow_missing=True)
    for binding in bindings:
        texture = binding["node"]
        if hw1_texture_coordinates(texture) is not None:
            existing = hw1_texture_coordinates(texture)
            upgrade_hw1_uv_animation(existing)
            if binding['slot'].get('usage') == 'envmask':
                time = existing.inputs['Time']
                animation = mat.node_tree.animation_data
                has_driver = animation is not None and any(curve.data_path == time.path_from_id('default_value') for curve in animation.drivers)
                if not has_driver:
                    driver = time.driver_add('default_value').driver
                    driver.expression = '(frame - 1) * fps_base / fps'
                    for name, path in (('fps', 'render.fps'), ('fps_base', 'render.fps_base')):
                        variable = driver.variables.new()
                        variable.name, variable.type = name, 'SINGLE_PROP'
                        variable.targets[0].id_type = 'SCENE'
                        variable.targets[0].id = bpy.context.scene
                        variable.targets[0].data_path = path
            continue
        node = mat.node_tree.nodes.new("ShaderNodeAndersHW1UV")
        node.label = str(binding["slot"].get("usage", "Texture")).capitalize() + " Coordinates"
        node.location = (texture.location.x - 270, texture.location.y)
        channel = binding["channel"]
        vector = texture.inputs["Vector"]
        source = vector.links[0].from_node if vector.is_linked else None
        if source is None:
            source = mat.node_tree.nodes.new("ShaderNodeUVMap")
            source.uv_map = f"UV{channel}"
            source.name = f"UGX UV{channel}"
        mat.node_tree.links.new(vector.links[0].from_socket if vector.is_linked else source.outputs["UV"], node.inputs["Vector"])
        speeds = v4_material_uvw_values(str(binding["slot"].get("usage", "")), binding["slot"])
        # Retain environment-map UVW values without scrolling surface coordinates.
        if str(binding["slot"].get("usage", "")) == "env":
            node.inputs["Time"].driver_remove("default_value")
            node.inputs["Time"].default_value = 0.0
        for name, value in zip(("Speed U", "Speed V", "Speed W"), speeds):
            node.inputs[name].default_value = value
        node.inputs["Map Flags"].default_value = v4_material_slot_flags(str(binding["slot"].get("usage", "")), binding["slot"])
        mat.node_tree.links.new(node.outputs["UV"], vector)
    refresh_hw1_texture_sampling(mat)
    cleanup_ugx_texture_nodes(mat)
    return next((node for node in mat.node_tree.nodes if node.bl_idname == "ShaderNodeAndersHW1UV"), None)


def arrange_ugx_material_nodes(mat: bpy.types.Material, cache_version: int):
    shader = v4_shader_node_for_material(mat, cache_version)
    if shader is None:
        return
    shader.label = "Anders HW2 Material" if cache_version == 6 else "Anders HW1 Material"
    shader.width = 350
    if cache_version == 6 and shader.inputs.get("Emission") is not None:
        attrs = json.loads(mat.get("hwek_ugx_ufx_attrs", "{}"))
        buffers = json.loads(mat.get("hwek_ugx_ufx_constant_buffers", "[]"))
        standard = str(attrs.get("shaderPermutationName0", {}).get("value", "")).upper().startswith("HOGAN_STANDARD_")
        shader.inputs["Emission"].hide = not standard or not any(
            buffer.get("name") == "PSCBData" and len(ufx_buffer_float_values_from_hex(str(buffer.get("hex", "")))) >= 8
            for buffer in buffers
        )
    shader.location = (0, 160)
    material_output_node(mat).location = (400, 160)
    for socket in shader.inputs:
        if socket.name in ("ufxVersion", "blendMode", "shadowRequiresConsts", "skinned", "terrainBlending"):
            socket.hide = True
    textures = [node for node in mat.node_tree.nodes if node.type == "TEX_IMAGE"]
    for index, node in enumerate(textures):
        node.width = 260
        node.location = (-550, 160 - index * 300)
        node.label = str(node.get("hwek_ugx_usage", "Texture")).capitalize()
    bsdf = next((node for node in mat.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
    if bsdf is not None and not any(socket.is_linked for socket in bsdf.outputs):
        mat.node_tree.nodes.remove(bsdf)
    mat["hwek_ugx_native_material"] = True
    ensure_ugx_parameter_nodes(mat)


def add_ugx_uv_animation(mat):
    if ugx_material_cache_version(mat) == 4:
        return connect_hw1_texture_coordinates(mat)
    bindings = ugx_material_node_bindings(mat, 6)
    attrs = json.loads(mat.get("hwek_ugx_ufx_attrs", "{}"))
    if not str(attrs.get("shaderPermutationName0", {}).get("value", "")).upper().startswith("HOGAN_STANDARD_"):
        raise UgxError(f"Material {mat.name}: UV Animation requires a standard HW2 material.")
    existing = material_uv_animation(mat, bindings)
    if existing is not None:
        return existing
    if not bindings:
        raise UgxError(f"Material {mat.name}: connect image textures before adding UV animation.")
    speeds = (0.0, 0.0)
    for index, buffer in enumerate(json.loads(mat.get("hwek_ugx_ufx_constant_buffers", "[]"))):
        if buffer.get("name") == "VSCBData":
            speeds = tuple(float(mat.get(ufx_buffer_float_property_name(index, "VSCBData", offset), 0.0)) for offset in (0, 1))
    nodes = {}
    for binding in bindings:
        channel = binding["channel"]
        if channel not in nodes:
            node = mat.node_tree.nodes.new("ShaderNodeAndersUVAnimation")
            uv = next((item for item in mat.node_tree.nodes if item.type == "UVMAP" and item.uv_map == f"UV{channel}"), None)
            if uv is None:
                uv = mat.node_tree.nodes.new("ShaderNodeUVMap")
                uv.name = f"UGX UV{channel}"
                uv.uv_map = f"UV{channel}"
            uv.location = (-1150, 160 - len(nodes) * 240)
            mat.node_tree.links.new(uv.outputs["UV"], node.inputs["Vector"])
            node.location = (-850, 160 - len(nodes) * 240)
            for name, value in zip(("Speed U", "Speed V"), speeds):
                node.inputs[name].default_value = value
            nodes[channel] = node
        target = binding["node"].inputs["Vector"]
        if target.is_linked and target.links[0].from_node.get("hwek_uv_scale"):
            target = target.links[0].from_node.inputs[0]
        mat.node_tree.links.new(nodes[channel].outputs["UV"], target)
    ensure_ugx_parameter_nodes(mat)
    cleanup_ugx_texture_nodes(mat)
    return next(iter(nodes.values()))


def refresh_hw1_diffuse_alpha(mat, slots):
    shader = v4_shader_node_for_material(mat, 4)
    if shader is None:
        return
    mask, diffuse = shader.inputs.get('Player_color (xf)'), shader.inputs.get('Diffuse (df)')
    if mask is None or diffuse is None:
        return
    current = mask.links[0].from_node if mask.is_linked else None
    fallback = current if current is not None and current.get('hwek_hw1_diffuse_alpha_fallback') else None
    if current is not None and fallback is None:
        return
    declared = any(slot.get('usage') == 'xform' and not disabled_texture_path(str(slot.get('path', ''))) for slot in slots)
    if declared:
        return
    try:
        texture = ugx_image_node_from_socket(diffuse, mat, True)
    except UgxError:
        return
    if texture is None:
        if fallback is not None:
            mat.node_tree.nodes.remove(fallback)
        return
    if fallback is None:
        fallback = mat.node_tree.nodes.new('ShaderNodeCombineColor')
        fallback.name = 'HW1 Diffuse Alpha Team Mask'
        fallback['hwek_hw1_diffuse_alpha_fallback'] = True
    for name in ('Red', 'Green', 'Blue'):
        target = fallback.inputs[name]
        if not target.is_linked or target.links[0].from_socket != texture.outputs['Alpha']:
            mat.node_tree.links.new(texture.outputs['Alpha'], target)
    if not mask.is_linked:
        mat.node_tree.links.new(fallback.outputs['Color'], mask)


def refresh_hw1_texture_sampling(mat):
    """Update image samplers from native flags and per-map animation."""
    if mat.node_tree is None or ugx_material_cache_version(mat) != 4:
        return
    slots = json.loads(mat.get('hwek_ugx_texture_slots', '[]'))
    refresh_hw1_diffuse_alpha(mat, slots)
    for texture in mat.node_tree.nodes:
        if texture.type != 'TEX_IMAGE':
            continue
        index = int(texture.get('hwek_ugx_slot_index', -1))
        slot = slots[index] if 0 <= index < len(slots) else {}
        usage = str(texture.get('hwek_ugx_usage', slot.get('usage', '')))
        try:
            _, _, velocity, coordinates = hw1_texture_mapping(texture, mat)
            flags = int(coordinates.inputs['Map Flags'].default_value) if coordinates is not None else v4_material_slot_flags(usage, slot)
        except UgxError:
            continue
        extension = hw1_sampler_extension(usage, flags, velocity)
        if texture.extension != extension:
            texture.extension = extension


@persistent
def sync_hw1_texture_sampling(_scene, depsgraph):
    materials = set()
    trees = set()
    for update in depsgraph.updates:
        data = getattr(update.id, 'original', update.id)
        if isinstance(data, bpy.types.Material):
            materials.add(data)
        elif isinstance(data, bpy.types.NodeTree):
            trees.add(data)
    if trees:
        materials.update(mat for mat in bpy.data.materials if mat.node_tree in trees)
    for mat in materials:
        if mat.get('hwek_ugx_native_material'):
            refresh_hw1_texture_sampling(mat)
