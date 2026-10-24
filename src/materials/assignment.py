"""Create native materials and assign game-specific shader settings."""

import bpy
import json
import os
from ..errors import UgxError
from ..ugx.data import ugx_game_name
from ..ugx.read import read_materials
from .context import is_ugx_material, material_output_node, ugx_material_cache_version, v4_shader_node_for_material
from .data import (
    Material,
    build_v4_material_bdt,
    editable_ugx_texture_slot,
    hw_shader_socket_name,
    ugx_material_display_name,
)
from .discovery import discover_material_textures
from .hw2 import build_v6_material_bdt
from .nodes import create_hw_shader_node
from .native import remember_native_material
from .ownership import remember_material_shader_state, sync_material_shader_state
from .properties import apply_ufx_material_metadata, apply_v4_material_metadata, ufx_buffer_float_values_from_hex
from .sockets import ugx_uv_map_name
from .textures import attach_material_textures
from .uv import (
    add_ugx_uv_animation,
    arrange_ugx_material_nodes,
    connect_hw1_texture_coordinates,
    ugx_texture_vector,
)

__all__ = ('make_material', 'clear_ugx_material_data', 'initialize_ugx_material_shader', 'assign_ugx_material_data')

def make_material(material: Material, index: int, source_path: str, cache_version: int):
    mat = bpy.data.materials.new(ugx_material_display_name(material, index))
    mat.use_nodes = True
    mat.diffuse_color = (
        ((index * 97) % 255) / 255.0,
        ((index * 57 + 80) % 255) / 255.0,
        ((index * 33 + 150) % 255) / 255.0,
        1.0,
    )
    mat["hwek_ugx_cache_version"] = cache_version
    mat["hwek_ugx_game"] = ugx_game_name(cache_version)
    mat["hwek_ugx_material_index"] = index
    mat["hwek_ugx_source_path"] = os.path.abspath(source_path)
    remember_native_material(mat, material)
    mat["hwek_ugx_texture_slots"] = json.dumps(material.slots)
    mat["hwek_ugx_source_name"] = material.name
    mat["hwek_ugx_blender_name"] = mat.name
    mat["hwek_ugx_export_name"] = material.name
    if material.name_capacity:
        mat["hwek_ugx_name_offset"] = material.name_offset
        mat["hwek_ugx_name_capacity"] = material.name_capacity
        if material.name_entry_offset:
            mat["hwek_ugx_name_entry_offset"] = material.name_entry_offset
    usage_sources = set()
    for slot_index, slot in enumerate(material.slots):
        usage = slot.get("usage", "").lower()
        if usage and editable_ugx_texture_slot(slot, cache_version):
            slot_path = str(slot.get("path", ""))
            mat[f"hwek_ugx_texture_slot_{slot_index}"] = slot_path
            mat[f"hwek_ugx_texture_slot_{slot_index}_synced"] = slot_path
            if usage not in usage_sources:
                mat[f"hwek_ugx_texture_{usage}"] = slot_path
                mat[f"hwek_ugx_texture_{usage}_source"] = slot_path
                usage_sources.add(usage)
    attach_material_textures(mat, material, source_path, cache_version)
    discover_material_textures(mat, os.path.dirname(os.path.abspath(source_path)), source_path)
    arrange_ugx_material_nodes(mat, cache_version)
    if cache_version == 4:
        connect_hw1_texture_coordinates(mat)
    if cache_version == 6:
        attrs = json.loads(mat.get("hwek_ugx_ufx_attrs", "{}"))
        standard = str(attrs.get("shaderPermutationName0", {}).get("value", "")).upper().startswith("HOGAN_STANDARD_")
        buffers = json.loads(mat.get("hwek_ugx_ufx_constant_buffers", "[]"))
        scrolling = any(
            buffer.get("name") == "VSCBData" and any(ufx_buffer_float_values_from_hex(str(buffer.get("hex", "")))[:2])
            for buffer in buffers
        )
        textures = [node for node in mat.node_tree.nodes if node.type == "TEX_IMAGE"]
        if standard and scrolling and textures and all(node.image is not None and not node.image.get("hwek_ugx_missing_texture") for node in textures):
            add_ugx_uv_animation(mat)
    remember_material_shader_state(mat, v4_shader_node_for_material(mat, cache_version))
    return mat


def clear_ugx_material_data(mat: bpy.types.Material):
    for key in list(mat.keys()):
        if str(key).startswith("hwek_ugx_"):
            del mat[key]


def initialize_ugx_material_shader(mat, shader, cache_version):
    if not is_ugx_material(mat):
        assign_ugx_material_data(mat, cache_version, shader_node=shader)
        return
    previous_version = ugx_material_cache_version(mat)
    if previous_version == cache_version:
        sync_material_shader_state(mat)
        previous = next((node for node in mat.node_tree.nodes
                         if node.get('hwek_ugx_shader_node_id') == mat.get('hwek_ugx_shader_state_id')
                         and node.get('hwek_ugx_shader_node_id')), None)
        if previous is None:
            previous = v4_shader_node_for_material(mat, cache_version)
        if previous is None or previous == shader:
            remember_material_shader_state(mat, shader)
            return
        remember_material_shader_state(mat, previous)
        source = read_materials(build_v4_material_bdt([mat.name]) if cache_version == 4
                                else build_v6_material_bdt([None], False), 1)[0]
        state = {'hwek_ugx_cache_version': cache_version, 'hwek_ugx_game': ugx_game_name(cache_version),
                 'hwek_ugx_native_material': True,
                 'hwek_ugx_source_name': mat.name, 'hwek_ugx_blender_name': mat.name,
                 'hwek_ugx_export_name': mat.name}
        apply_v4_material_metadata(state, source, shader)
        apply_ufx_material_metadata(state, source, shader, cache_version)
        for slot in source.slots:
            slot['path'] = ''
        state['hwek_ugx_texture_slots'] = json.dumps(source.slots)
        remember_material_shader_state(state, shader)
        return
    previous = v4_shader_node_for_material(mat, previous_version)
    connections = {}
    coordinates = []
    color = None
    if previous is not None:
        for usage in ("diffuse", "normal", "specular", "emissive"):
            socket = previous.inputs.get(hw_shader_socket_name(usage, previous_version))
            if socket is not None and socket.is_linked:
                connections[usage] = socket.links[0].from_socket
        color_socket = previous.inputs.get("Player Color" if previous_version == 6 else "Player_color")
        if color_socket is not None:
            color = tuple(color_socket.default_value)
        for node in mat.node_tree.nodes:
            if node.type != "TEX_IMAGE" or not node.inputs["Vector"].is_linked:
                continue
            source = node.inputs["Vector"].links[0].from_node
            if not source.get("hwek_uv_scale") and source.bl_idname not in ("ShaderNodeAndersUVAnimation", "ShaderNodeAndersHW1UV"):
                continue
            try:
                coordinates.append((node, ugx_uv_map_name(source, mat)))
            except UgxError:
                continue
    assign_ugx_material_data(mat, cache_version, shader_node=shader)
    # Parameters belong to one game. Reuse images and UV layers with the new
    # game's defaults rather than leaving drivers bound to removed properties.
    for node, uv_layer in coordinates:
        mat.node_tree.links.new(ugx_texture_vector(mat, uv_layer or "UV0"), node.inputs["Vector"])
    slots = json.loads(mat["hwek_ugx_texture_slots"])
    for usage, source in connections.items():
        socket_name = hw_shader_socket_name(usage, cache_version)
        target = shader.inputs.get(socket_name)
        if target is not None:
            mat.node_tree.links.new(source, target)
            if source.node.type == "TEX_IMAGE":
                for index, slot in enumerate(slots):
                    if hw_shader_socket_name(slot.get("usage", ""), cache_version) == socket_name:
                        source.node["hwek_ugx_slot_index"] = index
                        source.node["hwek_ugx_usage"] = slot["usage"]
                        break
    if color is not None:
        shader.inputs["Player Color" if cache_version == 6 else "Player_color"].default_value = color
    arrange_ugx_material_nodes(mat, cache_version)


def assign_ugx_material_data(mat: bpy.types.Material, cache_version: int, skinned: bool = False, shader_node=None):
    if cache_version not in (4, 6):
        raise UgxError(f"Unsupported UGX material version {cache_version}.")

    clear_ugx_material_data(mat)
    if cache_version == 4:
        source = read_materials(build_v4_material_bdt([mat.name]), 1)[0]
    else:
        source = read_materials(build_v6_material_bdt([mat], skinned), 1)[0]
    source.name = mat.name

    mat["hwek_ugx_cache_version"] = cache_version
    mat["hwek_ugx_game"] = ugx_game_name(cache_version)
    mat["hwek_ugx_texture_slots"] = json.dumps(source.slots)
    mat["hwek_ugx_source_name"] = mat.name
    mat["hwek_ugx_blender_name"] = mat.name
    mat["hwek_ugx_export_name"] = mat.name
    for slot_index, slot in enumerate(source.slots):
        if not editable_ugx_texture_slot(slot, cache_version):
            continue
        path = str(slot.get("path", ""))
        mat[f"hwek_ugx_texture_slot_{slot_index}"] = path
        mat[f"hwek_ugx_texture_slot_{slot_index}_synced"] = path
        usage = str(slot.get("usage", "")).lower()
        if usage:
            mat[f"hwek_ugx_texture_{usage}"] = path
            mat[f"hwek_ugx_texture_{usage}_source"] = path
    apply_v4_material_metadata(mat, source, None)
    apply_ufx_material_metadata(mat, source, None, cache_version)
    mat.use_nodes = True
    shader = shader_node if shader_node is not None else create_hw_shader_node(mat, cache_version)
    if shader_node is not None:
        mat.node_tree.links.new(shader.outputs["BSDF"], material_output_node(mat).inputs["Surface"])
    apply_v4_material_metadata(mat, source, shader)
    apply_ufx_material_metadata(mat, source, shader, cache_version)
    # Default texture paths belong to the game; create their image nodes when
    # source content is available, rather than loading placeholder files.
    for slot in source.slots:
        slot["path"] = ""
    mat["hwek_ugx_texture_slots"] = json.dumps(source.slots)
    for index in range(len(source.slots)):
        mat[f"hwek_ugx_texture_slot_{index}"] = ""
        mat[f"hwek_ugx_texture_slot_{index}_synced"] = ""
    arrange_ugx_material_nodes(mat, cache_version)
    if bpy.data.filepath:
        discover_material_textures(mat, os.path.dirname(bpy.data.filepath), bpy.data.filepath)
    remember_material_shader_state(mat, shader)
