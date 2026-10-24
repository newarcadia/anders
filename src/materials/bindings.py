"""Resolve and validate texture bindings on native material nodes."""

from typing import Any, Dict, List
import bpy
import json
import os
import re
from ..errors import UgxError
from ..textures.context import replacement_texture_path_for_image, ugx_texture_resource_key
from ..textures.paths import disabled_texture_path
from .context import ugx_material_cache_version, ugx_texture_node_for_slot
from .data import HW2_STATIC_MATERIAL_DEFAULT, V4_MATERIAL_MAP_USAGES, hw_shader_socket_name
from .properties import ufx_buffer_float_values_from_hex
from .mapping import hw1_texture_mapping
from .sockets import active_ugx_material_shader, ugx_uv_map_name, ugx_uv_source
from .uv_data import material_uv_animation

__all__ = ('ugx_image_node_from_socket', 'validate_ugx_texture_image_node', 'ugx_material_node_bindings')

def ugx_image_node_from_socket(socket, mat: bpy.types.Material, allow_missing: bool = False):
    if not socket.is_linked:
        return None
    link = socket.links[0]
    seen = set()
    while link.from_node.type == "REROUTE":
        node = link.from_node
        if node.as_pointer() in seen or not node.inputs[0].is_linked:
            raise UgxError(f"Material {mat.name}: texture reroute is incomplete.")
        seen.add(node.as_pointer())
        link = node.inputs[0].links[0]
    node = link.from_node
    if socket.name == 'Player_color (xf)' and node.get('hwek_hw1_diffuse_alpha_fallback'):
        return None
    if socket.name == "Player_color (xf)" and node.bl_idname == "ShaderNodeAndersHW1PlayerColorMask" and link.from_socket.name == "Mask":
        source = node.inputs["Color"]
        if not source.is_linked:
            raise UgxError(f"Material {mat.name}: connect an Image Texture to {node.name} before export.")
        return ugx_image_node_from_socket(source, mat, allow_missing)
    # HWDE imports replicate the team-mask channel through these two nodes.
    if socket.name in ("Player_color (xf)", "Emmissive_player_color (ef)", "Alpha (op)") and node.type == "COMBINE_COLOR" and node.label.endswith(" mask"):
        inputs = [node.inputs.get(name) for name in ("Red", "Green", "Blue")]
        if all(item and item.is_linked for item in inputs):
            source = inputs[0].links[0].from_socket
            if all(item.links[0].from_socket == source for item in inputs) and source.node.type == "SEPARATE_COLOR":
                return ugx_image_node_from_socket(source.node.inputs["Color"], mat, allow_missing)
    if node.type != "TEX_IMAGE" or link.from_socket.name not in ("Color", "Alpha"):
        raise UgxError(f"Material {mat.name}: {socket.name} needs an Image Texture, optionally through reroutes. Bake {node.name} first.")
    return validate_ugx_texture_image_node(node, mat, allow_missing)


def validate_ugx_texture_image_node(node, mat: bpy.types.Material, allow_missing: bool = False):
    if node.image is None:
        raise UgxError(f"Material {mat.name}: Image Texture {node.name} has no image.")
    if node.image.get("hwek_ugx_missing_texture") and not allow_missing:
        raise UgxError(f"Material {mat.name}: replace the missing image on {node.name}.")
    if node.projection != 'FLAT':
        raise UgxError(f"Material {mat.name}: bake {node.projection} projection on {node.name} before export.")
    if ugx_material_cache_version(mat) == 4:
        hw1_texture_mapping(node, mat)
        return node
    vector = node.inputs["Vector"]
    if vector.is_linked and ugx_uv_source(vector.links[0].from_node).type != "UVMAP" and ugx_uv_source(vector.links[0].from_node).bl_idname not in ("ShaderNodeAndersUVAnimation", "ShaderNodeAndersHW1UV"):
        raise UgxError(f"Material {mat.name}: bake the UV mapping on {node.name} before export.")
    return node


def ugx_material_node_bindings(mat: bpy.types.Material, cache_version: int, source_path: str = "", allow_missing: bool = False) -> List[Dict[str, Any]]:
    shader = active_ugx_material_shader(mat, cache_version)
    slots = json.loads(mat.get("hwek_ugx_texture_slots", "[]"))
    authored_hw2 = cache_version == 6 and not mat.get("hwek_ugx_source_path")
    default_paths = {ugx_texture_resource_key(path) for path in HW2_STATIC_MATERIAL_DEFAULT["textures"]}
    bindings = []
    visited = set()
    socket_indices = {}
    for index, slot in enumerate(slots):
        name = hw_shader_socket_name(str(slot.get("usage", "")).lower(), cache_version)
        if name:
            socket_indices.setdefault(name, []).append(index)
    preview_indices = {}
    for name, indices in socket_indices.items():
        if cache_version != 6 or len(indices) < 2:
            continue
        socket = shader.inputs.get(name)
        node = ugx_image_node_from_socket(socket, mat, True) if socket is not None else None
        tagged_index = int(node.get("hwek_ugx_slot_index", -1)) if node is not None else -1
        preview_indices[name] = tagged_index if tagged_index in indices else indices[-1]
    for index, slot in enumerate(slots):
        usage = str(slot.get("usage", "")).lower()
        socket_name = hw_shader_socket_name(usage, cache_version)
        socket = shader.inputs.get(socket_name) if socket_name else None
        original = str(slot.get("path", ""))
        if socket is None or (socket_name in preview_indices and index != preview_indices[socket_name]):
            # Preserve auxiliary textures used by shader variants without a preview socket.
            node = ugx_texture_node_for_slot(mat, index, usage)
            if node is None or node.image is None:
                continue
            validate_ugx_texture_image_node(node, mat, True)
        else:
            visited.add(socket_name)
            node = ugx_image_node_from_socket(socket, mat, True)
            if node is None:
                if original and not disabled_texture_path(original) and not authored_hw2:
                    raise UgxError(f"Material {mat.name}: reconnect {socket_name}, or replace its texture before export.")
                continue
        image = node.image
        preserve_missing_reference = (
            bool(mat.get("hwek_ugx_source_path")) and not disabled_texture_path(original)
            and ugx_texture_resource_key(str(image.get("hwek_ugx_texture_path", ""))) == ugx_texture_resource_key(original)
        )
        validate_ugx_texture_image_node(node, mat, allow_missing or preserve_missing_reference)
        path = replacement_texture_path_for_image(source_path, original, image)
        if authored_hw2 and ugx_texture_resource_key(original) in default_paths and ugx_texture_resource_key(path) in default_paths:
            # Older authored materials copied default resource paths onto user images.
            filename = image.filepath_raw or image.filepath or image.name
            stem = os.path.splitext(os.path.basename(filename.replace("\\", "/")))[0]
            if stem and ugx_texture_resource_key(stem) != ugx_texture_resource_key(path).rsplit("\\", 1)[-1]:
                path = stem
        if not path:
            path = original if original and not disabled_texture_path(original) else re.sub(r"[^\w.-]", "_", image.name)
            path = os.path.splitext(path)[0]
        channel = int(slot.get("channel", 0))
        if node.inputs["Vector"].is_linked:
            uv_name = hw1_texture_mapping(node, mat)[0] if cache_version == 4 else ugx_uv_map_name(node.inputs["Vector"].links[0].from_node, mat)
            if uv_name not in ("UV0", "UV1", "UV2", ""):
                raise UgxError(f"Material {mat.name}: {node.name} uses unsupported UV layer {uv_name}. Use UV0, UV1 or UV2.")
            channel = int(uv_name[-1]) if uv_name else 0
        bindings.append({"index": index, "slot": slot, "node": node, "image": image, "path": path, "channel": channel})
    usages = ("diffuse", "normal", "packedmask", "specular", "emissive") if cache_version == 6 else V4_MATERIAL_MAP_USAGES
    for usage in usages:
        socket_name = hw_shader_socket_name(usage, cache_version)
        if not socket_name or socket_name in visited:
            continue
        visited.add(socket_name)
        socket = shader.inputs.get(socket_name)
        if cache_version == 6:
            # Preview inputs outside the native shader array are unused by export.
            continue
        node = ugx_image_node_from_socket(socket, mat, allow_missing) if socket is not None else None
        if node is None:
            continue
        path = replacement_texture_path_for_image(source_path, "", node.image) or os.path.splitext(node.image.name)[0]
        bindings.append({"index": len(slots), "slot": {"usage": usage, "path": "", "channel": 0}, "node": node, "image": node.image, "path": path, "channel": 0})
        slots.append(bindings[-1]["slot"])
    animation = material_uv_animation(mat, bindings)
    if animation is not None:
        attrs = json.loads(mat.get("hwek_ugx_ufx_attrs", "{}"))
        standard = str(attrs.get("shaderPermutationName0", {}).get("value", "")).upper().startswith("HOGAN_STANDARD_")
        buffers = json.loads(mat.get("hwek_ugx_ufx_constant_buffers", "[]"))
        has_uv_buffer = any(buffer.get("name") == "VSCBData" and len(ufx_buffer_float_values_from_hex(str(buffer.get("hex", "")))) >= 4 for buffer in buffers)
        if cache_version != 6 or not standard or not has_uv_buffer:
            raise UgxError(f"Material {mat.name}: UV Animation requires a standard HW2 material with UV parameters.")
    return bindings
