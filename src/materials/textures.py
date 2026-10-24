"""Connect texture channels to native material preview inputs."""

import bpy
import hashlib
import json
from ..textures.context import texture_usage_is_non_color
from ..textures.discovery import resolve_texture_path
from ..textures.images import image_from_ddx, image_rgba_bytes, missing_texture_image
from ..textures.paths import disabled_texture_path
from .data import Material, hw_shader_socket_name, texture_usage_from_suffix
from .nodes import create_hw_shader_node
from .context import ugx_material_cache_version
from .sockets import active_ugx_material_shader
from .properties import apply_ufx_material_metadata, apply_v4_material_metadata
from .uv import ugx_texture_vector

__all__ = ('hw_shader_scalar_channel', 'replicated_texture_channel', 'link_texture_to_hw_shader', 'link_material_texture', 'repair_ugx_material_texture_links', 'attach_material_textures')

def hw_shader_scalar_channel(usage: str, cache_version: int) -> str:
    if cache_version == 6:
        return ""
    return {
        "effect": "Green",
        "emxform": "Green",
        "opacity": "Green",
        "xform": "Green",
    }.get(usage.lower(), "")


def replicated_texture_channel(mat: bpy.types.Material, tex, channel_name: str, index: int):
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    separate = nodes.new("ShaderNodeSeparateColor")
    separate.name = f"UGX {channel_name} Channel {index}"
    separate.label = channel_name
    separate.location = (tex.location.x + 240, tex.location.y)
    combine = nodes.new("ShaderNodeCombineColor")
    combine.name = f"UGX {channel_name} Mask {index}"
    combine.label = channel_name + " mask"
    combine.location = (tex.location.x + 430, tex.location.y)
    links.new(tex.outputs["Color"], separate.inputs["Color"])
    channel = separate.outputs.get(channel_name)
    if channel is None:
        return tex.outputs["Color"]
    for input_name in ("Red", "Green", "Blue"):
        links.new(channel, combine.inputs[input_name])
    if "Alpha" in combine.inputs:
        combine.inputs["Alpha"].default_value = 1.0
    return combine.outputs["Color"]


def link_texture_to_hw_shader(mat: bpy.types.Material, shader_node, usage: str, tex, cache_version: int) -> bool:
    socket_name = hw_shader_socket_name(usage, cache_version)
    if not shader_node or not socket_name or socket_name not in shader_node.inputs:
        return False
    output = tex.outputs["Color"]
    channel_name = hw_shader_scalar_channel(usage, cache_version)
    if channel_name:
        output = replicated_texture_channel(mat, tex, channel_name, len(shader_node.inputs[socket_name].links))
    mat.node_tree.links.new(output, shader_node.inputs[socket_name])
    return True


def link_material_texture(
    mat: bpy.types.Material,
    usage: str,
    image: bpy.types.Image,
    index: int,
    texture_path: str = "",
    uv_layer: str = "UV0",
    channel: int = 0,
    flags: int = 0,
    uvwvel: str = "",
    inferred: bool = False,
    shader_node=None,
    cache_version: int = 4,
):
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF") or next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.label = "UGX " + usage
    tex["hwek_ugx_usage"] = usage
    tex["hwek_ugx_slot_index"] = index
    ddx_path = image.get("hwek_ugx_ddx_path") if image else ""
    if ddx_path:
        tex["hwek_ugx_ddx_path"] = str(ddx_path)
    if texture_path:
        tex["hwek_ugx_texture_path"] = texture_path
    tex["hwek_ugx_uv_layer"] = uv_layer
    tex["hwek_ugx_channel"] = int(channel)
    tex["hwek_ugx_flags"] = int(flags)
    tex["hwek_ugx_inferred"] = bool(inferred)
    if uvwvel:
        tex["hwek_ugx_uvwvel"] = uvwvel
    tex.location = (-600, 220 - index * 240)
    vector = ugx_texture_vector(mat, uv_layer)
    if vector is not None and tex.inputs.get("Vector") is not None:
        links.new(vector, tex.inputs["Vector"])
    if link_texture_to_hw_shader(mat, shader_node, usage, tex, cache_version):
        return
    if not bsdf:
        return
    if usage in ("diffuse", "albedo") and "Base Color" in bsdf.inputs:
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    elif usage == "opacity" and "Alpha" in bsdf.inputs:
        links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
        if hasattr(mat, "blend_method"):
            mat.blend_method = "BLEND"
        if hasattr(mat, "surface_render_method"):
            mat.surface_render_method = "BLENDED"
    elif usage == "normal" and "Normal" in bsdf.inputs:
        normal_map = nodes.new("ShaderNodeNormalMap")
        normal_map.label = "UGX normal"
        normal_map.location = (-300, tex.location.y)
        links.new(tex.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
    elif usage == "gloss" and "Roughness" in bsdf.inputs:
        luma = nodes.new("ShaderNodeRGBToBW")
        luma.label = "UGX gloss"
        luma.location = (-360, tex.location.y)
        invert = nodes.new("ShaderNodeMapRange")
        invert.label = "UGX roughness"
        invert.location = (-160, tex.location.y)
        invert.inputs["From Min"].default_value = 0.0
        invert.inputs["From Max"].default_value = 1.0
        invert.inputs["To Min"].default_value = 1.0
        invert.inputs["To Max"].default_value = 0.0
        links.new(tex.outputs["Color"], luma.inputs["Color"])
        links.new(luma.outputs["Val"], invert.inputs["Value"])
        links.new(invert.outputs["Result"], bsdf.inputs["Roughness"])
    elif usage == "roughness" and "Roughness" in bsdf.inputs:
        luma = nodes.new("ShaderNodeRGBToBW")
        luma.label = "UGX roughness"
        luma.location = (-300, tex.location.y)
        links.new(tex.outputs["Color"], luma.inputs["Color"])
        links.new(luma.outputs["Val"], bsdf.inputs["Roughness"])
    elif usage in ("specular", "envmask") and "Specular IOR Level" in bsdf.inputs:
        luma = nodes.new("ShaderNodeRGBToBW")
        luma.label = "UGX " + usage
        luma.location = (-300, tex.location.y)
        links.new(tex.outputs["Color"], luma.inputs["Color"])
        links.new(luma.outputs["Val"], bsdf.inputs["Specular IOR Level"])
    elif usage == "emissive" and "Emission Color" in bsdf.inputs:
        links.new(tex.outputs["Color"], bsdf.inputs["Emission Color"])
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = max(float(bsdf.inputs["Emission Strength"].default_value), 1.0)


def repair_ugx_material_texture_links(mat: bpy.types.Material):
    version = ugx_material_cache_version(mat)
    shader = active_ugx_material_shader(mat, version)
    slots = json.loads(mat.get("hwek_ugx_texture_slots", "[]"))
    indices = {}
    for index, slot in enumerate(slots):
        name = hw_shader_socket_name(str(slot.get("usage", "")), version)
        if name:
            indices.setdefault(name, []).append(index)
    sources = {}
    for socket in shader.inputs:
        if not socket.is_linked:
            continue
        node = socket.links[0].from_node
        seen = set()
        while node.type == "REROUTE" and node.inputs[0].is_linked and node.as_pointer() not in seen:
            seen.add(node.as_pointer())
            node = node.inputs[0].links[0].from_node
        sources[socket.name] = node
    plan, skipped = {}, []
    for name, slot_indices in indices.items():
        socket = shader.inputs.get(name)
        current = sources.get(name)
        if socket is None or (current is not None and current.type != "TEX_IMAGE"):
            continue
        candidates = []
        for node in mat.node_tree.nodes:
            if node.type != "TEX_IMAGE" or node.image is None:
                continue
            usage = str(node.get("hwek_ugx_usage", ""))
            if version == 6 and usage == "gloss":
                usage = "specular"
            if hw_shader_socket_name(usage, version) == name:
                candidates.append(node)
        if current in candidates:
            chosen = current
        else:
            indexed = [node for node in candidates if int(node.get("hwek_ugx_slot_index", -1)) == slot_indices[-1]]
            choices = indexed or candidates
            if len(choices) != 1:
                if choices:
                    skipped.append(name)
                continue
            chosen = choices[0]
        slot_index = int(chosen.get("hwek_ugx_slot_index", -1))
        if slot_index not in slot_indices:
            slot_index = slot_indices[-1]
        plan[name] = (chosen, slot_index)
    for name, (node, _) in plan.items():
        socket = shader.inputs[name]
        for other in shader.inputs:
            if other == socket or sources.get(other.name) != node:
                continue
            if other.name in plan or other.name not in indices:
                for link in list(other.links):
                    mat.node_tree.links.remove(link)
    repaired = 0
    for name, (node, slot_index) in plan.items():
        socket = shader.inputs[name]
        if sources.get(name) != node:
            link_texture_to_hw_shader(mat, shader, str(slots[slot_index]["usage"]), node, version)
            repaired += 1
        node["hwek_ugx_slot_index"] = slot_index
        node["hwek_ugx_usage"] = str(slots[slot_index]["usage"])
    mat.node_tree.update_tag()
    mat.update_tag()
    return repaired, skipped


def attach_material_textures(mat: bpy.types.Material, material: Material, source_path: str, cache_version: int):
    loaded = []
    shader_node = create_hw_shader_node(mat, cache_version)
    # Source material slots define which textures the shader uses.
    for index, slot in enumerate(material.slots):
        usage = str(slot.get("usage", "")).lower() or texture_usage_from_suffix(str(slot.get("path", ""))) or "texture"
        texture_path = slot.get("path", "")
        if not usage or not texture_path:
            continue
        resolved = resolve_texture_path(source_path, texture_path)
        if not resolved:
            if disabled_texture_path(texture_path):
                mat[f"hwek_ugx_texture_{usage}_disabled"] = texture_path
                continue
            image = missing_texture_image(texture_path, usage)
            mat[f"hwek_ugx_texture_{usage}_missing"] = texture_path
        else:
            try:
                image = image_from_ddx(resolved, texture_usage_is_non_color(usage, cache_version), pack_preview=False)
            except Exception as exc:
                mat[f"hwek_ugx_texture_{usage}_load_error"] = str(exc)
                continue
        image["hwek_ugx_texture_path"] = texture_path
        if resolved:
            mat[f"hwek_ugx_texture_{usage}_file"] = resolved
            if "hwek_ugx_source_rgba_hash" not in image:
                image["hwek_ugx_source_rgba_hash"] = hashlib.sha256(image_rgba_bytes(image)).hexdigest()
        if bool(slot.get("inferred", False)):
            mat[f"hwek_ugx_texture_slot_{index}"] = texture_path
            mat[f"hwek_ugx_texture_slot_{index}_inferred"] = True
            if usage:
                usage_key = f"hwek_ugx_texture_{usage}"
                source_key = f"{usage_key}_source"
                if usage_key not in mat:
                    mat[usage_key] = texture_path
                if source_key not in mat:
                    mat[source_key] = texture_path
        loaded.append({"usage": usage, "file": resolved, "image": image.name})
        channel = max(0, min(2, int(slot.get("channel", 0))))
        link_material_texture(
            mat,
            usage,
            image,
            index,
            texture_path,
            f"UV{channel}",
            channel,
            int(slot.get("flags", 0) or 0),
            str(slot.get("uvwvel", "")),
            bool(slot.get("inferred", False)),
            shader_node,
            cache_version,
        )
    if loaded:
        mat["hwek_ugx_loaded_textures"] = json.dumps(loaded)
    apply_v4_material_metadata(mat, material, shader_node)
    apply_ufx_material_metadata(mat, material, shader_node, cache_version)
