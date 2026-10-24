"""Synchronize texture slots, source paths and preview connections."""

import bpy
import json
from ..errors import UgxError
from ..textures.context import (
    texture_usage_is_non_color,
    ensure_ugx_texture_context_from_image,
    replacement_texture_path_for_image,
    ugx_texture_context_path,
)
from ..textures.discovery import resolve_texture_path
from ..textures.images import image_from_ddx, set_ugx_image_source
from ..textures.paths import disabled_texture_path, texture_path_hint_for_slot
from ..ugx.scene import ugx_used_material_slots
from .context import (
    idprop_values_equal,
    material_output_node,
    material_texture_slot_current_value,
    ugx_material_cache_version,
    ugx_texture_node_for_slot,
    v4_shader_node_for_material,
)
from .data import V4_SOURCE_MATERIAL_ATTRS, texture_usage_from_suffix, v4_material_usage
from .nodes import create_hw_shader_node
from .sockets import active_ugx_material_shader
from .properties import (
    set_shader_input_default,
    set_shader_vector_default,
    ufx_idprop_value,
    v4_attr_property_name,
)
from .source import find_v4_source_material_for_texture
from .textures import link_material_texture

__all__ = ('assign_texture_path_to_ugx_texture_slot', 'material_texture_node_current_path', 'sync_v4_material_shader_inputs', 'clear_v4_texture_auto_context', 'sync_ugx_material_texture_slots_from_props', 'sync_ugx_materials_from_texture_props', 'apply_v4_source_material_context', 'apply_v4_source_material_context_for_diffuse_edits', 'assign_image_to_ugx_texture_node', 'refresh_v4_texture_preview', 'assign_image_to_ugx_texture_slot')

def assign_texture_path_to_ugx_texture_slot(obj: bpy.types.Object, mat: bpy.types.Material, slot_index: int, texture_path: str) -> str:
    slots = json.loads(mat.get("hwek_ugx_texture_slots", "[]"))
    if slot_index < 0 or slot_index >= len(slots):
        return ""
    slot = slots[slot_index]
    usage = v4_material_usage(slot.get("usage", ""))
    key = f"hwek_ugx_texture_slot_{slot_index}"
    mat[key] = texture_path
    mat[key + "_synced"] = texture_path
    if usage:
        mat[f"hwek_ugx_texture_{usage}"] = texture_path
    resolved = resolve_texture_path(ugx_texture_context_path(obj), texture_path)
    if not resolved:
        return texture_path
    try:
        image = image_from_ddx(resolved, texture_usage_is_non_color(usage, ugx_material_cache_version(mat)))
        set_ugx_image_source(image, resolved)
        return assign_image_to_ugx_texture_slot(obj, mat, slot_index, image, texture_path, apply_companions=False)
    except Exception:
        node = ugx_texture_node_for_slot(mat, slot_index, usage)
        if node is not None:
            node["hwek_ugx_texture_path"] = texture_path
        return texture_path


def material_texture_node_current_path(mat: bpy.types.Material, slot_index: int, usage: str) -> str:
    node = ugx_texture_node_for_slot(mat, slot_index, usage)
    if node is None:
        return ""
    value = str(node.get("hwek_ugx_texture_path", "")).strip()
    if value:
        return value
    image = getattr(node, "image", None)
    if image is not None:
        value = str(image.get("hwek_ugx_texture_path", "")).strip()
        if value:
            return value
    return ""


def sync_v4_material_shader_inputs(mat: bpy.types.Material, cache_version: int):
    if cache_version != 4:
        return
    try:
        attrs = json.loads(mat.get("hwek_ugx_v4_attrs", "{}"))
    except Exception:
        attrs = {}
    if not attrs:
        return
    shader_node = v4_shader_node_for_material(mat, cache_version)
    if shader_node is None:
        return
    for name, socket_name in {
        "SpecPower": "SpecPower",
        "EnvReflectivity": "Reflectivity",
        "EnvSharpness": "Sharpness",
        "EnvFresnelPower": "FresnelPower",
        "EnvFresnel": "Fresnel",
        "AccessoryIndex": "AccIndex",
        "BlendType": "BlendType",
        "Opacity": "Opacity",
        "Flags": "Flags",
    }.items():
        if name not in attrs:
            continue
        prop = v4_attr_property_name(name)
        if prop in mat:
            set_shader_input_default(shader_node, socket_name, mat.get(prop))
    spec = []
    for name in ("SpecColorR", "SpecColorG", "SpecColorB"):
        if name not in attrs:
            break
        prop = v4_attr_property_name(name)
        if prop not in mat:
            break
        try:
            spec.append(float(mat.get(prop)))
        except (TypeError, ValueError):
            break
    if len(spec) == 3:
        set_shader_vector_default(shader_node, "SpecColor", (spec[0], spec[1], spec[2], 1.0))


def clear_v4_texture_auto_context(mat: bpy.types.Material, slot_index: int):
    mat[f"hwek_ugx_texture_slot_{slot_index}_auto_context"] = False


def sync_ugx_material_texture_slots_from_props(obj: bpy.types.Object, mat: bpy.types.Material):
    try:
        slots = json.loads(mat.get("hwek_ugx_texture_slots", "[]"))
    except Exception:
        slots = []
    source_path = ugx_texture_context_path(obj)
    for slot_index, slot in enumerate(slots):
        key = f"hwek_ugx_texture_slot_{slot_index}"
        if key not in mat:
            continue
        usage = v4_material_usage(slot.get("usage", ""))
        target = material_texture_slot_current_value(mat, slot_index, slot)
        synced_key = key + "_synced"
        synced = str(mat.get(synced_key, str(slot.get("path", "")))).strip()
        if target == synced:
            continue
        clear_v4_texture_auto_context(mat, slot_index)
        mat[key] = target
        if usage:
            mat[f"hwek_ugx_texture_{usage}"] = target
        resolved = resolve_texture_path(source_path, target)
        if resolved:
            image = image_from_ddx(resolved, texture_usage_is_non_color(usage, ugx_material_cache_version(mat)))
            set_ugx_image_source(image, resolved)
            assign_image_to_ugx_texture_slot(obj, mat, slot_index, image, target, apply_companions=False)
        else:
            node = ugx_texture_node_for_slot(mat, slot_index, usage)
            if node is not None:
                node["hwek_ugx_texture_path"] = target
                image = getattr(node, "image", None)
                if image is not None:
                    image["hwek_ugx_texture_path"] = target
            mat[synced_key] = target
        if usage == "diffuse":
            apply_v4_source_material_context(obj, mat, "diffuse", target)
        mat[synced_key] = target
    sync_v4_material_shader_inputs(mat, ugx_material_cache_version(mat, obj))


def sync_ugx_materials_from_texture_props(obj: bpy.types.Object):
    if obj is None or getattr(obj, "type", "") != "MESH":
        return
    for _, mat in ugx_used_material_slots(obj):
        if mat is not None:
            sync_ugx_material_texture_slots_from_props(obj, mat)


def apply_v4_source_material_context(obj: bpy.types.Object, mat: bpy.types.Material, assigned_usage: str, texture_path: str):
    if ugx_material_cache_version(mat, obj) != 4 or v4_material_usage(assigned_usage) != "diffuse":
        return
    source_material = find_v4_source_material_for_texture(ugx_texture_context_path(obj), texture_path, "diffuse")
    if source_material is None:
        return
    try:
        slots = json.loads(mat.get("hwek_ugx_texture_slots", "[]"))
    except Exception:
        slots = []
    source_slots = {
        v4_material_usage(slot.get("usage", "")): slot
        for slot in source_material.slots
        if slot.get("path")
    }
    for slot_index, slot in enumerate(slots):
        usage = v4_material_usage(slot.get("usage", ""))
        if usage in ("", "diffuse"):
            continue
        source_slot = source_slots.get(usage)
        source_path = str(source_slot.get("path", "")) if source_slot else ""
        if not source_path or disabled_texture_path(source_path):
            continue
        current = material_texture_slot_current_value(mat, slot_index, slot)
        original = str(slot.get("path", ""))
        auto_key = f"hwek_ugx_texture_slot_{slot_index}_auto_context"
        auto_value_key = auto_key + "_value"
        auto_context = bool(mat.get(auto_key, False))
        auto_value = str(mat.get(auto_value_key, ""))
        if current and current != original and (not auto_context or current != auto_value):
            continue
        assign_texture_path_to_ugx_texture_slot(obj, mat, slot_index, source_path)
        mat[auto_key] = True
        mat[auto_value_key] = source_path

    try:
        current_attrs = json.loads(mat.get("hwek_ugx_v4_attrs", "{}"))
    except Exception:
        current_attrs = {}
    for name in V4_SOURCE_MATERIAL_ATTRS:
        source_info = source_material.v4_attrs.get(name)
        current_info = current_attrs.get(name)
        if not source_info or not current_info:
            continue
        prop = v4_attr_property_name(name)
        if prop not in mat:
            continue
        source_prop = prop + "_source"
        auto_prop = prop + "_auto_context"
        auto_value_prop = auto_prop + "_value"
        auto_context = bool(mat.get(auto_prop, False))
        auto_value = mat.get(auto_value_prop)
        if (
            source_prop in mat
            and not idprop_values_equal(mat.get(prop), mat.get(source_prop))
            and (not auto_context or not idprop_values_equal(mat.get(prop), auto_value))
        ):
            continue
        value = ufx_idprop_value(source_info)
        mat[prop] = value
        mat[auto_prop] = True
        mat[auto_value_prop] = value
    mat["hwek_ugx_v4_source_material_match"] = source_material.name
    sync_v4_material_shader_inputs(mat, ugx_material_cache_version(mat, obj))


def apply_v4_source_material_context_for_diffuse_edits(obj: bpy.types.Object):
    materials = [mat for _, mat in ugx_used_material_slots(obj) if mat is not None]
    if not any(ugx_material_cache_version(mat, obj) == 4 for mat in materials):
        return
    for mat in materials:
        if mat is None:
            continue
        try:
            slots = json.loads(mat.get("hwek_ugx_texture_slots", "[]"))
        except Exception:
            continue
        for slot_index, slot in enumerate(slots):
            if v4_material_usage(slot.get("usage", "")) != "diffuse":
                continue
            value = material_texture_slot_current_value(mat, slot_index, slot)
            if value and value != str(slot.get("path", "")):
                apply_v4_source_material_context(obj, mat, "diffuse", value)


def assign_image_to_ugx_texture_node(obj: bpy.types.Object, mat: bpy.types.Material, node, image: bpy.types.Image, texture_path: str = "") -> str:
    usage = str(node.get("hwek_ugx_usage", "")).lower()
    slot_index = int(node.get("hwek_ugx_slot_index", -1))
    slots = json.loads(mat.get("hwek_ugx_texture_slots", "[]"))
    original_texture_path = ""
    if 0 <= slot_index < len(slots):
        original_texture_path = str(slots[slot_index].get("path", ""))
    source_file = str(image.get("hwek_ugx_ddx_path", ""))
    if source_file:
        set_ugx_image_source(image, source_file)
    ensure_ugx_texture_context_from_image(obj, image)
    target_path = texture_path.strip() or replacement_texture_path_for_image(ugx_texture_context_path(obj), original_texture_path, image)
    node.image = image
    node["hwek_ugx_ddx_path"] = str(image.get("hwek_ugx_ddx_path", ""))
    if target_path:
        image["hwek_ugx_texture_path"] = target_path
        node["hwek_ugx_texture_path"] = target_path
        if slot_index >= 0:
            mat[f"hwek_ugx_texture_slot_{slot_index}"] = target_path
            mat[f"hwek_ugx_texture_slot_{slot_index}_synced"] = target_path
            clear_v4_texture_auto_context(mat, slot_index)
        if usage:
            mat[f"hwek_ugx_texture_{usage}"] = target_path
            mat[f"hwek_ugx_texture_{usage}_file"] = str(image.get("hwek_ugx_ddx_path", ""))
    return target_path


def refresh_v4_texture_preview(obj: bpy.types.Object, mat: bpy.types.Material):
    if ugx_material_cache_version(mat, obj) != 4 or not mat.use_nodes or not mat.node_tree:
        return
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output = material_output_node(mat)
    bsdf = nodes.get("Principled BSDF") or next((item for item in nodes if item.type == "BSDF_PRINCIPLED"), None)
    diffuse = next(
        (
            item for item in nodes
            if item.type == "TEX_IMAGE"
            and item.image is not None
            and v4_material_usage(str(item.get("hwek_ugx_usage", ""))) == "diffuse"
        ),
        None,
    )
    if output is None or bsdf is None or diffuse is None or "Surface" not in output.inputs or "Base Color" not in bsdf.inputs:
        return
    for link in list(output.inputs["Surface"].links):
        links.remove(link)
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    for link in list(bsdf.inputs["Base Color"].links):
        links.remove(link)
    links.new(diffuse.outputs["Color"], bsdf.inputs["Base Color"])
    for item in nodes:
        if item.type == "TEX_IMAGE" and item.image is not None:
            item.image.update()
    mat.diffuse_color = (1.0, 1.0, 1.0, 1.0)
    mat.node_tree.update_tag()
    mat.update_tag()
    obj.data.update()


def assign_image_to_ugx_texture_slot(
    obj: bpy.types.Object,
    mat: bpy.types.Material,
    slot_index: int,
    image: bpy.types.Image,
    texture_path: str = "",
    apply_companions: bool = True,
) -> str:
    slots = json.loads(mat.get("hwek_ugx_texture_slots", "[]"))
    if slot_index < 0 or slot_index >= len(slots):
        raise UgxError("UGX texture slot is no longer available on the active material.")
    slot = slots[slot_index]
    usage = str(slot.get("usage", "")).lower()
    node = ugx_texture_node_for_slot(mat, slot_index, usage)
    original_texture_path = texture_path_hint_for_slot(slot)
    ensure_ugx_texture_context_from_image(obj, image)
    target_path = texture_path.strip() or replacement_texture_path_for_image(ugx_texture_context_path(obj), original_texture_path, image)
    if node is None:
        if not target_path:
            raise UgxError(f"Could not derive a UGX texture path for material {mat.name} slot {slot_index}.")
        version = ugx_material_cache_version(mat, obj) or 4
        try:
            shader = active_ugx_material_shader(mat, version)
        except UgxError:
            shader = create_hw_shader_node(mat, version)
        link_material_texture(
            mat,
            usage or texture_usage_from_suffix(target_path) or "texture",
            image,
            slot_index,
            target_path,
            f"UV{max(0, min(2, int(slot.get('channel', 0) or 0)))}",
            max(0, min(2, int(slot.get("channel", 0) or 0))),
            int(slot.get("flags", 0) or 0),
            str(slot.get("uvwvel", "")),
            False,
            shader,
            version,
        )
        node = ugx_texture_node_for_slot(mat, slot_index, usage)
        if node is None:
            node = ugx_texture_node_for_slot(mat, slot_index, "")
    if node is None:
        raise UgxError(f"Material {mat.name} has no editable texture node for slot {slot_index}.")
    node["hwek_ugx_slot_index"] = slot_index
    if usage:
        node["hwek_ugx_usage"] = usage
    assigned = assign_image_to_ugx_texture_node(obj, mat, node, image, target_path)
    if apply_companions and assigned:
        apply_v4_source_material_context(obj, mat, usage, assigned)
    refresh_v4_texture_preview(obj, mat)
    return assigned
