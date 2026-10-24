"""Read HW1 material slots and attributes from Blender edits."""

from typing import Any, Dict, List, Optional
import bpy
import json
from ..textures.paths import disabled_texture_path
from .bindings import ugx_material_node_bindings
from .context import material_node_texture_path, material_texture_slot_current_value
from .data import (
    V4_DEFAULT_DIFFUSE_TEXTURE,
    V4_MATERIAL_MAP_USAGES,
    texture_usage_from_suffix,
    v4_material_slot_flags,
    v4_material_usage,
)
from .properties import parse_bdt_direct_edit_value, v4_attr_property_name
from .sockets import ugx_constant_socket_value
from .uv_data import hw1_texture_coordinates

__all__ = ('v4_material_slots_from_blender', 'v4_material_names_for_mesh', 'v4_material_attrs_from_blender', 'v4_material_attrs_for_mesh', 'v4_material_slots_for_mesh')



def v4_material_slots_from_blender(mat: Optional[bpy.types.Material]) -> List[Dict[str, Any]]:
    if mat is None:
        return []
    slots_by_usage: Dict[str, Dict[str, Any]] = {}

    if mat.get("hwek_ugx_native_material"):
        for binding in ugx_material_node_bindings(mat, int(mat.get('hwek_ugx_cache_version',4))):
            usage = v4_material_usage(binding["slot"].get("usage", ""))
            if usage not in V4_MATERIAL_MAP_USAGES:
                continue
            slot = binding["slot"]
            export_slot = {
                "usage": usage,
                "path": binding["path"],
                "channel": binding["channel"],
                "flags": v4_material_slot_flags(usage, slot),
            }
            for field in ('channel_info', 'flags_info'):
                if field in slot:
                    export_slot[field] = slot[field]
            coordinates = hw1_texture_coordinates(binding["node"])
            if coordinates is not None:
                export_slot["uv_scroll"] = [ugx_constant_socket_value(coordinates.inputs[name]) for name in ("Speed U", "Speed V", "Speed W")]
                export_slot["flags"] = int(ugx_constant_socket_value(coordinates.inputs["Map Flags"]))
            else:
                for name in ("uvwvel", "uv_scroll"):
                    if name in slot:
                        export_slot[name] = slot[name]
            slots_by_usage[usage] = export_slot
        return [slots_by_usage[usage] for usage in V4_MATERIAL_MAP_USAGES if usage in slots_by_usage]

    try:
        slots = json.loads(mat.get("hwek_ugx_texture_slots", "[]"))
    except Exception:
        slots = []
    for index, slot in enumerate(slots):
        usage = v4_material_usage(slot.get("usage", "") or texture_usage_from_suffix(str(slot.get("path", ""))))
        if usage not in V4_MATERIAL_MAP_USAGES:
            continue
        source_path = str(slot.get("path", "")).strip()
        slot_key = f"hwek_ugx_texture_slot_{index}"
        value = material_texture_slot_current_value(mat, index, slot)
        field_changed = slot_key in mat and str(mat.get(slot_key, source_path)).strip() != str(mat.get(slot_key + "_synced", source_path)).strip()
        node_value = material_node_texture_path(mat, usage, "", source_path, index)
        if not field_changed and node_value:
            value = node_value
        if value and not disabled_texture_path(value):
            export_slot = {
                "usage": usage,
                "path": value,
                "channel": max(0, min(2, int(slot.get("channel", 0) or 0))),
                "flags": v4_material_slot_flags(usage, slot),
            }
            for field in ('channel_info', 'flags_info'):
                if field in slot:
                    export_slot[field] = slot[field]
            if "uvwvel" in slot:
                export_slot["uvwvel"] = slot.get("uvwvel", "")
            if "uv_scroll" in slot:
                export_slot["uv_scroll"] = slot.get("uv_scroll", [])
            slots_by_usage[usage] = export_slot

    for usage in V4_MATERIAL_MAP_USAGES:
        prop_names = [usage]
        if usage == "diffuse":
            prop_names.append("albedo")
        elif usage == "gloss":
            prop_names.append("specular")
        elif usage == "env":
            prop_names.append("environment")
        elif usage == "emxform":
            prop_names.append("effect")
        for prop_usage in prop_names:
            value = str(mat.get(f"hwek_ugx_texture_{prop_usage}", "")).strip()
            if not value or disabled_texture_path(value):
                continue
            slots_by_usage.setdefault(usage, {
                "usage": usage,
                "path": value,
                "channel": 0,
                "flags": v4_material_slot_flags(usage, {}),
            })
            break

    return [slots_by_usage[usage] for usage in V4_MATERIAL_MAP_USAGES if usage in slots_by_usage]


def v4_material_names_for_mesh(mesh: bpy.types.Mesh, material_count: int) -> List[str]:
    names: List[str] = []
    for index in range(material_count):
        mat = mesh.materials[index] if index < len(mesh.materials) else None
        if mat is None:
            names.append(f"material_{index}")
            continue
        value = mat.get("hwek_ugx_export_name", mat.get("hwek_ugx_source_name", mat.name))
        if value == mat.get("hwek_ugx_source_name", "") and mat.name != mat.get("hwek_ugx_blender_name", mat.name):
            value = mat.name
        names.append(str(value or f"material_{index}"))
    return names


def v4_material_attrs_from_blender(mat: Optional[bpy.types.Material]) -> Dict[str, Dict[str, Any]]:
    if mat is None:
        return {}
    try:
        attrs = json.loads(mat.get("hwek_ugx_v4_attrs", "{}"))
    except Exception:
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for name, info in attrs.items():
        if not isinstance(info, dict):
            continue
        item = dict(info)
        prop = v4_attr_property_name(name)
        if prop in mat:
            value = mat.get(prop)
            item["value"] = str(value)
            if str(item.get("kind", "")) != "string" and bool(item.get("direct", False)):
                item["raw_value"] = parse_bdt_direct_edit_value(item, value)
        result[str(name)] = item
    return result


def v4_material_attrs_for_mesh(mesh: bpy.types.Mesh, material_count: int) -> List[Dict[str, Dict[str, Any]]]:
    result: List[Dict[str, Dict[str, Any]]] = []
    for index in range(material_count):
        mat = mesh.materials[index] if index < len(mesh.materials) else None
        result.append(v4_material_attrs_from_blender(mat))
    return result


def v4_material_slots_for_mesh(mesh: bpy.types.Mesh, material_count: int) -> List[List[Dict[str, Any]]]:
    result: List[List[Dict[str, Any]]] = []
    for index in range(material_count):
        mat = mesh.materials[index] if index < len(mesh.materials) else None
        slots = v4_material_slots_from_blender(mat)
        if not slots:
            slots = [{
                "usage": "diffuse",
                "path": V4_DEFAULT_DIFFUSE_TEXTURE,
                "channel": 0,
                "flags": 7,
            }]
        result.append(slots)
    return result
