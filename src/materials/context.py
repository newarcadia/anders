"""Resolve material ownership, source indices and active texture slots."""

from typing import Any, Dict, List, Optional, Tuple
import bpy
import json
import os
import re
from ..textures.context import ugx_texture_path_from_image
from ..ugx.data import UgxModel
from ..ugx.mesh import mesh_int_attribute
from ..ugx.scene import ugx_material_context_object
from .data import Material, editable_ugx_texture_slot, v4_material_usage
from .groups import hw_shader_group_name

__all__ = ('material_output_node', 'material_for_ugx_shader_node', 'material_node_texture_path', 'normalized_ugx_source_path', 'ufx_metadata_signature', 'blender_ufx_metadata_signature', 'parsed_ufx_metadata_signature', 'material_origin_matches_model', 'ugx_target_material_index_for_slot', 'ugx_material_index_for_slot', 'context_blender_material', 'active_ugx_texture_node', 'is_ugx_material', 'ugx_material_cache_version', 'ugx_materials_for_object', 'ensure_ugx_texture_slot_props', 'find_ugx_material', 'active_ugx_material', 'ugx_texture_node_for_slot', 'idprop_values_equal', 'material_texture_slot_current_value', 'v4_shader_node_for_material')

def material_output_node(mat: bpy.types.Material):
    nodes = mat.node_tree.nodes
    return nodes.get("Material Output") or next((node for node in nodes if node.type == "OUTPUT_MATERIAL"), None)


def material_for_ugx_shader_node(node, context=None) -> Optional[bpy.types.Material]:
    candidate = getattr(context, "material", None) if context is not None else None
    if candidate is not None and candidate.node_tree == node.id_data:
        return candidate
    return next((mat for mat in bpy.data.materials if mat.node_tree == node.id_data), None)


def material_node_texture_path(mat: bpy.types.Material, usage: str, source_path: str = "", original_texture_path: str = "", slot_index: int = -1) -> str:
    if not mat or not mat.use_nodes or not mat.node_tree:
        return ""
    usage = usage.lower()
    nodes = [
        node for node in mat.node_tree.nodes
        if node.type == "TEX_IMAGE" and str(node.get("hwek_ugx_usage", "")).lower() == usage
    ]
    if slot_index >= 0:
        indexed = next((node for node in nodes if int(node.get("hwek_ugx_slot_index", -1)) == slot_index), None)
        if indexed:
            nodes = [indexed]
        elif ugx_material_cache_version(mat) == 6:
            return ""
    for node in nodes:
        image = getattr(node, "image", None)
        node_path = node.get("hwek_ugx_texture_path")
        node_ddx = node.get("hwek_ugx_ddx_path")
        image_ddx = image.get("hwek_ugx_ddx_path") if image else ""
        if node_path and node_ddx and image_ddx and os.path.normcase(os.path.abspath(str(node_ddx))) == os.path.normcase(os.path.abspath(str(image_ddx))):
            return str(node_path)
        value = ugx_texture_path_from_image(source_path, original_texture_path, image)
        if value:
            return value
        if node_path:
            return str(node_path)
    return ""


def normalized_ugx_source_path(value: Any) -> str:
    text = str(value or "").strip()
    return os.path.normcase(os.path.abspath(text)) if text else ""


def ufx_metadata_signature(attrs: Dict[str, Any], buffers: List[Dict[str, Any]], slots: List[Dict[str, Any]], name: str) -> Tuple[Any, ...]:
    attr_values = tuple(sorted(
        (str(key), str(value.get("value", "")) if isinstance(value, dict) else str(value))
        for key, value in attrs.items()
    ))
    buffer_values = tuple(
        (str(buffer.get("name", "")), re.sub(r"\s+", "", str(buffer.get("hex", ""))).lower())
        for buffer in buffers
    )
    texture_values = tuple(str(slot.get("path", "")) for slot in slots)
    return str(name), attr_values, buffer_values, texture_values


def blender_ufx_metadata_signature(mat: bpy.types.Material) -> Tuple[Any, ...]:
    try:
        attrs = json.loads(mat.get("hwek_ugx_ufx_attrs", "{}"))
    except Exception:
        attrs = {}
    try:
        buffers = json.loads(mat.get("hwek_ugx_ufx_constant_buffers", "[]"))
    except Exception:
        buffers = []
    try:
        slots = json.loads(mat.get("hwek_ugx_texture_slots", "[]"))
    except Exception:
        slots = []
    return ufx_metadata_signature(attrs, buffers, slots, str(mat.get("hwek_ugx_source_name", "")))


def parsed_ufx_metadata_signature(material: Material) -> Tuple[Any, ...]:
    return ufx_metadata_signature(material.ufx_attrs, material.ufx_constant_buffers, material.slots, material.name)


def material_origin_matches_model(obj: bpy.types.Object, material: bpy.types.Material, model: UgxModel, material_index: int) -> bool:
    if material_index < 0 or material_index >= len(model.materials):
        return False
    material_source_path = normalized_ugx_source_path(material.get("hwek_ugx_source_path", ""))
    model_source_path = normalized_ugx_source_path(model.path)
    if material_source_path and model_source_path:
        return material_source_path == model_source_path
    return blender_ufx_metadata_signature(material) == parsed_ufx_metadata_signature(model.materials[material_index])


def ugx_target_material_index_for_slot(obj: bpy.types.Object, slot_index: int, model: Optional[UgxModel] = None) -> int:
    try:
        mapping = json.loads(obj.get("hwek_ugx_material_slot_indices", "[]"))
    except Exception:
        mapping = []
    if 0 <= slot_index < len(mapping):
        return int(mapping[slot_index])
    if model is not None:
        section_attr = mesh_int_attribute(obj.data, "UGX_Section")
        source_indices = {
            int(model.sections[section_attr[poly.index]].material_index)
            for poly in obj.data.polygons
            if int(poly.material_index) == slot_index
            and section_attr is not None
            and poly.index < len(section_attr)
            and 0 <= int(section_attr[poly.index]) < len(model.sections)
        }
        if len(source_indices) == 1:
            return next(iter(source_indices))
    return slot_index


def ugx_material_index_for_slot(obj: bpy.types.Object, slot_index: int, model: Optional[UgxModel] = None) -> int:
    if 0 <= slot_index < len(obj.material_slots):
        material = obj.material_slots[slot_index].material
        if material and material.get("hwek_ugx_native_material") and model is not None and model.cache_version == 6:
            return ugx_target_material_index_for_slot(obj, slot_index, model)
        if material and "hwek_ugx_material_index" in material:
            material_index = int(material["hwek_ugx_material_index"])
            if model is None or material_origin_matches_model(obj, material, model, material_index):
                return material_index
    return ugx_target_material_index_for_slot(obj, slot_index, model)


def context_blender_material(context) -> Optional[bpy.types.Material]:
    mat = getattr(context, "material", None)
    if mat is not None:
        return mat
    obj = getattr(context, "object", None)
    return getattr(obj, "active_material", None) if obj is not None else None


def active_ugx_texture_node(context) -> Tuple[Optional[bpy.types.Object], Optional[bpy.types.Material], Optional[Any]]:
    obj = ugx_material_context_object(getattr(context, "object", None))
    mat = active_ugx_material(obj, True)
    if obj is None or mat is None or not mat.use_nodes or not mat.node_tree:
        return None, None, None
    nodes = mat.node_tree.nodes
    active = getattr(nodes, "active", None)
    if active and active.type == "TEX_IMAGE":
        return obj, mat, active
    for node in nodes:
        if node.type == "TEX_IMAGE":
            return obj, mat, node
    return obj, mat, None


def is_ugx_material(mat: Optional[bpy.types.Material]) -> bool:
    return mat is not None and any(
        key in mat
        for key in ("hwek_ugx_cache_version", "hwek_ugx_material_index", "hwek_ugx_v4_attrs", "hwek_ugx_ufx_attrs")
    )


def ugx_material_cache_version(mat: Optional[bpy.types.Material], obj: Optional[bpy.types.Object] = None) -> int:
    if mat is None:
        return 0
    version = int(mat.get("hwek_ugx_cache_version", 0))
    if version in (4, 6):
        return version
    if "hwek_ugx_v4_attrs" in mat:
        return 4
    if "hwek_ugx_ufx_attrs" in mat:
        return 6
    if obj is not None:
        version = int(obj.get("hwek_ugx_cache_version", 0))
        if version in (4, 6):
            return version
    return 0


def ugx_materials_for_object(obj: Optional[bpy.types.Object]) -> List[bpy.types.Material]:
    if obj is None or getattr(obj, "type", "") != "MESH":
        return []
    return [
        mat for mat in obj.data.materials
        if is_ugx_material(mat)
    ]


def ensure_ugx_texture_slot_props(mat: Optional[bpy.types.Material]):
    if mat is None:
        return
    try:
        slots = json.loads(mat.get("hwek_ugx_texture_slots", "[]"))
    except Exception:
        return
    usage_sources = set()
    for slot_index, slot in enumerate(slots):
        usage = v4_material_usage(slot.get("usage", ""))
        source = str(slot.get("path", ""))
        if not editable_ugx_texture_slot(slot, 0):
            continue
        key = f"hwek_ugx_texture_slot_{slot_index}"
        if key not in mat:
            mat[key] = source
        if key + "_synced" not in mat:
            mat[key + "_synced"] = str(mat.get(key, source))
        if usage and usage not in usage_sources:
            usage_key = f"hwek_ugx_texture_{usage}"
            if usage_key not in mat:
                mat[usage_key] = str(mat.get(key, source))
            if usage_key + "_source" not in mat:
                mat[usage_key + "_source"] = source
            usage_sources.add(usage)


def find_ugx_material(obj: Optional[bpy.types.Object]) -> Optional[bpy.types.Material]:
    if obj is None or getattr(obj, "type", "") != "MESH":
        return None
    mat = obj.active_material
    if is_ugx_material(mat):
        return mat
    for candidate in obj.data.materials:
        if is_ugx_material(candidate):
            return candidate
    return mat


def active_ugx_material(obj: Optional[bpy.types.Object], ensure_active: bool = False) -> Optional[bpy.types.Material]:
    mat = find_ugx_material(obj)
    if obj is None or mat is None:
        return mat
    if ensure_active and obj.active_material != mat:
        try:
            obj.active_material_index = next(index for index, candidate in enumerate(obj.data.materials) if candidate == mat)
        except Exception:
            pass
    ensure_ugx_texture_slot_props(mat)
    return mat


def ugx_texture_node_for_slot(mat: bpy.types.Material, slot_index: int, usage: str = ""):
    if mat is None or not mat.use_nodes or not mat.node_tree:
        return None
    usage = usage.lower()
    indexed = [
        node for node in mat.node_tree.nodes
        if node.type == "TEX_IMAGE" and int(node.get("hwek_ugx_slot_index", -1)) == slot_index
        and (not usage or str(node.get("hwek_ugx_usage", "")).lower() == usage)
    ]
    if indexed:
        return indexed[0]
    if slot_index >= 0 and ugx_material_cache_version(mat) == 6:
        return None
    if usage:
        return next(
            (
                node for node in mat.node_tree.nodes
                if node.type == "TEX_IMAGE" and str(node.get("hwek_ugx_usage", "")).lower() == usage
            ),
            None,
        )
    return None


def idprop_values_equal(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= 0.00001
    except (TypeError, ValueError):
        return str(left) == str(right)


def material_texture_slot_current_value(mat: bpy.types.Material, slot_index: int, slot: Dict[str, Any]) -> str:
    usage = v4_material_usage(slot.get("usage", ""))
    source = str(slot.get("path", ""))
    key = f"hwek_ugx_texture_slot_{slot_index}"
    value = mat.get(key, source)
    synced = str(mat.get(key + "_synced", source))
    if key in mat and str(value).strip() != synced.strip():
        return str(value).strip()
    if ugx_material_cache_version(mat) == 6:
        slots = json.loads(mat.get("hwek_ugx_texture_slots", "[]"))
        if sum(str(item.get("usage", "")).lower() == str(slot.get("usage", "")).lower() for item in slots) > 1:
            return str(value).strip()
    if usage:
        legacy_key = f"hwek_ugx_texture_{usage}"
        legacy_source_key = f"{legacy_key}_source"
        legacy_value = mat.get(legacy_key, mat.get(legacy_source_key, source))
        if str(legacy_value) != str(mat.get(legacy_source_key, source)):
            value = legacy_value
    return str(value).strip()


def v4_shader_node_for_material(mat: bpy.types.Material, cache_version: int):
    if mat is None or not mat.use_nodes or not mat.node_tree:
        return None
    expected_names = {
        str(mat.get("hwek_ugx_shader_group", "")),
        hw_shader_group_name(cache_version),
    }
    for node in mat.node_tree.nodes:
        if node.type not in ("GROUP", "CUSTOM GROUP"):
            continue
        group = getattr(node, "node_tree", None)
        group_name = getattr(group, "name", "")
        if node.name in expected_names or group_name in expected_names:
            return node
    for node in mat.node_tree.nodes:
        if node.type in ("GROUP", "CUSTOM GROUP"):
            return node
    return None
