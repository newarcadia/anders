"""Retain source snapshots and expose editable native model metadata."""

from typing import Any, Dict, List, Optional, Tuple
import bpy
import json
import os
import re
import tempfile
from ..errors import UgxError
from ..formats.binary import sha256_bytes
from .data import Bone, Section, UgxModel, ugx_game_name
from .naming import sibling_damage_xmb

__all__ = ('UGX_SOURCE_SNAPSHOT_DIR', 'build_section_entry', 'apply_common_import_metadata', 'ugx_source_bytes_for_export', 'section_property_key', 'section_entry_source_index', 'SECTION_METADATA_KEYS', 'editable_section_metadata_keys', 'set_section_metadata_properties', 'section_metadata_value', 'cache_ugx_source_snapshot', 'set_ugx_source_metadata', 'bone_bound_property_name', 'set_bone_bound_properties', 'bone_bound_property_values', 'bone_bounds_are_empty', 'lod_section_indices')

UGX_SOURCE_SNAPSHOT_DIR = os.path.join(tempfile.gettempdir(), "hwek_ugx_source_snapshots")


def build_section_entry(section: Section, source_section_index: int, vertex_start: int, vertex_count: int, face_start: int, face_count: int) -> Dict[str, Any]:
    return {
        "source_section_index": source_section_index,
        "vertex_start": vertex_start,
        "vertex_count": vertex_count,
        "vertex_file_offset": section.vertex_file_offset,
        "vertex_stride": section.vertex_stride,
        "face_start": face_start,
        "face_count": face_count,
        "material_index": section.material_index,
        "material_index_offset": section.material_index_offset,
        "accessory_index": section.accessory_index,
        "accessory_index_offset": section.accessory_index_offset,
        "accessory_first_bone": section.accessory_first_bone,
        "accessory_bone_count": section.accessory_bone_count,
        "max_bones": section.max_bones,
        "max_bones_offset": section.max_bones_offset,
        "rigid_bone_index": section.rigid_bone_index,
        "rigid_bone_index_offset": section.rigid_bone_index_offset,
        "rigid_only": int(bool(section.rigid_only)),
        "rigid_only_offset": section.rigid_only_offset,
        "lod_min_distance": section.lod_min_distance,
        "lod_min_distance_offset": section.lod_min_distance_offset,
        "lod_max_distance": section.lod_max_distance,
        "lod_max_distance_offset": section.lod_max_distance_offset,
        "sorting_bias": section.sorting_bias,
        "sorting_bias_offset": section.sorting_bias_offset,
        "index_offset_field_offset": section.index_offset_field_offset,
        "tri_count_field_offset": section.tri_count_field_offset,
        "vertex_offset_field_offset": section.vertex_offset_field_offset,
        "vertex_size_field_offset": section.vertex_size_field_offset,
        "vertex_count_field_offset": section.vertex_count_field_offset,
        "big_endian": section.big_endian,
        "pack_order": section.pack_order,
        "decl_order": section.decl_order,
        "layout": [
            {
                "semantic": item.semantic,
                "type": item.type_id,
                "offset": item.offset,
                "size": item.size,
            }
            for item in section.layout
        ],
    }


def apply_common_import_metadata(obj: bpy.types.Object, model: UgxModel, import_id: str, total_lods: int):
    set_ugx_source_metadata(obj, model.path, model.original_bytes)
    obj["hwek_ugx_cache_version"] = model.cache_version
    obj["hwek_ugx_game"] = ugx_game_name(model.cache_version)
    obj["hwek_ugx_import_id"] = import_id
    obj["hwek_ugx_lod_object_count"] = total_lods
    obj["hwek_ugx_section_count"] = len(model.sections)
    obj["hwek_ugx_bone_count"] = len(model.bones)
    obj["hwek_ugx_accessory_count"] = len(model.accessories)
    obj["hwek_ugx_material_count"] = len(model.materials)
    if model.cache_version == 6:
        damage_path = sibling_damage_xmb(model.path, model.cache_version)
        if damage_path:
            obj["hwek_ugx_damage_path"] = damage_path
    obj["hwek_ugx_bone_patches"] = json.dumps([
        {"name": bone.name, "offset": bone.name_offset, "capacity": bone.name_capacity}
        for bone in model.bones if bone.name_capacity
    ])
    obj["hwek_ugx_bone_bounds"] = json.dumps([
        {
            "index": index,
            "name": bone.name,
            "low": list(bone.bounds_low),
            "high": list(bone.bounds_high),
            "low_offset": bone.bounds_low_offset,
            "high_offset": bone.bounds_high_offset,
            "big_endian": bone.big_endian,
        }
        for index, bone in enumerate(model.bones) if bone.bounds_low_offset or bone.bounds_high_offset
    ])


def ugx_source_bytes_for_export(obj: bpy.types.Object) -> Tuple[bytes, str]:
    source_path = str(obj.get("hwek_ugx_source_path", ""))
    expected_hash = str(obj.get("hwek_ugx_source_sha256", ""))
    source_bytes = b""
    source_hash = ""
    if source_path:
        try:
            with open(source_path, "rb") as handle:
                source_bytes = handle.read()
            source_hash = sha256_bytes(source_bytes)
            if not expected_hash or source_hash == expected_hash:
                return source_bytes, "source"
        except OSError:
            source_bytes = b""

    snapshot_path = str(obj.get("hwek_ugx_source_cache_path", ""))
    if snapshot_path:
        try:
            with open(snapshot_path, "rb") as handle:
                snapshot_bytes = handle.read()
            snapshot_hash = sha256_bytes(snapshot_bytes)
            if not expected_hash or snapshot_hash == expected_hash:
                return snapshot_bytes, "snapshot"
        except OSError:
            pass

    if source_bytes:
        return source_bytes, "rebased_source"
    raise UgxError("Source UGX changed since import and no import snapshot is available. Reimport the UGX, then export again.")


def section_property_key(index: int, key: str) -> str:
    return f"hwek_ugx_section_{index}_{key}"


def section_entry_source_index(display_index: int, entry: Dict[str, Any]) -> int:
    return int(entry.get("source_section_index", display_index))


SECTION_METADATA_KEYS = (
    "accessory_index",
    "max_bones",
    "rigid_bone_index",
    "rigid_only",
    "lod_min_distance",
    "lod_max_distance",
    "sorting_bias",
)


def editable_section_metadata_keys(section: Dict[str, Any]) -> List[str]:
    keys = []
    for key in SECTION_METADATA_KEYS:
        if int(section.get(f"{key}_offset", 0) or 0) > 0:
            keys.append(key)
    return keys


def set_section_metadata_properties(obj: bpy.types.Object, sections: List[Dict]):
    obj["hwek_ugx_section_edit_index"] = 0
    try:
        obj.id_properties_ui("hwek_ugx_section_edit_index").update(min=0, max=max(0, len(sections) - 1))
    except Exception:
        pass
    for display_index, section in enumerate(sections):
        source_index = section_entry_source_index(display_index, section)
        editable = set(editable_section_metadata_keys(section))
        for key in SECTION_METADATA_KEYS:
            prop_key = section_property_key(source_index, key)
            if key in editable:
                obj[prop_key] = section.get(key, 0)
            elif prop_key in obj:
                del obj[prop_key]


def section_metadata_value(obj: bpy.types.Object, index: int, entry: Dict, key: str, fallback):
    prop_key = section_property_key(section_entry_source_index(index, entry), key)
    return obj.get(prop_key, entry.get(key, fallback))


def cache_ugx_source_snapshot(source_path: str, data: bytes, digest: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.basename(source_path) or "source.ugx")
    snapshot_path = os.path.join(UGX_SOURCE_SNAPSHOT_DIR, f"{digest}_{safe_name}")
    if not os.path.exists(snapshot_path):
        os.makedirs(UGX_SOURCE_SNAPSHOT_DIR, exist_ok=True)
        with open(snapshot_path, "wb") as handle:
            handle.write(data)
    return snapshot_path


def set_ugx_source_metadata(obj: bpy.types.Object, source_path: str, data: bytes, digest=None):
    if digest is None:
        digest = sha256_bytes(data)
    obj["hwek_ugx_source_path"] = source_path
    if "hwek_ugx_texture_context_path" not in obj:
        obj["hwek_ugx_texture_context_path"] = source_path
    obj["hwek_ugx_source_sha256"] = digest
    try:
        snapshot_path = cache_ugx_source_snapshot(source_path, data, digest)
    except OSError:
        snapshot_path = ""
    if snapshot_path:
        obj["hwek_ugx_source_cache_path"] = snapshot_path
        obj["hwek_ugx_source_cache_sha256"] = digest


def bone_bound_property_name(key: str, axis: int) -> str:
    return f"hwek_ugx_bounds_{key}_{'xyz'[axis]}"


def set_bone_bound_properties(data_bone: bpy.types.Bone, bone: Bone):
    for key, values, offset in (
        ("low", bone.bounds_low, bone.bounds_low_offset),
        ("high", bone.bounds_high, bone.bounds_high_offset),
    ):
        if not offset:
            continue
        data_bone[f"hwek_ugx_bounds_{key}_offset"] = offset
        for axis, value in enumerate(values):
            data_bone[bone_bound_property_name(key, axis)] = float(value)


def bone_bound_property_values(data_bone: Optional[bpy.types.Bone], key: str, fallback) -> Optional[Tuple[float, float, float]]:
    if data_bone is None:
        return None
    keys = [bone_bound_property_name(key, axis) for axis in range(3)]
    if not all(prop in data_bone for prop in keys):
        return None
    return tuple(float(data_bone[prop]) for prop in keys)


def bone_bounds_are_empty(data_bone: Optional[bpy.types.Bone]) -> bool:
    low = bone_bound_property_values(data_bone, "low", None)
    high = bone_bound_property_values(data_bone, "high", None)
    return bool(low and high and all(low[axis] > high[axis] for axis in range(3)))


def lod_section_indices(obj: bpy.types.Object) -> List[int]:
    try:
        return [int(value) for value in json.loads(obj.get("hwek_ugx_lod_section_indices", "[]"))]
    except Exception:
        return []
