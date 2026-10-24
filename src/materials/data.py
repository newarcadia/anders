"""Native material records, texture semantics and BDT construction."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple
import math
import os
import re
import struct
from ..errors import UgxError
from ..formats.bdt import (
    BdtNode,
    BdtValue,
    bdt_direct_bool_flags,
    bdt_direct_byte_flags,
    bdt_direct_float_flags,
    bdt_direct_int_flags,
    bdt_direct_short_flags,
    bdt_null_flags,
    bdt_stored_byte_flags,
    bdt_stored_float_flags,
    bdt_stored_string_flags,
    build_bdt_node_chunk,
    clean_bdt_string,
)
from ..textures.paths import disabled_texture_path
from ..formats.bdt_edit import bdt_record_name, read_bdt_document

__all__ = ('V4_MATERIAL_MAP_USAGES', 'V4_MATERIAL_USAGE_ALIASES', 'V4_SOURCE_MATERIAL_ATTRS', 'V4_DEFAULT_DIFFUSE_TEXTURE', 'Material', 'parse_ufx_shader_hash', 'ufx_buffer_bytes_from_hex', 'bdt_value_kind', 'bdt_value_json', 'ufx_constant_buffer_json', 'hogan_texture_usage', 'HOGAN_TEXTURE_SLOT_ORDER', 'HW2_MATERIAL_FIELD_NAMES', 'hogan_texture_usages', 'hogan_material_name', 'ugx_material_display_name', 'texture_usage_from_suffix', 'editable_ugx_texture_slot', 'HW1_PARAMETER_SOCKETS', 'name_is_standard_hw2_material', 'hw_shader_socket_name', 'v4_material_usage', 'v4_material_slot_flags', 'v4_material_uvw_values', 'v4_material_attr_raw_flags', 'build_v4_material_bdt', 'HW2_STATIC_MATERIAL_DEFAULT', 'HW2_SKINNED_MATERIAL_DEFAULT', 'parse_bool_value', 'hw2_buffer_storage_flags', 'hw2_material_spec_from_parsed', 'build_v6_material_bdt_from_specs')

V4_MATERIAL_MAP_USAGES = (
    "diffuse", "normal", "gloss", "opacity", "xform", "emissive", "ao",
    "env", "envmask", "emxform", "distortion", "highlight", "modulate",
)


V4_MATERIAL_USAGE_ALIASES = {
    "albedo": "diffuse",
    "specular": "gloss",
    "environment": "env",
    "effect": "emxform",
}


V4_SOURCE_MATERIAL_ATTRS = (
    "Flags", "BlendType", "Opacity", "AccessoryIndex", "SpecPower",
    "SpecColorR", "SpecColorG", "SpecColorB",
    "EnvReflectivity", "EnvSharpness", "EnvFresnel", "EnvFresnelPower",
)


V4_DEFAULT_DIFFUSE_TEXTURE = r"\unsc\infantry\marine_01\marine_01_df"


@dataclass
class Material:
    name: str
    slots: List[Dict[str, Any]] = field(default_factory=list)
    name_offset: int = 0
    name_capacity: int = 0
    name_entry_offset: int = 0
    v4_attrs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    ufx_attrs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    ufx_constant_buffers: List[Dict[str, Any]] = field(default_factory=list)
    bdt_data: bytes = field(default=b'',repr=False,compare=False)
    bdt_index: int = -1
    source_texture_paths: Tuple[str, ...] = ()
    has_texture_array: bool = False
    has_hogan_material: bool = False
    native_name: Optional[str] = None
    legacy_containers: Optional[Tuple[str, ...]] = None


def parse_ufx_shader_hash(value: Any, label: str = "UFX header hash") -> int:
    text = str(value).strip()
    try:
        parsed = int(text, 16 if text.lower().startswith("0x") else 10)
    except (TypeError, ValueError):
        raise UgxError(f"{label} must be an unsigned 32-bit integer.")
    if parsed < 0 or parsed > 0xFFFFFFFF:
        raise UgxError(f"{label} must be between 0 and 4294967295.")
    return parsed


def ufx_buffer_bytes_from_hex(value: str) -> bytes:
    compact = re.sub(r"[^0-9A-Fa-f]", "", str(value))
    if not compact:
        return b""
    if len(compact) % 2:
        return b""
    try:
        return bytes.fromhex(compact)
    except ValueError:
        return b""


def bdt_value_kind(value: BdtValue) -> str:
    if value.value_type == 1:
        return "bool"
    if value.value_type == 2:
        return "int"
    if value.value_type == 3:
        return "float"
    if value.value_type == 4:
        return "string"
    return "unknown"


def bdt_value_json(value: BdtValue) -> Dict[str, Any]:
    return {
        "name": value.name,
        "value": value.text,
        "kind": bdt_value_kind(value),
        "entry_offset": value.entry_offset,
        "file_offset": value.file_offset,
        "capacity": value.string_capacity,
        "raw_value": int(value.raw_value),
        "flags": int(value.flags),
        "value_type": int(value.value_type),
        "value_size": int(value.value_size),
        "total_size": int(value.total_size),
        "direct": bool(value.direct),
        "value_file_offset": int(value.value_file_offset),
        "value_byte_size": int(value.value_byte_size),
    }


def ufx_constant_buffer_json(node: BdtNode) -> Optional[Dict[str, Any]]:
    value = node.value_info
    if value is None or value.direct or value.value_type not in (2,3):
        return None
    if value.value_byte_size and not value.value_file_offset:
        return None
    return {
        "name": node.name,
        "entry_offset": value.entry_offset,
        "value_file_offset": value.value_file_offset,
        "value_byte_size": value.value_byte_size,
        "flags": int(value.flags),
        "value_type": int(value.value_type),
        "value_size": int(value.value_size),
        "total_size": int(value.total_size),
        "hex": value.raw_bytes.hex(" "),
    }


def hogan_texture_usage(path: str) -> str:
    lower = path.replace("/", "\\").strip().lower()
    if not lower or lower.startswith("hogan_"):
        return ""
    for token, usage in (
        ("[al]", "diffuse"),
        ("[nm]", "normal"),
        ("[pm]", "packedmask"),
        ("[pa]", "packedmask"),
        ("[sp]", "specular"),
        ("[em]", "emissive"),
    ):
        if token in lower:
            return usage
    root = os.path.splitext(lower)[0]
    for suffix, usage in (
        ("_al", "diffuse"),
        ("_df", "diffuse"),
        ("_diffuse", "diffuse"),
        ("_albedo", "diffuse"),
        ("_basecolor", "diffuse"),
        ("_base_color", "diffuse"),
        ("_nm", "normal"),
        ("_normal", "normal"),
        ("_norm", "normal"),
        ("_sp", "specular"),
        ("_specular", "specular"),
        ("_spec", "specular"),
        ("_em", "emissive"),
        ("_emissive", "emissive"),
        ("_emission", "emissive"),
        ("_opacity", "opacity"),
        ("_alpha", "opacity"),
        ("_gloss", "gloss"),
        ("_roughness", "roughness"),
        ("_metallic", "metallic"),
        ("_ao", "ao"),
        ("_pm", "packedmask"),
        ("_pa", "packedmask"),
        ("_mask", "packedmask"),
        ("_packed", "packedmask"),
        ("_packedmask", "packedmask"),
    ):
        if root.endswith(suffix):
            return usage
    return "texture"


HOGAN_TEXTURE_SLOT_ORDER = ("diffuse", "normal", "packedmask", "specular", "emissive")

HW2_MATERIAL_FIELD_NAMES = tuple(f'shaderPermutation{kind}{index}' for kind in ('Name','Hash') for index in range(4))+(
    'basePermutationName','ufxVersion','blendMode','skinned','shadowRequiresConsts','terrainBlending',
)


def hogan_texture_usages(paths: Iterable[str]) -> List[str]:
    """Resolve Hogan's ordered texture array without discarding filename evidence."""
    paths = list(paths)
    usages = [hogan_texture_usage(path) for path in paths]
    known_positions = {
        index: HOGAN_TEXTURE_SLOT_ORDER.index(usage)
        for index, usage in enumerate(usages)
        if usage in HOGAN_TEXTURE_SLOT_ORDER
    }
    used = set(known_positions.values())
    for index, usage in enumerate(usages):
        if usage != "texture":
            continue
        previous = max((position for item, position in known_positions.items() if item < index), default=-1)
        following = min((position for item, position in known_positions.items() if item > index), default=len(HOGAN_TEXTURE_SLOT_ORDER))
        candidates = [position for position in range(previous + 1, following) if position not in used]
        if not candidates:
            candidates = [position for position in range(previous + 1, len(HOGAN_TEXTURE_SLOT_ORDER)) if position not in used]
        if not candidates:
            candidates = [position for position in range(len(HOGAN_TEXTURE_SLOT_ORDER)) if position not in used]
        if candidates:
            position = candidates[0]
            usages[index] = HOGAN_TEXTURE_SLOT_ORDER[position]
            known_positions[index] = position
            used.add(position)
    return usages


def hogan_material_name(index: int, name_value: BdtValue, hogan: Optional[BdtNode], textures: Optional[BdtNode]) -> str:
    if name_value.text:
        return name_value.text
    if textures:
        for texture_path in textures.strings:
            if not hogan_texture_usage(texture_path):
                continue
            root = os.path.splitext(os.path.basename(texture_path.replace("\\", "/")))[0]
            cut = root.find("[")
            if cut >= 0:
                root = root[:cut]
            for suffix in ("_diffuse", "_albedo", "_basecolor", "_base_color", "_normal", "_norm", "_specular", "_spec", "_emissive", "_emission", "_opacity", "_alpha", "_gloss", "_roughness", "_metallic", "_ao", "_mask", "_packed", "_packedmask"):
                if root.lower().endswith(suffix):
                    root = root[:-len(suffix)]
                    break
            root = root.strip(" ._-")
            if root:
                return root
    if hogan:
        shader = hogan.attrs.get("shaderPermutationName0")
        if shader and shader.text:
            return shader.text
    return f"material_{index}"


def ugx_material_display_name(material: Material, index: int) -> str:
    """Choose a useful Blender name without changing the source/export name."""
    source_name = str(material.name or "").strip()
    generic_name = not source_name or source_name.lower() == "default" or re.fullmatch(r"material_\d+", source_name.lower()) is not None
    if not generic_name:
        return source_name

    populated_slots = [slot for slot in material.slots if str(slot.get("path", "")).strip()]
    preferred_slots = [slot for slot in populated_slots if str(slot.get("usage", "")).lower() == "diffuse"]
    for slot in preferred_slots + [slot for slot in populated_slots if slot not in preferred_slots]:
        stem = os.path.splitext(os.path.basename(str(slot.get("path", "")).replace("\\", "/")))[0]
        lower_stem = stem.lower()
        for suffix in (
            "_diffuse", "_albedo", "_basecolor", "_base_color", "_normal", "_norm", "_specular", "_spec",
            "_emissive", "_emission", "_opacity", "_alpha", "_gloss", "_roughness", "_metallic", "_ao",
            "_mask", "_packed", "_packedmask", "_df", "_nm", "_sp", "_em", "_ef", "_op", "_gl", "_rm",
            "_xf", "_en", "_ds", "_hl", "_ml",
        ):
            if lower_stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        stem = stem.strip(" ._-")
        if stem:
            return stem

    shader = material.ufx_attrs.get("shaderPermutationName0")
    if isinstance(shader, dict):
        shader = shader.get("value", "")
    shader_name = str(shader or "").strip()
    if shader_name:
        return shader_name
    return source_name or f"material_{index}"


def texture_usage_from_suffix(path: str) -> str:
    root = os.path.splitext(path.replace("\\", "/").lower())[0]
    for suffix, usage in (
        ("_df", "diffuse"),
        ("_nm", "normal"),
        ("_sp", "specular"),
        ("_em", "emissive"),
        ("_ef", "effect"),
        ("_ao", "ao"),
        ("_op", "opacity"),
        ("_gl", "gloss"),
        ("_rm", "envmask"),
        ("_xf", "xform"),
        ("_en", "environment"),
        ("_ds", "distortion"),
        ("_hl", "highlight"),
        ("_ml", "modulate"),
    ):
        if root.endswith(suffix):
            return usage
    return ""


def editable_ugx_texture_slot(slot: Dict[str, Any], cache_version: int = 0) -> bool:
    usage = v4_material_usage(slot.get("usage", ""))
    path = str(slot.get("path", "")).strip()
    if path and not disabled_texture_path(path):
        return True
    if int(slot.get("offset", 0) or 0) > 0 or int(slot.get("capacity", 0) or 0) > 0:
        return True
    if usage in V4_MATERIAL_MAP_USAGES:
        return True
    return cache_version == 6


HW1_PARAMETER_SOCKETS = {
    "SpecPower": "SpecPower", "EnvReflectivity": "Reflectivity", "EnvSharpness": "Sharpness",
    "EnvFresnelPower": "FresnelPower", "EnvFresnel": "Fresnel", "AccessoryIndex": "AccIndex",
    "BlendType": "BlendType", "Opacity": "Opacity", "Flags": "Flags",
}


def name_is_standard_hw2_material(material: Material) -> bool:
    return str(material.ufx_attrs.get("shaderPermutationName0", {}).get("value", "")).upper().startswith("HOGAN_STANDARD_")


def hw_shader_socket_name(usage: str, cache_version: int) -> str:
    usage = usage.lower()
    if cache_version == 6:
        return {
            "diffuse": "Albedo",
            "albedo": "Albedo",
            "emissive": "Emissive",
            "normal": "Normal",
            "specular": "Specular",
            "packedmask": "Masks",
        }.get(usage, "")
    return {
        "diffuse": "Diffuse (df)",
        "albedo": "Diffuse (df)",
        "normal": "Normal (nm)",
        "specular": "Specular (sp)",
        "gloss": "Specular (sp)",
        "opacity": "Alpha (op)",
        "emissive": "Emmissive (em)",
        "effect": "Emmissive_player_color (ef)",
        "emxform": "Emmissive_player_color (ef)",
        "xform": "Player_color (xf)",
        "environment": "Environment (en)",
        "env": "Environment (en)",
        "envmask": "EnvMask (rm)",
        "ao": "Occlusion (ao)",
        "distortion": "Distortion (ds)",
        "highlight": "Highlight (hl)",
        "modulate": "Modulate (ml)",
    }.get(usage, "")


def v4_material_usage(usage: str) -> str:
    text = str(usage or "").strip().lower()
    return V4_MATERIAL_USAGE_ALIASES.get(text, text)


def v4_material_slot_flags(usage: str, slot: Dict[str, Any]) -> int:
    if "flags" in slot:
        try:
            return max(0, min(0xFFFF, int(slot.get("flags", 0))))
        except (TypeError, ValueError):
            pass
    return 7 if usage in ("diffuse", "xform", "emxform") else 0


def v4_material_uvw_values(usage: str, slot: Dict[str, Any]) -> Tuple[float, float, float]:
    source = slot.get("uv_scroll", None)
    if isinstance(source, (list, tuple)) and source:
        values = list(source)
    else:
        text = str(slot.get("uvwvel", ""))
        for char in ",;()[]\t":
            text = text.replace(char, " ")
        values = text.split()
    parsed: List[float] = []
    for value in values[:3]:
        try:
            number = float(value)
        except (TypeError, ValueError):
            parsed = []
            break
        if not math.isfinite(number):
            parsed = []
            break
        parsed.append(number)
    if parsed:
        while len(parsed) < 3:
            parsed.append(0.0)
        return float(parsed[0]), float(parsed[1]), float(parsed[2])
    return (0.0, 4.0, 0.0) if usage == "env" else (0.0, 0.0, 0.0)


def v4_material_attr_raw_flags(name: str, default_raw: int, default_flags: int, attrs: Dict[str, Dict[str, Any]]) -> Tuple[int, int]:
    info = attrs.get(name, {})
    if not isinstance(info, dict):
        return default_raw, default_flags
    try:
        raw = int(info.get("raw_value", default_raw))
    except (TypeError, ValueError):
        raw = default_raw
    try:
        flags = int(info.get("flags", default_flags))
    except (TypeError, ValueError):
        flags = default_flags
    return raw, flags


def _v4_map_scalar_flags(slot, field, value, default):
    info = slot.get(field.lower() + '_info', {})
    if not isinstance(info, dict) or not info.get('direct') or info.get('kind') != 'int':
        return default
    flags = int(info.get('flags', default))
    size = 1 << ((flags >> 5) & 7)
    if size not in (1, 2, 4) or (flags >> 2) & 7 != 2 or not flags & 2:
        raise UgxError(f"Texture map {field}: native scalar storage is unsupported.")
    maximum = (1 << (size * 8 - (0 if flags & 1 else 1))) - 1
    if not 0 <= value <= maximum:
        raise UgxError(f"Texture map {field}: {value} exceeds its native {size}-byte integer storage.")
    return (flags & ~256) | (default & 256)


def build_v4_material_bdt(
    material_names: List[str],
    slots_by_material: Optional[List[List[Dict[str, Any]]]] = None,
    attrs_by_material: Optional[List[Dict[str, Dict[str, Any]]]] = None,
    preserve_schema: bool = False,
    native_materials: Optional[List[Optional[Material]]] = None,
) -> bytes:
    if native_materials is not None and len(native_materials) != len(material_names):
        raise UgxError('HW1 retained material definitions do not match their material table.')
    values = bytearray()

    def add_value(data: bytes) -> int:
        if len(data) >= 127:
            while len(values) % 4:
                values.append(0)
            values.extend(struct.pack("<I", len(data)))
        offset = len(values)
        values.extend(data)
        return offset

    def add_string(value: str, fallback: str) -> Tuple[int, int]:
        raw = clean_bdt_string(value, fallback).encode("utf-8") + b"\0"
        return add_value(raw), len(raw)

    if slots_by_material is None:
        slots_by_material = [[{
            "usage": "diffuse",
            "path": V4_DEFAULT_DIFFUSE_TEXTURE,
            "channel": 0,
            "flags": 7,
        }] for _ in material_names]
    if attrs_by_material is None:
        attrs_by_material = [{} for _ in material_names]

    nodes: List[Dict[str, Any]] = [{
        "parent": 0xFFFF,
        "name_values": [(0, "Materials", bdt_null_flags())],
    }]
    for material_index, material_name in enumerate(material_names):
        native = native_materials[material_index] if native_materials is not None else None
        schema = preserve_schema or native is not None
        material_node_index = len(nodes)
        material_values=[(0, "Material", bdt_null_flags(schema and material_name is None))]
        if material_name is not None or not schema:
            name_offset, name_size = add_string(material_name, '' if schema else f"material_{material_index}")
            material_values.append((name_offset, "Name", bdt_stored_string_flags(name_size,schema)))
        if not schema:
            material_values.append((4, "Ver", bdt_direct_int_flags(True)))
        nodes.append({
            "parent": 0,
            "name_values": material_values,
        })

        name_values_index = len(nodes)
        nodes.append({
            "parent": material_node_index,
            "name_values": [(0, "NameValues", bdt_null_flags())],
        })
        attrs = attrs_by_material[material_index] if material_index < len(attrs_by_material) else {}
        default_attrs = (
            ("SpecPower", struct.unpack("<I", struct.pack("<f", 25.0))[0], bdt_direct_float_flags()),
            ("SpecColorR", struct.unpack("<I", struct.pack("<f", 1.0))[0], bdt_direct_float_flags()),
            ("SpecColorG", struct.unpack("<I", struct.pack("<f", 1.0))[0], bdt_direct_float_flags()),
            ("SpecColorB", struct.unpack("<I", struct.pack("<f", 1.0))[0], bdt_direct_float_flags()),
            ("EnvReflectivity", struct.unpack("<I", struct.pack("<f", 1.0))[0], bdt_direct_float_flags()),
            ("EnvSharpness", struct.unpack("<I", struct.pack("<f", 1.0))[0], bdt_direct_float_flags()),
            ("EnvFresnel", struct.unpack("<I", struct.pack("<f", 0.5))[0], bdt_direct_float_flags()),
            ("EnvFresnelPower", struct.unpack("<I", struct.pack("<f", 4.0))[0], bdt_direct_float_flags()),
            ("AccessoryIndex", 0, bdt_direct_int_flags()),
            ("Flags", 1, bdt_direct_int_flags()),
            ("BlendType", 0, bdt_direct_byte_flags()),
            ("Opacity", 255, bdt_direct_byte_flags()),
        )
        emitted_attrs = set()
        for name, default_value, default_flags in default_attrs:
            if schema and name not in attrs:
                continue
            value, flags = v4_material_attr_raw_flags(name, default_value, default_flags, attrs)
            nodes.append({"parent": name_values_index, "name_values": [(value, name, flags)]})
            emitted_attrs.add(name)
        for name in sorted(set(attrs) - emitted_attrs):
            info = attrs.get(name, {})
            if not isinstance(info,dict):
                continue
            if info.get("kind")=="string":
                value,size=add_string(str(info.get("value","")),"")
                nodes.append({"parent":name_values_index,"name_values":[(value,name,bdt_stored_string_flags(size))]})
                continue
            if not info.get("direct",True):
                continue
            try:
                value = int(info.get("raw_value", 0))
                flags = int(info.get("flags", bdt_direct_int_flags()))
            except (TypeError, ValueError):
                continue
            nodes.append({"parent": name_values_index, "name_values": [(value, name, flags)]})

        maps_index = len(nodes)
        nodes.append({
            "parent": material_node_index,
            "name_values": [(0, "Maps", bdt_null_flags())],
        })
        slots = {
            v4_material_usage(slot.get("usage", "")): slot
            for slot in (slots_by_material[material_index] if material_index < len(slots_by_material) else [])
        }
        for usage in V4_MATERIAL_MAP_USAGES:
            if schema and usage not in slots:
                continue
            map_category_index = len(nodes)
            slot = slots.get(usage)
            if schema and slot and slot.get('generated_category') and not slot.get('path'):
                continue
            uvw_offset = add_value(struct.pack("<3f", *v4_material_uvw_values(usage, slot or {})))
            nodes.append({
                "parent": maps_index,
                "name_values": [
                    (0, usage, bdt_null_flags(False)),
                    (uvw_offset, "UVWVel", bdt_stored_float_flags(12, True)),
                ],
            })
            path = str(slot.get("path", "")).strip() if slot else ""
            if not path or disabled_texture_path(path):
                continue
            path_offset, path_size = add_string(path, path)
            channel = max(0, min(2, int(slot.get("channel", 0) or 0)))
            map_flags = v4_material_slot_flags(usage, slot)
            nodes.append({
                "parent": map_category_index,
                "name_values": [
                    (0, "Map", bdt_null_flags(False)),
                    (path_offset, "Name", bdt_stored_string_flags(path_size, False)),
                    (channel, "Channel", _v4_map_scalar_flags(slot, 'Channel', channel, bdt_direct_short_flags(False))),
                    (map_flags, "Flags", _v4_map_scalar_flags(slot, 'Flags', map_flags, bdt_direct_short_flags(True, True))),
                ],
            })

    if native_materials is not None:
        omitted = set()
        material_index = -1
        parents = {node['parent'] for node in nodes}
        for index, node in enumerate(nodes):
            name = node['name_values'][0][1]
            if name == 'Material':
                material_index += 1
            native = native_materials[material_index] if material_index >= 0 else None
            if (native is not None and native.legacy_containers is not None
                    and name in ('NameValues', 'Maps') and name not in native.legacy_containers and index not in parents):
                omitted.add(index)
        if omitted:
            indices = {old: new for new, old in enumerate(index for index in range(len(nodes)) if index not in omitted)}
            nodes = [dict(node, parent=indices.get(node['parent'], node['parent'])) for index, node in enumerate(nodes) if index not in omitted]
    return build_bdt_node_chunk(nodes, bytes(values))


HW2_STATIC_MATERIAL_DEFAULT = {
    # Retail-backed opaque defaults from mesh_banishedshell.ugx. These use
    # stock HW2 textures instead of making a new mesh depend on another unit.
    "shader_names": (
        "HOGAN_STANDARD_00000000003009A0",
        "HOGAN_STANDARD_00000000005009A0",
        "HOGAN_STANDARD_00080000000009A0",
        "HOGAN_STANDARD_00180000000009A0",
    ),
    "shader_hashes": (3773723191, 3381528805, 3867325916, 1220853483),
    "ufx_version": 9,
    "blend_mode": 0,
    "shadow_requires_consts": False,
    "skinned": False,
    "terrain_blending": False,
    "vscb": bytes(16),
    "pscb": struct.pack("<8f", 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
    "textures": (
        r"environment_tiling\generic\default_[al]",
        r"environment_tiling\generic\default_[nm]",
        r"environment_tiling\generic\default_[pm]",
        r"environment_tiling\generic\default_[sp]",
    ),
}


HW2_SKINNED_MATERIAL_DEFAULT = {
    "shader_names": (
        "HOGAN_STANDARD_00000100003009A0",
        "HOGAN_STANDARD_00000100005009A0",
        "HOGAN_STANDARD_00080100000009A0",
        "HOGAN_STANDARD_00180100000009A0",
    ),
    "shader_hashes": (2819901374, 3801621453, 1497689608, 3206855925),
    "ufx_version": 9,
    "blend_mode": 0,
    "shadow_requires_consts": False,
    "skinned": True,
    "terrain_blending": False,
    "vscb": bytes(16),
    "pscb": struct.pack("<8f", 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
    "textures": (
        r"environment_tiling\generic\default_[al]",
        r"environment_tiling\generic\default_[nm]",
        r"environment_tiling\generic\default_[pm]",
        r"environment_tiling\generic\default_[sp]",
    ),
}


def parse_bool_value(value: Any, default: bool = False) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if value is None:
        return default
    return bool(value)


def hw2_buffer_storage_flags(buffers):
    default=bdt_stored_byte_flags(0,False)
    return {str(buffer['name']):int(buffer.get('flags',default))&255 for buffer in buffers
            if int(buffer.get('flags',default))&255 != default}


def hw2_material_spec_from_parsed(material: Material, skinned: bool) -> Dict[str, Any]:
    defaults = HW2_SKINNED_MATERIAL_DEFAULT if skinned else HW2_STATIC_MATERIAL_DEFAULT

    def attr_value(name: str, default: Any) -> Any:
        info = material.ufx_attrs.get(name, {})
        return info.get("value", default) if isinstance(info, dict) else default

    def buffer_value(name: str, default: bytes) -> bytes:
        for buffer in material.ufx_constant_buffers:
            if str(buffer.get("name", "")) != name:
                continue
            return ufx_buffer_bytes_from_hex(str(buffer.get("hex", "")))
        return default

    shader_names = tuple(str(attr_value(f"shaderPermutationName{index}", defaults["shader_names"][index])) or "" for index in range(4))
    shader_hashes = tuple(
        parse_ufx_shader_hash(
            attr_value(f"shaderPermutationHash{index}", defaults["shader_hashes"][index]),
            f"Shader pass {index} UFX header hash",
        )
        for index in range(4)
    )
    paths=list(material.source_texture_paths)
    for index,slot in enumerate(material.slots):
        register=int(slot.get('hogan_index',index))
        if register<0:
            raise UgxError('Native texture register cannot be negative.')
        while len(paths)<=register:
            paths.append('0')
        paths[register]=str(slot.get('path','')) or '0'
    textures=tuple(paths)
    native_hogan=material.has_hogan_material or bool(material.ufx_attrs)
    if not textures and not native_hogan:
        textures = tuple(defaults["textures"])
    spec = {
        "shader_names": shader_names,
        "shader_hashes": shader_hashes,
        "ufx_version": int(attr_value("ufxVersion", defaults["ufx_version"])),
        "blend_mode": int(attr_value("blendMode", defaults["blend_mode"])),
        "shadow_requires_consts": parse_bool_value(attr_value("shadowRequiresConsts", defaults["shadow_requires_consts"])),
        "skinned": parse_bool_value(attr_value("skinned", defaults["skinned"]), bool(defaults["skinned"])),
        "terrain_blending": parse_bool_value(attr_value("terrainBlending", defaults["terrain_blending"])),
        "vscb": buffer_value("VSCBData", defaults["vscb"]),
        "pscb": buffer_value("PSCBData", defaults["pscb"]),
        "textures": textures,
        "has_textures": material.has_texture_array or not native_hogan,
        "buffer_names": [str(buffer["name"]) for buffer in material.ufx_constant_buffers] if native_hogan else ["VSCBData", "PSCBData"],
        "native_attrs": material.ufx_attrs,
    }
    storage=hw2_buffer_storage_flags(material.ufx_constant_buffers)
    if storage:
        spec['buffer_flags']=storage
    if material.bdt_data and not material.has_hogan_material:
        spec['legacy']={'name':material.native_name,'slots':material.slots,'attrs':material.v4_attrs}
        if material.legacy_containers is not None and set(material.legacy_containers)!={'NameValues','Maps'}:
            spec['legacy']['containers']=material.legacy_containers
    elif material.bdt_data and not all(name in material.ufx_attrs for name in HW2_MATERIAL_FIELD_NAMES if name!='basePermutationName'):
        spec['field_names']=list(material.ufx_attrs)
    return spec


def build_v6_material_bdt_from_specs(specs: List[Dict[str, Any]]) -> bytes:
    values = bytearray()

    def add_value(data: bytes) -> int:
        if len(data) >= 127:
            while len(values) % 4:
                values.append(0)
            values.extend(struct.pack("<I", len(data)))
        offset = len(values)
        values.extend(data)
        return offset

    nodes: List[Dict[str, Any]] = [{
        "parent": 0xFFFF,
        "name_values": [(0, "Materials", bdt_null_flags())],
    }]
    for material_index, spec in enumerate(specs):
        if 'legacy' in spec:
            legacy=spec['legacy']
            payload=build_v4_material_bdt([legacy['name']],[legacy['slots']],[legacy['attrs']],preserve_schema=True)
            document=read_bdt_document(payload)
            containers=legacy.get('containers',('NameValues','Maps'))
            parents={node.parent for node in document.nodes}
            omitted={index for index,node in enumerate(document.nodes)
                     if bdt_record_name(document,node.entries[0]) in ('NameValues','Maps')
                     and bdt_record_name(document,node.entries[0]) not in containers
                     and index not in parents}
            values.extend(bytes((-len(values))%16))
            value_base=len(values)
            values.extend(document.values)
            indices={0:0}
            for index,record in enumerate(document.nodes[1:],1):
                if index in omitted:
                    continue
                indices[index]=len(nodes)
                entries=[]
                for raw,name,flags in record.entries:
                    if not flags&2 and (flags>>2)&7 not in (0,1):
                        raw+=value_base
                    entries.append((raw,bdt_record_name(document,(raw,name,flags)),flags))
                nodes.append({'parent':indices[record.parent],'name_values':entries})
            continue
        material_node_index = len(nodes)
        material_values=[(0,"Material",bdt_null_flags())]
        if "name" in spec:
            raw=str(spec["name"]).encode('utf-8')+b'\0'
            material_values[0]=(0,"Material",bdt_null_flags(False))
            material_values.append((add_value(raw),"Name",bdt_stored_string_flags(len(raw))))
        nodes.append({"parent":0,"name_values":material_values})

        hogan_values: List[Tuple[int, str, int]] = [(0, "HoganMaterial", bdt_null_flags(False))]
        for index, shader_name in enumerate(spec["shader_names"]):
            raw = (str(shader_name).replace("\0", "").encode("utf-8") + b"\0") if shader_name else b"\0"
            hogan_values.append((add_value(raw), f"shaderPermutationName{index}", bdt_stored_string_flags(max(1, len(raw)), False)))
            hogan_values.append((int(spec["shader_hashes"][index]), f"shaderPermutationHash{index}", bdt_direct_int_flags(False)))
        hogan_values.extend((
            (int(spec["ufx_version"]), "ufxVersion", bdt_direct_int_flags(False)),
            (int(spec["blend_mode"]), "blendMode", bdt_direct_int_flags(False)),
            (1 if spec["shadow_requires_consts"] else 0, "shadowRequiresConsts", bdt_direct_bool_flags(False)),
            (1 if spec["skinned"] else 0, "skinned", bdt_direct_bool_flags(False)),
            (1 if spec["terrain_blending"] else 0, "terrainBlending", bdt_direct_bool_flags(True)),
        ))
        known={name for _,name,_ in hogan_values}
        for name,info in spec.get("native_attrs",{}).items():
            if name in known:
                continue
            flags=int(info.get("flags",0))
            if info.get("direct"):
                value=int(info.get("raw_value",0))
            elif info.get("kind")=="string":
                raw=str(info.get("value","")).encode('utf-8')+b'\0'
                value=add_value(raw)
                flags=bdt_stored_string_flags(len(raw))
            else:
                continue
            hogan_values.append((value,name,flags))
        if 'field_names' in spec:
            fields=set(spec['field_names'])
            hogan_values=[value for index,value in enumerate(hogan_values) if index==0 or value[1] in fields]
        hogan_values=[(raw,name,(flags&~256)|(256 if index==len(hogan_values)-1 else 0))
                      for index,(raw,name,flags) in enumerate(hogan_values)]
        hogan_node_index = len(nodes)
        nodes.append({
            "parent": material_node_index,
            "name_values": hogan_values,
        })

        vscb = bytes(spec["vscb"])
        pscb = bytes(spec["pscb"])
        textures = [str(path).replace("\0", "") for path in spec["textures"] if str(path).strip()]
        if textures:
            texture_blob = b"\0".join(path.encode("utf-8") for path in textures) + b"\0\0"
            texture_total = 127 if len(texture_blob) >= 127 else max(1, len(texture_blob))
        else:
            texture_blob = b"\0\0\0\0"
            texture_total = 4
        for name, raw in (("VSCBData", vscb), ("PSCBData", pscb)):
            if name not in spec.get("buffer_names", ("VSCBData", "PSCBData")):
                continue
            flags=int(spec.get('buffer_flags',{}).get(name,bdt_stored_byte_flags(0,False)))&255
            kind=(flags>>2)&7
            element=1<<((flags>>5)&7)
            if flags&2 or kind not in (2,3) or element not in (1,4) or (kind==3 and element!=4):
                raise UgxError(f'HW2 {name} requires stored byte or four-byte numeric data.')
            if len(raw)%element:
                raise UgxError(f'HW2 {name} requires whole {element}-byte elements.')
            values.extend(bytes((-len(values))%element))
            nodes.append({
                "parent": hogan_node_index,
                "name_values": [(add_value(raw), name, flags|256|(min(127,len(raw))<<9))],
            })
        if spec.get('has_textures',True):
            nodes.append({
                "parent": hogan_node_index,
                "name_values": [(add_value(texture_blob), "textures", bdt_stored_string_flags(texture_total, True))],
            })

    return build_bdt_node_chunk(nodes, bytes(values))
