"""Patch edited material fields while retaining native BDT records."""

from typing import Any, Dict, Optional, Tuple
import bpy
import json
import math
import re
import struct
from ..errors import UgxError
from ..ugx.scene import ugx_used_material_slots
from ..ugx.read import read_materials
from ..formats.bdt import (bdt_rewrite_cstrings, refresh_bdt_crc_in_place, bdt_null_flags,
                           bdt_direct_bool_flags, bdt_direct_int_flags, bdt_stored_string_flags)
from ..formats.bdt_edit import BdtRecord, bdt_record_name, read_bdt_document, write_bdt_document
from ..formats.binary import write_u32_le
from ..formats.ecf import MATERIAL_CHUNK, ecf_chunk_records, replace_ecf_chunk
from .data import parse_ufx_shader_hash, v4_material_uvw_values, v4_material_usage
from .hw1 import v4_material_slots_from_blender
from .properties import (
    is_ufx_shader_hash_name,
    parse_bdt_direct_edit_value,
    ufx_attr_property_name,
    ufx_buffer_float_property_name,
    ufx_buffer_property_name,
    ufx_idprop_value,
    v4_attr_property_name,
)

__all__ = ('patch_hw2_shader_fields', 'patch_material_bdt_strings', 'add_bdt_replacement', 'refresh_material_bdt_crc', 'parse_ufx_edit_value', 'patch_material_v4_fields', 'parse_hex_bytes', 'material_ufx_buffer_target_bytes', 'patch_material_ufx_fields')


def _create_shader_layout_fields(data, materials, targets):
    missing = {index:[name for name,value in fields.items() if name not in materials[index].ufx_attrs and value not in ('',0,False)]
               for index,fields in targets.items()}
    if not any(missing.values()):
        return data
    document = read_bdt_document(data)
    def name(node):
        return bdt_record_name(document,node.entries[0]) if node.entries else ''
    roots = [index for index,node in enumerate(document.nodes) if node.parent==0xffff and name(node)=='Materials']
    records = [index for root in roots for index,node in enumerate(document.nodes) if node.parent==root and name(node)=='Material']
    for index,fields in missing.items():
        if not fields:
            continue
        if index>=len(records):
            raise UgxError(f'Shader layout edit references missing native material {index}.')
        parent = records[index]
        hogan = next((node for node in document.nodes if node.parent==parent and name(node)=='HoganMaterial'),None)
        if hogan is None:
            hogan = BdtRecord(parent,[(0,'HoganMaterial',bdt_null_flags())])
            document.nodes.append(hogan)
        for field in fields:
            if field in ('skinned','shadowRequiresConsts','terrainBlending'):
                entry = (0,field,bdt_direct_bool_flags())
            elif field in ('ufxVersion','blendMode') or is_ufx_shader_hash_name(field):
                entry = (0,field,bdt_direct_int_flags())
            elif field=='basePermutationName' or re.fullmatch(r'shaderPermutationName[0-3]',field):
                entry = (len(document.values),field,bdt_stored_string_flags(1))
                document.values.append(0)
            else:
                raise UgxError(f'Material {index}: shader layout field {field} has no supported native type.')
            hogan.entries.append(entry)
    return write_bdt_document(document)


def patch_hw2_shader_fields(data, targets):
    """Keep provisional shader contracts consistent with rebuilt vertex streams."""
    if not targets:
        return data
    record = next((item for item in ecf_chunk_records(bytes(data)) if item.id == MATERIAL_CHUNK), None)
    if record is None:
        raise UgxError('Changing a native skin layout requires its material chunk.')
    materials = read_materials(record.data, 0, record.offset)
    for index in targets:
        if not 0 <= index < len(materials):
            raise UgxError(f'Shader layout edit references missing material {index}.')
    payload = _create_shader_layout_fields(record.data,materials,targets)
    if payload != record.data:
        data = replace_ecf_chunk(bytearray(data),MATERIAL_CHUNK,payload)
        record = next(item for item in ecf_chunk_records(bytes(data)) if item.id==MATERIAL_CHUNK)
        materials = read_materials(record.data,0,record.offset)
    result = bytearray(data)
    strings = {}
    for index, fields in targets.items():
        if not 0 <= index < len(materials):
            raise UgxError(f'Shader layout edit references missing material {index}.')
        for name, value in fields.items():
            info = materials[index].ufx_attrs.get(name)
            if info is None:
                if value in ('', 0, False):
                    continue
                raise UgxError(f'Material {index}: shader layout field {name} is missing.')
            before = ufx_idprop_value(info)
            if str(value).lower() == str(before).lower():
                continue
            offset = int(info.get('entry_offset', 0))
            if offset <= 0 or offset+4 > len(result):
                raise UgxError(f'Material {index}: shader layout field {name} is outside its source chunk.')
            if info.get('kind') == 'string':
                add_bdt_replacement(strings, offset, str(value), f'Material {index} {name}')
            elif info.get('direct'):
                write_u32_le(result, offset, parse_ufx_edit_value(info, value))
            else:
                raise UgxError(f'Material {index}: shader layout field {name} is not directly editable.')
    result = patch_material_bdt_strings(result, {}, strings)
    refresh_material_bdt_crc(result)
    return result

def patch_material_bdt_strings(data: bytearray, replacements: Dict[int, str], entry_replacements: Optional[Dict[int, str]] = None) -> bytearray:
    entry_replacements = entry_replacements or {}
    if not replacements and not entry_replacements:
        return data
    records = ecf_chunk_records(bytes(data))
    record = next((item for item in records if item.id == MATERIAL_CHUNK), None)
    if not record:
        raise UgxError("UGX is missing the material chunk.")
    local_replacements = {offset - record.offset: value for offset, value in replacements.items()}
    local_entry_replacements = {offset - record.offset: value for offset, value in entry_replacements.items()}
    chunk = bytes(data[record.offset:record.offset + record.size])
    rewritten = bdt_rewrite_cstrings(chunk, local_replacements, local_entry_replacements)
    return replace_ecf_chunk(data, MATERIAL_CHUNK, rewritten)


def add_bdt_replacement(replacements: Dict[int, str], offset: int, value: str, label: str):
    previous = replacements.get(offset)
    if previous is not None and previous != value:
        raise UgxError(f"{label} shares a BinaryDataTree string with another edited value.")
    replacements[offset] = value


def refresh_material_bdt_crc(data: bytearray):
    record = next((item for item in ecf_chunk_records(bytes(data)) if item.id == MATERIAL_CHUNK), None)
    if record:
        refresh_bdt_crc_in_place(data, record.offset, record.size)


def parse_ufx_edit_value(info: Dict[str, Any], value: Any) -> int:
    kind = str(info.get("kind", ""))
    if kind == "bool":
        if isinstance(value, str):
            return 1 if value.strip().lower() in ("1", "true", "yes", "on") else 0
        return 1 if bool(value) else 0
    if kind == "int":
        if is_ufx_shader_hash_name(str(info.get("name", ""))):
            return parse_ufx_shader_hash(value, str(info.get("name", "UFX header hash")))
        return int(value)
    raise UgxError(f"UFX value {info.get('name', '')} is not a direct editable scalar.")


def patch_material_v4_fields(data: bytearray, obj: bpy.types.Object) -> bool:
    changed_any = False
    record = next((item for item in ecf_chunk_records(bytes(data)) if item.id == MATERIAL_CHUNK), None)
    source_materials = read_materials(record.data, 0, record.offset) if record is not None else []
    for _, mat in ugx_used_material_slots(obj):
        if mat is None:
            continue
        for info in json.loads(mat.get("hwek_ugx_v4_attrs", "{}")).values():
            name = str(info.get("name", ""))
            prop = v4_attr_property_name(name)
            if prop not in mat:
                continue
            if str(info.get("kind", "")) == "string":
                continue
            if not bool(info.get("direct", False)):
                continue
            entry_offset = int(info.get("entry_offset", 0))
            if entry_offset <= 0 or entry_offset + 4 > len(data):
                continue
            raw = parse_bdt_direct_edit_value(info, mat.get(prop))
            if raw == int(info.get("raw_value", 0)):
                continue
            write_u32_le(data, entry_offset, raw)
            changed_any = True
        if int(mat.get('hwek_ugx_cache_version', 0)) != 4:
            continue
        index = int(mat.get('hwek_ugx_material_index', -1))
        if not 0 <= index < len(source_materials):
            continue
        source_slots = {v4_material_usage(slot['usage']): slot for slot in source_materials[index].slots}
        for slot in v4_material_slots_from_blender(mat):
            source = source_slots.get(slot['usage'], {})
            for field in ('channel', 'flags'):
                if int(slot[field]) == int(source.get(field, 0)):
                    continue
                info = source.get(field + '_info', {})
                offset = int(info.get('entry_offset', 0))
                if not info.get('direct') or offset <= 0 or offset + 4 > len(data):
                    raise UgxError(f"Material {mat.name}: {slot['usage']} {field} cannot be patched. Export as a new UGX.")
                write_u32_le(data, offset, parse_bdt_direct_edit_value(info, slot[field]))
                changed_any = True
            values = v4_material_uvw_values(slot['usage'], slot)
            if values == v4_material_uvw_values(slot['usage'], source):
                continue
            info = source.get('uvwvel_info', {})
            offset = int(info.get('value_file_offset', 0))
            if int(info.get('value_byte_size', 0)) != 12 or offset <= 0 or offset + 12 > len(data):
                raise UgxError(f"Material {mat.name}: {slot['usage']} UVW velocity cannot be patched. Export as a new UGX.")
            data[offset:offset + 12] = struct.pack('<3f', *values)
            changed_any = True
    return changed_any


def parse_hex_bytes(value: str, expected_size: int, label: str) -> bytes:
    compact = re.sub(r"[^0-9A-Fa-f]", "", str(value))
    if len(compact) != expected_size * 2:
        raise UgxError(f"{label} must contain exactly {expected_size} bytes of hex.")
    try:
        return bytes.fromhex(compact)
    except ValueError as exc:
        raise UgxError(f"{label} is not valid hex.") from exc


def material_ufx_buffer_target_bytes(mat: bpy.types.Material, buffer_index: int, buffer: Dict[str, Any]) -> Tuple[Optional[bytes], bool]:
    name = str(buffer.get("name", "Buffer"))
    size = int(buffer.get("value_byte_size", 0))
    if size <= 0:
        return None, False
    label = f"Material {mat.name} {name}"
    hex_prop = ufx_buffer_property_name(buffer_index, name)
    source_hex = str(buffer.get("hex", ""))
    current_hex = str(mat.get(hex_prop, source_hex))
    target = bytearray(parse_hex_bytes(current_hex, size, label))
    source = parse_hex_bytes(source_hex, size, label)
    changed = target != source

    if size % 4 == 0:
        float_count = size // 4
        for float_index in range(float_count):
            float_prop = ufx_buffer_float_property_name(buffer_index, name, float_index)
            if float_prop not in mat:
                continue
            try:
                current = float(mat.get(float_prop))
            except (TypeError, ValueError):
                raise UgxError(f"{label} float {float_index} is not a valid number.")
            if not math.isfinite(current):
                raise UgxError(f"{label} float {float_index} must be finite.")
            packed = struct.pack("<f", current)
            start = float_index * 4
            if packed != source[start:start + 4]:
                target[start:start + 4] = packed
                changed = True

    if changed and hex_prop in mat:
        mat[hex_prop] = bytes(target).hex(" ")
    return bytes(target), changed


def patch_material_ufx_fields(data: bytearray, obj: bpy.types.Object) -> bool:
    changed_any = False
    for _, mat in ugx_used_material_slots(obj):
        if mat is None:
            continue
        for info in json.loads(mat.get("hwek_ugx_ufx_attrs", "{}")).values():
            name = str(info.get("name", ""))
            prop = ufx_attr_property_name(name)
            if prop not in mat:
                continue
            kind = str(info.get("kind", ""))
            if kind == "string":
                continue
            if not bool(info.get("direct", False)):
                continue
            entry_offset = int(info.get("entry_offset", 0))
            if entry_offset <= 0 or entry_offset + 4 > len(data):
                continue
            source = ufx_idprop_value(info)
            current = mat.get(prop)
            if current == source:
                continue
            raw = parse_ufx_edit_value(info, current)
            write_u32_le(data, entry_offset, raw)
            changed_any = True
        for index, buffer in enumerate(json.loads(mat.get("hwek_ugx_ufx_constant_buffers", "[]"))):
            name = str(buffer.get("name", "Buffer"))
            target, changed = material_ufx_buffer_target_bytes(mat, index, buffer)
            if not changed or target is None:
                continue
            offset = int(buffer.get("value_file_offset", 0))
            size = int(buffer.get("value_byte_size", 0))
            if offset <= 0 or size <= 0 or offset + size > len(data):
                raise UgxError(f"Material {mat.name} {name} constant buffer points outside the UGX file.")
            data[offset:offset + size] = target
            changed_any = True
    return changed_any
