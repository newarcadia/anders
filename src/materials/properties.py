"""Store editable native shader fields on Blender materials."""

from typing import Any, Dict, List, Optional, Tuple
import bpy
import json
import math
import re
import struct
from ..errors import UgxError
from .data import Material, name_is_standard_hw2_material, parse_ufx_shader_hash, ufx_buffer_bytes_from_hex

__all__ = ('ufx_prop_suffix', 'ufx_attr_property_name', 'is_ufx_shader_hash_name', 'ufx_shader_hash_text_from_info', 'ensure_ufx_shader_hash_string_props', 'ufx_shader_file_property_name', 'v4_attr_property_name', 'ufx_buffer_property_name', 'ufx_buffer_float_property_name', 'ufx_buffer_float_values_from_hex', 'ufx_buffer_float_label', 'ufx_shader_pass_label', 'split_ufx_permutation_name', 'set_ufx_buffer_material_properties', 'parse_ufx_property_value', 'ufx_idprop_value', 'set_shader_vector_default', 'set_shader_input_default', 'apply_v4_material_metadata', 'apply_ufx_material_metadata', 'parse_bdt_direct_edit_value')

def ufx_prop_suffix(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "value"


def ufx_attr_property_name(name: str) -> str:
    return f"hwek_ugx_ufx_{ufx_prop_suffix(name)}"


def is_ufx_shader_hash_name(name: str) -> bool:
    return re.fullmatch(r"shaderPermutationHash[0-3]", str(name or "")) is not None


def ufx_shader_hash_text_from_info(info: Dict[str, Any]) -> str:
    return str(parse_ufx_shader_hash(info.get("value", ""), str(info.get("name", "UFX header hash"))))


def ensure_ufx_shader_hash_string_props(mat: Optional[bpy.types.Material], attrs: Dict[str, Any]):
    """Keep UFX uint32 hashes out of Blender's signed integer ID-property UI."""
    if mat is None:
        return
    for index in range(4):
        name = f"shaderPermutationHash{index}"
        prop = ufx_attr_property_name(name)
        info = attrs.get(name, {})
        source_text = ""
        if isinstance(info, dict) and "value" in info:
            try:
                source_text = ufx_shader_hash_text_from_info(info)
            except UgxError:
                source_text = ""
        if prop not in mat:
            if source_text:
                mat[prop] = source_text
                mat[prop + "_source"] = source_text
            continue

        current = mat.get(prop)
        if isinstance(current, str):
            continue
        try:
            numeric = int(current)
        except (TypeError, ValueError):
            continue
        if numeric == -0x80000000 and source_text:
            # Blender clamps overflowing signed integer ID properties to INT_MIN.
            # The imported BDT metadata still contains the original uint32 value.
            text = source_text
        elif numeric < 0:
            text = str(numeric & 0xFFFFFFFF)
        else:
            text = str(numeric)
        mat[prop] = text
        if source_text:
            mat[prop + "_source"] = source_text


def ufx_shader_file_property_name(index: int) -> str:
    return f"hwek_ugx_ufx_shaderPermutationFile{index}"


def v4_attr_property_name(name: str) -> str:
    return f"hwek_ugx_v4_{ufx_prop_suffix(name)}"


def ufx_buffer_property_name(buffer_index: int, name: str) -> str:
    return f"hwek_ugx_ufx_{ufx_prop_suffix(name)}_{buffer_index}_hex"


def ufx_buffer_float_property_name(buffer_index: int, name: str, float_index: int) -> str:
    return f"hwek_ugx_ufx_{ufx_prop_suffix(name)}_{buffer_index}_f{float_index}"


def ufx_buffer_float_values_from_hex(value: str) -> List[float]:
    data = ufx_buffer_bytes_from_hex(value)
    if not data or len(data) % 4:
        return []
    return [float(item) for item in struct.unpack(f"<{len(data) // 4}f", data)]


def ufx_buffer_float_label(buffer_name: str, index: int, total: int) -> str:
    if buffer_name == "VSCBData" and total >= 4:
        labels = (
            "Animated UV speed U",
            "Animated UV speed V",
            "Reserved vertex scalar A",
            "Reserved vertex scalar B",
        )
        if index < len(labels):
            return labels[index]
    if buffer_name == "PSCBData" and total >= 8:
        labels = (
            "Albedo UV scale",
            "Normal UV scale",
            "Packed map UV scale",
            "Specular UV scale",
            "Specular team-tint strength",
            "Emissive UV scale",
            "Emissive intensity",
            "Reserved pixel scalar",
        )
        if index < len(labels):
            return labels[index]
    return f"Float {index}"


def ufx_shader_pass_label(index: int) -> str:
    return {
        0: "Surface pass",
        1: "Surface variant pass",
        2: "Depth/shadow pass",
        3: "Constant-color pass",
    }.get(index, f"Shader pass {index}")


def split_ufx_permutation_name(name: str) -> Tuple[str, str]:
    text = str(name or "")
    if "_" not in text:
        return text, ""
    family, key = text.rsplit("_", 1)
    return family, key


def set_ufx_buffer_material_properties(mat: bpy.types.Material, buffer_index: int, buffer: Dict[str, Any]):
    name = str(buffer.get("name", "Buffer"))
    hex_value = str(buffer.get("hex", ""))
    prop = ufx_buffer_property_name(buffer_index, name)
    mat[prop] = hex_value
    mat[prop + "_source"] = hex_value
    for float_index, value in enumerate(ufx_buffer_float_values_from_hex(hex_value)):
        float_prop = ufx_buffer_float_property_name(buffer_index, name, float_index)
        mat[float_prop] = float(value)
        mat[float_prop + "_source"] = float(value)


def parse_ufx_property_value(info: Dict[str, Any]) -> Any:
    kind = str(info.get("kind", ""))
    value = info.get("value", "")
    if kind == "bool":
        return str(value).lower() == "true"
    if kind == "int":
        try:
            return int(str(value), 0)
        except ValueError:
            return 0
    if kind == "float":
        try:
            return float(str(value))
        except ValueError:
            return 0.0
    return str(value)


def ufx_idprop_value(info: Dict[str, Any]) -> Any:
    value = parse_ufx_property_value(info)
    if isinstance(value, int) and not (-0x80000000 <= value <= 0x7FFFFFFF):
        return str(value)
    return value


def set_shader_vector_default(shader_node, name: str, values: Tuple[float, ...]):
    if shader_node is None or name not in shader_node.inputs:
        return
    socket = shader_node.inputs[name]
    try:
        current = socket.default_value
    except Exception:
        return
    if not hasattr(current, "__len__") or isinstance(current, str):
        return
    for index, value in enumerate(values[:len(current)]):
        current[index] = float(value)


def set_shader_input_default(shader_node, name: str, value: Any):
    if shader_node is None or name not in shader_node.inputs:
        return
    socket = shader_node.inputs[name]
    try:
        current = socket.default_value
    except Exception:
        return
    if isinstance(current, bool):
        socket.default_value = bool(value)
    elif isinstance(current, int):
        socket.default_value = int(value)
    elif isinstance(current, float):
        socket.default_value = float(value)
    elif hasattr(current, "__len__") and not isinstance(current, str):
        numeric = float(value)
        try:
            for index in range(len(current)):
                current[index] = numeric
        except Exception:
            pass


def apply_v4_material_metadata(mat: bpy.types.Material, material: Material, shader_node):
    if not material.v4_attrs:
        return
    mat["hwek_ugx_v4_attrs"] = json.dumps(material.v4_attrs)
    for name, info in material.v4_attrs.items():
        prop = v4_attr_property_name(name)
        value = ufx_idprop_value(info)
        mat[prop] = value
        mat[prop + "_source"] = value
        socket_name = {
            "SpecPower": "SpecPower",
            "EnvReflectivity": "Reflectivity",
            "EnvSharpness": "Sharpness",
            "EnvFresnelPower": "FresnelPower",
            "EnvFresnel": "Fresnel",
            "AccessoryIndex": "AccIndex",
            "BlendType": "BlendType",
            "Opacity": "Opacity",
            "Flags": "Flags",
        }.get(name)
        if socket_name:
            set_shader_input_default(shader_node, socket_name, value)
    spec = []
    for name in ("SpecColorR", "SpecColorG", "SpecColorB"):
        info = material.v4_attrs.get(name)
        if info is None:
            break
        try:
            spec.append(float(ufx_idprop_value(info)))
        except (TypeError, ValueError):
            break
    if len(spec) == 3:
        set_shader_vector_default(shader_node, "SpecColor", (spec[0], spec[1], spec[2], 1.0))


def apply_ufx_material_metadata(mat: bpy.types.Material, material: Material, shader_node, cache_version: int):
    if cache_version==6:
        if material.legacy_containers is not None:
            mat['hwek_ugx_legacy_containers']=json.dumps(material.legacy_containers)
        mat['hwek_ugx_has_hogan_material']=material.has_hogan_material
        if material.has_hogan_material:
            mat['hwek_ugx_ufx_attrs']=json.dumps(material.ufx_attrs)
            mat['hwek_ugx_ufx_constant_buffers']=json.dumps(material.ufx_constant_buffers)
    if not material.ufx_attrs and not material.ufx_constant_buffers:
        return
    if material.ufx_attrs:
        mat["hwek_ugx_ufx_attrs"] = json.dumps(material.ufx_attrs)
        mat["hwek_ugx_ufx_constant_buffers"] = json.dumps(material.ufx_constant_buffers)
        for name, info in material.ufx_attrs.items():
            prop = ufx_attr_property_name(name)
            value = ufx_shader_hash_text_from_info(info) if is_ufx_shader_hash_name(name) else ufx_idprop_value(info)
            mat[prop] = value
            mat[prop + "_source"] = value
            if cache_version == 6 and name in ("ufxVersion", "blendMode", "shadowRequiresConsts", "skinned", "terrainBlending"):
                set_shader_input_default(shader_node, name, value)
        permutations = [
            {
                "name": material.ufx_attrs.get(f"shaderPermutationName{index}", {}).get("value", ""),
                "hash": material.ufx_attrs.get(f"shaderPermutationHash{index}", {}).get("value", ""),
            }
            for index in range(4)
            if material.ufx_attrs.get(f"shaderPermutationName{index}") or material.ufx_attrs.get(f"shaderPermutationHash{index}")
        ]
        if permutations:
            mat["hwek_ugx_shader_permutations"] = json.dumps(permutations)
        ensure_ufx_shader_hash_string_props(mat, material.ufx_attrs)
    if material.ufx_constant_buffers:
        mat["hwek_ugx_ufx_constant_buffers"] = json.dumps(material.ufx_constant_buffers)
        for index, buffer in enumerate(material.ufx_constant_buffers):
            set_ufx_buffer_material_properties(mat, index, buffer)
            if cache_version == 6 and name_is_standard_hw2_material(material):
                if buffer.get("name") == "PSCBData":
                    values = ufx_buffer_float_values_from_hex(str(buffer.get("hex", "")))
                    if len(values) >= 8:
                        set_shader_input_default(shader_node, "Emission", values[6])


def parse_bdt_direct_edit_value(info: Dict[str, Any], value: Any) -> int:
    kind = str(info.get("kind", ""))
    if kind == "bool":
        if isinstance(value, str):
            return 1 if value.strip().lower() in ("1", "true", "yes", "on") else 0
        return 1 if bool(value) else 0
    if kind == "int":
        try:
            raw = int(value)
        except (TypeError, ValueError):
            raise UgxError(f"Material value {info.get('name', '')} is not an integer.")
        value_size = int(info.get("value_size", 4) or 4)
        if value_size == 1 and not (-128 <= raw <= 255):
            raise UgxError(f"Material value {info.get('name', '')} must fit in one byte.")
        if value_size == 2 and not (-32768 <= raw <= 65535):
            raise UgxError(f"Material value {info.get('name', '')} must fit in two bytes.")
        return raw
    if kind == "float":
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise UgxError(f"Material value {info.get('name', '')} is not a float.")
        if not math.isfinite(number):
            raise UgxError(f"Material value {info.get('name', '')} must be finite.")
        return struct.unpack("<I", struct.pack("<f", number))[0]
    raise UgxError(f"Material value {info.get('name', '')} is not a direct editable scalar.")
