"""Copy edited node values into native material export properties."""

from typing import Iterable
import bpy
import json
from ..errors import UgxError
from ..textures.context import ugx_texture_context_path
from ..ugx.data import ugx_game_name
from ..ugx.scene import ugx_used_material_slots
from .bindings import ugx_material_node_bindings
from .context import idprop_values_equal, ugx_material_cache_version
from .data import hw_shader_socket_name
from .parameters import ensure_ugx_parameter_nodes
from .properties import ufx_buffer_float_property_name, ufx_buffer_float_values_from_hex, v4_attr_property_name
from .slots import sync_ugx_material_texture_slots_from_props
from .sockets import active_ugx_material_shader, ugx_constant_socket_value, ugx_parameter_socket_property
from .uv import connect_hw1_texture_coordinates
from .uv_data import hw1_texture_coordinates, material_uv_animation
from .ownership import sync_material_shader_state

__all__ = ('sync_ugx_node_material_values', 'sync_ugx_material_from_nodes', 'sync_ugx_export_material_nodes')

def sync_ugx_node_material_values(mat: bpy.types.Material, cache_version: int):
    shader = active_ugx_material_shader(mat, cache_version)
    if cache_version == 6:
        attrs = json.loads(mat.get("hwek_ugx_ufx_attrs", "{}"))
        if not str(attrs.get("shaderPermutationName0", {}).get("value", "")).upper().startswith("HOGAN_STANDARD_"):
            return
        animation = material_uv_animation(mat, ugx_material_node_bindings(mat, cache_version))
        for index, buffer in enumerate(json.loads(mat.get("hwek_ugx_ufx_constant_buffers", "[]"))):
            if buffer.get("name") == "VSCBData" and animation is not None:
                for offset, name in enumerate(("Speed U", "Speed V")):
                    mat[ufx_buffer_float_property_name(index, "VSCBData", offset)] = ugx_constant_socket_value(animation.inputs[name])
            if buffer.get("name") != "PSCBData" or len(ufx_buffer_float_values_from_hex(str(buffer.get("hex", "")))) < 8:
                continue
            prop = ufx_buffer_float_property_name(index, "PSCBData", 6)
            socket = shader.inputs.get("Emission")
            if socket is not None and prop in mat:
                if ugx_parameter_socket_property(socket):
                    continue
                if socket.is_linked:
                    mat[prop] = ugx_constant_socket_value(socket)
                    continue
                original = float(mat.get(prop + "_source", mat[prop]))
                if abs(float(socket.default_value) - original) > 0.00001:
                    mat[prop] = float(socket.default_value)
                else:
                    socket.default_value = float(mat[prop])
        return
    for name, socket_name in {
        "SpecPower": "SpecPower", "EnvReflectivity": "Reflectivity", "EnvSharpness": "Sharpness",
        "EnvFresnelPower": "FresnelPower", "EnvFresnel": "Fresnel", "Opacity": "Opacity",
    }.items():
        prop = v4_attr_property_name(name)
        socket = shader.inputs.get(socket_name)
        if socket is not None and prop in mat and not socket.is_linked and not ugx_parameter_socket_property(socket):
            original = mat.get(prop + "_source", mat[prop])
            if not idprop_values_equal(socket.default_value, original):
                mat[prop] = socket.default_value


def sync_ugx_material_from_nodes(mat: bpy.types.Material, cache_version: int, source_path: str):
    if not mat.get("hwek_ugx_native_material"):
        return
    bindings = ugx_material_node_bindings(mat, cache_version, source_path)
    slots = json.loads(mat.get("hwek_ugx_texture_slots", "[]"))
    if cache_version == 6 and not mat.get("hwek_ugx_source_path"):
        shader = active_ugx_material_shader(mat, cache_version)
        bound_indices = {binding["index"] for binding in bindings}
        for index, slot in enumerate(slots):
            socket = shader.inputs.get(hw_shader_socket_name(slot.get("usage", ""), cache_version))
            if index in bound_indices or socket is None or socket.is_linked:
                continue
            slot["path"] = "0"
            mat[f"hwek_ugx_texture_slot_{index}"] = "0"
            mat[f"hwek_ugx_texture_slot_{index}_synced"] = "0"
            mat[f"hwek_ugx_texture_{slot['usage']}"] = "0"
    for binding in bindings:
        index = binding["index"]
        if index >= len(slots):
            slots.append(binding["slot"])
        slots[index]["channel"] = binding["channel"]
        coordinates = hw1_texture_coordinates(binding["node"]) if cache_version == 4 else None
        if coordinates is not None:
            slots[index]["uv_scroll"] = [ugx_constant_socket_value(coordinates.inputs[name]) for name in ("Speed U", "Speed V", "Speed W")]
            slots[index]["flags"] = int(ugx_constant_socket_value(coordinates.inputs["Map Flags"]))
        path = binding["path"]
        mat[f"hwek_ugx_texture_slot_{index}"] = path
        mat[f"hwek_ugx_texture_slot_{index}_synced"] = path
        mat[f"hwek_ugx_texture_{binding['slot']['usage']}"] = path
        binding["node"]["hwek_ugx_texture_path"] = path
        binding["image"]["hwek_ugx_texture_path"] = path
    mat["hwek_ugx_texture_slots"] = json.dumps(slots)


def sync_ugx_export_material_nodes(objects: Iterable[bpy.types.Object], cache_version: int):
    seen = set()
    for obj in objects:
        for _, mat in ugx_used_material_slots(obj):
            if mat is None or mat.as_pointer() in seen:
                continue
            seen.add(mat.as_pointer())
            sync_material_shader_state(mat)
            if (cache_version == 6 and not mat.get("hwek_ugx_native_material")
                    and mat.node_tree is not None
                    and any(node.bl_idname == "ShaderNodeAndersHW2" for node in mat.node_tree.nodes)):
                raise UgxError(f"Material {mat.name}: HW2 shader data is missing. Restore the material from its source UGX, or assign new HW2 material data before export.")
            version = ugx_material_cache_version(mat, obj)
            if mat.get("hwek_ugx_native_material") and version != cache_version:
                raise UgxError(f"Material {mat.name} targets {ugx_game_name(version)}, but the export targets {ugx_game_name(cache_version)}.")
            if mat.get("hwek_ugx_native_material"):
                ensure_ugx_parameter_nodes(mat)
                sync_ugx_node_material_values(mat, cache_version)
                if cache_version == 4:
                    connect_hw1_texture_coordinates(mat)
            sync_ugx_material_texture_slots_from_props(obj, mat)
            sync_ugx_material_from_nodes(mat, cache_version, ugx_texture_context_path(obj))
