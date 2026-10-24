"""Upgrade built-in shader groups after startup and file loading."""

from bpy.app.handlers import persistent
import bpy
import json
from ..errors import UgxError
from .context import is_ugx_material, ugx_material_cache_version
from .groups import (
    hw_shader_group_legacy_copy,
    ensure_hw_shader_group,
    hw_shader_group_interface_names,
    hw_shader_group_signature,
    hw_shader_groups_have_same_graph,
    hw_shader_socket,
    migrate_builtin_hw_shader_group_users,
)
from .node_types import ensure_ugx_material_node_types
from .nodes import upgrade_ugx_material_node
from .parameters import ensure_ugx_parameter_nodes
from .player_color import initialize_all_ugx_player_colors
from .properties import set_shader_input_default, ufx_buffer_float_property_name
from .sockets import active_ugx_material_shader
from .uv import arrange_ugx_material_nodes, connect_hw1_texture_coordinates, migrate_ugx_uv_inputs

__all__ = ('ensure_builtin_hw_shader_groups_after_startup', 'ensure_builtin_hw_shader_groups_on_load')

def ensure_builtin_hw_shader_groups_after_startup():
    if getattr(bpy.data, "node_groups", None) is None:
        return 0.1
    for version in (4, 6):
        group = ensure_hw_shader_group(version)
        signature = hw_shader_group_signature(version)
        for duplicate in list(bpy.data.node_groups):
            if duplicate == group or not duplicate.get("hwek_ugx_shader_builtin"):
                continue
            if int(duplicate.get("hwek_ugx_shader_game_version", 0)) != version:
                continue
            if int(duplicate.get("hwek_ugx_shader_builtin_version", 0)) < 9:
                decode = duplicate.nodes.get("Decode Normal")
                invert = duplicate.nodes.get("Decode Normal Green")
                if decode is not None and invert is not None and decode.bl_idname == "ShaderNodeSeparateColor" and invert.bl_idname == "ShaderNodeInvert":
                    factor = hw_shader_socket(invert.inputs, "Factor", "Fac")
                    color = invert.inputs.get("Color")
                    if color is not None and color.is_linked and color.links[0].from_node == decode and color.links[0].from_socket.name == "Green":
                        for link in list(factor.links):
                            if link.from_node == decode and link.from_socket.name == "Green":
                                duplicate.links.remove(link)
                                factor.default_value = 1.0
            names = hw_shader_group_interface_names(duplicate)
            expected = list(signature["interface"])
            if version == 6 and "Shader Passes" not in names:
                expected.remove("Shader Passes")
            if names != expected:
                continue
            candidate = hw_shader_group_legacy_copy(duplicate, version)
            try:
                matches = hw_shader_groups_have_same_graph(candidate, group)
            finally:
                bpy.data.node_groups.remove(candidate)
            if not matches:
                continue
            migrate_builtin_hw_shader_group_users(duplicate, group)
            duplicate.use_fake_user = False
            if duplicate.users == 0:
                bpy.data.node_groups.remove(duplicate)
    initialize_all_ugx_player_colors()
    ensure_ugx_material_node_types()
    for mat in bpy.data.materials:
        if is_ugx_material(mat):
            try:
                upgrade_ugx_material_node(mat, ugx_material_cache_version(mat))
                migrate_ugx_uv_inputs(mat)
                ensure_ugx_parameter_nodes(mat)
                if ugx_material_cache_version(mat) == 4:
                    connect_hw1_texture_coordinates(mat)
            except UgxError:
                # Keep unsupported graphs intact for export validation.
                pass
    for mat in bpy.data.materials:
        if not is_ugx_material(mat) or mat.get("hwek_ugx_native_material"):
            continue
        version = ugx_material_cache_version(mat)
        try:
            shader = active_ugx_material_shader(mat, version)
        except UgxError:
            continue
        if version == 6:
            attrs = json.loads(mat.get("hwek_ugx_ufx_attrs", "{}"))
            if str(attrs.get("shaderPermutationName0", {}).get("value", "")).upper().startswith("HOGAN_STANDARD_"):
                for index, buffer in enumerate(json.loads(mat.get("hwek_ugx_ufx_constant_buffers", "[]"))):
                    prop = ufx_buffer_float_property_name(index, "PSCBData", 6)
                    if buffer.get("name") == "PSCBData" and prop in mat:
                        set_shader_input_default(shader, "Emission", float(mat[prop]))
        arrange_ugx_material_nodes(mat, version)
    return None


@persistent
def ensure_builtin_hw_shader_groups_on_load(_unused):
    ensure_builtin_hw_shader_groups_after_startup()
