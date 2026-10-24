"""Draw native shader parameters and texture-slot controls."""

from __future__ import annotations
from ..materials.context import material_for_ugx_shader_node
from ..materials.operators import HWEK_OT_validate_ugx_material
from ..materials.parameters import draw_inline_ugx_parameters
from .panels import draw_ugx_material_controls

__all__ = ('draw_ugx_shader_node_controls',)

def draw_ugx_shader_node_controls(node, context, layout, cache_version: int):
    mat = material_for_ugx_shader_node(node, context)
    if mat is None:
        layout.label(text="Place this node in a material.")
        return
    layout.use_property_split = False
    if mat.get('hwek_ugx_shader_state_id') != node.get('hwek_ugx_shader_node_id'):
        layout.label(text='Connect to Surface to edit game parameters.')
        return
    draw_ugx_material_controls(layout, getattr(context, "object", None), mat, shader_node=node)
    draw_inline_ugx_parameters(layout, mat, cache_version)
    operator = layout.operator("hwek.discover_ugx_textures", text="Discover Textures", icon="FILE_REFRESH")
    operator.material_name = mat.name
    operator = layout.operator("hwek.repair_ugx_texture_links", text="Repair Texture Links", icon="NODETREE")
    operator.material_name = mat.name
    operator = layout.operator(HWEK_OT_validate_ugx_material.bl_idname, text="Validate Material", icon="CHECKMARK")
    operator.material_name = mat.name
    operator = layout.operator("hwek.bake_ugx_texture_input", text="Bake Texture Input", icon="RENDER_STILL")
    operator.material_name = mat.name
    if cache_version in (4, 6):
        operator = layout.operator("hwek.add_uv_animation", text="Connect UV Animation", icon="UV")
        operator.material_name = mat.name
