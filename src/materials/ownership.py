"""Carry native shader data with Blender's copied shader graph."""

import uuid
import bpy
from bpy.app.handlers import persistent
from .context import material_for_ugx_shader_node

__all__ = ('remember_material_shader_state', 'copy_material_shader_state', 'sync_material_shader_state', 'sync_material_shader_states')

STATE = 'hwek_ugx_shader_state'
STATE_ID = 'hwek_ugx_shader_node_id'
ACTIVE_STATE = 'hwek_ugx_shader_state_id'


def remember_material_shader_state(mat, shader):
    values = {key: mat[key] for key in mat.keys()
              if key.startswith('hwek_ugx_') and key != ACTIVE_STATE}
    state = shader.get(STATE)
    if state is None or state.to_dict() != values:
        shader[STATE] = values
    if not shader.get(STATE_ID):
        shader[STATE_ID] = uuid.uuid4().hex
    if mat.get(ACTIVE_STATE) != shader[STATE_ID]:
        mat[ACTIVE_STATE] = shader[STATE_ID]


def copy_material_shader_state(shader, source):
    mat = material_for_ugx_shader_node(source)
    if mat is not None and (not source.get(STATE_ID) or mat.get(ACTIVE_STATE) == source.get(STATE_ID)):
        remember_material_shader_state(mat, source)
    state = source.get(STATE)
    if state is not None:
        shader[STATE] = state.to_dict()
    # A pasted node must not inherit the recipient material's cached state.
    shader[STATE_ID] = uuid.uuid4().hex


def sync_material_shader_state(mat):
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return
    output = next((node for node in mat.node_tree.nodes
                   if node.type == 'OUTPUT_MATERIAL' and node.is_active_output), None)
    surface = output.inputs.get('Surface') if output is not None else None
    if surface is None or not surface.is_linked:
        return
    shader = surface.links[0].from_node
    seen = set()
    while shader.type == 'REROUTE':
        if shader.as_pointer() in seen or not shader.inputs[0].is_linked:
            return
        seen.add(shader.as_pointer())
        shader = shader.inputs[0].links[0].from_node
    if shader.bl_idname not in ('ShaderNodeAndersHW2', 'ShaderNodeAndersHWDE'):
        return
    state_id = shader.get(STATE_ID)
    state = shader.get(STATE)
    if state is None:
        if mat.get('hwek_ugx_cache_version'):
            remember_material_shader_state(mat, shader)
        return
    if mat.get(ACTIVE_STATE) == state_id:
        remember_material_shader_state(mat, shader)
        return
    previous = next((node for node in mat.node_tree.nodes
                     if node.get(STATE_ID) == mat.get(ACTIVE_STATE)), None)
    if previous is not None:
        remember_material_shader_state(mat, previous)
    # Material Paste replaces the graph but keeps the recipient's ID properties.
    # Restore those properties from the shader that now drives Surface.
    for key in list(mat.keys()):
        if key.startswith('hwek_ugx_'):
            del mat[key]
    for key, value in state.to_dict().items():
        mat[key] = value
    mat[ACTIVE_STATE] = state_id
    from .parameters import retarget_ugx_parameter_drivers
    retarget_ugx_parameter_drivers(mat)


@persistent
def sync_material_shader_states(_scene, depsgraph):
    materials, trees = set(), set()
    for update in depsgraph.updates:
        item = getattr(update.id, 'original', update.id)
        if isinstance(item, bpy.types.Material):
            materials.add(item)
        elif isinstance(item, bpy.types.ShaderNodeTree):
            trees.add(item)
    if trees:
        materials.update(mat for mat in bpy.data.materials if mat.node_tree in trees)
    for mat in materials:
        sync_material_shader_state(mat)
