"""Follow animated attachment origins and native Action Editor selections."""
import json
import math
import bpy
from bpy.app.handlers import persistent
from ..animation.scene import assign_uax_action
from ..animation.loading import queue_pending_uax_actions

__all__ = ('synchronize_visual_playback', 'update_visual_attachments', 'synchronize_child_actions', 'linked_visual_action')

_synchronizing = False


def linked_visual_action(action, candidates):
    states = set(json.loads(action.get('hwek_vis_states', '[]')))
    source = str(action.get('hwek_uax_source_path', ''))
    looping = action.get('hwek_uax_looping')
    choices = []
    for candidate in candidates:
        same_state = bool(states.intersection(json.loads(candidate.get('hwek_vis_states', '[]'))))
        same_source = bool(source and candidate.get('hwek_uax_source_path') == source)
        if same_state or same_source:
            choices.append(((same_state, same_source, looping == candidate.get('hwek_uax_looping')), candidate))
    return max(choices, key=lambda item: item[0])[1] if choices else None


def synchronize_child_actions(owner, action):
    for part in bpy.data.objects:
        if part.get('hwek_vis_assembly_root') != owner or part.type != 'ARMATURE':
            continue
        anchor = part.get('hwek_vis_anchor')
        marker = anchor.parent if anchor is not None else None
        if marker is None or str(marker.get('vis_syncanims', '')).casefold() != 'true':
            continue
        if action is None:
            if part.animation_data is not None:
                part.animation_data.action = None
            if 'hwek_vis_requested_action' in part:
                del part['hwek_vis_requested_action']
            continue
        choice = linked_visual_action(action, (candidate for candidate in bpy.data.actions if candidate.get('hwek_vis_owner') == part))
        if choice is None:
            continue
        part['hwek_vis_requested_action'] = choice
        if choice.get('hwek_uax_pending'):
            queue_pending_uax_actions([choice], active=True)
        else:
            assign_uax_action(part, choice, [])


@persistent
def update_visual_attachments(_scene=None, _depsgraph=None):
    for anchor in bpy.data.objects:
        if not anchor.get('hwek_vis_attachment_anchor'):
            continue
        part = anchor.get('hwek_vis_model_owner')
        bone = part.pose.bones.get(anchor.get('hwek_vis_from_bone', '')) if part is not None and part.type == 'ARMATURE' else None
        if bone is None:
            continue
        matrix = bone.matrix.inverted_safe()
        marker = anchor.parent
        if marker is not None and str(marker.get('vis_disregardorient', '')).casefold() == 'true':
            parent = marker.parent
            world = marker.matrix_world @ matrix
            oriented = parent.matrix_world.copy()
            oriented.translation = world.translation
            matrix = marker.matrix_world.inverted_safe() @ oriented
        if any(abs(anchor.matrix_basis[r][c] - matrix[r][c]) > 1e-6 for r in range(4) for c in range(4)):
            anchor.matrix_basis = matrix


@persistent
def synchronize_visual_playback(scene=None, depsgraph=None):
    global _synchronizing
    if _synchronizing:
        return
    _synchronizing = True
    try:
        for owner in bpy.data.objects:
            if owner.type != 'ARMATURE' or owner.get('hwek_vis_assembly_root') or not owner.get('hwek_vis_text'):
                continue
            action = owner.animation_data.action if owner.animation_data else None
            if owner.get('hwek_vis_active_action') != action:
                if action is None:
                    del owner['hwek_vis_active_action']
                else:
                    owner['hwek_vis_active_action'] = action
                synchronize_child_actions(owner, action)
                if scene is not None and action is not None and action.get('hwek_uax_fps'):
                    fps = float(action['hwek_uax_fps'])
                    scene.render.fps = round(fps)
                    scene.render.fps_base = round(fps) / fps
                    scene.frame_end = max(1, math.ceil(action.frame_range[1]))
        update_visual_attachments(scene, depsgraph)
    finally:
        _synchronizing = False
