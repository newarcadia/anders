"""Apply matrix-basis animation curves during Blender playback."""

from bpy.app.handlers import persistent
from mathutils import Matrix
from typing import Dict, List
import bpy
from .actions import uax_iter_action_fcurves

__all__ = ('uax_matrix_basis_bone_from_path', 'apply_uax_matrix_basis_action', '_UAX_PLAYBACK_ACTIVE', 'hwek_uax_matrix_basis_playback', 'ensure_uax_playback_handler', 'remove_uax_playback_handler')

def uax_matrix_basis_bone_from_path(data_path: str) -> str:
    prefix = 'pose.bones["'
    suffix = '"].matrix_basis'
    if not data_path.startswith(prefix) or not data_path.endswith(suffix):
        return ""
    value = data_path[len(prefix):-len(suffix)]
    return value.replace('\\"', '"').replace("\\\\", "\\")


def apply_uax_matrix_basis_action(armature: bpy.types.Object, action: bpy.types.Action, frame: float):
    if armature is None or getattr(armature, "type", "") != "ARMATURE" or action is None:
        return
    if action.get("hwek_uax_fcurve_mode") != "matrix_basis":
        return
    pose_bones = getattr(getattr(armature, "pose", None), "bones", None)
    if pose_bones is None:
        return
    matrices: Dict[str, List[float]] = {}
    for fcurve in uax_iter_action_fcurves(action):
        if fcurve.mute or (fcurve.group is not None and fcurve.group.mute):
            continue
        bone_name = uax_matrix_basis_bone_from_path(getattr(fcurve, "data_path", ""))
        array_index = int(getattr(fcurve, "array_index", -1))
        if not bone_name or array_index < 0 or array_index >= 16:
            continue
        values = matrices.get(bone_name)
        if values is None:
            pose_bone = pose_bones.get(bone_name)
            if pose_bone is None:
                continue
            values = [float(value) for row in pose_bone.matrix_basis for value in row]
            matrices[bone_name] = values
        values[array_index] = float(fcurve.evaluate(frame))
    for bone_name, values in matrices.items():
        pose_bone = pose_bones.get(bone_name)
        if pose_bone is None:
            continue
        pose_bone.matrix_basis = Matrix((
            (values[0], values[1], values[2], values[3]),
            (values[4], values[5], values[6], values[7]),
            (values[8], values[9], values[10], values[11]),
            (values[12], values[13], values[14], values[15]),
        ))
    update_tag = getattr(armature, "update_tag", None)
    if callable(update_tag):
        update_tag()


_UAX_PLAYBACK_ACTIVE = False


@persistent
def hwek_uax_matrix_basis_playback(scene, depsgraph=None):
    global _UAX_PLAYBACK_ACTIVE
    if _UAX_PLAYBACK_ACTIVE:
        return
    _UAX_PLAYBACK_ACTIVE = True
    try:
        frame = float(getattr(scene, "frame_current_final", getattr(scene, "frame_current", 1)))
        for obj in getattr(bpy.data, "objects", []):
            if getattr(obj, "type", "") != "ARMATURE":
                continue
            animation_data = getattr(obj, "animation_data", None)
            action = getattr(animation_data, "action", None) if animation_data is not None else None
            apply_uax_matrix_basis_action(obj, action, frame)
    finally:
        _UAX_PLAYBACK_ACTIVE = False


def ensure_uax_playback_handler():
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    if handlers is None:
        return
    frame_change_post = getattr(handlers, "frame_change_post", None)
    if frame_change_post is None:
        return
    for handler in list(frame_change_post):
        if getattr(handler, "__name__", "") == hwek_uax_matrix_basis_playback.__name__:
            frame_change_post.remove(handler)
    frame_change_post.append(hwek_uax_matrix_basis_playback)


def remove_uax_playback_handler():
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    frame_change_post = getattr(handlers, "frame_change_post", None) if handlers is not None else None
    if frame_change_post is None:
        return
    for handler in list(frame_change_post):
        if getattr(handler, "__name__", "") == hwek_uax_matrix_basis_playback.__name__:
            frame_change_post.remove(handler)
