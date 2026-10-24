"""Bind native animation tracks to editable Blender armature actions."""

from __future__ import annotations
from dataclasses import dataclass, field
from mathutils import Matrix, Quaternion, Vector
from typing import Any, Dict, List, Optional, Tuple
import bpy
import hashlib
import json
import math
import os
from time import perf_counter
import zlib
import copy
from bisect import bisect_left
from ..errors import UgxError
from .binding import granny_model_group_index, granny_track_index
from .auxiliary import AUXILIARY_SOURCE, capture_uax_auxiliary_source
from ..formats.granny import GrannyTransform, GrannyAnimationInfo, GrannyCurveInfo, GrannyTrackGroupInfo, GrannyTransformTrackInfo
from ..formats.matrix import mat3_mul, mat4_inverse, mat4_mul, uax_matrix3_identity, uax_matrix4_identity
from ..ugx.transforms import bone_rest_display_rows, game_rows_to_blender_matrix, parse_matrix_rows
from ..ugx.scene import ugx_ordered_armature_bones
from .actions import (
    blender_data_path_string,
    uax_action_signature,
    uax_fcurve_signature,
    uax_armature_signature,
    uax_iter_action_fcurves,
    uax_animation_name_for_bone,
    uax_source_name_for_bone,
    uax_track_action_signatures,
)
from .playback import apply_uax_matrix_basis_action, ensure_uax_playback_handler
from .uax import UaxFile, parse_uax
from .placement import PLACEMENT_PROPERTY, native_uax_group_placement
from .curve_fitting import fit_bezier_steps, fit_bezier_batch_steps, native_bezier_steps, fit_native_candidate_steps, float32_curve_error
from .batch_sampling import NativePoseBatchSampler, channel_fit_errors, channel_matrices
from .keyframes import (
    uax_frame, uax_reduce_scalar_keys, uax_bezier_channel_keys, uax_bezier_transform_keys,
    uax_pulse_channel, add_uax_pulse_envelope,
)
from .channels import UaxSampleKeys, prepare_uax_track, sample_uax_keys, uax_curve_break_times
from .local_sampling import local_channel_sampler, local_matrix_channels_representable
from ..formats.granny_transform import granny_inverse_transform, granny_multiply_transforms, granny_retarget_transform, granny_transform_rows
from ..ugx.transforms import granny_transform_from_local_rows

__all__ = ('UaxArmatureBoneBinding', 'uax_frame', 'uax_reduce_scalar_keys', 'add_uax_fcurve', 'uax_target_armature', 'assign_uax_action', 'uax_armature_bindings', 'uax_binding_depth', 'uax_binding_receives_export_axis', 'uax_binding_for_track', 'uax_track_key_count', 'uax_all_key_times', 'uax_animation_sample_times', 'uax_find_key_interval', 'uax_eval_vec3', 'uax_eval_quaternion', 'uax_eval_mat3', 'uax_quaternion_to_game_rows', 'uax_track_local_rows', 'build_uax_sample_world_rows', 'uax_view_layer_update', 'uax_reset_pose_basis', 'uax_set_pose_sample', 'uax_matrix_values', 'uax_matrix_delta', 'uax_loc_rot_scale_matrix', 'uax_continuous_quaternion_values', 'bake_uax_pose_transform_action', 'bake_uax_matrix_basis_action', 'bake_uax_matrix_basis_steps', 'uax_curve_metadata', 'uax_track_group_metadata', 'set_uax_action_fake_user', 'expand_uax_action_channels', 'set_uax_action_metadata', 'focus_uax_armature_for_editing', 'uax_action_name', 'activate_uax_action', 'import_uax', 'uax_context_armature')

@dataclass
class UaxArmatureBoneBinding:
    source_name: str
    blender_name: str
    parent_source_name: str = ""
    rest_world_rows: List[List[float]] = field(default_factory=list)
    rest_local_rows: List[List[float]] = field(default_factory=list)
    import_world_rows: List[List[float]] = field(default_factory=list)
    rest_transform: Optional[GrannyTransform] = None
    parent: Optional[UaxArmatureBoneBinding] = field(default=None, repr=False, compare=False)



def uax_tracks_for_armature(animation, armature, bindings_by_source, native_binding=False, *, bound_group_index=None):
    groups = animation.track_groups
    if not groups:
        return ([] if native_binding else animation.transform_tracks), -1
    wanted = armature.get('hwek_ugx_granny_model_name')
    if wanted is None and native_binding:
        # Fresh UGX exports use the first authored bone as their model name.
        bones = ugx_ordered_armature_bones(armature)
        wanted = uax_source_name_for_bone(bones[0]) if bones else None
    native = granny_model_group_index(groups, str(wanted)) if wanted is not None else -1
    if native_binding and bound_group_index is not None:
        native = int(bound_group_index)
        if native != -1 and (not 0 <= native < len(groups) or groups[native].flags & 8):
            raise UgxError(f'Animation {animation.name}: its imported track group is unavailable.')
    if native_binding and native < 0:
        return [], -1
    candidates = [index for index, group in enumerate(groups) if not group.flags & 8]
    if native >= 0:
        index = native
    elif len(candidates) == 1:
        index = candidates[0]
    else:
        scores = {index: sum(uax_binding_for_track(track.name, bindings_by_source) is not None
                             for track in groups[index].transform_tracks) for index in candidates}
        best = max(scores.values(), default=0)
        candidates = [index for index, score in scores.items() if score == best and score > 0]
        if len(candidates) != 1:
            names = ', '.join(group.name for group in groups)
            raise UgxError(f'Animation {animation.name}: cannot choose a track group for armature {armature.name} '
                           f'from {names}. Use the matching model or retarget the animation first.')
        index = candidates[0]
    group = groups[index]
    names = [track.name for track in group.transform_tracks]
    tracks = [track for track_index, track in enumerate(group.transform_tracks)
              if granny_track_index(names, track.name, group.flags) == track_index]
    return tracks, index



def _uax_action_curve_collection(action, datablock):
    legacy = getattr(action, 'fcurves', None)
    if legacy is not None:
        return legacy
    from bpy_extras.anim_utils import action_ensure_channelbag_for_slot, action_get_first_suitable_slot
    slot = action_get_first_suitable_slot(action, 'OBJECT')
    if slot is None:
        slot = action.slots.new(id_type='OBJECT', name=datablock.name)
    return action_ensure_channelbag_for_slot(action, slot).fcurves


def add_uax_fcurve(action: bpy.types.Action, data_path: str, index: int, keys: List[Tuple[float, float]], group: str, datablock=None, handles=None, interpolations=None, curve_collection=None) -> bool:
    if not keys:
        return False
    keys = uax_reduce_scalar_keys(keys) if handles is None else keys
    fcurve = None
    fcurves = getattr(action, "fcurves", None)
    if curve_collection is not None and fcurves is None:
        fcurve = curve_collection.find(data_path, index=index)
        if fcurve is None:
            fcurve = curve_collection.new(data_path, index=index, group_name=group)
    elif fcurves is not None:
        try:
            fcurve = fcurves.new(data_path=data_path, index=index, action_group=group)
        except TypeError:
            fcurve = fcurves.new(data_path=data_path, index=index)
    else:
        ensure = getattr(action, "fcurve_ensure_for_datablock", None)
        if callable(ensure) and datablock is not None:
            animation_data_create = getattr(datablock, "animation_data_create", None)
            if callable(animation_data_create):
                animation_data_create()
            animation_data = getattr(datablock, "animation_data", None)
            previous = animation_data.action if animation_data is not None else None
            slot = getattr(animation_data, 'action_slot', None)
            if animation_data is not None and previous is not action:
                animation_data.action = action
            try:
                try:
                    fcurve = ensure(datablock, data_path, index=index, group_name=group)
                except TypeError:
                    fcurve = ensure(datablock, data_path, index=index)
            finally:
                if animation_data is not None and previous is not action:
                    animation_data.action = previous
                    if previous is not None and slot is not None:
                        animation_data.action_slot = slot
    if fcurve is None:
        return False
    points = getattr(fcurve, "keyframe_points", None)
    if points is None:
        return False
    if len(points) == 0:
        try:
            points.add(len(keys))
            points.foreach_set("co", [component for frame, value in keys for component in (float(frame), float(value))])
        except Exception:
            for frame, value in keys:
                try:
                    points.insert(float(frame), float(value), options={"FAST"})
                except TypeError:
                    points.insert(float(frame), float(value))
    else:
        for frame, value in keys:
            try:
                points.insert(float(frame), float(value), options={"FAST"})
            except TypeError:
                points.insert(float(frame), float(value))
    interpolation_items = bpy.types.Keyframe.bl_rna.properties['interpolation'].enum_items
    values = ([interpolation_items[name].value for name in interpolations] if interpolations is not None else
              [interpolation_items['BEZIER' if handles is not None else 'LINEAR'].value] * len(points))
    points.foreach_set('interpolation', values)
    if handles is not None:
        free = [bpy.types.Keyframe.bl_rna.properties['handle_left_type'].enum_items['FREE'].value] * len(points)
        points.foreach_set('handle_left_type', free)
        points.foreach_set('handle_right_type', free)
    if handles is not None:
        points.foreach_set('handle_left', [v for left, _right in handles for v in left])
        points.foreach_set('handle_right', [v for _left, right in handles for v in right])
    if handles is not None:
        # These keys are already ordered and have explicit free handles.
        # FCurve.update also deletes nearby keys, including a loop reset.
        action.update_tag()
    else:
        update = getattr(fcurve, "update", None)
        if callable(update):
            update()
    return True


def uax_target_armature(target):
    if target is None:
        return None
    if getattr(target, "type", "") == "ARMATURE":
        return target
    parent = getattr(target, "parent", None)
    if parent is not None and getattr(parent, "type", "") == "ARMATURE":
        return parent
    for modifier in getattr(target, "modifiers", []):
        if getattr(modifier, "type", "") == "ARMATURE" and getattr(modifier, "object", None) is not None:
            return modifier.object
    return None


def assign_uax_action(target, action: bpy.types.Action, tracks: List[GrannyTransformTrackInfo]):
    target = uax_target_armature(target)
    if target is None:
        return
    pose_bones = getattr(getattr(target, "pose", None), "bones", None)
    if pose_bones is not None:
        for track in tracks:
            try:
                pose_bones[track.name].rotation_mode = "QUATERNION"
            except (KeyError, TypeError, AttributeError):
                continue
    animation_data_create = getattr(target, "animation_data_create", None)
    if callable(animation_data_create):
        animation_data_create()
    animation_data = getattr(target, "animation_data", None)
    if animation_data is not None:
        animation_data.action = action


def uax_armature_bindings(armature: bpy.types.Object, track_names_by_bone=None) -> Tuple[List[UaxArmatureBoneBinding], Dict[str, UaxArmatureBoneBinding]]:
    if armature is None or getattr(armature, "type", "") != "ARMATURE":
        return [], {}
    data_bones = list(getattr(getattr(armature, "data", None), "bones", []))
    rest_world_by_bone: Dict[str, List[List[float]]] = {}
    import_world_by_bone: Dict[str, List[List[float]]] = {}
    for bone in data_bones:
        current_rows = bone_rest_display_rows(armature, bone)
        source_stored = bone.get("hwek_ugx_source_display_matrix", "")
        import_stored = bone.get("hwek_ugx_import_display_matrix", "")
        try:
            rest_world_by_bone[bone.name] = parse_matrix_rows(source_stored, f"{bone.name} source display matrix") if source_stored else current_rows
        except UgxError:
            rest_world_by_bone[bone.name] = current_rows
        try:
            import_world_by_bone[bone.name] = parse_matrix_rows(import_stored, f"{bone.name} import display matrix") if import_stored else current_rows
        except UgxError:
            import_world_by_bone[bone.name] = current_rows
    bindings: List[UaxArmatureBoneBinding] = []
    by_source: Dict[str, UaxArmatureBoneBinding] = {}
    for bone in data_bones:
        source_name = uax_animation_name_for_bone(armature, bone, track_names_by_bone)
        parent = getattr(bone, "parent", None)
        parent_source_name = uax_animation_name_for_bone(armature, parent, track_names_by_bone) if parent is not None else ""
        rest_world = rest_world_by_bone.get(bone.name, uax_matrix4_identity())
        parent_world = rest_world_by_bone.get(parent.name) if parent is not None else None
        rest_local = mat4_mul(rest_world, mat4_inverse(parent_world)) if parent_world is not None else [row[:] for row in rest_world]
        import_world = import_world_by_bone.get(bone.name, rest_world)
        stored_transform = bone.get('hwek_ugx_granny_local_transform')
        rest_transform = GrannyTransform(*json.loads(stored_transform)) if stored_transform else None
        binding = UaxArmatureBoneBinding(source_name, bone.name, parent_source_name, rest_world, rest_local, import_world, rest_transform)
        bindings.append(binding)
        by_source.setdefault(source_name, binding)
        native_name = uax_source_name_for_bone(bone)
        by_source.setdefault(native_name, binding)
    by_blender = {binding.blender_name: binding for binding in bindings}
    for bone, binding in zip(data_bones, bindings):
        if bone.parent is not None:
            binding.parent = by_blender[bone.parent.name]
    return bindings, by_source


def uax_bone_track_names(armature, tracks):
    names = {track.name for track in tracks}
    return {bone.name: uax_source_name_for_bone(bone) for bone in armature.data.bones
            if uax_source_name_for_bone(bone) in names and
            uax_source_name_for_bone(bone) != uax_animation_name_for_bone(armature, bone)}


def uax_binding_depth(binding: UaxArmatureBoneBinding, by_source: Dict[str, UaxArmatureBoneBinding]) -> int:
    depth = 0
    seen = {binding.blender_name}
    parent = binding.parent or (by_source.get(binding.parent_source_name) if binding.parent_source_name else None)
    while parent is not None:
        if parent.blender_name in seen:
            break
        seen.add(parent.blender_name)
        parent = parent.parent or (by_source.get(parent.parent_source_name) if parent.parent_source_name else None)
        depth += 1
    return depth


def uax_binding_receives_export_axis(
    binding: UaxArmatureBoneBinding,
    by_source: Dict[str, UaxArmatureBoneBinding],
) -> bool:
    if binding.parent is None and not binding.parent_source_name:
        return binding.source_name.lower() != "grannyrootbone"
    parent = binding.parent or by_source.get(binding.parent_source_name)
    return bool(
        parent is not None
        and parent.parent is None and not parent.parent_source_name
        and parent.source_name.lower() == "grannyrootbone"
    )


def uax_binding_for_track(track_name: str, by_source: Dict[str, UaxArmatureBoneBinding]) -> Optional[UaxArmatureBoneBinding]:
    return by_source.get(track_name)


def uax_track_key_count(track: GrannyTransformTrackInfo) -> int:
    return len(track.rotation_keys) + len(track.position_keys) + len(track.scale_shear_keys or track.scale_keys)


def uax_all_key_times(animation: GrannyAnimationInfo) -> List[float]:
    times: List[float] = []
    for track in animation.transform_tracks:
        times.extend(time for time, _ in track.rotation_keys)
        times.extend(time for time, _ in track.position_keys)
        times.extend(time for time, _ in track.scale_shear_keys)
    return [time for time in times if math.isfinite(float(time)) and float(time) >= 0.0]


def uax_animation_sample_times(animation: GrannyAnimationInfo) -> List[float]:
    key_times = uax_all_key_times(animation)
    duration = float(animation.duration) if animation.duration and animation.duration > 0.0 else (max(key_times) if key_times else 0.0)
    samples = {0.0, max(0.0, duration)}
    # Native knots define spline intervals. Uniform keyframe curves already
    # carry their frame times; other channels refine only where the fit needs it.
    for time in key_times:
        samples.add(min(duration, max(0.0, float(time))))
    return sorted(samples)


def uax_find_key_interval(keys: List[Tuple[float, Any]], time: float) -> Tuple[Tuple[float, Any], Tuple[float, Any], float]:
    if not keys:
        return (0.0, None), (0.0, None), 0.0
    if isinstance(keys, UaxSampleKeys):
        ordered, times = keys.values, keys.times
    else:
        ordered = sorted(keys, key=lambda item: item[0])
        times = [item[0] for item in ordered]
    if time <= ordered[0][0]:
        return ordered[0], ordered[0], 0.0
    if time >= ordered[-1][0]:
        return ordered[-1], ordered[-1], 0.0
    index = bisect_left(times, time)
    left, right = ordered[index - 1], ordered[index]
    span = right[0] - left[0]
    alpha = 0.0 if abs(span) < 0.0000001 else (time - left[0]) / span
    return left, right, max(0.0, min(1.0, alpha))


def uax_eval_vec3(keys: List[Tuple[float, Tuple[float, float, float]]], time: float, default: Tuple[float, float, float], left_limit=False) -> Tuple[float, float, float]:
    if not keys:
        return default
    return sample_uax_keys(keys, time, left_limit=left_limit)


def uax_eval_quaternion(keys: List[Tuple[float, Tuple[float, float, float, float]]], time: float, left_limit=False) -> Tuple[float, float, float, float]:
    if not keys:
        return (1.0, 0.0, 0.0, 0.0)
    return sample_uax_keys(keys, time, normalize=True, left_limit=left_limit)



def uax_eval_mat3(keys: List[Tuple[float, Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]], time: float, left_limit=False) -> List[List[float]]:
    if not keys:
        return uax_matrix3_identity()
    values = sample_uax_keys(keys, time, matrix=True, left_limit=left_limit)
    return [list(values[row * 3:row * 3 + 3]) for row in range(3)]


def uax_quaternion_to_game_rows(value: Tuple[float, float, float, float]) -> List[List[float]]:
    quat = Quaternion(tuple(float(component) for component in value))
    if math.sqrt(quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z) <= 0.000001:
        return uax_matrix3_identity()
    quat.normalize()
    matrix = quat.to_matrix()
    return [[float(matrix[column][row]) for column in range(3)] for row in range(3)]


def uax_retarget_offsets(bindings, source_bones):
    source = {}
    for bone in source_bones:
        source.setdefault(bone.name, bone)
    offsets = {}
    for binding in bindings:
        source_bone = source.get(binding.source_name)
        if source_bone is None:
            continue
        target = binding.rest_transform or granny_transform_from_local_rows(binding.rest_local_rows)
        offsets[binding.blender_name] = granny_multiply_transforms(target, granny_inverse_transform(source_bone.transform))
    return offsets


def uax_sample_retargeted_transform(track, time, left_limit=False):
    rest = track.sample_rest
    rotation = uax_eval_quaternion(track.rotation_keys, time, left_limit) if track.rotation_keys else (rest.orientation[3], *rest.orientation[:3])
    sample = GrannyTransform(7, uax_eval_vec3(track.position_keys, time, rest.position, left_limit),
                            (*rotation[1:], rotation[0]),
                            tuple(tuple(row) for row in uax_eval_mat3(track.scale_shear_keys, time, left_limit)) if track.scale_shear_keys else rest.scale_shear)
    return granny_retarget_transform(sample, track.retarget_offset)


def uax_track_local_rows(track: GrannyTransformTrackInfo, time: float, rest_local_rows: List[List[float]], left_limit=False) -> List[List[float]]:
    if not isinstance(track.position_keys, UaxSampleKeys):
        track = prepare_uax_track(track)
    if getattr(track, 'retarget_offset', None) is not None:
        return granny_transform_rows(uax_sample_retargeted_transform(track, time, left_limit))
    if not track.position_keys and not track.rotation_keys and not track.scale_shear_keys:
        return [row[:] for row in rest_local_rows]
    position_default = tuple(float(component) for component in rest_local_rows[3][:3])
    position = uax_eval_vec3(track.position_keys, time, position_default, left_limit)
    if track.rotation_keys:
        rotation_rows = uax_quaternion_to_game_rows(uax_eval_quaternion(track.rotation_keys, time, left_limit))
    else:
        rotation_rows = [[float(rest_local_rows[row][column]) for column in range(3)] for row in range(3)]
    scale_shear = uax_eval_mat3(track.scale_shear_keys, time, left_limit)
    linear = mat3_mul([list(row) for row in zip(*scale_shear)], rotation_rows) if track.rotation_keys or track.scale_shear_keys else rotation_rows
    return [
        [linear[0][0], linear[0][1], linear[0][2], 0.0],
        [linear[1][0], linear[1][1], linear[1][2], 0.0],
        [linear[2][0], linear[2][1], linear[2][2], 0.0],
        [position[0], position[1], position[2], 1.0],
    ]


def uax_track_local_matrix(track, time, rest_local, left_limit=False):
    if not isinstance(track.position_keys, UaxSampleKeys):
        track = prepare_uax_track(track)
    if getattr(track, 'retarget_offset', None) is not None:
        return Matrix(granny_transform_rows(uax_sample_retargeted_transform(track, time, left_limit))).transposed()
    if not track.position_keys and not track.rotation_keys and not track.scale_shear_keys:
        return rest_local
    position = uax_eval_vec3(track.position_keys, time, rest_local.translation, left_limit)
    if track.rotation_keys:
        rotation = Quaternion(uax_eval_quaternion(track.rotation_keys, time, left_limit))
        if rotation.magnitude <= 0.000001:
            rotation = Quaternion()
        else:
            rotation.normalize()
        linear = rotation.to_matrix()
    else:
        linear = rest_local.to_3x3()
    if track.scale_shear_keys:
        linear = linear @ Matrix(uax_eval_mat3(track.scale_shear_keys, time, left_limit))
    matrix = linear.to_4x4()
    matrix.translation = Vector(position)
    return matrix


def build_uax_sample_world_rows(
    bindings: List[UaxArmatureBoneBinding],
    by_source: Dict[str, UaxArmatureBoneBinding],
    tracks_by_source: Dict[str, GrannyTransformTrackInfo],
    time: float,
) -> Dict[str, List[List[float]]]:
    worlds: Dict[str, List[List[float]]] = {}
    ordered = sorted(bindings, key=lambda item: uax_binding_depth(item, by_source))
    for binding in ordered:
        track = tracks_by_source.get(binding.source_name)
        local_rows = uax_track_local_rows(track, time, binding.rest_local_rows) if track is not None else [row[:] for row in binding.rest_local_rows]
        parent_rows = worlds.get(binding.parent.blender_name if binding.parent is not None else binding.parent_source_name)
        worlds[binding.blender_name] = mat4_mul(local_rows, parent_rows) if parent_rows is not None else local_rows
    return worlds


def uax_view_layer_update():
    view_layer = getattr(getattr(bpy, "context", None), "view_layer", None)
    update = getattr(view_layer, "update", None)
    if callable(update):
        update()


def uax_reset_pose_basis(armature: bpy.types.Object):
    pose_bones = getattr(getattr(armature, "pose", None), "bones", None)
    if pose_bones is None:
        return
    identity = Matrix.Identity(4)
    for pose_bone in pose_bones:
        pose_bone.matrix_basis = identity.copy()
    uax_view_layer_update()


def uax_set_pose_sample(
    armature: bpy.types.Object,
    matched_bindings: List[UaxArmatureBoneBinding],
    worlds: Dict[str, List[List[float]]],
    bindings_by_source: Optional[Dict[str, UaxArmatureBoneBinding]] = None,
):
    pose_bones = getattr(getattr(armature, "pose", None), "bones", None)
    if pose_bones is None:
        return
    if bindings_by_source is None:
        bindings_by_source = {}
        for binding in matched_bindings:
            bindings_by_source[binding.source_name] = binding
    by_depth: Dict[int, List[UaxArmatureBoneBinding]] = {}
    for binding in matched_bindings:
        by_depth.setdefault(uax_binding_depth(binding, bindings_by_source), []).append(binding)
    # Blender's pose-matrix setter needs evaluated parent matrices. Updating
    # once per hierarchy level preserves that contract while replacing the old
    # once-per-bone dependency-graph update with only a handful per sample.
    for depth in sorted(by_depth):
        for binding in by_depth[depth]:
            pose_bone = pose_bones.get(binding.blender_name)
            rows = worlds.get(binding.blender_name)
            if pose_bone is None or rows is None:
                continue
            source_world = game_rows_to_blender_matrix(rows)
            source_rest = game_rows_to_blender_matrix(binding.rest_world_rows)
            import_rest = game_rows_to_blender_matrix(binding.import_world_rows or binding.rest_world_rows)
            pose_bone.matrix = source_world @ source_rest.inverted_safe() @ import_rest
        uax_view_layer_update()


def uax_matrix_values(matrix: Matrix) -> List[float]:
    return [float(value) for row in matrix for value in row]


def uax_matrix_delta(left: Matrix, right: Matrix) -> float:
    return max(abs(float(left[row][column]) - float(right[row][column])) for row in range(4) for column in range(4))


def uax_loc_rot_scale_matrix(location: Tuple[float, float, float], rotation: Quaternion, scale: Tuple[float, float, float]) -> Matrix:
    loc = Vector(location)
    scl = Vector(scale)
    try:
        return Matrix.LocRotScale(loc, rotation, scl)
    except AttributeError:
        return Matrix.Translation(loc) @ rotation.to_matrix().to_4x4() @ Matrix.Diagonal((scl.x, scl.y, scl.z, 1.0))


def uax_continuous_quaternion_values(rotation: Quaternion, previous: Optional[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    quat = rotation.copy()
    if quat.magnitude == 0.0:
        quat = Quaternion((1.0, 0.0, 0.0, 0.0))
    else:
        quat.normalize()
    values = (float(quat.w), float(quat.x), float(quat.y), float(quat.z))
    if previous is not None and sum(previous[index] * values[index] for index in range(4)) < 0.0:
        values = tuple(-value for value in values)
    return values


def uax_editable_sample_times(times, fps, keep_end_break=False, breaks=()):
    # FCurve.update merges keys within 0.01 frames and replaces their handles.
    # Fit across closely spaced knots, retaining them as error-check locations.
    # Keep other native knots so linear corners do not require deep refinement.
    minimum = .011 / fps
    result = [times[0]]
    for time in times[1:-1]:
        if time-result[-1] > minimum and times[-1]-time > minimum:
            result.append(time)
    if keep_end_break and len(times) > 2 and times[-1]-times[-2] <= minimum and times[-2] != result[-1]:
        result.append(times[-2])
    if times[-1] != result[-1]:
        result.append(times[-1])
    return sorted(set(result) | set(breaks))


def uax_channel_matrix(values):
    rotation = Quaternion(values[3:7])
    if rotation.magnitude > 1e-12:
        rotation.normalize()
    else:
        rotation = Quaternion()
    return uax_loc_rot_scale_matrix(values[:3], rotation, values[7:10])


def uax_channel_fit_error(expected, actual):
    error = float32_curve_error(uax_matrix_values(uax_channel_matrix(expected)),
                                uax_matrix_values(uax_channel_matrix(actual)))
    if len(expected) > 10:
        error = max(error, float32_curve_error(expected[10:], actual[10:]))
    return error


def add_uax_native_channel(action, data_path, index, group, armature, spans, component, fps, curve_collection=None, prepared_keys=None):
    pulse = uax_pulse_channel(spans, component, fps)
    try:
        if pulse is None:
            keys, handles, interpolations = (prepared_keys if prepared_keys is not None else
                uax_bezier_channel_keys(spans, component, fps, jump_tolerance=1e-5))
        else:
            base, native, windows = pulse
            keys, handles, interpolations = uax_bezier_channel_keys(base, 0, fps, jump_tolerance=1e-5)
        if not add_uax_fcurve(action, data_path, index, keys, group, armature, handles, interpolations, curve_collection):
            return 0
        if pulse is not None:
            curve = next(curve for curve in uax_iter_action_fcurves(action)
                         if curve.data_path == data_path and curve.array_index == index)
            add_uax_pulse_envelope(curve, native, fps, windows)
            return len(curve.keyframe_points)
        return len(keys)
    except UgxError as exc:
        raise UgxError(f'Animation {action.name}, bone {group}, {data_path.rsplit(".", 1)[-1]}[{index}]: {exc}') from exc


def bake_uax_pose_transform_action(
    action: bpy.types.Action,
    basis_samples: Dict[str, List[Tuple[float, Matrix]]],
    matched_bindings: List[UaxArmatureBoneBinding],
    armature: bpy.types.Object,
    fitted_spans=None,
    fps=30.0,
) -> Dict[str, Any]:
    pose_bones = getattr(getattr(armature, "pose", None), "bones", None)
    fcurve_count = 0
    key_count = 0
    max_decompose_error = 0.0
    curve_collection = _uax_action_curve_collection(action, armature)
    for binding in matched_bindings:
        samples_for_bone = basis_samples.get(binding.blender_name, [])
        if not samples_for_bone:
            continue
        pose_bone = pose_bones.get(binding.blender_name) if pose_bones is not None else None
        if pose_bone is not None:
            pose_bone.rotation_mode = "QUATERNION"
        spans = fitted_spans.get(binding.blender_name, []) if fitted_spans is not None else []
        if spans:
            escaped_name = blender_data_path_string(binding.blender_name)
            transform_keys = uax_bezier_transform_keys(spans, fps, jump_tolerance=1e-5)
            for channel, count, offset in (('location', 3, 0), ('rotation_quaternion', 4, 3), ('scale', 3, 7)):
                for index in range(count):
                    data_path = f'pose.bones["{escaped_name}"].{channel}'
                    added_keys = add_uax_native_channel(action, data_path, index, binding.blender_name,
                                                       armature, spans, offset+index, fps, curve_collection, transform_keys[offset+index])
                    if added_keys:
                        fcurve_count += 1
                        key_count += added_keys
            continue
        location_keys: List[List[Tuple[float, float]]] = [[] for _ in range(3)]
        rotation_keys: List[List[Tuple[float, float]]] = [[] for _ in range(4)]
        scale_keys: List[List[Tuple[float, float]]] = [[] for _ in range(3)]
        previous_rotation: Optional[Tuple[float, float, float, float]] = None
        for frame, matrix in samples_for_bone:
            location, rotation, scale = matrix.decompose()
            location_values = (float(location.x), float(location.y), float(location.z))
            rotation_values = uax_continuous_quaternion_values(rotation, previous_rotation)
            scale_values = (float(scale.x), float(scale.y), float(scale.z))
            previous_rotation = rotation_values
            max_decompose_error = max(max_decompose_error, uax_matrix_delta(matrix, uax_loc_rot_scale_matrix(location_values, Quaternion(rotation_values), scale_values)))
            for index, value in enumerate(location_values):
                location_keys[index].append((frame, value))
            for index, value in enumerate(rotation_values):
                rotation_keys[index].append((frame, value))
            for index, value in enumerate(scale_values):
                scale_keys[index].append((frame, value))
        escaped_name = blender_data_path_string(binding.blender_name)
        group = binding.blender_name
        for data_path, values_by_index in (
            (f'pose.bones["{escaped_name}"].location', location_keys),
            (f'pose.bones["{escaped_name}"].rotation_quaternion', rotation_keys),
            (f'pose.bones["{escaped_name}"].scale', scale_keys),
        ):
            for index, keys in enumerate(values_by_index):
                if add_uax_fcurve(action, data_path, index, keys, group, armature, curve_collection=curve_collection):
                    fcurve_count += 1
                    key_count += len(uax_reduce_scalar_keys(keys))
    return {
        "fcurve_count": fcurve_count,
        "key_count": key_count,
        "max_decompose_error": max_decompose_error,
    }


def bake_uax_matrix_basis_steps(
    action: Optional[bpy.types.Action],
    animation: GrannyAnimationInfo,
    armature: bpy.types.Object,
    fps: float,
    editable_action: Optional[bpy.types.Action] = None,
    source_bones=(),
    looping=False,
    native_binding=False,
) -> Dict[str, Any]:
    wrap_boundaries = bool(looping and animation.accumulation_flags & 5 and animation.duration > 0)
    bindings, bindings_by_source = uax_armature_bindings(armature)
    # Deferred preparation must keep the same group as an eager import, even
    # when the model's export identity changes before its curves are loaded.
    bound_group_index = (editable_action.get('hwek_uax_target_group_index')
                         if native_binding and editable_action is not None and editable_action.get('hwek_uax_pending') else None)
    source_tracks, group_index = uax_tracks_for_armature(animation, armature, bindings_by_source, native_binding,
                                                       bound_group_index=bound_group_index)
    track_names_by_bone = uax_bone_track_names(armature, source_tracks)
    if track_names_by_bone:
        bindings, bindings_by_source = uax_armature_bindings(armature, track_names_by_bone)
    offsets = uax_retarget_offsets(bindings, source_bones) if not native_binding or group_index >= 0 else {}
    source_rest = {}
    for bone in source_bones:
        source_rest.setdefault(bone.name, bone.transform)
    tracks_by_bone: Dict[str, GrannyTransformTrackInfo] = {}
    track_breaks = {}
    matched_bindings: List[UaxArmatureBoneBinding] = []
    matched_tracks: List[str] = []
    missing_tracks: List[str] = []
    targets_by_name = {}
    for binding in bindings:
        targets_by_name.setdefault(binding.source_name, []).append(binding)
    for track in source_tracks:
        targets = targets_by_name.get(track.name, [])
        if not targets:
            alias = uax_binding_for_track(track.name, bindings_by_source)
            targets = [alias] if alias is not None else []
        if not targets:
            missing_tracks.append(track.name)
            continue
        matched_tracks.append(track.name)
        for binding in targets:
            prepared = prepare_uax_track(track, animation.duration, wrap_boundaries)
            track_breaks[binding.blender_name] = set().union(*(uax_curve_break_times(keys) for keys in
                (prepared.position_keys, prepared.rotation_keys, prepared.scale_shear_keys)))
            if binding.blender_name in offsets:
                prepared.retarget_offset = offsets[binding.blender_name]
                prepared.sample_rest = source_rest[binding.source_name]
            tracks_by_bone[binding.blender_name] = prepared
            matched_bindings.append(binding)
    for binding in bindings:
        if (native_binding or binding.blender_name in offsets) and binding.blender_name not in tracks_by_bone:
            prepared = prepare_uax_track(GrannyTransformTrackInfo(binding.source_name))
            if binding.blender_name in offsets:
                prepared.retarget_offset = offsets[binding.blender_name]
                prepared.sample_rest = source_rest[binding.source_name]
            tracks_by_bone[binding.blender_name] = prepared
            matched_bindings.append(binding)
    matched_bindings = sorted(matched_bindings, key=lambda item: uax_binding_depth(item, bindings_by_source))
    sampled_animation = copy.copy(animation)
    sampled_animation.transform_tracks = [track for track in source_tracks
                                          if uax_binding_for_track(track.name, bindings_by_source) is not None]
    samples = uax_animation_sample_times(sampled_animation)
    basis_samples: Dict[str, List[Tuple[float, Matrix]]] = {binding.blender_name: [] for binding in matched_bindings}
    sample_bones = []
    axis = Matrix(((-1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    bone_local = Matrix.Diagonal((-1, 1, 1, 1))
    for binding in sorted(bindings, key=lambda item: uax_binding_depth(item, bindings_by_source)):
        bone = armature.data.bones.get(binding.blender_name)
        if bone is None:
            continue
        correction = None
        if binding.blender_name in basis_samples:
            source_rest = game_rows_to_blender_matrix(binding.rest_world_rows)
            import_rest = game_rows_to_blender_matrix(binding.import_world_rows or binding.rest_world_rows)
            correction = bone_local @ source_rest.inverted_safe() @ import_rest
        track = tracks_by_bone.get(binding.blender_name)
        rest_local = Matrix(binding.rest_local_rows).transposed()
        constant = track is None or all(len(keys) <= 1 for keys in (track.position_keys, track.rotation_keys, track.scale_shear_keys))
        if constant and track is not None:
            rest_local = uax_track_local_matrix(track, 0.0, rest_local)
        sample_bones.append((binding, bone, correction, track if not constant else None, rest_local))
    fast_samplers, fast_sample_times, fast_parents, corrections = {}, {}, {}, {}
    projected_samplers = {}
    max_channel_decompose_error = 0.0
    for binding, bone, _correction, track, rest_local in sample_bones:
        source_rest = game_rows_to_blender_matrix(binding.rest_world_rows)
        import_rest = game_rows_to_blender_matrix(binding.import_world_rows or binding.rest_world_rows)
        right = bone_local @ source_rest.inverted_safe() @ import_rest
        corrections[bone.name] = right
        parent = bone.parent
        eligible = (bone.use_inherit_rotation and bone.inherit_scale == 'FULL' and bone.use_local_location
                    and (parent is None or fast_parents.get(parent.name, False)))
        left = bone.matrix_local.inverted_safe() @ (parent.matrix_local @ corrections[parent.name].inverted_safe() if parent is not None else axis)
        sampler = local_channel_sampler(left, right, track, rest_local, action is not None) if eligible else None
        if sampler is None and action is None and bone.use_inherit_rotation and bone.inherit_scale == 'FULL' and bone.use_local_location:
            projected = local_channel_sampler(left, right, track, rest_local, project_basis=True)
            if projected is not None:
                projected_samplers[bone.name] = projected.batch
        faithful_parent = sampler is not None
        if sampler is None and eligible:
            faithful_parent = local_matrix_channels_representable(left, right, track, rest_local)
            left_rows, right_rows = [list(row) for row in left], [list(row) for row in right]
            rest_rows = [list(row) for row in rest_local.transposed()]
            # Work locally in double precision before creating Blender values.
            # World-space cancellation loses small motion on distant bones.
            def sampler(time, left_limit=False, left_rows=left_rows, right_rows=right_rows,
                        rest_rows=rest_rows, track=track):
                nonlocal max_channel_decompose_error
                rows = uax_track_local_rows(track, time, rest_rows, left_limit) if track is not None else rest_rows
                native_local = [list(row) for row in zip(*rows)]
                basis = Matrix(mat4_mul(mat4_mul(left_rows, native_local), right_rows))
                location, rotation, scale = basis.decompose()
                rotation.normalize()
                matrix = uax_loc_rot_scale_matrix(location, rotation, scale)
                max_channel_decompose_error = max(max_channel_decompose_error, uax_matrix_delta(basis, matrix))
                values = tuple(location) + tuple(rotation) + tuple(scale)
                return values + (tuple(uax_matrix_values(matrix)) if action is not None else ())
            if track is None:
                constant = sampler(0.0)
                sampler = lambda _time, left_limit=False, constant=constant: constant
        if sampler is not None and _correction is None and uax_matrix_delta(left @ rest_local @ right, Matrix.Identity(4)) > 1e-5:
            sampler = None
        fast_parents[bone.name] = sampler is not None and faithful_parent
        if sampler is not None and binding.blender_name in basis_samples:
            fast_samplers[binding.blender_name] = sampler
            times = {samples[0], samples[-1]}
            if track is not None:
                for keys in (track.position_keys, track.rotation_keys, track.scale_shear_keys):
                    times.update(min(samples[-1], max(0., time)) for time in keys.times)
            fast_sample_times[binding.blender_name] = sorted(times)
    batch_sampler = NativePoseBatchSampler.for_rig(sample_bones, corrections, axis) if action is None else None
    batch_samplers = {name: sampler.batch for name, sampler in fast_samplers.items()
        if hasattr(sampler, 'batch')} if action is None else {}
    local_batch_names = set(batch_samplers)
    if batch_sampler is not None:
        for binding in matched_bindings:
            name = binding.blender_name
            if name not in batch_samplers and name not in batch_sampler.excluded:
                batch_samplers[name] = lambda times, left_limit=False, name=name: batch_sampler.sample(name, times, left_limit)
    batch_values = {}
    batch_basis_deltas = {}
    sample_bones_by_target = {}
    slow_sample_times = {}
    for binding in matched_bindings:
        if binding.blender_name in fast_samplers and binding.blender_name not in batch_samplers:
            continue
        required = {binding.blender_name}
        for ancestor, bone, _correction, _track, _rest in reversed(sample_bones):
            if ancestor.blender_name in required:
                required.add(ancestor.parent.blender_name if ancestor.parent is not None else ancestor.parent_source_name)
                if bone.parent is not None:
                    required.add(bone.parent.name)
        records = [record for record in sample_bones if record[0].blender_name in required]
        sample_bones_by_target[binding.blender_name] = records
        times = {samples[0], samples[-1]}
        for _ancestor, _bone, _correction, track, _rest in records:
            if track is not None:
                for keys in (track.position_keys, track.rotation_keys, track.scale_shear_keys):
                    times.update(min(samples[-1], max(0., time)) for time in keys.times)
        slow_sample_times[binding.blender_name] = sorted(times)
    sample_cache = {}
    inherited_breaks = {}
    for binding, _bone, _correction, _track, _rest in sample_bones:
        inherited_breaks[binding.blender_name] = (track_breaks.get(binding.blender_name, set()) |
            inherited_breaks.get(binding.parent.blender_name if binding.parent is not None else binding.parent_source_name, set()))
    def sample_pose(sample_time, target_name, left_limit=False):
        nonlocal max_channel_decompose_error
        cache_key = (sample_time, left_limit)
        cached = sample_cache.get(cache_key)
        if cached is None:
            worlds, pose_matrices, channels = {}, {}, {}
            sample_cache[cache_key] = (worlds, pose_matrices, channels)
        else:
            worlds, pose_matrices, channels = cached
            if target_name in channels:
                return channels
        records = sample_bones_by_target[target_name]
        for binding, bone, correction, track, rest_local in records:
            if bone.name in pose_matrices:
                continue
            local = uax_track_local_matrix(track, sample_time, rest_local, left_limit) if track is not None else rest_local
            parent_world = worlds.get(binding.parent.blender_name if binding.parent is not None else binding.parent_source_name)
            world = (parent_world if parent_world is not None else axis) @ local
            worlds[binding.blender_name] = world
            parent = bone.parent
            parent_matrices = ({'parent_matrix': pose_matrices[parent.name], 'parent_matrix_local': parent.matrix_local}
                               if parent is not None else {})
            basis = Matrix.Identity(4)
            if correction is not None:
                target = world @ correction
                basis = bone.convert_local_to_pose(target, bone.matrix_local, invert=True, **parent_matrices)
                location, rotation, scale = basis.decompose()
                rotation.normalize()
                channel_basis = uax_loc_rot_scale_matrix(location, rotation, scale)
                max_channel_decompose_error = max(max_channel_decompose_error, uax_matrix_delta(basis, channel_basis))
                basis = channel_basis
                values = tuple(location) + tuple(rotation) + tuple(scale)
                channels[binding.blender_name] = values + (tuple(uax_matrix_values(basis)) if action is not None else ())
            pose_matrices[bone.name] = bone.convert_local_to_pose(basis, bone.matrix_local, **parent_matrices)
        return channels
    fitted_spans = {}
    deadline = perf_counter() + 0.008
    if matched_bindings and samples:
        slow_bindings = [binding for binding in matched_bindings
            if binding.blender_name in batch_samplers or binding.blender_name not in fast_samplers]
        for binding in slow_bindings:
            times = (fast_sample_times[binding.blender_name] if binding.blender_name in local_batch_names
                else slow_sample_times[binding.blender_name])
            sampler = batch_samplers.get(binding.blender_name)
            if sampler is not None:
                values = sampler(times)
                batch_values[binding.blender_name] = values
                matrices = channel_matrices(values).astype('float32')
                batch_basis_deltas[binding.blender_name] = float(abs(matrices-matrices[0]).max())
                basis_samples[binding.blender_name] = [(uax_frame(times[index], fps), Matrix(matrices[index].tolist()))
                    for index in ((0, len(times)-1) if len(times) > 1 else (0,))]
                if perf_counter() >= deadline:
                    yield
                    deadline = perf_counter() + 0.008
                continue
            else:
                values = [sample_pose(time, binding.blender_name)[binding.blender_name] for time in times]
            for sample_time, value in zip(times, values):
                basis_samples[binding.blender_name].append((uax_frame(sample_time, fps), uax_channel_matrix(value)))
                if perf_counter() >= deadline:
                    yield
                    deadline = perf_counter() + 0.008
        for binding in matched_bindings:
            sampler = (fast_samplers.get(binding.blender_name)
                if binding.blender_name not in batch_samplers else None)
            if sampler is not None:
                for sample_time in fast_sample_times[binding.blender_name]:
                    basis_samples[binding.blender_name].append((uax_frame(sample_time, fps), uax_channel_matrix(sampler(sample_time))))
            if perf_counter() >= deadline:
                yield
                deadline = perf_counter() + 0.008
        for binding in matched_bindings:
            spans = []
            sampler = fast_samplers.get(binding.blender_name)
            use_batch = binding.blender_name in batch_samplers
            if use_batch:
                sampler = batch_samplers.get(binding.blender_name)
                if binding.blender_name in local_batch_names:
                    breaks = track_breaks.get(binding.blender_name, set())
                    times = fast_sample_times[binding.blender_name]
                else:
                    breaks = inherited_breaks.get(binding.blender_name, set())
                    times = slow_sample_times[binding.blender_name]
                last_values = batch_values[binding.blender_name][-2:]
                keep_end_break = len(times) > 2 and channel_fit_errors(last_values[:1], last_values[1:])[0] > 1e-5
            elif sampler is None:
                sampler = lambda time, left_limit=False: sample_pose(time, binding.blender_name, left_limit)[binding.blender_name]
                breaks = inherited_breaks.get(binding.blender_name, set())
                times = slow_sample_times[binding.blender_name]
            else:
                breaks = track_breaks.get(binding.blender_name, set())
                times = fast_sample_times[binding.blender_name]
            if not use_batch:
                keep_end_break = len(times) > 2 and uax_channel_fit_error(sampler(times[-2]), sampler(times[-1])) > 1e-5
            check_times = set(times)
            for a, b in zip(times, times[1:]):
                check_times.update(a+(b-a)*ratio for ratio in (.1464466094067262, .5, .8535533905932737))
            editable_times = uax_editable_sample_times(times, fps, keep_end_break, breaks)
            if use_batch and getattr(sampler, 'polynomial', False) and editable_times == times:
                fitting = native_bezier_steps(times, sampler, channel_fit_errors)
            elif use_batch and binding.blender_name in projected_samplers:
                fitting = fit_native_candidate_steps(editable_times, projected_samplers[binding.blender_name], sampler,
                    channel_fit_errors, check_times, breaks)
            else:
                fitting = (fit_bezier_batch_steps(editable_times, sampler, channel_fit_errors, check_times, breaks)
                    if use_batch else fit_bezier_steps(editable_times, sampler,
                    uax_channel_fit_error, (3,), check_times=check_times,
                    sample_left=lambda time: sampler(time, True) if time in breaks else None, native_clock=True))
            for span in fitting:
                if span is not None:
                    spans.append(span)
                if perf_counter() >= deadline:
                    yield
                    deadline = perf_counter() + 0.008
            fitted_spans[binding.blender_name] = spans
        if batch_sampler is not None:
            max_channel_decompose_error = max(max_channel_decompose_error, batch_sampler.max_decompose_error)
    fcurve_count = 0
    key_count = 0
    max_basis_delta = 0.0
    curve_collection = _uax_action_curve_collection(action, armature) if action is not None else None
    for binding in matched_bindings:
        if perf_counter() >= deadline:
            yield
            deadline = perf_counter() + 0.008
        samples_for_bone = basis_samples.get(binding.blender_name, [])
        if not samples_for_bone:
            continue
        first_values = uax_matrix_values(samples_for_bone[0][1])
        if binding.blender_name in batch_basis_deltas:
            max_basis_delta = max(max_basis_delta, batch_basis_deltas[binding.blender_name])
        else:
            for _, matrix in samples_for_bone[1:]:
                values = uax_matrix_values(matrix)
                max_basis_delta = max(max_basis_delta, max(abs(values[index] - first_values[index]) for index in range(16)))
        if action is not None:
            data_path = f'pose.bones["{blender_data_path_string(binding.blender_name)}"].matrix_basis'
            group = binding.blender_name
            values_by_index: List[List[Tuple[float, float]]] = [[] for _ in range(16)]
            for frame, matrix in samples_for_bone:
                values = uax_matrix_values(matrix)
                for index, value in enumerate(values):
                    values_by_index[index].append((frame, value))
            for index, keys in enumerate(values_by_index):
                spans = fitted_spans.get(binding.blender_name, [])
                if spans:
                    added_keys = add_uax_native_channel(action, data_path, index, group, armature, spans, 10+index, fps, curve_collection)
                elif add_uax_fcurve(action, data_path, index, keys, group, armature):
                    added_keys = len(uax_reduce_scalar_keys(keys))
                else:
                    added_keys = 0
                if added_keys:
                    fcurve_count += 1
                    key_count += added_keys
    editable_bake = {"fcurve_count": 0, "key_count": 0, "max_decompose_error": 0.0}
    if editable_action is not None:
        for binding in matched_bindings:
            result = bake_uax_pose_transform_action(editable_action, basis_samples, [binding], armature, fitted_spans, fps)
            editable_bake['fcurve_count'] += result['fcurve_count']
            editable_bake['key_count'] += result['key_count']
            editable_bake['max_decompose_error'] = max(editable_bake['max_decompose_error'], result['max_decompose_error'])
            if perf_counter() >= deadline:
                yield
                deadline = perf_counter() + 0.008
    return {
        "key_count": editable_bake["key_count"] if editable_action is not None else key_count,
        "fcurve_count": editable_bake["fcurve_count"] if editable_action is not None else fcurve_count,
        "matched_tracks": matched_tracks,
        "target_group_index": group_index,
        "looping": bool(looping),
        "track_names_by_bone": track_names_by_bone,
        "missing_tracks": missing_tracks,
        "sample_count": len(samples),
        "decoded_key_count": sum(uax_track_key_count(track) for track in source_tracks),
        "matrix_basis_fcurves": fcurve_count,
        "matrix_basis_key_count": key_count,
        "editable_fcurve_count": editable_bake["fcurve_count"],
        "editable_key_count": editable_bake["key_count"],
        "editable_max_decompose_error": max(max_channel_decompose_error, editable_bake["max_decompose_error"]),
        "max_basis_delta": max_basis_delta,
        "mode": "pose_trs" if editable_action is not None else "matrix_basis",
    }


def bake_uax_matrix_basis_action(action, animation, armature, fps, editable_action=None, source_bones=(), looping=False, native_binding=False):
    steps = bake_uax_matrix_basis_steps(action, animation, armature, fps, editable_action, source_bones, looping, native_binding)
    while True:
        try:
            next(steps)
        except StopIteration as result:
            return result.value


def uax_curve_metadata(curve: Optional[GrannyCurveInfo]) -> Dict[str, Any]:
    if curve is None:
        return {}
    return {
        "type": curve.type_name,
        "type_offset": curve.type_offset,
        "object_offset": curve.object_offset,
        "format": curve.header_format,
        "degree": curve.header_degree,
    }


def uax_track_group_metadata(groups: List[GrannyTrackGroupInfo]) -> List[Dict[str, Any]]:
    return [
        {
            "name": group.name,
            "flags": group.flags,
            "loop_translation": list(group.loop_translation),
            "periodic_loop_offset": group.periodic_loop_offset,
            "extended_data_type_offset": group.extended_data_type_offset,
            "extended_data_object_offset": group.extended_data_object_offset,
            "vector_tracks": [
                {
                    "name": track.name,
                    "track_key": track.track_key,
                    "dimension": track.dimension,
                    "curve": uax_curve_metadata(track.value_curve),
                }
                for track in group.vector_tracks
            ],
            "transform_lod_errors": group.transform_lod_errors,
            "text_tracks": [
                {
                    "name": track.name,
                    "entries": [
                        {"time": entry.time_stamp, "text": entry.text}
                        for entry in track.entries
                    ],
                }
                for track in group.text_tracks
            ],
        }
        for group in groups
    ]


def set_uax_action_fake_user(action: bpy.types.Action):
    try:
        action.use_fake_user = True
    except Exception:
        pass


def expand_uax_action_channels(action: Optional[bpy.types.Action]):
    if action is None:
        return
    for fcurve in uax_iter_action_fcurves(action):
        group = getattr(fcurve, "group", None)
        if group is None:
            continue
        for attr in ("show_expanded", "show_expanded_graph"):
            try:
                setattr(group, attr, True)
            except Exception:
                pass


def set_uax_action_metadata(
    action: bpy.types.Action,
    uax: UaxFile,
    animation: GrannyAnimationInfo,
    bake: Dict[str, Any],
    mode: str,
    fcurve_count: int,
    key_count: int,
    matrix_source_action: str = "",
    editable_action: str = "",
):
    set_uax_action_fake_user(action)
    action["hwek_uax_source_path"] = uax.path
    action["hwek_uax_source_sha256"] = hashlib.sha256(uax.original_bytes).hexdigest()
    action["hwek_uax_duration"] = animation.duration
    action["hwek_uax_time_step"] = animation.time_step
    action["hwek_uax_oversampling"] = animation.oversampling
    action['hwek_uax_accumulation_flags'] = animation.accumulation_flags
    action['hwek_uax_default_loop_count'] = animation.default_loop_count
    action['hwek_uax_animation_flags'] = animation.flags
    action["hwek_uax_fps"] = (1.0 / animation.time_step) if animation.time_step and animation.time_step > 0.0 else 30.0
    action.use_frame_range = True
    action.frame_start = 1
    action.frame_end = max(1, uax_frame(animation.duration, action['hwek_uax_fps']))
    action["hwek_uax_key_count"] = key_count
    action["hwek_uax_decoded_key_count"] = bake["decoded_key_count"]
    action["hwek_uax_fcurve_count"] = fcurve_count
    action["hwek_uax_fcurve_mode"] = mode
    action["hwek_uax_baked_frame_count"] = bake["sample_count"]
    action['hwek_uax_baked_looping'] = bake.get('looping', False)
    if bake.get('track_names_by_bone'):
        action['hwek_uax_track_names_by_bone'] = json.dumps(bake['track_names_by_bone'], sort_keys=True)
    action["hwek_uax_matrix_basis_fcurves"] = bake.get("matrix_basis_fcurves", 0)
    action["hwek_uax_matrix_basis_key_count"] = bake.get("matrix_basis_key_count", 0)
    action["hwek_uax_editable_fcurves"] = bake.get("editable_fcurve_count", 0)
    action["hwek_uax_editable_key_count"] = bake.get("editable_key_count", 0)
    action["hwek_uax_editable_decompose_error"] = bake.get("editable_max_decompose_error", 0.0)
    action["hwek_uax_max_basis_delta"] = bake.get("max_basis_delta", 0.0)
    action["hwek_uax_matched_tracks"] = json.dumps(bake["matched_tracks"])
    action["hwek_uax_missing_tracks"] = json.dumps(bake["missing_tracks"])
    action["hwek_uax_skipped_track_count"] = len(bake["missing_tracks"])
    action["hwek_uax_track_groups"] = json.dumps(animation.track_group_names)
    action["hwek_uax_track_groups_metadata"] = json.dumps(uax_track_group_metadata(animation.track_groups or uax.granny_info.track_groups))
    placement_groups = animation.track_groups or uax.granny_info.track_groups
    group_index = bake.get('target_group_index', 0 if len(placement_groups) == 1 else -1)
    if AUXILIARY_SOURCE not in action:
        action[AUXILIARY_SOURCE] = capture_uax_auxiliary_source(uax, animation, group_index)
    action[PLACEMENT_PROPERTY] = native_uax_group_placement(uax, animation, group_index).hex()
    if action.get('hwek_uax_native_binding'):
        action['hwek_uax_target_group_index'] = group_index
    if group_index >= 0:
        action['hwek_uax_target_group_index'] = group_index
        action['hwek_uax_target_group_name'] = placement_groups[group_index].name
    action["hwek_uax_transform_tracks"] = json.dumps(animation.transform_track_names)
    action["hwek_uax_transform_curves"] = json.dumps([
        {
            "name": track.name,
            "orientation": uax_curve_metadata(track.orientation_curve_info),
            "position": uax_curve_metadata(track.position_curve_info),
            "scale_shear": uax_curve_metadata(track.scale_shear_curve_info),
            "rotation_keys": len(track.rotation_keys),
            "position_keys": len(track.position_keys),
            "scale_keys": len(track.scale_keys),
            "scale_shear_keys": len(track.scale_shear_keys),
        }
        for track in animation.transform_tracks
    ])
    if matrix_source_action:
        action["hwek_uax_matrix_source_action"] = matrix_source_action
    if editable_action:
        action["hwek_uax_editable_action"] = editable_action
    action["hwek_uax_template"] = "UAX.bt/Granny.bt"


def focus_uax_armature_for_editing(armature: Optional[bpy.types.Object], action: Optional[bpy.types.Action], duration: float, fps: float):
    if armature is None or getattr(armature, "type", "") != "ARMATURE":
        return
    try:
        armature.hide_viewport = False
        armature.hide_set(False)
    except Exception:
        pass
    try:
        armature.show_in_front = True
    except Exception:
        pass
    data = getattr(armature, "data", None)
    if data is not None:
        try:
            data.display_type = "STICK"
        except Exception:
            pass
        for bone in getattr(data, "bones", []) or []:
            try:
                bone.hide = False
                bone.select = True
            except Exception:
                pass
        bones = list(getattr(data, "bones", []) or [])
        if bones:
            try:
                data.bones.active = bones[0]
            except Exception:
                pass
    scene = getattr(bpy.context, "scene", None)
    if scene is not None:
        end_frame = int(math.ceil(uax_frame(duration, fps)))
        scene.frame_start = 1
        scene.frame_end = max(1, end_frame)
        try:
            scene.frame_set(1)
        except Exception:
            pass
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass
    try:
        bpy.ops.object.select_all(action="DESELECT")
    except Exception:
        pass
    try:
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature
    except Exception:
        pass
    if action is not None:
        animation_data_create = getattr(armature, "animation_data_create", None)
        if callable(animation_data_create):
            animation_data_create()
        animation_data = getattr(armature, "animation_data", None)
        if animation_data is not None:
            animation_data.action = action
    try:
        bpy.ops.object.mode_set(mode="POSE")
    except Exception:
        pass
    pose_bones = getattr(getattr(armature, "pose", None), "bones", None)
    if pose_bones is not None:
        for pose_bone in pose_bones:
            try:
                pose_bone.select = True
            except Exception:
                pass
    try:
        bpy.ops.pose.select_all(action="SELECT")
    except Exception:
        pass
    expand_uax_action_channels(action)
    uax_view_layer_update()


def uax_action_name(path: str, animation_index: int = 0, animation_count: int = 1) -> str:
    """Use the UAX filename, never Granny's authoring-machine source path."""
    stem = os.path.splitext(os.path.basename(path))[0] or "UAX Animation"
    if animation_count <= 1 or animation_index <= 0:
        return stem
    return f"{stem} {animation_index + 1}"


def activate_uax_action(
    armature_target: Optional[bpy.types.Object],
    action: Optional[bpy.types.Action],
    animation: Optional[GrannyAnimationInfo] = None,
):
    if armature_target is None or action is None:
        return
    tracks = animation.transform_tracks if animation is not None else []
    assign_uax_action(armature_target, action, tracks)
    if action.get("hwek_uax_fcurve_mode") == "matrix_basis":
        ensure_uax_playback_handler()
        scene = getattr(bpy.context, "scene", None)
        frame = float(getattr(scene, "frame_current_final", getattr(scene, "frame_current", 1))) if scene is not None else 1.0
        apply_uax_matrix_basis_action(armature_target, action, frame)
    else:
        focus_uax_armature_for_editing(
            armature_target,
            action,
            float(action.get("hwek_uax_duration", 0.0)),
            float(action.get("hwek_uax_fps", 30.0)),
        )


def import_uax(
    path: str,
    target=None,
    keep_matrix_source: bool = False,
    activate: bool = True,
    defer: bool = False,
    action_prefix: str = '',
    looping: Optional[bool] = None,
    action_suffix: str = '',
    native_binding: bool = False,
) -> List[bpy.types.Action]:
    from .loading import remove_orphaned_uax_actions
    remove_orphaned_uax_actions()
    uax = parse_uax(path, decode_animation_curves=not defer)
    actions: List[bpy.types.Action] = []
    animations = uax.granny_info.animations or [GrannyAnimationInfo(os.path.splitext(os.path.basename(path))[0])]
    armature_target = uax_target_armature(target)
    animation_bindings = []
    if armature_target is not None:
        _, bindings_by_source = uax_armature_bindings(armature_target)
        for animation in animations:
            tracks, group_index = uax_tracks_for_armature(animation, armature_target, bindings_by_source, native_binding)
            animation_bindings.append((tracks, group_index))
            if not native_binding and tracks and not any(
                uax_binding_for_track(track.name, bindings_by_source) is not None
                for track in tracks
            ):
                raise UgxError(
                    f"UAX {os.path.basename(path)}: no bone tracks match armature {armature_target.name}. "
                    "Select the matching source rig or retarget the animation first."
                )
    for animation_index, animation in enumerate(animations):
        action_name = uax_action_name(path, animation_index, len(animations))
        if action_prefix:
            action_name = action_prefix + ' / ' + action_name.replace('_', ' ').title()
        action_name += action_suffix
        fps = (1.0 / animation.time_step) if animation.time_step and animation.time_step > 0.0 else 30.0
        if armature_target is not None and not defer:
            action = bpy.data.actions.new(action_name)
            matrix_action = bpy.data.actions.new(f"{action.name} [UAX Matrix Source]") if keep_matrix_source else None
            if matrix_action is not None:
                matrix_action['hwek_uax_owner'] = armature_target
            if native_binding:
                action['hwek_uax_native_binding'] = True
                if matrix_action is not None:
                    matrix_action['hwek_uax_native_binding'] = True
            bake = bake_uax_matrix_basis_action(matrix_action, animation, armature_target, fps, action, uax.granny_info.bones, bool(looping), native_binding)
            set_uax_action_metadata(
                action,
                uax,
                animation,
                bake,
                bake["mode"],
                bake.get("editable_fcurve_count", 0),
                bake.get("editable_key_count", 0),
                matrix_source_action=matrix_action.name if matrix_action is not None else "",
            )
            if matrix_action is not None:
                set_uax_action_metadata(
                    matrix_action,
                    uax,
                    animation,
                    bake,
                    "matrix_basis",
                    bake.get("matrix_basis_fcurves", 0),
                    bake.get("matrix_basis_key_count", 0),
                    editable_action=action.name,
                )
        else:
            action = bpy.data.actions.new(action_name)
            if native_binding:
                action['hwek_uax_native_binding'] = True
            bake = {
                "key_count": sum(uax_track_key_count(track) for track in animation.transform_tracks),
                "fcurve_count": 0,
                "matched_tracks": [],
                "missing_tracks": animation.transform_track_names,
                "sample_count": len(uax_animation_sample_times(animation)),
                "decoded_key_count": sum(uax_track_key_count(track) for track in animation.transform_tracks),
                "matrix_basis_fcurves": 0,
                "matrix_basis_key_count": 0,
                "editable_fcurve_count": 0,
                "editable_key_count": 0,
                "editable_max_decompose_error": 0.0,
                "max_basis_delta": 0.0,
                "mode": "metadata_only",
                "target_group_index": animation_bindings[animation_index][1] if armature_target is not None
                    else (0 if len(animation.track_groups) == 1 else -1),
            }
            set_uax_action_metadata(action, uax, animation, bake, bake['mode'], 0, bake['key_count'])
            if armature_target is not None:
                if hasattr(action, 'slots'):
                    action.slots.new(id_type='OBJECT', name=armature_target.name)
                else:
                    action.id_root = 'OBJECT'
                action['hwek_uax_pending'] = True
                action['hwek_uax_pending_owner'] = armature_target
                action.use_frame_range = True
                action.frame_start = 1
                action.frame_end = max(1, uax_frame(animation.duration, fps))
        if armature_target is not None:
            action['hwek_uax_owner'] = armature_target
            tracks, group_index = animation_bindings[animation_index]
            if native_binding:
                action['hwek_uax_target_group_index'] = group_index
                if group_index >= 0:
                    action['hwek_uax_target_group_name'] = animation.track_groups[group_index].name
            mapping = uax_bone_track_names(armature_target, tracks)
            if mapping:
                action['hwek_uax_track_names_by_bone'] = json.dumps(mapping, sort_keys=True)
        if looping is not None:
            action['hwek_uax_looping'] = bool(looping)
            action.use_cyclic = bool(looping)
        action["hwek_uax_source_animation_name"] = animation.name
        action["hwek_uax_source_animation_index"] = animation_index
        action["hwek_uax_source_snapshot"] = zlib.compress(uax.original_bytes).hex()
        action['hwek_uax_signature_version'] = 2
        curve_signatures = [(curve, uax_fcurve_signature(curve)) for curve in uax_iter_action_fcurves(action)]
        action["hwek_uax_import_action_signature"] = uax_action_signature(action, curve_signatures)
        action["hwek_uax_import_armature_signature"] = uax_armature_signature(armature_target)
        action["hwek_uax_import_track_signatures"] = json.dumps(uax_track_action_signatures(action, armature_target, curve_signatures))
        action["hwek_uax_import_fps"] = fps
        if action.get('hwek_uax_pending'):
            action['hwek_uax_deferred_metadata'] = json.dumps({name: action.get(name) for name in (
                'hwek_uax_fps', 'hwek_uax_duration', 'hwek_uax_oversampling',
                'hwek_uax_default_loop_count', 'hwek_uax_animation_flags',
                'hwek_uax_track_groups_metadata', 'hwek_uax_source_animation_name', PLACEMENT_PROPERTY)})
        actions.append(action)
    if actions and activate and not defer:
        activate_uax_action(armature_target, actions[0], animations[0])
    return actions


def uax_context_armature(context):
    """Resolve the active rig even while Blender's file browser owns the context."""
    candidates = []

    def add_candidate(candidate):
        if candidate is not None and candidate not in candidates:
            candidates.append(candidate)

    add_candidate(getattr(context, "object", None))
    add_candidate(getattr(context, "active_object", None))
    view_layer = getattr(context, "view_layer", None)
    add_candidate(getattr(getattr(view_layer, "objects", None), "active", None))

    global_context = getattr(bpy, "context", None)
    add_candidate(getattr(global_context, "object", None))
    add_candidate(getattr(global_context, "active_object", None))
    global_view_layer = getattr(global_context, "view_layer", None)
    add_candidate(getattr(getattr(global_view_layer, "objects", None), "active", None))

    for candidate in candidates:
        armature = uax_target_armature(candidate)
        if armature is not None:
            return armature

    selected = list(getattr(context, "selected_objects", []) or [])
    selected.extend(list(getattr(global_context, "selected_objects", []) or []))
    for candidate in selected:
        armature = uax_target_armature(candidate)
        if armature is not None:
            return armature
    return None
