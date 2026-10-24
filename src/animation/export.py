"""Export Blender actions while retaining unchanged native UAX tracks."""

from typing import Any, Dict, Iterable, List, Optional, Tuple
from array import array
import bpy
from mathutils import Matrix
from ..ugx.model_space import ugx_model_matrix, ugx_model_transform_objects
from .loading import finish_pending_uax_action, initialize_uax_animation_options
from .placement import write_uax_group_placement
from .auxiliary import write_uax_auxiliary_source
import hashlib
import json
import math
import os
import struct
import zlib
from ..errors import UgxError
from .binding import granny_model_name_key, granny_track_group_sort_flags
from ..formats.ecf import EcfFile, UAX_FILE_ID, build_ecf_file, replace_ecf_chunk
from ..formats.granny import (
    GRX_FILE_INFO_ANIMATIONS,
    GRX_FILE_INFO_TRACK_GROUPS,
    GRX_ROOT_SIZE,
    GRX_TRACK_GROUP_SIZE,
    GRX_TRANSFORM_TRACK_SIZE,
    GrxWriter,
    grx_array_header,
    grx_cstring,
    grx_ref_array,
    grx_ref_array_from_header,
    grx_u64,
    identity_grx_transform,
    write_grx_array_header,
    write_grx_transform,
    write_grx_type_schema,
)
from ..formats.matrix import mat3_mul, mat4_inverse, mat4_mul
from ..formats.granny_transform import granny_unretarget_transform, granny_transform_rows
from ..formats.granny import GrannyTransform
from .curve_fitting import fit_bezier_steps, bezier_granny_controls, float32_curve_error
from .local_sampling import local_export_channel_sampler
from .keyframes import uax_scene_evaluation_frame
from .scene import uax_retarget_offsets
from ..ugx.transforms import (
    blender_matrix_to_game_rows,
    game_rows_to_blender_matrix,
    granny_transform_from_local_rows,
    uax_export_axis_rows,
)
from .actions import (
    uax_action_signature,
    uax_armature_signature,
    uax_fcurve_signature,
    uax_source_name_for_bone,
    uax_track_action_signatures,
    uax_iter_action_fcurves,
    uax_action_group_metadata,
)
from .scene import (
    uax_armature_bindings,
    uax_binding_depth,
    uax_binding_receives_export_axis,
    uax_view_layer_update,
)
from .uax import (
    UAX_CURVE_SCHEMA_DEFS,
    UAX_CURVE_TYPE_NAMES,
    UAX_GRANNY_CHUNK,
    parse_uax,
    parse_uax_data,
    write_uax_float_curve,
)

__all__ = ('uax_export_action_fps', 'uax_export_action_frames', 'set_scene_frame_float', 'sample_uax_action', 'uax_track_group_name', 'uax_track_group_flags', 'build_uax_granny_info_chunk', 'uax_import_source_bytes', 'patch_imported_uax_animation', 'prepare_uax_export', 'export_uax')

def uax_export_action_fps(action: bpy.types.Action, scene) -> float:
    try:
        metadata_fps = float(action.get("hwek_uax_fps", 0.0))
    except (TypeError, ValueError):
        metadata_fps = 0.0
    if math.isfinite(metadata_fps) and metadata_fps > 0.0:
        return metadata_fps
    render = getattr(scene, "render", None)
    fps = float(getattr(render, "fps", 30.0))
    base = float(getattr(render, "fps_base", 1.0))
    return fps / base if math.isfinite(base) and base > 0.0 else max(1.0, fps)


def uax_export_action_frames(action: bpy.types.Action) -> Tuple[float, float, List[float]]:
    frame_range = getattr(action, "frame_range", (1.0, 1.0))
    start = float(frame_range[0])
    end = max(start, float(frame_range[1]))
    count = int(math.ceil(end - start))
    if count > 100000:
        raise UgxError("The active action is longer than the 100,000-frame UAX export limit.")
    frames = [min(end, start + index) for index in range(count + 1)]
    if not frames or abs(frames[-1] - end) > 0.000001:
        frames.append(end)
    return start, end, frames


def set_scene_frame_float(scene, frame: float):
    whole = math.floor(float(frame))
    subframe = float(frame) - whole
    try:
        scene.frame_set(int(whole), subframe=subframe)
    except TypeError:
        scene.frame_set(float(frame))


def uax_export_transform_error(expected, actual):
    maximum = 0.0
    for offset in range(0, len(expected), 16):
        matrices = []
        for values in (expected, actual):
            data = values[offset:offset+16]
            rotation = data[3:7]
            length = math.sqrt(sum(v*v for v in rotation))
            if length < 1e-12:
                return math.inf
            transform = GrannyTransform(7, data[:3], tuple(v/length for v in rotation),
                                       tuple(tuple(data[i:i+3]) for i in (7, 10, 13)))
            matrices.append(granny_transform_rows(transform))
        maximum = max(maximum, float32_curve_error((v for row in matrices[0] for v in row),
                                                  (v for row in matrices[1] for v in row)))
    return maximum


def uax_stepped_break_times(curve, start, end, fps):
    result = set()
    for modifier in curve.modifiers:
        if modifier.type != 'STEPPED' or modifier.mute:
            continue
        step, offset = float(modifier.frame_step), float(modifier.frame_offset)
        if step <= 0:
            continue
        first, last = start, end
        if modifier.use_frame_start:
            first = max(first, float(modifier.frame_start))
        if modifier.use_frame_end:
            last = min(last, float(modifier.frame_end))
        if last < first:
            continue
        count = math.ceil((last-offset)/step)-math.floor((first-offset)/step)
        if count > 100000:
            raise UgxError('Stepped animation exceeds the 100,000-boundary UAX export limit.')
        boundaries = {offset+index*step for index in range(
            math.floor((first-offset)/step), math.ceil((last-offset)/step)+1)}
        boundaries = sorted({first, last} | {frame for frame in boundaries if first <= frame <= last})

        def block(time):
            # Blender evaluates the modifier's subtraction and division as
            # float32, then truncates toward zero, including negative frames.
            frame = uax_scene_evaluation_frame(start+time*fps)
            if modifier.use_frame_start and frame < modifier.frame_start:
                return None
            if modifier.use_frame_end and frame > modifier.frame_end:
                return None
            relative = struct.unpack('<f', struct.pack('<f', frame-offset))[0]
            return int(struct.unpack('<f', struct.pack('<f', relative/step))[0])

        for index, frame in enumerate(boundaries):
            if not start < frame <= end:
                continue
            previous = boundaries[index-1] if index else frame-step
            following = boundaries[index+1] if index+1 < len(boundaries) else frame+step
            before = max(0., ((frame+previous)*.5-start)/fps)
            after = min((end-start)/fps, ((frame+following)*.5-start)/fps)
            initial = block(before)
            if initial == block(after):
                continue
            low = struct.unpack('<I', struct.pack('<f', before))[0]
            high = struct.unpack('<I', struct.pack('<f', after))[0]
            while high-low > 1:
                middle = (low+high)//2
                time = struct.unpack('<f', struct.pack('<I', middle))[0]
                if block(time) == initial:
                    low = middle
                else:
                    high = middle
            result.add(struct.unpack('<f', struct.pack('<I', high))[0])
    return result


def uax_cyclic_break_times(curve, modifier, start, end, fps):
    points = list(curve.keyframe_points)
    if len(points) < 2 or any(key.interpolation != 'CONSTANT' for key in points[:-1]):
        return set()
    if ((modifier.use_influence and modifier.influence != 1.)
            or modifier.blend_in != 0. or modifier.blend_out != 0.):
        return set()
    first, last = float(points[0].co.x), float(points[-1].co.x)
    period = struct.unpack('<f', struct.pack('<f', last-first))[0]
    if period <= 0:
        return set()
    begin = math.floor((start-first)/period)
    finish = math.floor((end-first)/period)
    if modifier.mode_before == 'NONE':
        begin = max(begin, 0)
    elif modifier.cycles_before:
        begin = max(begin, -modifier.cycles_before)
    if modifier.mode_after == 'NONE':
        finish = min(finish, 0)
    elif modifier.cycles_after:
        finish = min(finish, modifier.cycles_after)
    if (finish-begin+1)*len(points) > 100000:
        raise UgxError('Cyclic animation exceeds the 100,000-boundary UAX export limit.')
    boundaries = {start, end}
    for cycle in range(begin, finish+1):
        mode, count = ('NONE', 0)
        if cycle < 0:
            mode, count = modifier.mode_before, modifier.cycles_before
        elif cycle > 0:
            mode, count = modifier.mode_after, modifier.cycles_after
        if cycle and (mode == 'NONE' or (count and abs(cycle) > count)):
            continue
        for key in points:
            offset = last-float(key.co.x) if mode == 'MIRROR' and cycle % 2 else float(key.co.x)-first
            offset = struct.unpack('<f', struct.pack('<f', offset))[0]
            frame = first+cycle*period+offset
            if start < frame < end:
                boundaries.add(frame)
    if modifier.use_restricted_range:
        boundaries.update(frame for frame in (modifier.frame_start, modifier.frame_end) if start < frame < end)
    boundaries = sorted(boundaries)
    probes = set()
    duration = (end-start)/fps
    for index, frame in enumerate(boundaries):
        time = max(0., (frame-start)/fps)
        bits = struct.unpack('<I', struct.pack('<f', time))[0]
        # A repeated endpoint can briefly expose its last key before wrapping.
        # Keep adjacent native clocks as well as the intervals on either side.
        for delta in range(-4, 5):
            if bits+delta >= 0:
                value = struct.unpack('<f', struct.pack('<I', bits+delta))[0]
                if 0 <= value <= duration:
                    probes.add(value)
        if index:
            probes.add(struct.unpack('<f', struct.pack('<f', ((frame+boundaries[index-1])*.5-start)/fps))[0])
    probes = sorted(probes)
    result = set()
    for first_time, last_time in zip(probes, probes[1:]):
        initial = curve.evaluate(uax_scene_evaluation_frame(start+first_time*fps))
        if initial == curve.evaluate(uax_scene_evaluation_frame(start+last_time*fps)):
            continue
        low = struct.unpack('<I', struct.pack('<f', first_time))[0]
        high = struct.unpack('<I', struct.pack('<f', last_time))[0]
        while high-low > 1:
            middle = (low+high)//2
            time = struct.unpack('<f', struct.pack('<I', middle))[0]
            if curve.evaluate(uax_scene_evaluation_frame(start+time*fps)) == initial:
                low = middle
            else:
                high = middle
        result.add(struct.unpack('<f', struct.pack('<I', high))[0])
    return result


def uax_constant_key_break_times(curve, start, end, fps):
    result = set()
    points = list(curve.keyframe_points)
    for previous, key in zip(points, points[1:]):
        frame = float(key.co.x)
        if previous.interpolation != 'CONSTANT' or previous.co.y == key.co.y or frame <= start or previous.co.x >= end:
            continue
        first = max(0., (max(start, float(previous.co.x))-start)/fps)
        last = (min(end, frame)-start)/fps
        target = float(key.co.y)
        if (curve.evaluate(uax_scene_evaluation_frame(start+first*fps)) == target
                or curve.evaluate(uax_scene_evaluation_frame(start+last*fps)) != target):
            continue
        low = struct.unpack('<I', struct.pack('<f', first))[0]
        high = struct.unpack('<I', struct.pack('<f', last))[0]
        # Blender snaps evaluations near a key to that key's value. Find
        # the first native float32 time that evaluates after the jump.
        while high-low > 1:
            middle = (low+high)//2
            time = struct.unpack('<f', struct.pack('<I', middle))[0]
            if curve.evaluate(uax_scene_evaluation_frame(start+time*fps)) == target:
                high = middle
            else:
                low = middle
        result.add(struct.unpack('<f', struct.pack('<I', high))[0])
    return result


def uax_action_break_times(curves, start, end, fps):
    result = set()
    for curve in curves:
        if curve.mute:
            continue
        result.update(uax_stepped_break_times(curve, start, end, fps))
        modifiers = [modifier for modifier in curve.modifiers if not modifier.mute
                     and not (modifier.use_influence and modifier.influence == 0.)]
        if not modifiers:
            result.update(uax_constant_key_break_times(curve, start, end, fps))
        elif len(modifiers) == 1 and modifiers[0].type == 'CYCLES':
            result.update(uax_cyclic_break_times(curve, modifiers[0], start, end, fps))
        elif all(modifier.type == 'ENVELOPE' for modifier in modifiers):
            # Value envelopes cannot move a base key's jump. Inspect its timing
            # without their value offsets, restoring them before pose sampling.
            try:
                for modifier in modifiers:
                    modifier.mute = True
                result.update(uax_constant_key_break_times(curve, start, end, fps))
            finally:
                for modifier in modifiers:
                    modifier.mute = False
    return result


def uax_frame_boundary_times(frames, start, end, fps):
    duration = struct.unpack('<f', struct.pack('<f', (end-start)/fps))[0]
    last = struct.unpack('<I', struct.pack('<f', duration))[0]
    result = set()
    for frame in frames:
        low, high = 0, last
        while low < high:
            middle = (low+high)//2
            time = struct.unpack('<f', struct.pack('<I', middle))[0]
            if uax_scene_evaluation_frame(start+time*fps) < frame:
                low = middle+1
            else:
                high = middle
        for bits in (low-1, low, low+1):
            if 0 <= bits <= last:
                result.add(struct.unpack('<f', struct.pack('<I', bits))[0])
    return result


def sample_uax_action(
    armature: bpy.types.Object,
    action: bpy.types.Action,
    axis_forward: str = "Z",
    axis_up: str = "Y",
    source_bones=(),
    preserve_tracks=(),
) -> Tuple[float, float, float, List[Dict[str, Any]]]:
    if armature is None or getattr(armature, "type", "") != "ARMATURE":
        raise UgxError("Select an armature with an active action before exporting UAX.")
    scene = getattr(bpy.context, "scene", None)
    if scene is None:
        raise UgxError("UAX export requires an active Blender scene.")
    fps = uax_export_action_fps(action, scene)
    start, end, frame_grid = uax_export_action_frames(action)
    mapping = json.loads(action.get('hwek_uax_track_names_by_bone', '{}'))
    bindings, by_source = uax_armature_bindings(armature, mapping)
    preserve_tracks = set(preserve_tracks)
    bindings = sorted(bindings, key=lambda item: uax_binding_depth(item, by_source))
    sampled_by_name = {}
    for binding in bindings:
        if binding.source_name not in preserve_tracks:
            sampled_by_name.setdefault(binding.source_name, binding)
    sampled_bindings = list(sampled_by_name.values())
    if not bindings:
        raise UgxError("The selected armature has no bones to export.")
    if not sampled_bindings:
        return fps, (end-start)/fps, float(action.get('hwek_uax_oversampling', 1.0)), []
    curves = list(uax_iter_action_fcurves(action))
    for owner in ugx_model_transform_objects(armature):
        animation = owner.animation_data
        if owner != armature and animation is not None and animation.action is not None:
            curves.extend(uax_iter_action_fcurves(animation.action))
    rig_signature = uax_armature_signature(armature)
    if rig_signature and preserve_tracks:
        needed_bones = set()
        for binding in bindings:
            if binding.source_name in preserve_tracks:
                continue
            bone = armature.data.bones[binding.blender_name]
            while bone is not None and bone.name not in needed_bones:
                needed_bones.add(bone.name)
                bone = bone.parent
        # Copied siblings cannot affect the edited pose. Retain parent clocks
        # because inheritance can couple their motion to the sampled bones.
        copied_paths = {pose.path_from_id('location').rsplit('.', 1)[0]
                        for pose in armature.pose.bones if pose.name not in needed_bones}
        curves = [curve for curve in curves if curve.data_path.rsplit('.', 1)[0] not in copied_paths]
    evaluated_rig = not rig_signature or any(curve.modifiers for curve in curves)
    frames = set(frame_grid) if evaluated_rig else {start, end}
    boundary_frames = set()
    curve_boundaries = {}
    for curve in curves:
        local_boundaries = set()
        for points in (curve.keyframe_points, curve.sampled_points):
            if points:
                coordinates = array('f', [0]) * (len(points) * 2)
                points.foreach_get('co', coordinates)
                frames.update(frame for frame in coordinates[::2] if start <= frame <= end)
        # Even smooth keys snap inside Blender's exact-key window. Retain
        # significant edges when a correction envelope is muted or removed.
        epsilon = struct.unpack('<f', struct.pack('<f', .0001))[0]
        has_envelope = any(modifier.type == 'ENVELOPE' for modifier in curve.modifiers)
        for key in curve.keyframe_points:
            for frame in (key.co.x-epsilon, key.co.x+epsilon):
                bits = struct.unpack('<I', struct.pack('<f', frame))[0]
                probes = [struct.unpack('<f', struct.pack('<I', bits+delta))[0]
                          for delta in (-1, 0, 1) if bits+delta >= 0]
                values = [curve.evaluate(value) for value in probes] if not has_envelope else ()
                if has_envelope or max(values)-min(values) > 1e-6:
                    local_boundaries.update(value for value in probes if start <= value <= end)
        for modifier in curve.modifiers:
            if modifier.type != 'ENVELOPE' or modifier.mute:
                continue
            for point in modifier.control_points:
                if start <= point.frame <= end:
                    local_boundaries.add(float(point.frame))
        boundary_frames.update(local_boundaries)
        curve_boundaries[curve.as_pointer()] = local_boundaries
    boundary_times = uax_frame_boundary_times(boundary_frames, start, end, fps)
    breaks = uax_action_break_times(curves, start, end, fps)
    times = sorted({(frame-start)/fps for frame in frames} | breaks | boundary_times)
    offsets = uax_retarget_offsets(bindings, source_bones)
    if not bindings:
        raise UgxError("The selected armature has no bones to export.")
    if not sampled_bindings:
        return fps, (end-start)/fps, float(action.get('hwek_uax_oversampling', 1.0)), []
    axis_rows = uax_export_axis_rows(axis_forward, axis_up)
    axis_linear = [row[:3] for row in axis_rows[:3]]
    prepared = {}
    for binding in bindings:
        pose_bone = armature.pose.bones.get(binding.blender_name)
        if pose_bone is None:
            raise UgxError(f'Armature {armature.name}: bone {binding.blender_name} has no pose to export.')
        source_rest = game_rows_to_blender_matrix(binding.rest_world_rows)
        import_rest = game_rows_to_blender_matrix(binding.import_world_rows or binding.rest_world_rows)
        prepared[binding.blender_name] = (binding, pose_bone, import_rest.inverted_safe() @ source_rest,
                                         uax_binding_receives_export_axis(binding, by_source))
    source_entries = {name: [prepared[binding.blender_name] for binding in bindings
                             if binding.source_name == name] for name in sampled_by_name}
    animation_data = armature.animation_data_create()
    previous_action = animation_data.action
    previous_slot = getattr(animation_data, 'action_slot', None)
    previous_frame = float(getattr(scene, 'frame_current_final', scene.frame_current))
    animation_data.action = action
    cache = {}
    current_time = None
    model_matrix = None
    worlds, inverses = {}, {}

    def source_world(binding):
        rows = worlds.get(binding.blender_name)
        if rows is None:
            _binding, pose_bone, correction, _rotate_axis = prepared[binding.blender_name]
            rows = blender_matrix_to_game_rows(model_matrix @ pose_bone.matrix @ correction)
            worlds[binding.blender_name] = rows
        return rows

    def sample_pose(time, source_name=None):
        nonlocal current_time, model_matrix
        cached = cache.setdefault(time, {})
        names = sampled_by_name if source_name is None else (source_name,)
        missing = [name for name in names if name not in cached]
        if missing and current_time != time:
            set_scene_frame_float(scene, start+time*fps)
            uax_view_layer_update()
            model_matrix = ugx_model_matrix(armature)
            current_time = time
            worlds.clear()
            inverses.clear()
        for name in missing:
            previous = None
            for binding, _pose_bone, _correction, rotate_axis in source_entries[name]:
                rows = source_world(binding)
                parent = binding.parent
                if parent is None and binding.parent_source_name:
                    parent = by_source.get(binding.parent_source_name)
                if parent is not None:
                    if parent.blender_name not in inverses:
                        inverses[parent.blender_name] = mat4_inverse(source_world(parent))
                    local = mat4_mul(rows, inverses[parent.blender_name])
                else:
                    local = [row[:] for row in rows]
                if rotate_axis:
                    linear = mat3_mul([row[:3] for row in local[:3]], axis_linear)
                    for row in range(3):
                        local[row][:3] = linear[row]
                transform = granny_transform_from_local_rows(local)
                offset = offsets.get(binding.blender_name)
                if offset is not None:
                    transform = granny_unretarget_transform(transform, offset)
                components = (*transform.position, *transform.orientation, *(v for row in transform.scale_shear for v in row))
                if previous is not None:
                    if uax_export_transform_error(previous[0], components) > 1e-5:
                        raise UgxError(f'Bones {previous[1]} and {binding.blender_name} animate differently under the same native '
                                       f'name {name!r}. Give them distinct bone names to animate them independently.')
                else:
                    previous = (components, binding.blender_name)
            cached[name] = previous[0]
        return cached[source_name] if source_name is not None else tuple(
            value for name in sampled_by_name for value in cached[name])
    try:
        def sample_left(time, source_name=None):
            if time not in breaks:
                return None
            bits = struct.unpack('<I', struct.pack('<f', time))[0]
            return sample_pose(struct.unpack('<f', struct.pack('<I', bits-1))[0], source_name)
        tracks = []
        local_samplers = {}
        layers = getattr(action, 'layers', ())
        simple_action = (not layers or (len(layers) == 1 and len(layers[0].strips) == 1
                         and len(layers[0].strips[0].channelbags) == 1))
        # Unrestricted envelope correction stays within its F-curve and adds
        # no dependencies between bones.
        independent_tracks = (bool(rig_signature) and simple_action and not offsets
                              and not any(any(modifier.type != 'ENVELOPE' or modifier.use_restricted_range
                                              for modifier in curve.modifiers)
                                          or curve.mute or not curve.is_valid or curve.is_empty
                                          or (curve.group and curve.group.mute)
                                          for curve in curves)
                              and animation_data.action_blend_type == 'REPLACE'
                              and animation_data.action_influence == 1.
                              and not animation_data.use_tweak_mode
                              and armature.data.pose_position == 'POSE'
                              and (not layers or layers[0].strips[0].channelbags[0].slot_handle
                                   == animation_data.action_slot_handle)
                              and all(bone.use_inherit_rotation and bone.inherit_scale == 'FULL'
                                      and bone.use_local_location for bone in armature.data.bones)
                              and all(len(entries) == 1 for entries in source_entries.values())
                              and axis_forward == 'Z' and axis_up == 'Y')
        if independent_tracks:
            sample_pose(0.)
            mirror = Matrix.Diagonal((-1, 1, 1, 1))
            inverse_axis = Matrix(((-1., 0., 0., 0.), (0., 0., 1., 0.),
                                   (0., -1., 0., 0.), (0., 0., 0., 1.)))
            paths = {pose.path_from_id(field): binding.blender_name
                     for binding, pose, _correction, _axis in prepared.values()
                     for field in ('location', 'rotation_quaternion', 'scale')}
            bone_curves = {binding.blender_name: [] for binding in bindings}
            for curve in curves:
                name = paths.get(curve.data_path)
                if name is None:
                    independent_tracks = False
                    break
                bone_curves[name].append(curve)
            if independent_tracks:
                for binding in sampled_bindings:
                    name = binding.blender_name
                    _binding, pose, correction, _axis = prepared[name]
                    bone = armature.data.bones[name]
                    if bone.parent is None:
                        left = inverse_axis @ model_matrix @ bone.matrix_local
                    else:
                        parent_correction = prepared[bone.parent.name][2]
                        left = (mirror @ parent_correction.inverted_safe() @
                                bone.parent.matrix_local.inverted_safe() @ bone.matrix_local)
                    sampler = local_export_channel_sampler(left, correction @ mirror, pose, bone_curves[name])
                    if sampler is None:
                        # Evaluated bone matrices still have independent key
                        # grids under standard inheritance and an unconstrained rig.
                        local_samplers.clear()
                        break
                    local_samplers[binding.source_name] = sampler
        if independent_tracks:
            for binding in sampled_bindings:
                name = binding.source_name
                local_curves = bone_curves[binding.blender_name]
                local_frames = {start, end}
                for curve in local_curves:
                    for points in (curve.keyframe_points, curve.sampled_points):
                        local_frames.update(float(key.co[0]) for key in points if start <= key.co[0] <= end)
                local_breaks = uax_action_break_times(local_curves, start, end, fps)
                local_boundaries = set().union(*(curve_boundaries[curve.as_pointer()] for curve in local_curves))
                local_boundary_times = uax_frame_boundary_times(local_boundaries, start, end, fps)
                local_times = sorted({(frame-start)/fps for frame in local_frames} | local_breaks | local_boundary_times)
                sampler = local_samplers.get(name)
                def sample_local(time):
                    if sampler is None:
                        return sample_pose(time, name)
                    return sampler(uax_scene_evaluation_frame(start+time*fps))
                def sample_local_left(time):
                    if time not in local_breaks:
                        return None
                    bits = struct.unpack('<I', struct.pack('<f', time))[0]
                    return sample_local(struct.unpack('<f', struct.pack('<I', bits-1))[0])
                spans = [span for span in fit_bezier_steps(
                    local_times, sample_local, uax_export_transform_error, (3,), tolerance=1e-5/math.sqrt(1+uax_binding_depth(binding, by_source)),
                    sample_left=sample_local_left, native_clock=True) if span is not None]
                if spans:
                    knots, controls = bezier_granny_controls(spans)
                    degree = 3
                else:
                    knots, controls, degree = [0.0], [sample_local(0.0)], 1
                tracks.append({'name': name, 'knots': knots, 'degree': degree,
                               'position': [v for point in controls for v in point[:3]],
                               'orientation': [v for point in controls for v in point[3:7]],
                               'scale_shear': [v for point in controls for v in point[7:16]]})
        else:
            spans = [span for span in fit_bezier_steps(times, sample_pose, uax_export_transform_error,
                                                      tuple(index*16+3 for index in range(len(sampled_bindings))),
                                                      sample_left=sample_left, native_clock=True) if span is not None]
            if spans:
                knots, controls = bezier_granny_controls(spans)
                degree = 3
            else:
                knots, controls, degree = [0.0], [sample_pose(0.0)], 1
            for index, binding in enumerate(sampled_bindings):
                offset = index*16
                tracks.append({'name': binding.source_name, 'knots': knots, 'degree': degree,
                               'position': [v for point in controls for v in point[offset:offset+3]],
                               'orientation': [v for point in controls for v in point[offset+3:offset+7]],
                               'scale_shear': [v for point in controls for v in point[offset+7:offset+16]]})
    finally:
        animation_data.action = previous_action
        if previous_action is not None and previous_slot is not None:
            animation_data.action_slot = previous_slot
        set_scene_frame_float(scene, previous_frame)
        uax_view_layer_update()
    return fps, (end-start)/fps, float(action.get('hwek_uax_oversampling', 1.0)), tracks


def uax_track_group_name(action: bpy.types.Action, armature: bpy.types.Object) -> str:
    if 'hwek_ugx_granny_model_name' in armature:
        return str(armature['hwek_ugx_granny_model_name'])
    roots = [bone for bone in armature.data.bones if bone.parent is None]
    if roots:
        return uax_source_name_for_bone(roots[0])
    selected = action.get('hwek_uax_target_group_name')
    if selected:
        return str(selected)
    try:
        names = json.loads(str(action.get("hwek_uax_track_groups", "[]")))
        if names and str(names[0]).strip():
            return str(names[0])
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return armature.name


def uax_track_group_flags(action: bpy.types.Action) -> int:
    if 'hwek_uax_accumulation_flags' in action:
        return int(action['hwek_uax_accumulation_flags'])
    try:
        groups = json.loads(str(action.get("hwek_uax_track_groups_metadata", "[]")))
        index = int(action.get('hwek_uax_target_group_index', 0))
        return int(groups[index].get("flags", 0)) if groups else 0
    except (TypeError, ValueError, json.JSONDecodeError, AttributeError, IndexError):
        return 0


def write_uax_group_loop_translation(data, offset, action, group_index):
    metadata = json.loads(action.get('hwek_uax_track_groups_metadata', '[]'))
    original = struct.unpack_from('<3f', data, offset+128)
    translation = original
    if 0 <= group_index < len(metadata) and 'loop_translation' in metadata[group_index]:
        try:
            translation = tuple(float(value) for value in metadata[group_index]['loop_translation'])
        except (TypeError, ValueError) as exc:
            raise UgxError(f'Animation {action.name}: loop translation must contain three numbers.') from exc
    if len(translation) != 3:
        raise UgxError(f'Animation {action.name}: loop translation must contain three numbers.')
    finite = all(math.isfinite(value) for value in translation)
    if (uax_track_group_flags(action) & 5) == 1 and not grx_u64(data, offset+140) and not finite:
        raise UgxError(f'Animation {action.name}: constant motion extraction requires a finite loop translation.')
    if all(left == right or (math.isnan(left) and math.isnan(right)) for left, right in zip(translation, original)):
        return
    if not finite:
        raise UgxError(f'Animation {action.name}: edited loop translation must be finite.')
    try:
        struct.pack_into('<3f', data, offset+128, *translation)
    except (OverflowError, struct.error) as exc:
        raise UgxError(f'Animation {action.name}: loop translation exceeds native float32 range.') from exc


def write_uax_animation_options(data, offset, action):
    for name, field in (('hwek_uax_default_loop_count', 32), ('hwek_uax_animation_flags', 36)):
        if name not in action:
            continue
        value = action[name]
        if not isinstance(value, int) or not -2147483648 <= value <= 2147483647:
            raise UgxError(f'Animation {action.name}: {name.removeprefix("hwek_uax_")} must be a signed 32-bit integer.')
        struct.pack_into('<i', data, offset+field, value)
    count, flags = struct.unpack_from('<ii', data, offset+32)
    if flags & 1 and count < 0:
        raise UgxError(f'Animation {action.name}: the enabled native default loop count cannot be negative.')


def build_uax_granny_info_chunk(
    output_path: str,
    armature: bpy.types.Object,
    action: bpy.types.Action,
    fps: float,
    duration: float,
    oversampling: float,
    tracks: List[Dict[str, Any]],
) -> bytes:
    initialize_uax_animation_options(action)
    writer = GrxWriter()
    root_offset = writer.reserve(GRX_ROOT_SIZE, 1)
    group_refs_offset = writer.reserve(8, 16)
    group_offset = writer.reserve(GRX_TRACK_GROUP_SIZE, 16)
    struct.pack_into("<Q", writer.data, group_refs_offset, group_offset)
    writer.string_ref(group_offset, uax_track_group_name(action, armature))

    transform_tracks_offset = writer.reserve(len(tracks) * GRX_TRANSFORM_TRACK_SIZE, 16)
    curve_variants = []
    for track_index, track in enumerate(tracks):
        track_offset = transform_tracks_offset + track_index * GRX_TRANSFORM_TRACK_SIZE
        writer.string_ref(track_offset, str(track["name"]))
        struct.pack_into("<I", writer.data, track_offset + 8, 0)
        for variant_offset, controls in (
            (track_offset + 12, track["orientation"]),
            (track_offset + 28, track["position"]),
            (track_offset + 44, track["scale_shear"]),
        ):
            curve_offset = write_uax_float_curve(writer, track["knots"], controls, track.get("degree", 1))
            struct.pack_into("<Q", writer.data, variant_offset + 8, curve_offset)
            curve_variants.append((variant_offset, writer.data[curve_offset]))
    write_grx_array_header(writer.data, group_offset + 20, len(tracks), transform_tracks_offset)
    # Without measured mesh-space errors, let native binding retain motion.
    # Zero errors tell the game every track can use its static LOD pose.
    write_grx_array_header(writer.data, group_offset + 32, 0, 0)
    write_grx_transform(writer.data, group_offset + 56, identity_grx_transform())
    write_uax_group_placement(writer.data, group_offset, action)
    _, metadata_index = uax_action_group_metadata(action)
    write_uax_auxiliary_source(writer, group_offset, action, metadata_index)
    write_uax_group_loop_translation(writer.data, group_offset, action, metadata_index)
    flags = granny_track_group_sort_flags(uax_track_group_flags(action), [track['name'] for track in tracks])
    struct.pack_into("<I", writer.data, group_offset + 124, flags)
    write_grx_array_header(writer.data, root_offset + GRX_FILE_INFO_TRACK_GROUPS, 1, group_refs_offset)

    animation_refs_offset = writer.reserve(8, 16)
    animation_offset = writer.reserve(56, 16)
    struct.pack_into("<Q", writer.data, animation_refs_offset, animation_offset)
    animation_name = os.path.splitext(os.path.basename(output_path))[0] or action.name
    writer.string_ref(animation_offset, animation_name)
    struct.pack_into("<3f", writer.data, animation_offset + 8, float(duration), 1.0 / float(fps), max(1.0, float(oversampling)))
    write_uax_animation_options(writer.data, animation_offset, action)
    animation_group_refs_offset = writer.reserve(8, 16)
    struct.pack_into("<Q", writer.data, animation_group_refs_offset, group_offset)
    write_grx_array_header(writer.data, animation_offset + 20, 1, animation_group_refs_offset)
    write_grx_array_header(writer.data, root_offset + GRX_FILE_INFO_ANIMATIONS, 1, animation_refs_offset)

    schema_offsets = write_grx_type_schema(writer, UAX_CURVE_SCHEMA_DEFS)
    for variant_offset, format_index in curve_variants:
        struct.pack_into("<Q", writer.data, variant_offset, schema_offsets[UAX_CURVE_TYPE_NAMES[format_index]])
    # HWDE rejects the entire animation before binding unless Granny's
    # FromFileName marker is the literal emitted by the original gr2ugx tool.
    writer.string_ref(root_offset + 16, "gr2ugx")
    writer.finish_strings()
    return bytes(writer.data)


def uax_import_source_bytes(action: bpy.types.Action, source_cache=None) -> bytes:
    snapshot = action.get("hwek_uax_source_snapshot", "")
    if snapshot:
        key = ('snapshot', str(snapshot))
        if source_cache is not None and key in source_cache:
            return source_cache[key]
        payload = zlib.decompress(bytes.fromhex(str(snapshot)))
        if source_cache is not None:
            source_cache[key] = payload
        return payload
    path = bpy.path.abspath(str(action.get("hwek_uax_source_path", "")))
    key = ('file', path, action.get("hwek_uax_source_sha256"))
    if source_cache is not None and key in source_cache:
        return source_cache[key]
    if path and os.path.isfile(path):
        with open(path, "rb") as handle:
            payload = handle.read()
        if hashlib.sha256(payload).hexdigest() == action.get("hwek_uax_source_sha256"):
            if source_cache is not None:
                source_cache[key] = payload
            return payload
    return b""


def uax_model_group_order(data, groups, group_index, target_name):
    if group_index <= 0 or struct.unpack_from('<I', data, groups[group_index] + 124)[0] & 8:
        return groups
    wanted = granny_model_name_key(target_name)
    for group in groups[:group_index]:
        if (not struct.unpack_from('<I', data, group + 124)[0] & 8
                and granny_model_name_key(grx_cstring(data, grx_u64(data, group))) == wanted):
            if group == groups[group_index]:
                return groups
            # Native binding picks the first eligible model-name match.
            return [groups[group_index], *groups[:group_index], *groups[group_index + 1:]]
    return groups


def patch_imported_uax_animation(
    payload: bytes, action: bpy.types.Action, tracks: List[Dict[str, Any]],
    fps: float, duration: float, oversampling: float, preserve_tracks: Iterable[str] = (),
    target_model_name: Optional[str] = None,
) -> bytes:
    ecf = EcfFile.parse(payload)
    writer = GrxWriter()
    writer.data = bytearray(ecf.chunks[UAX_GRANNY_CHUNK].data)
    animations = grx_ref_array_from_header(writer.data, GRX_FILE_INFO_ANIMATIONS)
    index = int(action.get("hwek_uax_source_animation_index", 0))
    if not 0 <= index < len(animations):
        raise UgxError("Imported UAX animation index is outside its source animation table.")
    animation_offset = animations[index]
    groups = grx_ref_array(writer.data, *grx_array_header(writer.data, animation_offset + 20))
    group_index = int(action.get('hwek_uax_target_group_index', 0 if len(groups) == 1 else -1))
    if not 0 <= group_index < len(groups):
        if not (action.get('hwek_uax_native_binding') and group_index == -1 and target_model_name is not None):
            raise UgxError(f'Animation {action.name}: its source track group is unavailable. Reimport the UAX onto its model before exporting.')
        # An unbound native clip has no curves for this model. Author a separate
        # group rather than repurposing a different model's retained tracks.
        group_index = len(groups)
        group = writer.reserve(GRX_TRACK_GROUP_SIZE, 16)
        writer.string_ref(group, target_model_name)
        write_grx_transform(writer.data, group + 56, identity_grx_transform())
        struct.pack_into('<I', writer.data, group + 124, 2)
        groups.append(group)
        group_refs = writer.reserve(len(groups) * 8)
        struct.pack_into(f'<{len(groups)}Q', writer.data, group_refs, *groups)
        write_grx_array_header(writer.data, animation_offset + 20, len(groups), group_refs)
    target_group = groups[group_index]
    write_uax_group_placement(writer.data, target_group, action)
    original_track_count, _ = grx_array_header(writer.data, target_group + 20)
    changed_groups = {target_group}
    write_uax_group_loop_translation(writer.data, target_group, action, group_index)
    if 'hwek_uax_accumulation_flags' in action:
        root_groups = grx_ref_array_from_header(writer.data, GRX_FILE_INFO_TRACK_GROUPS)
        flags = int(action['hwek_uax_accumulation_flags'])
        if not root_groups and flags:
            root_refs = writer.reserve(8)
            struct.pack_into('<Q', writer.data, root_refs, target_group)
            write_grx_array_header(writer.data, GRX_FILE_INFO_TRACK_GROUPS, 1, root_refs)
            root_groups = [target_group]
        if root_groups:
            struct.pack_into('<I', writer.data, root_groups[0] + 124, flags)
            changed_groups.add(root_groups[0])
    if target_model_name is not None and grx_cstring(writer.data, grx_u64(writer.data, target_group)) != target_model_name:
        writer.string_ref(target_group, target_model_name)
    struct.pack_into('<3f', writer.data, animation_offset + 8, duration, 1.0 / fps, oversampling)
    write_uax_animation_options(writer.data, animation_offset, action)
    if len(animations) > 1:
        _, refs = grx_array_header(writer.data, GRX_FILE_INFO_ANIMATIONS)
        write_grx_array_header(writer.data, GRX_FILE_INFO_ANIMATIONS, 1, refs + index * 8)
    count, offset = grx_array_header(writer.data, target_group + 20)
    if count and tracks:
        # Groups can share their source track array. Edit a private copy so
        # another model's curves keep their original variants and controls.
        copied = writer.reserve(count * GRX_TRANSFORM_TRACK_SIZE)
        writer.data[copied:copied + count * GRX_TRANSFORM_TRACK_SIZE] = writer.data[offset:offset + count * GRX_TRANSFORM_TRACK_SIZE]
        write_grx_array_header(writer.data, target_group + 20, count, copied)
    sampled = {str(track["name"]): track for track in tracks}
    preserve_tracks = set(preserve_tracks)
    curve_types = write_grx_type_schema(writer, UAX_CURVE_SCHEMA_DEFS) if tracks else {}
    assigned = set()
    count, offset = grx_array_header(writer.data, target_group + 20)
    for track_index in range(count):
        track_offset = offset + track_index * GRX_TRANSFORM_TRACK_SIZE
        name = grx_cstring(writer.data, grx_u64(writer.data, track_offset))
        track = sampled.get(name)
        if track is None:
            continue
        assigned.add(name)
        if name in preserve_tracks:
            continue
        for variant_offset, curve_field in ((track_offset + 12, "orientation"), (track_offset + 28, "position"), (track_offset + 44, "scale_shear")):
            curve_offset = write_uax_float_curve(writer, track["knots"], track[curve_field], track.get("degree", 1))
            struct.pack_into("<QQ", writer.data, variant_offset, curve_types[UAX_CURVE_TYPE_NAMES[writer.data[curve_offset]]], curve_offset)
    new_tracks = [track for track in tracks if str(track["name"]) not in assigned]
    if new_tracks and groups:
        group_offset = target_group
        count, offset = grx_array_header(writer.data, group_offset + 20)
        new_offset = writer.reserve((count + len(new_tracks)) * GRX_TRANSFORM_TRACK_SIZE)
        writer.data[new_offset:new_offset + count * GRX_TRANSFORM_TRACK_SIZE] = writer.data[offset:offset + count * GRX_TRANSFORM_TRACK_SIZE]
        for index, track in enumerate(new_tracks, count):
            track_offset = new_offset + index * GRX_TRANSFORM_TRACK_SIZE
            writer.string_ref(track_offset, str(track["name"]))
            for variant_offset, curve_field in ((track_offset + 12, "orientation"), (track_offset + 28, "position"), (track_offset + 44, "scale_shear")):
                curve_offset = write_uax_float_curve(writer, track["knots"], track[curve_field], track.get("degree", 1))
                struct.pack_into("<QQ", writer.data, variant_offset, curve_types[UAX_CURVE_TYPE_NAMES[writer.data[curve_offset]]], curve_offset)
        write_grx_array_header(writer.data, group_offset + 20, count + len(new_tracks), new_offset)
    lod_count, lod_offset = grx_array_header(writer.data, target_group + 32)
    if lod_count and tracks:
        if lod_count != original_track_count or lod_offset <= 0 or lod_offset + lod_count * 4 > len(writer.data):
            raise UgxError(f'Animation {action.name}: source LOD errors do not match its transform tracks.')
        count, offset = grx_array_header(writer.data, target_group + 20)
        new_lod = writer.reserve(count * 4)
        writer.data[new_lod:new_lod + lod_count * 4] = writer.data[lod_offset:lod_offset + lod_count * 4]
        for track_index in range(count):
            name = grx_cstring(writer.data, grx_u64(writer.data, offset + track_index * GRX_TRANSFORM_TRACK_SIZE))
            if name in sampled and name not in preserve_tracks:
                # Changed curves invalidate the source mesh-space estimate.
                # Use Granny's finite no-LOD value until errors are measured.
                struct.pack_into('<I', writer.data, new_lod + track_index * 4, 0x7f0fffff)
        write_grx_array_header(writer.data, target_group + 32, count, new_lod)
    writer.finish_strings()
    for group in changed_groups if tracks else ():
        count, tracks_offset = grx_array_header(writer.data, group + 20)
        names = [grx_cstring(writer.data, grx_u64(writer.data, tracks_offset + index * GRX_TRANSFORM_TRACK_SIZE))
                 for index in range(count)]
        flags = struct.unpack_from('<I', writer.data, group + 124)[0]
        struct.pack_into('<I', writer.data, group + 124, granny_track_group_sort_flags(flags, names))
    ordered = uax_model_group_order(writer.data, groups, group_index,
                                    grx_cstring(writer.data, grx_u64(writer.data, target_group)))
    if ordered is not groups:
        refs = writer.reserve(len(ordered) * 8)
        struct.pack_into(f'<{len(ordered)}Q', writer.data, refs, *ordered)
        write_grx_array_header(writer.data, animation_offset + 20, len(ordered), refs)
    return bytes(replace_ecf_chunk(bytearray(payload), UAX_GRANNY_CHUNK, bytes(writer.data)))


def prepare_uax_export(
    armature: bpy.types.Object,
    action: bpy.types.Action,
    path: str,
    axis_forward: str = "Z",
    axis_up: str = "Y",
    *,
    source_cache=None,
) -> Tuple[bytes, Dict[str, Any]]:
    initialize_uax_animation_options(action)
    source = uax_import_source_bytes(action, source_cache)
    signature_version = int(action.get('hwek_uax_signature_version', 1))
    curve_signatures = [(curve, uax_fcurve_signature(curve, signature_version))
                        for curve in uax_iter_action_fcurves(action)] if source else []
    signature = uax_action_signature(action, curve_signatures) if source else ''
    rig_signature = uax_armature_signature(armature) if source else ''
    unchanged = (source and signature and rig_signature
                 and signature == action.get("hwek_uax_import_action_signature")
                 and rig_signature == action.get("hwek_uax_import_armature_signature")
                 and axis_forward == "Z" and axis_up == "Y")
    if unchanged:
        payload = source
        data = EcfFile.parse(payload).chunks[UAX_GRANNY_CHUNK].data
        count, offset = grx_array_header(data, GRX_FILE_INFO_ANIMATIONS)
        index = int(action.get("hwek_uax_source_animation_index", 0))
        if not 0 <= index < count:
            raise UgxError(f'Animation {action.name}: source animation index {index} exceeds {count} animations.')
        animation = grx_ref_array(data, count, offset)[index]
        groups = grx_ref_array(data, *grx_array_header(data, animation + 20))
        group_index = int(action.get('hwek_uax_target_group_index', 0 if len(groups) == 1 else -1))
        target_name = uax_track_group_name(action, armature)
        unbound = action.get('hwek_uax_native_binding') and group_index < 0
        if group_index < 0 and not unbound:
            group_index = next((i for i, group in enumerate(groups)
                                if granny_model_name_key(grx_cstring(data, grx_u64(data, group))) == granny_model_name_key(target_name)), -1)
        if groups and not unbound and not 0 <= group_index < len(groups):
            raise UgxError(f'Animation {action.name}: its source track group is unavailable. Reimport the UAX onto its model before exporting.')
        if groups and not unbound:
            ordered = uax_model_group_order(data, groups, group_index, target_name)
            renamed = grx_cstring(data, grx_u64(data, groups[group_index])) != target_name
            if renamed or ordered is not groups:
                writer = GrxWriter()
                writer.data = bytearray(data)
                if renamed:
                    writer.string_ref(groups[group_index], target_name)
                if ordered is not groups:
                    refs = writer.reserve(len(ordered) * 8)
                    struct.pack_into(f'<{len(ordered)}Q', writer.data, refs, *ordered)
                    write_grx_array_header(writer.data, animation + 20, len(ordered), refs)
                writer.finish_strings()
                data = bytes(writer.data)
                payload = bytes(replace_ecf_chunk(bytearray(payload), UAX_GRANNY_CHUNK, data))
        if count > 1:
            data = bytearray(data)
            write_grx_array_header(data, GRX_FILE_INFO_ANIMATIONS, 1, offset + index * 8)
            payload = bytes(replace_ecf_chunk(bytearray(payload), UAX_GRANNY_CHUNK, bytes(data)))
        return payload, {"tracks": len(json.loads(action.get('hwek_uax_transform_tracks', '[]'))), "samples": 0,
                "fps": uax_export_action_fps(action, bpy.context.scene), "duration": float(action.get('hwek_uax_duration', 0)),
                "bytes": len(payload), "axis_forward": axis_forward, "axis_up": axis_up}
    if action.get('hwek_uax_pending'):
        finish_pending_uax_action(action)
        return prepare_uax_export(armature, action, path, axis_forward, axis_up, source_cache=source_cache)
    source_info = parse_uax_data(path, source).granny_info if source else None
    preserve_tracks = []
    fps = uax_export_action_fps(action, bpy.context.scene)
    if source_info is not None:
        animation_index = int(action.get('hwek_uax_source_animation_index', 0))
        if not 0 <= animation_index < len(source_info.animations):
            raise UgxError(f'Animation {action.name}: source animation index {animation_index} exceeds its source animation table.')
        animation = source_info.animations[animation_index]
        end = struct.unpack('<f', struct.pack('<f', 1+animation.duration*fps))[0]
        if (rig_signature and rig_signature == action.get("hwek_uax_import_armature_signature")
                and axis_forward == "Z" and axis_up == "Y"
                and fps == action.get("hwek_uax_import_fps")
                and tuple(action.frame_range) == (1., end)):
            original_signatures = json.loads(action.get("hwek_uax_import_track_signatures", "{}"))
            preserve_tracks = [name for name, value in uax_track_action_signatures(action, armature, curve_signatures).items()
                               if value and value == original_signatures.get(name)]
    fps, duration, oversampling, tracks = sample_uax_action(armature, action, axis_forward, axis_up,
                                                          source_info.bones if source_info else (), preserve_tracks)
    if source:
        payload = patch_imported_uax_animation(source, action, tracks, fps, duration, oversampling, preserve_tracks,
                                               uax_track_group_name(action, armature))
    else:
        granny = build_uax_granny_info_chunk(path, armature, action, fps, duration, oversampling, tracks)
        payload = build_ecf_file([(UAX_GRANNY_CHUNK, granny)], file_id=UAX_FILE_ID)
    parsed = parse_uax_data(path, payload)
    if len(parsed.granny_info.animations) != 1:
        raise UgxError("Generated UAX did not reparse with exactly one animation.")
    animation = parsed.granny_info.animations[0]
    expected_names = [str(track["name"]) for track in tracks]
    if any(name not in animation.transform_track_names for name in expected_names):
        raise UgxError("Generated UAX did not preserve the armature transform-track order.")
    if any(not track.rotation_keys or not track.position_keys or not track.scale_shear_keys for track in animation.transform_tracks if track.name in expected_names):
        raise UgxError("Generated UAX did not preserve all sampled transform curves.")
    if not source and parsed.granny_info.from_file_name.lower() != "gr2ugx":
        raise UgxError("Written UAX does not contain HWDE's required gr2ugx source marker.")
    return payload, {
        "tracks": len(animation.track_groups[int(action.get('hwek_uax_target_group_index', 0))].transform_tracks) if source else len(tracks),
        "samples": len(tracks[0]["knots"]) if tracks else 0,
        "fps": fps,
        "duration": duration,
        "bytes": len(payload),
        "axis_forward": axis_forward,
        "axis_up": axis_up,
    }


def export_uax(
    armature: bpy.types.Object,
    action: bpy.types.Action,
    path: str,
    axis_forward: str = "Z",
    axis_up: str = "Y",
) -> Dict[str, Any]:
    payload, result = prepare_uax_export(armature, action, path, axis_forward, axis_up)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as handle:
        handle.write(payload)
    if result['samples'] and len(parse_uax(path).granny_info.animations) != 1:
        raise UgxError('Written UAX failed its on-disk reparse check.')
    return result
