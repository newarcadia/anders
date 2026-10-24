"""Track action edits and resolve native animation bone names."""

from typing import Dict, Optional
import bpy
from ..ugx.model_space import ugx_model_matrix, ugx_model_transform_objects
import hashlib
import json
import struct
from array import array

__all__ = ('uax_fcurve_signature', 'uax_track_action_signatures', 'uax_action_signature', 'uax_armature_signature', 'blender_data_path_string', 'uax_source_name_for_bone', 'uax_animation_name_for_bone', 'uax_iter_action_fcurves', 'set_uax_animation_flag', 'uax_action_group_metadata', 'uax_loop_translation', 'set_uax_loop_translation')


def set_uax_animation_flag(action, flag, enabled):
    flags = int(action.get('hwek_uax_animation_flags', 0))
    action['hwek_uax_animation_flags'] = flags | flag if enabled else flags & ~flag


def uax_action_group_metadata(action):
    groups = json.loads(action.get('hwek_uax_track_groups_metadata', '[]'))
    index = int(action.get('hwek_uax_target_group_index', 0))
    if index == -1:
        # An unbound Action appends a model group on export. Its original
        # group-name list stays fixed while editable metadata can grow.
        index = len(json.loads(action.get('hwek_uax_track_groups', '[]')))
    return groups, index


def uax_loop_translation(action):
    groups, index = uax_action_group_metadata(action)
    return tuple(groups[index].get('loop_translation', (0., 0., 0.))) if index < len(groups) else (0., 0., 0.)


def set_uax_loop_translation(action, value):
    groups, index = uax_action_group_metadata(action)
    groups.extend({} for _ in range(index + 1 - len(groups)))
    groups[index]['loop_translation'] = list(value)
    action['hwek_uax_track_groups_metadata'] = json.dumps(groups)

def uax_fcurve_signature(curve, version=2) -> str:
    group = curve.group
    if group is not None and group.mute:
        return ""
    if version < 2:
        return legacy_fcurve_signature(curve)
    if any(modifier.type != 'ENVELOPE' for modifier in curve.modifiers):
        return ""
    digest = hashlib.sha256()
    digest.update(repr((curve.data_path, curve.array_index, curve.extrapolation, curve.mute)).encode())
    for collection, properties in (
        (curve.keyframe_points, (('co', 2, 'f'), ('handle_left', 2, 'f'), ('handle_right', 2, 'f'),
                                 ('interpolation', 1, 'i'), ('easing', 1, 'i'),
                                 ('back', 1, 'f'), ('amplitude', 1, 'f'), ('period', 1, 'f'))),
        (curve.sampled_points, (('co', 2, 'f'),)),
    ):
        digest.update(struct.pack('<Q', len(collection)))
        for name, width, code in properties:
            values = array(code, [0]) * (len(collection) * width)
            collection.foreach_get(name, values)
            digest.update(values)
    for modifier in curve.modifiers:
        digest.update(repr((modifier.type, modifier.mute, modifier.use_restricted_range,
                            modifier.frame_start, modifier.frame_end, modifier.blend_in, modifier.blend_out,
                            modifier.use_influence, modifier.influence, modifier.reference_value,
                            modifier.default_min, modifier.default_max)).encode())
        digest.update(repr(tuple((point.frame, point.min, point.max)
                                 for point in modifier.control_points)).encode())
    return digest.hexdigest()


def legacy_fcurve_signature(curve):
    digest = hashlib.sha256()
    digest.update(repr((curve.data_path, curve.array_index, curve.extrapolation, curve.mute)).encode())
    for point in curve.keyframe_points:
        digest.update(repr((tuple(point.co), tuple(point.handle_left), tuple(point.handle_right),
                            point.interpolation, point.easing, point.back, point.amplitude, point.period)).encode())
    for point in curve.sampled_points:
        digest.update(struct.pack("<2f", *point.co))
    # Modified curves are sampled instead of using the unchanged-source path.
    return "" if curve.modifiers else digest.hexdigest()


def uax_track_action_signatures(action: bpy.types.Action, armature: Optional[bpy.types.Object], curve_signatures=None) -> Dict[str, str]:
    if armature is None:
        return {}
    mapping = json.loads(action.get('hwek_uax_track_names_by_bone', '{}'))
    prefixes = {f'pose.bones["{blender_data_path_string(bone.name)}"]': uax_animation_name_for_bone(armature, bone, mapping) for bone in armature.data.bones}
    signatures = {}
    if curve_signatures is None:
        curve_signatures = [(curve, uax_fcurve_signature(curve, int(action.get('hwek_uax_signature_version', 1)))) for curve in uax_iter_action_fcurves(action)]
    for curve, signature in curve_signatures:
        if not curve.mute and curve.data_path in (
                'location', 'rotation_euler', 'rotation_quaternion', 'rotation_axis_angle', 'scale',
                'delta_location', 'delta_rotation_euler', 'delta_rotation_quaternion', 'delta_scale',
                'matrix_basis', 'matrix_local', 'matrix_world', 'matrix_parent_inverse'):
            # Object motion changes native root tracks even when every
            # pose-bone curve still matches its source fingerprint.
            return {}
        prefix = curve.data_path.rsplit(".", 1)[0]
        name = prefixes.get(prefix)
        if name is not None:
            signatures.setdefault(name, []).append(signature)
    return {name: hashlib.sha256(repr(values).encode()).hexdigest() if all(values) else "" for name, values in signatures.items()}


def uax_action_signature(action: bpy.types.Action, curve_signatures=None) -> str:
    digest = hashlib.sha256()
    if curve_signatures is None:
        curve_signatures = [(curve, uax_fcurve_signature(curve, int(action.get('hwek_uax_signature_version', 1)))) for curve in uax_iter_action_fcurves(action)]
    for _curve, signature in curve_signatures:
        if not signature:
            return ""
        digest.update(signature.encode())
    digest.update(repr((action.name, tuple(action.frame_range), action.use_frame_range,
                        action.get("hwek_uax_fps"), action.get("hwek_uax_duration"),
                        action.get("hwek_uax_oversampling"), action.get("hwek_uax_track_groups_metadata"),
                        action.get("hwek_uax_source_animation_name"))).encode())
    if 'hwek_uax_accumulation_flags' in action:
        digest.update(repr(('hwek_uax_accumulation_flags', action['hwek_uax_accumulation_flags'])).encode())
    for name in ('hwek_uax_default_loop_count', 'hwek_uax_animation_flags', 'hwek_uax_initial_placement'):
        if name in action:
            digest.update(repr((name, action[name])).encode())
    if 'hwek_uax_track_names_by_bone' in action:
        digest.update(str(action['hwek_uax_track_names_by_bone']).encode())
    if 'hwek_uax_native_binding' in action:
        digest.update(repr(('hwek_uax_native_binding', action['hwek_uax_native_binding'])).encode())
    return digest.hexdigest()


def uax_armature_signature(armature: Optional[bpy.types.Object]) -> str:
    if armature is None:
        return ""
    animation = armature.animation_data
    if armature.constraints or (animation is not None and (animation.drivers or animation.nla_tracks)):
        return ""
    for owner in ugx_model_transform_objects(armature):
        if owner == armature:
            continue
        animation = owner.animation_data
        if owner.constraints or (animation is not None and
                                 (animation.action is not None or animation.drivers or animation.nla_tracks)):
            return ""
    digest = hashlib.sha256()
    digest.update(repr(tuple(tuple(row) for row in ugx_model_matrix(armature))).encode())
    for bone in armature.data.bones:
        pose = armature.pose.bones.get(bone.name)
        if pose is not None and pose.constraints:
            return ""
        digest.update(repr((bone.name, bone.parent.name if bone.parent else "", uax_animation_name_for_bone(armature, bone),
                            tuple(tuple(row) for row in bone.matrix_local),
                            bone.get("hwek_ugx_source_display_matrix"), bone.get("hwek_ugx_import_display_matrix"),
                            bone.use_inherit_rotation, bone.inherit_scale, bone.use_local_location,
                            pose.rotation_mode if pose is not None else "")).encode())
        native_transform = bone.get('hwek_ugx_granny_local_transform')
        if native_transform is not None:
            digest.update(str(native_transform).encode())
    return digest.hexdigest()


def blender_data_path_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def uax_source_name_for_bone(bone: bpy.types.Bone) -> str:
    source_name = str(bone.get("hwek_ugx_source_name", bone.name))
    blender_name = str(bone.get("hwek_ugx_blender_name", bone.name))
    value = str(bone.get("hwek_ugx_export_name", source_name))
    if value == source_name and bone.name != blender_name:
        value = bone.name
    return value


def uax_animation_name_for_bone(armature, bone, track_names_by_bone=None):
    native_name = uax_source_name_for_bone(bone)
    if track_names_by_bone and track_names_by_bone.get(bone.name) == native_name:
        return native_name
    if bone.name == armature.get('hwek_uax_root_bone'):
        return str(armature.get('hwek_uax_root_track', native_name))
    return uax_source_name_for_bone(bone)


def uax_iter_action_fcurves(action: bpy.types.Action):
    if action is None:
        return
    fcurves = getattr(action, "fcurves", None)
    if fcurves is not None:
        for fcurve in fcurves:
            yield fcurve
        return
    for layer in getattr(action, "layers", []) or []:
        for strip in getattr(layer, "strips", []) or []:
            for channelbag in getattr(strip, "channelbags", []) or []:
                for fcurve in getattr(channelbag, "fcurves", []) or []:
                    yield fcurve
