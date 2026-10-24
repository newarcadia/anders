"""Prepare editable animation curves in short UI time slices."""
from collections import OrderedDict
import json
import time
import traceback
import zlib
import bpy
from bpy.app.handlers import persistent
from ..errors import UgxError
from .auxiliary import AUXILIARY_SOURCE, capture_uax_auxiliary_source
from .actions import uax_action_signature, uax_armature_signature, uax_fcurve_signature, uax_iter_action_fcurves, uax_track_action_signatures
from .scene import assign_uax_action, bake_uax_matrix_basis_steps, set_uax_action_metadata, uax_frame, uax_target_armature
from .uax import parse_uax_data
from .placement import PLACEMENT_PROPERTY, native_uax_group_placement

__all__ = ('queue_pending_uax_actions', 'finish_pending_uax_action', 'resume_pending_uax_actions', 'clear_pending_uax_actions', 'prepare_active_uax_actions', 'synchronize_uax_action_editors', 'remove_orphaned_uax_actions')

_pending = OrderedDict()


def remove_orphaned_uax_actions():
    owner_keys = ('hwek_vis_owner', 'hwek_uax_owner', 'hwek_uax_pending_owner')
    live_owners, orphaned_owners, candidates = {}, set(), []
    for action in bpy.data.actions:
        if action.library is not None or action.asset_data is not None or not any(key in action for key in owner_keys):
            continue
        owners = {action.get(key) for key in owner_keys if isinstance(action.get(key), bpy.types.Object)}
        for owner in owners:
            if owner not in live_owners:
                live_owners[owner] = bool(owner.users_scene) or owner.asset_data is not None
        if any(live_owners[owner] for owner in owners):
            continue
        orphaned_owners.update(owners)
        candidates.append(action)
    if not candidates:
        return
    users = bpy.data.user_map(subset=set(candidates))
    for action in candidates:
        # Deleted rigs can still reference their active, default or NLA actions.
        # References from other datablocks keep a shared action alive.
        if users.get(action, set()) - orphaned_owners:
            continue
        _pending.pop(action, None)
        bpy.data.actions.remove(action)


def select_action_bones(owner):
    bones = [bone if hasattr(bone, 'select') else bone.bone for bone in owner.pose.bones]
    if not any(bone.select for bone in bones):
        for bone in bones:
            if not bone.hide:
                bone.select = True


def prepare_action(action):
    owner = action.get('hwek_uax_pending_owner')
    if not isinstance(owner, bpy.types.Object) or owner.type != 'ARMATURE':
        raise UgxError(f'Animation {action.name}: its armature is unavailable.')
    action['hwek_uax_owner'] = owner
    source = zlib.decompress(bytes.fromhex(action['hwek_uax_source_snapshot']))
    uax = parse_uax_data(str(action.get('hwek_uax_source_path', action.name)), source)
    animation = uax.granny_info.animations[int(action.get('hwek_uax_source_animation_index', 0))]
    fields = ('hwek_uax_fps', 'hwek_uax_duration', 'hwek_uax_oversampling', 'hwek_uax_accumulation_flags',
              'hwek_uax_default_loop_count', 'hwek_uax_animation_flags',
              'hwek_uax_track_groups_metadata', 'hwek_uax_source_animation_name', PLACEMENT_PROPERTY)
    previous = {name: action.get(name) for name in fields}
    deferred = json.loads(action.get('hwek_uax_deferred_metadata', '{}'))
    previous_range = (action.use_frame_range, action.frame_start, action.frame_end)
    original_rig = action.get('hwek_uax_import_armature_signature')
    fps = 1.0 / animation.time_step if animation.time_step and animation.time_step > 0 else 30.0
    bake = yield from bake_uax_matrix_basis_steps(None, animation, owner, fps, action, uax.granny_info.bones,
                                                bool(action.get('hwek_uax_looping')), bool(action.get('hwek_uax_native_binding')))
    set_uax_action_metadata(action, uax, animation, bake, bake['mode'], bake['editable_fcurve_count'], bake['editable_key_count'])
    action['hwek_uax_source_animation_name'] = animation.name
    action.use_frame_range = bool(deferred)
    action.frame_start = 1
    action.frame_end = max(1, uax_frame(animation.duration, fps))
    action['hwek_uax_signature_version'] = 2
    signatures = []
    deadline = time.perf_counter() + 0.008
    for curve in uax_iter_action_fcurves(action):
        signatures.append((curve, uax_fcurve_signature(curve)))
        if time.perf_counter() >= deadline:
            yield
            deadline = time.perf_counter() + 0.008
    action['hwek_uax_import_action_signature'] = uax_action_signature(action, signatures)
    action['hwek_uax_import_armature_signature'] = original_rig if original_rig is not None else uax_armature_signature(owner)
    action['hwek_uax_import_track_signatures'] = json.dumps(uax_track_action_signatures(action, owner, signatures))
    for name, value in previous.items():
        if value is not None and (name not in deferred or value != deferred[name]):
            action[name] = value
    action.use_frame_range, action.frame_start, action.frame_end = previous_range
    del action['hwek_uax_pending']
    del action['hwek_uax_pending_owner']
    if 'hwek_uax_deferred_metadata' in action:
        del action['hwek_uax_deferred_metadata']
    requested = owner.get('hwek_vis_requested_action')
    current = owner.animation_data.action if owner.animation_data else None
    root = owner.get('hwek_vis_assembly_root')
    default_allowed = root is None or (root.animation_data is not None and root.animation_data.action is not None)
    if requested == action or (default_allowed and current is None and owner.get('hwek_vis_default_action') == action):
        assign_uax_action(owner, action, [])
    if current == action and bpy.context.object == owner:
        select_action_bones(owner)


def process_pending_actions():
    deadline = time.perf_counter() + 0.008
    while _pending and time.perf_counter() < deadline:
        action = next(iter(_pending))
        try:
            if action.name not in bpy.data.actions:
                del _pending[action]
                continue
            next(_pending[action])
        except StopIteration:
            del _pending[action]
        except ReferenceError:
            del _pending[action]
        except (UgxError, ValueError, KeyError, RuntimeError) as exc:
            traceback.print_exc()
            if action.name in bpy.data.actions:
                action['hwek_uax_load_error'] = str(exc)
            del _pending[action]
    return 0.01 if _pending else None


def queue_pending_uax_actions(actions, active=False):
    for action in actions:
        if not action.get('hwek_uax_pending'):
            continue
        if action in _pending:
            if active:
                _pending.move_to_end(action, last=False)
            continue
        owner = action.get('hwek_uax_pending_owner')
        if owner is None:
            continue
        current = owner.animation_data.action if owner.animation_data else None
        if not active and action not in (current, owner.get('hwek_vis_requested_action'), owner.get('hwek_vis_default_action')):
            continue
        if bpy.app.background:
            finish_pending_uax_action(action)
            continue
        _pending[action] = prepare_action(action)
        if active:
            _pending.move_to_end(action, last=False)
    if _pending and not bpy.app.timers.is_registered(process_pending_actions):
        bpy.app.timers.register(process_pending_actions, first_interval=0.01)


def finish_pending_uax_action(action):
    if not action.get('hwek_uax_pending'):
        return
    if action.get('hwek_uax_load_error'):
        raise UgxError(f"Animation {action.name}: {action['hwek_uax_load_error']}")
    steps = _pending.pop(action, None) or prepare_action(action)
    for _step in steps:
        pass


def initialize_uax_animation_options(action):
    fields = ('hwek_uax_default_loop_count', 'hwek_uax_animation_flags', PLACEMENT_PROPERTY)
    if (all(name in action for name in fields) and AUXILIARY_SOURCE in action) or action.get('hwek_uax_options_error'):
        return False
    snapshot = action.get('hwek_uax_source_snapshot')
    if not snapshot:
        return False
    try:
        source = zlib.decompress(bytes.fromhex(snapshot))
        uax = parse_uax_data(str(action.get('hwek_uax_source_path', action.name)), source,
                             decode_animation_curves=False)
        info = uax.granny_info
        index = int(action.get('hwek_uax_source_animation_index', 0))
        if not 0 <= index < len(info.animations):
            raise UgxError(f'Animation {action.name}: its native animation record is unavailable.')
        native = info.animations[index]
        group_index = int(action.get('hwek_uax_target_group_index', 0 if len(native.track_groups) == 1 else -1))
        if AUXILIARY_SOURCE not in action:
            action[AUXILIARY_SOURCE] = capture_uax_auxiliary_source(uax, native, group_index)
        deferred = json.loads(action.get('hwek_uax_deferred_metadata', '{}'))
        signature = uax_action_signature(action)
        unchanged = bool(signature and signature == action.get('hwek_uax_import_action_signature'))
        placement = native_uax_group_placement(uax, native, group_index).hex()
        for name, value in zip(fields, (native.default_loop_count, native.flags, placement)):
            if name not in action:
                action[name] = value
            deferred.setdefault(name, value)
        if unchanged:
            action['hwek_uax_import_action_signature'] = uax_action_signature(action)
        if 'hwek_uax_deferred_metadata' in action:
            action['hwek_uax_deferred_metadata'] = json.dumps(deferred)
    except (UgxError, ValueError, zlib.error) as exc:
        action['hwek_uax_options_error'] = str(exc)
        traceback.print_exc()
    return True


def synchronize_uax_action_editors():
    for window in bpy.context.window_manager.windows:
        target = window.view_layer.objects.active
        for area in window.screen.areas:
            if area.type != 'DOPESHEET_EDITOR' or area.spaces.active.mode != 'ACTION':
                continue
            space = area.spaces.active
            # Blender 5 no longer exposes SpaceDopeSheetEditor.action. Its
            # native picker assigns the Action to the active object instead.
            pinned = bool(getattr(space, 'pin', False))
            action = getattr(space, 'action', None) if pinned else None
            if not pinned and target is not None and target.animation_data is not None:
                action = target.animation_data.action
            if action is None:
                continue
            if initialize_uax_animation_options(action):
                area.tag_redraw()
            owner = action.get('hwek_vis_owner') or action.get('hwek_uax_owner') or action.get('hwek_uax_pending_owner')
            if not pinned and target is not None and target.type == 'MESH' and target.mode == 'OBJECT' and owner == uax_target_armature(target):
                if owner is None or owner.name not in window.view_layer.objects:
                    continue
                assign_uax_action(owner, action, [])
                if target.animation_data is not None and target.animation_data.action == action:
                    target.animation_data.action = None
                owner.hide_set(False, view_layer=window.view_layer)
                owner.select_set(True, view_layer=window.view_layer)
                target.select_set(False, view_layer=window.view_layer)
                window.view_layer.objects.active = owner
                target = owner
                select_action_bones(owner)
                area.tag_redraw()
            if action.get('hwek_uax_pending') and not action.get('hwek_uax_load_error'):
                queue_pending_uax_actions([action], active=True)


@persistent
def prepare_active_uax_actions(_scene, depsgraph):
    for update in depsgraph.updates:
        obj = getattr(update.id, 'original', update.id)
        if not isinstance(obj, bpy.types.Object) or obj.type != 'ARMATURE' or obj.animation_data is None:
            continue
        actions = [obj.animation_data.action]
        for track in obj.animation_data.nla_tracks:
            if not track.mute:
                actions.extend(strip.action for strip in track.strips if strip.type == 'CLIP' and not strip.mute)
        for action in actions:
            if action is not None and action.get('hwek_uax_pending') and not action.get('hwek_uax_load_error'):
                queue_pending_uax_actions([action], active=True)


@persistent
def resume_pending_uax_actions(_unused=None):
    if not hasattr(bpy.data, 'actions'):
        return 0.1
    clear_pending_uax_actions()
    for action in bpy.data.actions:
        if not action.get('hwek_uax_pending'):
            continue
        if 'hwek_uax_load_error' in action:
            del action['hwek_uax_load_error']
        # A saved scene can contain a partially prepared action. Rebuild only
        # those derived curves from the retained native animation snapshot.
        for layer in list(getattr(action, 'layers', [])):
            action.layers.remove(layer)
        curves = getattr(action, 'fcurves', None)
        if curves is not None:
            for curve in list(curves):
                curves.remove(curve)
    queue_pending_uax_actions(bpy.data.actions)


def clear_pending_uax_actions():
    if bpy.app.timers.is_registered(process_pending_actions):
        bpy.app.timers.unregister(process_pending_actions)
    _pending.clear()
