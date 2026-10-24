"""Own editable VIS text, bone markers and animation actions in Blender."""
from pathlib import Path
import json
import hashlib
import xml.etree.ElementTree as ET
import zlib
import bpy
from ..errors import UgxError
from ..formats.vis import read_visual_document, visual_elements, visual_xml
from ..animation.actions import uax_source_name_for_bone
from ..animation.scene import import_uax, assign_uax_action
from ..animation.loading import queue_pending_uax_actions, synchronize_uax_action_editors, remove_orphaned_uax_actions
from ..ugx.scene import ugx_armature_for
from .paths import find_visual_documents, resolve_visual_resource, visual_animation_references, visual_model_names
from .damage import load_associated_damage
from .properties import apply_visual_uv_edits, migrate_visual_uv_objects, populate_visual_settings, visual_marker_collection
from .damage_properties import populate_damage_settings
from .damage_format import model_damage_version

__all__ = ('visual_owner', 'visual_text', 'load_visual_document', 'load_visual_animations', 'load_ugx_visual', 'create_visual_marker', 'visual_markers', 'refresh_visual_markers', 'synchronize_visual_documents', 'apply_marker_edits', 'apply_visual_document_edits')


def visual_owner(obj):
    if obj is None:
        return None
    if obj.get('hwek_vis_marker'):
        return obj.parent
    return ugx_armature_for(obj) or obj


def visual_text(obj):
    owner = visual_owner(obj)
    text = owner.get('hwek_vis_text') if owner is not None else None
    return text if isinstance(text, bpy.types.Text) else None


def visual_markers(owner):
    return [obj for obj in owner.children if obj.get('hwek_vis_marker')]


def marker_values(marker):
    values = {key[4:]: format(value, '.9g') if isinstance(value, float) else str(value) for key, value in marker.items() if key.startswith('vis_')}
    if marker['hwek_vis_marker'] == 'uvOffset':
        original = json.loads(marker['hwek_vis_original_attrs'])
        for key in ('uOfs', 'vOfs'):
            if key in values and float(values[key]) == float(original.get(key, '0')):
                values[key] = original.get(key, '0')
        return values
    bone = marker.constraints.get('VIS Bone')
    key = 'bone' if marker['hwek_vis_marker'] == 'point' else 'tobone'
    values[key] = bone.subtarget if bone is not None else str(marker.get('vis_bone', ''))
    if bone is not None and marker.parent.type == 'ARMATURE':
        target = marker.parent.data.bones.get(bone.subtarget)
        if target is not None:
            values[key] = uax_source_name_for_bone(target)
    return values


def apply_marker_edits(root, owner):
    elements = visual_elements(root)
    markers = visual_markers(owner)
    remaining = {tuple(json.loads(marker['hwek_vis_element_path'])) for marker in markers}
    removals = []
    originals = json.loads(owner.get('hwek_vis_marker_originals', '{}'))
    for path in json.loads(owner.get('hwek_vis_marker_paths', '[]')):
        path = tuple(path)
        if path not in remaining:
            element = elements.get(path)
            if element is None or element.tag not in ('point', 'attach', 'uvOffset'):
                raise UgxError('VIS XML structure changed around a deleted marker. Resolve the changed point in the visual definition before exporting.')
            if element.attrib != originals.get(json.dumps(path), element.attrib):
                raise UgxError('VIS XML changed a deleted marker. Resolve the changed point in the visual definition before exporting.')
            removals.append((elements[path[:-1]], element))
    for marker in markers:
        original = json.loads(marker['hwek_vis_original_attrs'])
        values = {key: value for key, value in marker_values(marker).items() if value != original.get(key, '')}
        if not values:
            continue
        path = tuple(json.loads(marker['hwek_vis_element_path']))
        element = elements.get(path)
        if element is None or element.tag != marker['hwek_vis_marker']:
            raise UgxError(f'VIS XML structure changed around {marker.name}. Resolve the changed point in the visual definition before exporting.')
        for key, value in values.items():
            if element.get(key, '') not in (original.get(key, ''), value):
                raise UgxError(f'VIS {marker.name}: {key} was edited differently in XML and on the marker.')
            element.set(key, value)
    for parent, child in removals:
        parent.remove(child)


def create_visual_marker(owner, element, path, text):
    kind = element.tag
    if kind not in ('point', 'attach'):
        raise UgxError('Only bone-linked points and attachments have scene markers.')
    bone = element.get('bone' if kind == 'point' else 'tobone', '')
    label = element.get('pointType', 'Point') if kind == 'point' else element.get('name', element.get('type', 'Attachment'))
    if bone:
        label += ' (' + bone + ')'
    obj = bpy.data.objects.new(label, None)
    collection = visual_marker_collection(owner, kind)
    collection.objects.link(obj)
    obj.parent = owner
    obj.empty_display_type = 'SPHERE' if kind == 'point' else 'ARROWS'
    obj.empty_display_size = 0.08 if kind == 'point' else 0.15
    obj.show_name = False
    obj.show_in_front = True
    obj.lock_location = obj.lock_rotation = obj.lock_scale = (True, True, True)
    obj['hwek_vis_marker'] = kind
    obj['hwek_vis_text'] = text
    obj['hwek_vis_element_path'] = json.dumps(path)
    obj['hwek_vis_original_attrs'] = json.dumps(element.attrib)
    for key, value in element.attrib.items():
        if key not in ('bone', 'tobone'):
            obj['vis_' + key] = value
    if owner.type == 'ARMATURE':
        constraint = obj.constraints.new('COPY_TRANSFORMS')
        constraint.name = 'VIS Bone'
        target = next((item for item in owner.data.bones if bone.casefold() in (item.name.casefold(), str(item.get('hwek_ugx_source_name', '')).casefold())), None)
        constraint.mute = target is None
        constraint.target = owner if target is not None else None
        constraint.subtarget = target.name if target is not None else bone
        if target is None:
            obj.color = (1, 0.1, 0.1, 1)
    else:
        obj['vis_bone'] = bone
    return obj


def refresh_visual_markers(owner):
    text = visual_text(owner)
    root = read_visual_document(text.as_string().encode('utf-8')).root
    changed = apply_visual_uv_edits(root, owner)
    changed |= migrate_visual_uv_objects(owner, root)
    if changed:
        text.clear()
        text.write(ET.tostring(root, encoding='unicode'))
    names = json.loads(owner.get('hwek_vis_models', '[]'))
    model_nodes = set(model for model in root.findall('model') if model.get('name', '') in names)
    parent = {child: item for item in root.iter() for child in item}
    active = bpy.context.view_layer.objects.active
    active_path = active.get('hwek_vis_element_path') if active is not None and active.parent == owner else None
    attached = {marker['hwek_vis_element_path']: [(child, child.matrix_basis.copy(), child.matrix_parent_inverse.copy())
                for child in marker.children if child.get('hwek_vis_attachment_anchor')]
                for marker in visual_markers(owner)}
    for marker in visual_markers(owner):
        bpy.data.objects.remove(marker, do_unlink=True)
    paths = []
    originals = {}
    for path, element in visual_elements(root).items():
        if element.tag not in ('point', 'attach'):
            continue
        ancestor = parent.get(element)
        if element.tag in ('point', 'attach') and (ancestor is None or ancestor.tag != 'component'):
            continue
        while ancestor is not None and ancestor.tag != 'model':
            ancestor = parent.get(ancestor)
        if ancestor not in model_nodes:
            continue
        marker = create_visual_marker(owner, element, path, text)
        for child, basis, inverse in attached.get(marker['hwek_vis_element_path'], []):
            child.parent = marker
            child.matrix_parent_inverse = inverse
            child.matrix_basis = basis
        if active_path == json.dumps(path):
            marker.select_set(True)
            bpy.context.view_layer.objects.active = marker
        paths.append(path)
        originals[json.dumps(path)] = dict(element.attrib)
    owner['hwek_vis_marker_paths'] = json.dumps(paths)
    owner['hwek_vis_marker_originals'] = json.dumps(originals)
    owner['hwek_vis_text_signature'] = hashlib.sha256(text.as_string().encode()).hexdigest()
    populate_visual_settings(owner, root)


def apply_visual_document_edits(root, owner):
    text = visual_text(owner)
    owners = [obj for obj in bpy.data.objects if obj.get('hwek_vis_text') == text and not obj.get('hwek_vis_marker')]
    for obj in owners:
        apply_visual_uv_edits(root, obj)
    for obj in owners:
        apply_marker_edits(root, obj)


def synchronize_visual_documents():
    remove_orphaned_uax_actions()
    synchronize_uax_action_editors()
    for obj in bpy.data.objects:
        damage = obj.get('hwek_damage_text')
        if isinstance(damage, bpy.types.Text):
            signature = hashlib.sha256(damage.as_string().encode()).hexdigest()
            if (signature != obj.get('hwek_damage_projection_signature') or
                    obj.get('hwek_damage_projection_game') != model_damage_version(obj)):
                try:
                    populate_damage_settings(obj)
                except (UgxError, ValueError):
                    continue
    editing = {area.spaces.active.text for window in bpy.context.window_manager.windows for area in window.screen.areas
               if area.type == 'TEXT_EDITOR'}
    owners = [obj for obj in bpy.data.objects if obj.get('hwek_vis_text') and not obj.get('hwek_vis_marker')]
    for owner in owners:
        text = owner.get('hwek_vis_text')
        if not isinstance(text, bpy.types.Text) or owner.get('hwek_vis_marker') or owner.mode == 'EDIT':
            continue
        if not owner.get('hwek_visual_properties_ready') or owner.get('hwek_vis_uv_parameters_version') != 1:
            try:
                populate_visual_settings(owner)
            except (UgxError, ValueError):
                continue
        signature = hashlib.sha256(text.as_string().encode()).hexdigest()
        if signature != owner.get('hwek_vis_text_signature') and text not in editing:
            try:
                root = read_visual_document(text.as_string().encode('utf-8')).root
                apply_visual_uv_edits(root, owner)
                apply_marker_edits(root, owner)
                text.clear()
                text.write(ET.tostring(root, encoding='unicode'))
                refresh_visual_markers(owner)
            except (UgxError, ValueError):
                # A text edit can temporarily contain incomplete XML.
                continue
        for marker in visual_markers(owner):
            constraint = marker.constraints.get('VIS Bone')
            if constraint is not None and owner.type == 'ARMATURE':
                valid = constraint.subtarget in owner.data.bones
                target = owner if valid else None
                if constraint.target != target:
                    constraint.target = target
                if constraint.mute == valid:
                    constraint.mute = not valid
    return 0.5


def load_visual_document(obj, path, animations=True):
    owner = visual_owner(obj)
    if owner is None:
        raise UgxError('Select a model or armature before loading its VIS.')
    document = read_visual_document(Path(path).read_bytes())
    source = str(owner.get('hwek_ugx_source_path', obj.get('hwek_ugx_source_path', '')))
    names = visual_model_names(document.root, source) if source else []
    if not names:
        raise UgxError(f'VIS {Path(path).name} does not reference the selected UGX.')
    if visual_text(owner) is not None:
        raise UgxError('This model already has an editable VIS. Export it before loading a replacement.')
    text = bpy.data.texts.new(Path(path).name + ' XML')
    text.write(visual_xml(document))
    text.use_fake_user = True
    text['hwek_vis_source_path'] = str(Path(path).absolute())
    text['hwek_vis_source'] = zlib.compress(document.source).hex()
    owner['hwek_vis_text'] = text
    owner['hwek_vis_models'] = json.dumps(names)
    owner['hwek_vis_model_source'] = source
    refresh_visual_markers(owner)
    report = load_visual_animations(owner) if animations else {'imported': 0, 'issues': []}
    report['issues'].extend(load_associated_damage(owner, document.root, path, names))
    missing = [marker.constraints['VIS Bone'].subtarget for marker in visual_markers(owner)
               if marker.constraints.get('VIS Bone') is not None and marker.constraints['VIS Bone'].target is None]
    if missing:
        report['issues'].append('VIS points or attachments reference missing bones: ' + ', '.join(sorted(set(missing))))
    owner['hwek_vis_report'] = json.dumps(report)
    return report


def load_visual_animations(obj, include_folder=True):
    owner = visual_owner(obj)
    text = visual_text(owner)
    source = str(owner.get('hwek_ugx_source_path', ''))
    references, origin = [], source
    if text is not None:
        root = read_visual_document(text.as_string().encode('utf-8')).root
        names = visual_model_names(root, owner.get('hwek_vis_model_source', source))
        references = visual_animation_references(root, names)
        origin = text['hwek_vis_source_path']
    paths, issues = {}, []
    for reference, state, exit_action, model_name in references:
        path = resolve_visual_resource(origin, reference, '.uax')
        if path is None:
            issues.append(f'Missing animation: {reference}')
        else:
            looping = exit_action.casefold() not in ('freeze', 'transition')
            paths.setdefault((path, looping), []).append({'reference': reference, 'state': state, 'exit_action': exit_action, 'model': model_name})
    referenced = {path for path, _looping in paths}
    if include_folder and source:
        for path in sorted(Path(source).parent.iterdir()):
            if path.suffix.casefold() == '.uax' and path.absolute() not in referenced:
                paths.setdefault((path.absolute(), None), [])
    imported = 0
    known = {(str(action.get('hwek_uax_source_path', '')), action.get('hwek_uax_looping')): action
             for action in bpy.data.actions if action.get('hwek_vis_owner') == owner}
    def priority(item):
        (path, looping), bindings = item
        idle = any(binding['state'].casefold() == 'idle' for binding in bindings)
        return not idle, 'idle' not in path.stem.casefold(), str(path), looping is False
    for (path, looping), bindings in sorted(paths.items(), key=priority):
        if (str(path), looping) in known:
            continue
        states = {binding['state'] for binding in bindings}
        mixed = sum(candidate == path for candidate, _mode in paths) > 1
        suffix = (' (Loop)' if looping else ' (Held)') if mixed else ''
        try:
            actions = import_uax(str(path), owner if owner.type == 'ARMATURE' else None, activate=False, defer=True,
                                 action_prefix=Path(source).stem.replace('_', ' ').title(), looping=looping, action_suffix=suffix,
                                 native_binding=True)
            if not bindings and owner.type == 'ARMATURE':
                # Nearby clips can belong to another attached model. Explicit
                # VIS bindings still retain their native held-pose behavior.
                matched = []
                for action in actions:
                    if action.get('hwek_uax_target_group_index', -1) < 0:
                        bpy.data.actions.remove(action)
                    else:
                        matched.append(action)
                actions = matched
            for action in actions:
                action['hwek_vis_owner'] = owner
                action['hwek_vis_states'] = json.dumps(sorted(states))
                action['hwek_vis_animation_bindings'] = json.dumps(bindings)
                if not states:
                    action['hwek_vis_unlinked'] = True
                    action['vis_type'] = ''
                if owner.type == 'ARMATURE' and not owner.get('hwek_vis_default_action') and (any(state.casefold() == 'idle' for state in states) or (not bindings and 'idle' in path.stem.casefold())):
                    owner['hwek_vis_default_action'] = action
                    if not action.get('hwek_uax_pending') and (owner.animation_data is None or owner.animation_data.action is None):
                        assign_uax_action(owner, action, [])
            queue_pending_uax_actions(actions)
            imported += len(actions)
        except (UgxError, OSError, ValueError) as exc:
            issues.append(f'{path.name}: {exc}')
    report = {'imported': imported, 'issues': issues}
    owner['hwek_vis_report'] = json.dumps(report)
    return report


def load_ugx_visual(obj, animations=True):
    source = str(obj.get('hwek_ugx_source_path', ''))
    paths, issues = find_visual_documents(source)
    if len(paths) == 1:
        report = load_visual_document(obj, str(paths[0]), animations)
        report['issues'][:0] = issues
    else:
        if len(paths) > 1:
            issues.append('Several VIS documents reference this model. Use Load VIS to choose one: ' + ', '.join(path.name for path in paths))
        report = load_visual_animations(obj) if animations else {'imported': 0, 'issues': []}
        report['issues'].extend(load_associated_damage(visual_owner(obj)))
        report['issues'][:0] = issues
    visual_owner(obj)['hwek_vis_report'] = json.dumps(report)
    return report
