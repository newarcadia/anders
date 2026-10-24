"""Load separate UGX parts referenced by VIS model attachments."""
import json
from mathutils import Matrix
import bpy
from ..errors import UgxError
from ..formats.vis import read_visual_document
from ..ugx.scene_import import import_ugx
from .damage import load_associated_damage
from .paths import resolve_visual_resource
from .playback import synchronize_child_actions
from .scene import load_visual_animations, refresh_visual_markers, visual_markers, visual_owner, visual_text

__all__ = ('load_visual_attachments',)


def load_visual_attachments(obj, animations=True):
    owner = visual_owner(obj)
    text = visual_text(owner)
    if text is None:
        return {'imported': 0, 'issues': []}
    root = read_visual_document(text.as_string().encode()).root
    models = {model.get('name', '').casefold(): model for model in root.findall('model')}
    report = {'imported': 0, 'issues': []}
    selected = list(bpy.context.selected_objects)
    active = bpy.context.view_layer.objects.active

    def load_parts(parent, ancestors):
        for marker in visual_markers(parent):
            if marker.get('hwek_vis_marker') != 'attach' or str(marker.get('vis_type', '')).casefold() != 'modelref':
                continue
            if any(child.get('hwek_vis_attachment_anchor') for child in marker.children):
                continue
            name = str(marker.get('vis_name', ''))
            model = models.get(name.casefold())
            if model is None:
                report['issues'].append(f'Missing VIS model: {name}')
                continue
            if name.casefold() in ancestors:
                report['issues'].append(f'Cyclic VIS attachment: {name}')
                continue
            constraint = marker.constraints.get('VIS Bone')
            if parent.type == 'ARMATURE' and (constraint is None or constraint.target is None):
                continue
            for asset in model.findall('./component/asset'):
                if asset.get('type', '').casefold() != 'model':
                    continue
                reference = asset.findtext('file', '')
                path = resolve_visual_resource(text['hwek_vis_source_path'], reference, '.ugx')
                if path is None:
                    report['issues'].append(f'Missing attached model {name}: {reference}')
                    continue
                try:
                    mesh = import_ugx(str(path), int(owner.get('hwek_ugx_cache_version', 0)) or None)
                    imported_meshes = list(bpy.context.selected_objects)
                    part = visual_owner(mesh)
                    label = name + ' (' + (constraint.subtarget if constraint is not None else str(marker.get('vis_bone', ''))) + ')'
                    part.name = label + ' Rig' if part.type == 'ARMATURE' else part.name
                    part_collection = part.users_collection[0]
                    part_collection.name = label
                    parent_collection = parent.users_collection[0]
                    if part_collection.name not in parent_collection.children:
                        parent_collection.children.link(part_collection)
                    if part_collection.name in bpy.context.scene.collection.children:
                        bpy.context.scene.collection.children.unlink(part_collection)
                    part['hwek_vis_text'] = text
                    part['hwek_vis_models'] = json.dumps([model.get('name', '')])
                    part['hwek_vis_model_source'] = str(path)
                    part['hwek_vis_assembly_root'] = owner
                    from_name = str(marker.get('vis_frombone', ''))
                    bone = None
                    if part.type == 'ARMATURE':
                        bone = next((b for b in part.data.bones if from_name.casefold() in (b.name.casefold(), str(b.get('hwek_ugx_source_name', '')).casefold())), None)
                        if bone is None:
                            bone = next((b for b in part.data.bones if b.parent is None), None)
                            if from_name and bone is not None:
                                part['hwek_uax_root_bone'] = bone.name
                                part['hwek_uax_root_track'] = from_name
                    refresh_visual_markers(part)
                    report['issues'].extend(load_associated_damage(part, root, text['hwek_vis_source_path'], [model.get('name', '')]))
                    if animations:
                        loaded = load_visual_animations(part, include_folder=False)
                        report['imported'] += loaded['imported']
                        report['issues'].extend(loaded['issues'])
                    load_parts(part, ancestors | {name.casefold()})
                    anchor = bpy.data.objects.new('VIS Model ' + name, None)
                    parent.users_collection[0].objects.link(anchor)
                    anchor.parent = marker
                    anchor['hwek_vis_attachment_anchor'] = True
                    anchor['hwek_vis_model_owner'] = part
                    anchor.hide_select = True
                    anchor.empty_display_size = 0.05
                    for item in imported_meshes:
                        item['hwek_vis_anchor'] = anchor
                    if part.type == 'ARMATURE':
                        anchor['hwek_vis_from_bone'] = bone.name if bone is not None else ''
                        anchor.matrix_basis = bone.matrix_local.inverted_safe() if bone is not None else Matrix.Identity(4)
                        part.parent = anchor
                        part['hwek_vis_anchor'] = anchor
                        for item in imported_meshes:
                            if item.parent is None:
                                item.parent = anchor
                    else:
                        for item in imported_meshes:
                            item.parent = anchor
                            item['hwek_vis_anchor'] = anchor
                    anchor.hide_set(True)
                except (UgxError, OSError, ValueError) as exc:
                    report['issues'].append(f'Attached model {name}: {exc}')

    try:
        load_parts(owner, {name.casefold() for name in json.loads(owner.get('hwek_vis_models', '[]'))})
        if owner.type == 'ARMATURE' and owner.animation_data and owner.animation_data.action:
            synchronize_child_actions(owner, owner.animation_data.action)
    finally:
        for item in bpy.context.selected_objects:
            item.select_set(False)
        for item in selected:
            item.select_set(True)
        bpy.context.view_layer.objects.active = active
    return report
