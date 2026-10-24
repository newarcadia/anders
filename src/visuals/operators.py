"""VIS loading, document editing, point creation and export controls."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import zlib
import bpy
from bpy.props import BoolProperty, StringProperty
from bpy_extras.io_utils import ImportHelper, ExportHelper
from ..errors import UgxError
from ..formats.files import write_ugx_export_files
from ..formats.vis import read_visual_document
from ..ugx.scene_import import import_ugx
from ..animation.scene import activate_uax_action, assign_uax_action
from ..animation.loading import finish_pending_uax_action
from .export import prepare_visual_export
from .paths import resolve_visual_resource
from .scene import apply_visual_document_edits, load_visual_document, refresh_visual_markers, visual_owner, visual_text
from .assembly import load_visual_attachments
from .playback import linked_visual_action

__all__ = ('HWEK_OT_import_vis', 'HWEK_OT_export_vis', 'HWEK_OT_edit_vis', 'HWEK_OT_add_vis_point', 'HWEK_OT_create_vis', 'HWEK_OT_activate_vis_action', 'HWEK_MT_visual_actions', 'HWEK_PT_visual', 'HWEK_PT_visual_uv', 'create_visual_document')


class HWEK_OT_import_vis(bpy.types.Operator, ImportHelper):
    bl_idname = 'hwek.import_vis'
    bl_label = 'Load Halo Wars VIS'
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = '.vis'
    filter_glob: StringProperty(default='*.vis;*.vis.xmb', options={'HIDDEN'})
    load_animations: BoolProperty(name='Load UAX Animations', default=True)

    def execute(self, context):
        try:
            obj = context.object
            if obj is None or not obj.get('hwek_ugx_source_path'):
                document = read_visual_document(Path(self.filepath).read_bytes())
                model = next((node for node in document.root.findall('model') if node.get('name') == document.root.get('defaultmodel')), None)
                model = model if model is not None else document.root.find('model')
                reference = model.findtext('./component/asset[@type="Model"]/file', '') if model is not None else ''
                path = resolve_visual_resource(self.filepath, reference, '.ugx')
                if path is None:
                    raise UgxError(f'Could not resolve the default VIS model: {reference}')
                obj = import_ugx(str(path))
            report = load_visual_document(obj, self.filepath, self.load_animations)
            attached = load_visual_attachments(obj, self.load_animations)
            report['imported'] += attached['imported']
            report['issues'].extend(attached['issues'])
        except (UgxError, OSError, ValueError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'WARNING'} if report['issues'] else {'INFO'}, f"Loaded VIS and {report['imported']} actions. " + '; '.join(report['issues'][:3]))
        return {'FINISHED'}


class HWEK_OT_export_vis(bpy.types.Operator, ExportHelper):
    bl_idname = 'hwek.export_vis'
    bl_label = 'Export Halo Wars VIS'
    filename_ext = '.vis'
    filter_glob: StringProperty(default='*.vis;*.vis.xmb', options={'HIDDEN'})
    hw2_binary: BoolProperty(name='HW2 Binary VIS (.vis.xmb)', default=False)
    export_animations: BoolProperty(name='Export Linked UAX Animations', default=True)

    def invoke(self, context, event):
        owner = visual_owner(context.object)
        self.hw2_binary = owner is not None and int(owner.get('hwek_ugx_cache_version', owner.get('hwek_ugx_port_target', 4))) == 6
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        try:
            outputs = prepare_visual_export(context.object, self.filepath, binary=self.hw2_binary or self.filepath.lower().endswith('.xmb'), animations=self.export_animations)
            write_ugx_export_files(outputs)
        except (UgxError, OSError, ValueError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, f'Exported {outputs[0][0]} and {len(outputs) - 1} UAX files.')
        return {'FINISHED'}


class HWEK_OT_edit_vis(bpy.types.Operator):
    bl_idname = 'hwek.edit_vis'
    bl_label = 'Edit VIS XML'

    def execute(self, context):
        text = visual_text(context.object)
        if text is None or context.area is None:
            return {'CANCELLED'}
        owner = visual_owner(context.object)
        root = read_visual_document(text.as_string().encode('utf-8')).root
        apply_visual_document_edits(root, owner)
        text.clear()
        text.write(ET.tostring(root, encoding='unicode'))
        refresh_visual_markers(owner)
        context.area.type = 'TEXT_EDITOR'
        context.area.spaces.active.text = text
        return {'FINISHED'}


def create_visual_document(owner):
    if visual_text(owner) is not None:
        return visual_text(owner)
    path = str(Path(bpy.path.abspath('//')) / (owner.name + '.vis'))
    outputs = prepare_visual_export(owner, path, binary=False, animations=False)
    text = bpy.data.texts.new(owner.name + ' VIS XML')
    text.write(outputs[0][1].decode('utf-8'))
    text.use_fake_user = True
    text['hwek_vis_source_path'] = path
    text['hwek_vis_source'] = zlib.compress(outputs[0][1]).hex()
    owner['hwek_vis_text'] = text
    owner['hwek_vis_models'] = json.dumps(['Default'])
    owner['hwek_vis_model_source'] = str(owner.get('hwek_ugx_source_path') or Path(path).with_suffix('.ugx'))
    refresh_visual_markers(owner)
    return text


class HWEK_OT_create_vis(bpy.types.Operator):
    bl_idname = 'hwek.create_vis'
    bl_label = 'Create VIS Document'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        owner = visual_owner(context.object)
        if owner is None or visual_text(owner) is not None:
            return {'CANCELLED'}
        create_visual_document(owner)
        return {'FINISHED'}


class HWEK_OT_add_vis_point(bpy.types.Operator):
    bl_idname = 'hwek.add_vis_point'
    bl_label = 'Add VIS Point'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        owner = visual_owner(context.object)
        text = visual_text(owner)
        if text is None:
            bpy.ops.hwek.create_vis()
            text = visual_text(owner)
        try:
            root = read_visual_document(text.as_string().encode('utf-8')).root
            apply_visual_document_edits(root, owner)
            names = json.loads(owner.get('hwek_vis_models', '[]'))
            model = next(node for node in root.findall('model') if node.get('name') in names)
            component = model.find('component')
            if component is None:
                component = ET.SubElement(model, 'component')
            bone = owner.data.bones.active if owner.type == 'ARMATURE' else None
            ET.SubElement(component, 'point', pointType='Impact', bone=bone.name if bone else '', pointData='Metal')
            text.clear()
            text.write(ET.tostring(root, encoding='unicode'))
            refresh_visual_markers(owner)
            owner.hwek_visual.show_points = True
            marker = next(marker for marker in reversed(list(owner.children)) if marker.get('hwek_vis_marker') == 'point')
            for obj in context.selected_objects:
                obj.select_set(False)
            marker.select_set(True)
            context.view_layer.objects.active = marker
        except (UgxError, ValueError, StopIteration) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class HWEK_OT_activate_vis_action(bpy.types.Operator):
    bl_idname = 'hwek.activate_vis_action'
    bl_label = 'Select VIS Animation'
    bl_options = {'REGISTER', 'UNDO'}
    action_name: StringProperty()

    def execute(self, context):
        owner = visual_owner(context.object)
        action = bpy.data.actions.get(self.action_name)
        if owner is None or owner.type != 'ARMATURE' or action is None or action.get('hwek_vis_owner') != owner:
            return {'CANCELLED'}
        finish_pending_uax_action(action)
        for part in bpy.data.objects:
            if part.get('hwek_vis_assembly_root') != owner or part.type != 'ARMATURE':
                continue
            anchor = part.get('hwek_vis_anchor')
            if anchor is None or anchor.parent is None or str(anchor.parent.get('vis_syncanims', '')).casefold() != 'true':
                continue
            choice = linked_visual_action(action, (candidate for candidate in bpy.data.actions if candidate.get('hwek_vis_owner') == part))
            if choice is not None:
                finish_pending_uax_action(choice)
                assign_uax_action(part, choice, [])
        activate_uax_action(owner, action)
        context.scene.frame_set(1)
        return {'FINISHED'}


class HWEK_MT_visual_actions(bpy.types.Menu):
    bl_idname = 'HWEK_MT_visual_actions'
    bl_label = 'VIS Animations'

    def draw(self, context):
        owner = visual_owner(context.object)
        for action in bpy.data.actions:
            if action.get('hwek_vis_owner') == owner:
                self.layout.operator('hwek.activate_vis_action', text=action.name, icon='ACTION').action_name = action.name


class HWEK_PT_visual(bpy.types.Panel):
    bl_label = 'Halo Wars'
    bl_idname = 'HWEK_PT_visual'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'object'

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and visual_text(obj) is not None

    def draw(self, context):
        obj, layout = context.object, self.layout
        owner = visual_owner(obj)
        layout.use_property_split = True
        layout.use_property_decorate = False
        if obj.get('hwek_vis_marker'):
            labels = {'pointType': 'Type', 'pointData': 'Surface', 'name': 'Model', 'type': 'Attachment Type',
                      'frombone': 'Origin Bone', 'syncanims': 'Sync Animation', 'disregardorient': 'Ignore Orientation'}
            for key in obj.keys():
                if key.startswith('vis_') and key not in ('vis_channel', 'vis_uOfs', 'vis_vOfs'):
                    layout.prop(obj, '["' + key + '"]', text=labels.get(key[4:], key[4:]))
            constraint = obj.constraints.get('VIS Bone')
            if constraint is not None:
                layout.prop_search(constraint, 'subtarget', owner.data, 'bones', text='Bone')
            return
        layout.prop(owner.hwek_visual, 'show_points')
        row = layout.row()
        row.enabled = owner.hwek_visual.show_points
        row.prop(owner.hwek_visual, 'show_names')
        layout.operator('hwek.add_vis_point', text='Add Attachment Point', icon='ADD')


class HWEK_PT_visual_uv(bpy.types.Panel):
    bl_label = 'Texture Offsets'
    bl_idname = 'HWEK_PT_visual_uv'
    bl_parent_id = 'HWEK_PT_visual'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'object'

    @classmethod
    def poll(cls, context):
        owner = visual_owner(context.object)
        return owner is not None and len(owner.hwek_visual.uv_sets) > 0 and not context.object.get('hwek_vis_marker')

    def draw(self, context):
        settings = visual_owner(context.object).hwek_visual
        layout = self.layout
        if len(settings.uv_sets) > 1:
            layout.prop_search(settings, 'uv_variant', settings, 'uv_sets', text='Variant')
        group = settings.uv_sets[min(settings.active_uv_set, len(settings.uv_sets) - 1)]
        for channel in group.channels:
            row = layout.row(align=True)
            row.label(text='UV ' + channel.channel)
            row.prop(channel, 'u')
            row.prop(channel, 'v')
