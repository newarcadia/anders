"""Edit damage events and their complete native action attributes."""
import json
import re
import xml.etree.ElementTree as ET
import bpy
from bpy.props import CollectionProperty, EnumProperty, StringProperty
from bpy_extras.io_utils import ImportHelper
from ..errors import UgxError
from ..formats.native_xml import read_native_xml_document, xml_elements
from .damage import damage_text, load_damage_document
from .scene import visual_owner
from .damage_properties import damage_document, write_damage_document, populate_damage_settings, damage_action_types
from .damage_format import model_damage_version, require_damage_version

__all__ = ('HWEK_DamageAttribute', 'HWEK_OT_load_damage', 'HWEK_OT_edit_damage_xml', 'HWEK_OT_edit_damage_action', 'HWEK_OT_select_damage_event', 'HWEK_MT_damage_events', 'HWEK_PT_damage', 'HWEK_UL_damage_events', 'HWEK_UL_damage_actions', 'HWEK_OT_damage_entry')


def damage_owner(obj):
    owner = visual_owner(obj)
    if owner is not None and damage_text(owner) is None:
        root = owner.get('hwek_vis_assembly_root')
        if isinstance(root, bpy.types.Object) and model_damage_version(owner) == model_damage_version(root):
            return root
    return owner


def damage_root(owner):
    text = damage_text(owner)
    if text is None:
        raise UgxError('Load a damage document first.')
    require_damage_version(owner, model_damage_version(owner))
    root = read_native_xml_document(text.as_string().encode('utf-8')).root
    if root.tag != 'damagetemplate':
        raise UgxError('Damage document must have a damagetemplate root.')
    return text, root


def damage_event_label(node, parent):
    names = {'percentagebased': 'Health Level', 'cryopercentagebased': 'Freeze Level', 'impactpoint': 'Impact',
             'shatterdeathevent': 'Shatter Death', 'deathevent': 'Death'}
    return names.get(parent.tag, parent.tag) + ' ' + parent.get('bone', '') + ' / ' + str(list(parent).index(node) + 1)


class HWEK_DamageAttribute(bpy.types.PropertyGroup):
    value: StringProperty(name='Value')


class HWEK_OT_load_damage(bpy.types.Operator, ImportHelper):
    bl_idname = 'hwek.load_damage'
    bl_label = 'Load Damage Template'
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = '.dmg'
    filter_glob: StringProperty(default='*.dmg;*.dmg.xmb;*.xml', options={'HIDDEN'})

    def execute(self, context):
        try:
            load_damage_document(visual_owner(context.object), self.filepath)
        except (UgxError, OSError, ValueError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class HWEK_OT_edit_damage_xml(bpy.types.Operator):
    bl_idname = 'hwek.edit_damage_xml'
    bl_label = 'Edit Complete Damage XML'

    def execute(self, context):
        text = damage_text(damage_owner(context.object))
        if text is None or context.area is None:
            return {'CANCELLED'}
        context.area.type = 'TEXT_EDITOR'
        context.area.spaces.active.text = text
        return {'FINISHED'}


class HWEK_OT_edit_damage_action(bpy.types.Operator):
    bl_idname = 'hwek.edit_damage_action'
    bl_label = 'Edit Damage Entry'
    bl_options = {'REGISTER', 'UNDO'}
    element_path: StringProperty()
    original: StringProperty(options={'HIDDEN'})
    target: StringProperty(name='Value / Targets')
    attributes: CollectionProperty(type=HWEK_DamageAttribute)

    def invoke(self, context, event):
        try:
            _, root = damage_root(damage_owner(context.object))
            element = xml_elements(root)[tuple(json.loads(self.element_path))]
            self.original = ET.tostring(element, encoding='unicode')
            self.target = element.text or ''
            self.attributes.clear()
            for key, value in element.attrib.items():
                item = self.attributes.add()
                item.name, item.value = key, value
        except (UgxError, ValueError, KeyError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return context.window_manager.invoke_props_dialog(self, width=560)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, 'target')
        for item in self.attributes:
            layout.prop(item, 'value', text=item.name)

    def execute(self, context):
        try:
            text, root = damage_root(damage_owner(context.object))
            element = xml_elements(root)[tuple(json.loads(self.element_path))]
            if self.original and ET.tostring(element, encoding='unicode') != self.original:
                raise UgxError('The damage entry changed while its editor was open. Reopen the entry.')
            element.text = self.target or None
            for item in self.attributes:
                element.set(item.name, item.value)
            text.clear()
            text.write(ET.tostring(root, encoding='unicode'))
        except (UgxError, ValueError, KeyError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class HWEK_OT_select_damage_event(bpy.types.Operator):
    bl_idname = 'hwek.select_damage_event'
    bl_label = 'Select Damage Event'
    element_path: StringProperty()

    def execute(self, context):
        damage_owner(context.object)['hwek_damage_event_path'] = self.element_path
        return {'FINISHED'}


class HWEK_MT_damage_events(bpy.types.Menu):
    bl_idname = 'HWEK_MT_damage_events'
    bl_label = 'Damage Events'

    def draw(self, context):
        _, root = damage_root(damage_owner(context.object))
        parents = {child: parent for parent in root.iter() for child in parent}
        for path, node in xml_elements(root).items():
            if node.tag != 'event' and not node.tag.endswith('event'):
                continue
            parent = parents.get(node, root)
            label = damage_event_label(node, parent)
            self.layout.operator('hwek.select_damage_event', text=label).element_path = json.dumps(path)


class HWEK_UL_damage_events(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        layout.label(text=item.name, icon='CHECKMARK' if index == 0 else 'KEYTYPE_KEYFRAME_VEC')


class HWEK_UL_damage_actions(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        labels = {kind.casefold(): label for kind, label, _ in damage_action_types(item)}
        layout.label(text=labels.get(item.kind.casefold(), item.kind), icon='MESH_DATA' if item.mesh else 'PROPERTIES')
        if item.mesh:
            layout.label(text=item.mesh.name)


class HWEK_OT_damage_entry(bpy.types.Operator):
    bl_idname = 'hwek.damage_entry'
    bl_label = 'Edit Damage Entries'
    bl_options = {'REGISTER', 'UNDO'}
    operation: EnumProperty(items=[(name, label, '') for name, label in (
        ('ADD_EVENT', 'Add Stage'), ('REMOVE_EVENT', 'Remove Stage'),
        ('ADD_ACTION', 'Add Action'), ('REMOVE_ACTION', 'Remove Action'))])

    def execute(self, context):
        owner = damage_owner(context.object)
        try:
            require_damage_version(owner, model_damage_version(owner))
        except UgxError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        settings = owner.hwek_damage
        region = settings.regions[settings.active_region]
        _, root = damage_document(owner)
        elements = xml_elements(root)
        parent = elements[tuple(json.loads(region.path))]
        selected = region.active_event
        event = region.events[min(selected, len(region.events) - 1)]
        if self.operation == 'ADD_EVENT':
            if parent.tag.endswith('event'):
                return {'CANCELLED'}
            ET.SubElement(parent, 'event')
        elif self.operation == 'REMOVE_EVENT':
            if not event.path or parent.tag.endswith('event'):
                return {'CANCELLED'}
            parent.remove(elements[tuple(json.loads(event.path))])
        elif self.operation == 'ADD_ACTION':
            if not event.path:
                return {'CANCELLED'}
            ET.SubElement(elements[tuple(json.loads(event.path))], 'action', type='swappart').text = ','
        elif self.operation == 'REMOVE_ACTION':
            if not event.actions:
                return {'CANCELLED'}
            action = event.actions[min(event.active_action, len(event.actions) - 1)]
            elements[tuple(json.loads(event.path))].remove(elements[tuple(json.loads(action.path))])
        write_damage_document(owner, root)
        populate_damage_settings(owner)
        region = settings.regions[settings.active_region]
        if self.operation == 'ADD_EVENT':
            region.active_event = len(region.events) - 1
        if self.operation == 'ADD_ACTION':
            region.events[region.active_event].active_action = len(region.events[region.active_event].actions) - 1
        return {'FINISHED'}


def draw_fields(layout, fields):
    labels = {'fromindex': 'From Frame', 'toindex': 'To Frame', 'effect': 'Effect', 'streamereffect': 'Trail Effect',
              'releaseeffect': 'Release Effect', 'impactsound': 'Impact Sound', 'show': 'Visible', 'delay': 'Delay'}
    for field in fields:
        prop = 'number' if field.format == 'NUMBER' else 'enabled' if field.format == 'BOOL' else 'value'
        layout.prop(field, prop, text=field.label or labels.get(field.name, re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', field.name).replace('_', ' ').title()))


class HWEK_PT_damage(bpy.types.Panel):
    bl_idname = 'HWEK_PT_damage'
    bl_label = 'Damage'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'object'

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and obj.type in ('MESH', 'ARMATURE') and not obj.get('hwek_damage_preview_source')

    def draw(self, context):
        owner = damage_owner(context.object)
        settings, layout = owner.hwek_damage, self.layout
        layout.use_property_decorate = False
        if damage_text(owner) is None:
            layout.label(text='Select a separate part to make it detachable.')
            layout.operator('hwek.make_detachable', icon='MOD_EXPLODE')
            return
        try:
            require_damage_version(owner, model_damage_version(owner))
        except UgxError:
            layout.label(text='Damage template belongs to the other game.', icon='ERROR')
            layout.operator('hwek.load_damage', text='Load Matching Damage Template', icon='FILE_FOLDER')
            return
        row = layout.row(align=True)
        row.operator('hwek.make_detachable', text='Make Detachable', icon='MOD_EXPLODE')
        row.operator('hwek.reset_damage_preview', text='', icon='LOOP_BACK')
        layout.prop(settings, 'preview')
        preview = layout.column()
        preview.enabled = settings.preview
        hw2 = model_damage_version(owner) == 6
        if hw2 or any(region.name == 'Health' for region in settings.regions):
            preview.prop(settings, 'health', slider=True)
        if hw2:
            preview.prop(settings, 'hw2_weapon')
            preview.prop(settings, 'hw2_timed')
            if settings.hw2_timed:
                preview.prop(settings, 'hw2_time')
        preview.prop(settings, 'show_debris')
        if settings.show_debris:
            preview.prop(settings, 'separation')
        if settings.fields:
            draw_fields(layout, settings.fields)
        if not settings.regions:
            return
        layout.prop_search(settings, 'region', settings, 'regions', text='Region')
        region = settings.regions[min(settings.active_region, len(settings.regions) - 1)]
        if region.bone and owner.type == 'ARMATURE':
            layout.prop_search(region, 'bone', owner.data, 'bones')
        row = layout.row()
        row.template_list('HWEK_UL_damage_events', '', region, 'events', region, 'active_event', rows=3)
        col = row.column(align=True)
        col.operator('hwek.damage_entry', text='', icon='ADD').operation = 'ADD_EVENT'
        col.operator('hwek.damage_entry', text='', icon='REMOVE').operation = 'REMOVE_EVENT'
        if not region.events:
            return
        event = region.events[min(region.active_event, len(region.events) - 1)]
        if not event.path:
            return
        if not event.actions:
            layout.label(text='No additional change at this stage.', icon='INFO')
        elif all(action.kind.casefold() in ('multiframetextureindex', 'attachparticle', 'playsound') for action in event.actions):
            layout.label(text='No part replacement at this stage.', icon='INFO')
        row = layout.row()
        row.template_list('HWEK_UL_damage_actions', '', event, 'actions', event, 'active_action', rows=3)
        col = row.column(align=True)
        col.operator('hwek.damage_entry', text='', icon='ADD').operation = 'ADD_ACTION'
        col.operator('hwek.damage_entry', text='', icon='REMOVE').operation = 'REMOVE_ACTION'
        layout.use_property_split = True
        draw_fields(layout, region.fields)
        draw_fields(layout, event.fields)
        if event.actions:
            action = event.actions[min(event.active_action, len(event.actions) - 1)]
            layout.prop(action, 'action_type')
            if action.action_type == 'CUSTOM':
                layout.prop(action, 'kind')
            if action.kind.casefold() in ('swappart', 'throwpart', 'togglepart', 'hidepart'):
                layout.prop(action, 'mesh')
                if action.kind.casefold() == 'swappart':
                    layout.prop(action, 'replacement')
                if action.mesh is None:
                    layout.prop(action, 'target', text='Part Names')
            else:
                layout.prop(action, 'target', text='Sound' if action.kind.casefold() == 'playsound' else 'Target Bone')
            draw_fields(layout, action.fields)
