"""Create native detachment events from selected vehicle parts."""
from functools import lru_cache
import json
import xml.etree.ElementTree as ET
import bpy
from bpy.props import EnumProperty, FloatProperty, IntProperty, StringProperty
from mathutils import Vector
from ..errors import UgxError
from ..formats.native_xml import xml_elements
from ..formats.vis import read_visual_document
from ..materials.context import ugx_material_cache_version
from ..ugx.bones import new_export_bone_name
from ..ugx.naming import current_ugx_mesh_name
from ..ugx.scene import ugx_armature_for
from .damage import damage_text
from .damage_format import model_damage_version, require_damage_version
from .damage_hw2 import health_region_node
from .damage_properties import damage_document, write_damage_document, populate_damage_settings, reset_preview
from .operators import create_visual_document
from .scene import visual_owner, visual_text, refresh_visual_markers, apply_visual_document_edits

__all__ = ('HWEK_OT_make_detachable', 'HWEK_OT_reset_damage_preview', 'menu_damage_authoring')


def vehicle_for_operator(operator):
    obj = bpy.data.objects.get(operator.vehicle)
    return visual_owner(obj) if obj is not None else None


def health_event_count(owner):
    if damage_text(owner) is not None:
        _, root = damage_document(owner)
        health = health_region_node(root) if model_damage_version(owner) == 6 else root.find('percentagebased')
        if health is not None and len(health.findall('event')) > 1:
            return len(health.findall('event'))
    return 5


@lru_cache(maxsize=64)
def health_stage_choices(count):
    # The native tracker indexes ceil(health * (count - 1)); event zero is intact.
    return tuple((str(index), format(100 * (count - 1 - index) / (count - 1), '.4g') + '% health',
                  'Detach when health reaches this level') for index in range(1, count))


def health_stage_items(operator, _context):
    return health_stage_choices(health_event_count(vehicle_for_operator(operator)))


def get_health_stage(operator):
    count = health_event_count(vehicle_for_operator(operator))
    return min(int(operator.get('health_stage_index', max(0, count // 2 - 1))), count - 2)


def set_health_stage(operator, index):
    operator['health_stage_index'] = index


def part_center(obj):
    return obj.matrix_world @ (sum((Vector(corner) for corner in obj.bound_box), Vector()) / 8)


def bind_static_part(obj, rig):
    matrix = obj.matrix_world.copy()
    root = next(bone for bone in rig.data.bones if bone.parent is None)
    obj.parent = rig
    obj.matrix_world = matrix
    modifier = obj.modifiers.new('Vehicle', 'ARMATURE')
    modifier.object = rig
    bone_groups = {group.index for group in obj.vertex_groups if group.name in rig.data.bones}
    unweighted = [vertex.index for vertex in obj.data.vertices if not any(group.group in bone_groups and group.weight > 0 for group in vertex.groups)]
    if unweighted:
        group = obj.vertex_groups.get(root.name) or obj.vertex_groups.new(name=root.name)
        group.add(unweighted, 1.0, 'REPLACE')


def create_vehicle_rig(context, body, parts):
    rig = bpy.data.objects.new(body.name + ' Rig', bpy.data.armatures.new(body.name + ' Rig'))
    body.users_collection[0].objects.link(rig)
    context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    rig.data.edit_bones.new('GrannyRootBone').tail = (0, 0, 1)
    bpy.ops.object.mode_set(mode='OBJECT')
    objects = list(dict.fromkeys([body] + list(body.children_recursive) + parts))
    for obj in objects:
        if obj.get('hwek_vis_marker') and obj.parent == body:
            matrix = obj.matrix_world.copy()
            obj.parent = rig
            obj.matrix_world = matrix
            continue
        if obj.type != 'MESH' or ugx_armature_for(obj) is not None:
            continue
        bind_static_part(obj, rig)
    damage_keys = ('hwek_damage_text', 'hwek_damage_event_path', 'hwek_damage_projection_signature')
    for key in tuple(body.keys()):
        if key.startswith(('hwek_ugx_', 'hwek_vis_')) or key in damage_keys:
            rig[key] = body[key]
            if key.startswith('hwek_vis_') or key in damage_keys:
                del body[key]
    context.view_layer.update()
    return rig


def create_impact_bone(context, owner, part):
    context.view_layer.objects.active = owner
    bpy.ops.object.mode_set(mode='EDIT')
    bone = owner.data.edit_bones.new('bone_impact_' + part.name)
    bone.head = owner.matrix_world.inverted_safe() @ part_center(part)
    bone.tail = bone.head + Vector((0, 0, 0.15))
    bone.parent = next((item for item in owner.data.edit_bones if item != bone and item.parent is None), None)
    name = bone.name
    bpy.ops.object.mode_set(mode='OBJECT')
    return name


def link_impact_point(owner, bone):
    text = visual_text(owner)
    root = read_visual_document(text.as_string().encode()).root
    apply_visual_document_edits(root, owner)
    names = json.loads(owner.get('hwek_vis_models', '[]'))
    model = next((item for item in root.findall('model') if item.get('name') in names), root.find('model'))
    component = model.find('component')
    if component is None:
        component = ET.SubElement(model, 'component')
    if not any(point.get('bone') == bone and point.get('pointType') == 'Impact' for point in component.findall('point')):
        ET.SubElement(component, 'point', pointType='Impact', bone=bone, pointData='Metal')
    text.clear()
    text.write(ET.tostring(root, encoding='unicode'))
    refresh_visual_markers(owner)
    owner.hwek_visual.show_points = True


class HWEK_OT_make_detachable(bpy.types.Operator):
    bl_idname = 'hwek.make_detachable'
    bl_label = 'Make Detachable'
    bl_description = 'Detach selected separate mesh parts when the vehicle takes damage'
    bl_options = {'REGISTER', 'UNDO'}
    vehicle: StringProperty(name='Vehicle', description='Vehicle armature or body mesh')
    part_names: StringProperty(options={'HIDDEN'})
    trigger: EnumProperty(name='When', items=(('HEALTH', 'Health Level', 'Detach as vehicle health drops'),
        ('IMPACT', 'Impact on Part', 'Detach when an impact region advances')))
    health_stage: EnumProperty(name='Detach At', items=health_stage_items, get=get_health_stage, set=set_health_stage)
    bone: StringProperty(name='Impact Bone', description='Leave empty to create an impact bone at the part center')
    impact_stage: IntProperty(name='Hit Stage', min=1, default=1)
    force: FloatProperty(name='Force Multiplier', min=0, default=1, soft_max=5)

    @classmethod
    def poll(cls, context):
        cls.poll_message_set('Select separate mesh parts in Object Mode.')
        return context.mode == 'OBJECT' and context.object is not None and any(obj.type == 'MESH' and not obj.get('hwek_vis_marker') and not obj.get('hwek_damage_preview_source') for obj in context.selected_objects)

    def invoke(self, context, event):
        parts = [obj for obj in context.selected_objects if obj.type == 'MESH' and not obj.get('hwek_damage_preview_source')]
        owner = ugx_armature_for(context.object)
        if owner is None and context.object.parent is not None and context.object.parent.type == 'MESH':
            owner = visual_owner(context.object.parent)
        if owner is None:
            owner = next((obj for obj in context.selected_objects if obj.type == 'ARMATURE'), None)
        if owner is None:
            owner = next((obj for obj in parts if obj != context.object), None)
        self.vehicle = owner.name if owner else ''
        self.part_names = json.dumps([obj.name for obj in parts])
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop_search(self, 'vehicle', bpy.data, 'objects')
        owner = vehicle_for_operator(self)
        names = [name for name in json.loads(self.part_names or '[]') if owner is None or name != owner.name]
        layout.label(text='Parts: ' + ', '.join(names))
        if owner is not None and owner.type == 'MESH':
            layout.label(text='Creates a shared vehicle rig.', icon='ARMATURE_DATA')
        layout.prop(self, 'trigger')
        if self.trigger == 'HEALTH':
            layout.prop(self, 'health_stage')
        else:
            if owner is not None and owner.type == 'ARMATURE':
                layout.prop_search(self, 'bone', owner.data, 'bones')
            else:
                layout.prop(self, 'bone')
            if not self.bone:
                layout.label(text='Creates an impact point at the part.', icon='EMPTY_AXIS')
            layout.prop(self, 'impact_stage')
        layout.prop(self, 'force')

    def execute(self, context):
        try:
            owner = vehicle_for_operator(self)
            if owner is None or owner.type not in ('ARMATURE', 'MESH'):
                raise UgxError('Choose the vehicle armature or body mesh.')
            names = json.loads(self.part_names) if self.part_names else [obj.name for obj in context.selected_objects if obj.type == 'MESH']
            parts = [bpy.data.objects.get(name) for name in names if name != owner.name]
            if not parts or any(obj is None or obj.type != 'MESH' or not obj.data.polygons for obj in parts):
                raise UgxError('Select a separate mesh part to detach, then choose its vehicle.')
            for part in parts:
                rig = ugx_armature_for(part)
                if rig is not None and rig != owner:
                    raise UgxError(f'{part.name} belongs to another armature. Choose that vehicle instead.')
                if not part.get('hwek_ugx_export_enabled', True):
                    raise UgxError(f'{part.name} is disabled for UGX export. Enable it before making it detachable.')
                if ',' in current_ugx_mesh_name(part):
                    raise UgxError(f'{part.name}: part names cannot contain commas in damage actions.')
            if owner.type == 'ARMATURE' and not owner.data.bones:
                raise UgxError('The vehicle armature needs a root bone.')
            if self.trigger == 'IMPACT' and self.bone and (owner.type != 'ARMATURE' or self.bone not in owner.data.bones):
                raise UgxError(f'Impact bone {self.bone} is not in the vehicle armature.')
            if not owner.get('hwek_ugx_cache_version') and not owner.get('hwek_ugx_port_target'):
                meshes = [owner] + parts if owner.type == 'MESH' else list(owner.children_recursive) + parts
                versions = {ugx_material_cache_version(slot.material) for mesh in meshes if mesh.type == 'MESH' for slot in mesh.material_slots}
                versions.discard(0)
                if len(versions) > 1:
                    raise UgxError('Choose materials from one game before creating the damage template.')
                owner['hwek_ugx_port_target'] = versions.pop() if versions else 4
            visual = None
            if visual_text(owner) is not None:
                visual = read_visual_document(visual_text(owner).as_string().encode()).root
                apply_visual_document_edits(visual, owner)
            if damage_text(owner) is not None:
                require_damage_version(owner, model_damage_version(owner))
                _, root = damage_document(owner)
                if root.tag != 'damagetemplate':
                    raise UgxError('Damage document must have a damagetemplate root.')
            else:
                root = ET.Element('damagetemplate')
                if model_damage_version(owner) == 6:
                    ET.SubElement(root, 'intactToFracturedThreshold').text = '0.66'
                    ET.SubElement(root, 'fracturedToDestructibleThreshold').text = '0.66'
            if owner.type == 'MESH':
                if visual is not None:
                    text = visual_text(owner)
                    text.clear()
                    text.write(ET.tostring(visual, encoding='unicode'))
                owner = create_vehicle_rig(context, owner, parts)
                if visual is not None:
                    refresh_visual_markers(owner)
            for part in parts:
                if ugx_armature_for(part) is None:
                    bind_static_part(part, owner)
            create_visual_document(owner)
            if damage_text(owner) is None:
                text = bpy.data.texts.new(owner.name + ' Damage XML')
                text.use_fake_user = True
                text['hwek_damage_game'] = model_damage_version(owner)
                text.write(ET.tostring(root, encoding='unicode'))
                owner['hwek_damage_text'] = text
            if self.trigger == 'HEALTH':
                region = health_region_node(root) if model_damage_version(owner) == 6 else root.find('percentagebased')
                if region is None:
                    region = ET.SubElement(root, 'percentagebased')
                count = health_event_count(owner)
                while len(region.findall('event')) < count:
                    ET.SubElement(region, 'event')
                event = region.findall('event')[int(self.health_stage)]
            else:
                bone_name = self.bone or create_impact_bone(context, owner, parts[0])
                native = new_export_bone_name(owner.data.bones[bone_name], list(owner.data.bones).index(owner.data.bones[bone_name]))
                link_impact_point(owner, native)
                group = root.find('impactpointbased')
                if group is None:
                    group = ET.SubElement(root, 'impactpointbased')
                region = next((item for item in group if item.tag == 'impactpoint' and item.get('bone') == native), None)
                if region is None:
                    region = ET.SubElement(group, 'impactpoint', name=bone_name, bone=native)
                while len(region.findall('event')) < self.impact_stage:
                    ET.SubElement(region, 'event')
                event = region.findall('event')[self.impact_stage - 1]
            for part in parts:
                name = current_ugx_mesh_name(part)
                action = next((item for item in event.findall('action') if item.get('type') == 'throwpart' and item.text == name), None)
                if action is None:
                    action = ET.SubElement(event, 'action', type='throwpart')
                    action.text = name
                action.set('forcemultiplier', format(self.force, '.9g'))
            write_damage_document(owner, root)
            populate_damage_settings(owner)
            paths = {node: json.dumps(path) for path, node in xml_elements(root).items()}
            settings = owner.hwek_damage
            settings.active_region = next(index for index, item in enumerate(settings.regions) if item.path == paths[region])
            reset_preview(settings)
            settings.preview = True
            settings.show_debris = True
            context.view_layer.objects.active = owner
            for obj in context.selected_objects:
                obj.select_set(False)
            owner.select_set(True)
            if context.area is not None:
                for area in context.screen.areas:
                    if area.type == 'PROPERTIES':
                        area.spaces.active.context = 'OBJECT'
            self.report({'INFO'}, f'{len(parts)} detachable part(s) added. Preview with Health or the impact stage in Object Properties > Damage.')
        except (UgxError, ValueError, KeyError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class HWEK_OT_reset_damage_preview(bpy.types.Operator):
    bl_idname = 'hwek.reset_damage_preview'
    bl_label = 'Restore Vehicle'
    bl_description = 'Return every damage region to its intact preview'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        owner = visual_owner(context.object)
        return owner is not None and (damage_text(owner) is not None or isinstance(owner.get('hwek_vis_assembly_root'), bpy.types.Object))

    def execute(self, context):
        owner = visual_owner(context.object)
        if owner is None:
            return {'CANCELLED'}
        if damage_text(owner) is None:
            root = owner.get('hwek_vis_assembly_root')
            owner = root if isinstance(root, bpy.types.Object) and model_damage_version(root) == model_damage_version(owner) else None
        if owner is None or damage_text(owner) is None:
            return {'CANCELLED'}
        reset_preview(owner.hwek_damage)
        return {'FINISHED'}


def menu_damage_authoring(self, context):
    if HWEK_OT_make_detachable.poll(context):
        self.layout.separator()
        self.layout.operator('hwek.make_detachable', icon='MOD_EXPLODE')
