"""Object display settings and native UV offset parameters."""
import json
import xml.etree.ElementTree as ET
import bpy
from bpy.props import BoolProperty, CollectionProperty, FloatProperty, IntProperty, StringProperty
from ..errors import UgxError
from ..formats.vis import read_visual_document, visual_elements

__all__ = ('HWEK_VisualUV', 'HWEK_VisualUVSet', 'HWEK_VisualSettings', 'populate_visual_settings',
           'visual_marker_collection', 'apply_visual_uv_edits', 'migrate_visual_uv_objects')


def update_point_display(settings, _context):
    for obj in settings.id_data.children:
        if obj.get('hwek_vis_marker') not in ('point', 'attach'):
            continue
        obj.hide_set(not settings.show_points)
        obj.show_name = settings.show_names


class HWEK_VisualUV(bpy.types.PropertyGroup):
    path: StringProperty(options={'HIDDEN'})
    channel: StringProperty(options={'HIDDEN'})
    original_attrs: StringProperty(options={'HIDDEN'})
    original_u: FloatProperty(options={'HIDDEN'})
    original_v: FloatProperty(options={'HIDDEN'})
    u: FloatProperty(name='U', precision=4, options=set())
    v: FloatProperty(name='V', precision=4, options=set())


class HWEK_VisualUVSet(bpy.types.PropertyGroup):
    path: StringProperty(options={'HIDDEN'})
    channels: CollectionProperty(type=HWEK_VisualUV)


class HWEK_VisualSettings(bpy.types.PropertyGroup):
    show_points: BoolProperty(name='Attachment Points', default=True, update=update_point_display)
    show_names: BoolProperty(name='Point Names', default=False, update=update_point_display)
    uv_sets: CollectionProperty(type=HWEK_VisualUVSet)
    active_uv_set: IntProperty(name='Variant', min=0)
    uv_variant: StringProperty(name='Variant',
        get=lambda self: self.uv_sets[min(self.active_uv_set, len(self.uv_sets) - 1)].name if self.uv_sets else '',
        set=lambda self, name: setattr(self, 'active_uv_set', max(0, self.uv_sets.find(name))))


def visual_marker_collection(owner, _kind):
    parent = owner.users_collection[0] if owner.users_collection else bpy.context.scene.collection
    collection = owner.get('hwek_vis_points_collection')
    owned = isinstance(collection, bpy.types.Collection) and collection.name in parent.children
    if owned:
        previous = collection.get('hwek_vis_owner')
        owned = previous == owner or (previous is None and all(obj.parent == owner for obj in collection.objects))
    if not owned:
        collection = bpy.data.collections.new('Attachment Points')
        parent.children.link(collection)
        owner['hwek_vis_points_collection'] = collection
    collection['hwek_vis_owner'] = owner
    return collection


def merge_uv_values(elements, path, original, values):
    if not values:
        return False
    element = elements.get(path)
    if element is None or element.tag != 'uvOffset' or element.get('channel', '0') != original.get('channel', '0'):
        raise UgxError('VIS XML structure changed around a texture offset. Resolve the changed UV channel in the visual definition before exporting.')
    for key, value in values.items():
        if element.get(key, '') not in (original.get(key, ''), value):
            raise UgxError(f'VIS UV{original.get("channel", "0")}: {key} was edited differently in XML and in Texture Offsets.')
    element.attrib.update(values)
    return True


def apply_visual_uv_edits(root, owner):
    elements = visual_elements(root)
    changed = False
    for group in owner.hwek_visual.uv_sets:
        for channel in group.channels:
            if not channel.path:
                continue
            original = json.loads(channel.original_attrs)
            # RNA floats have less precision than native XML. Compare RNA baselines
            # so opening a model never rewrites untouched native decimal strings.
            values = {key: format(value, '.9g') for key, value, baseline in (
                ('uOfs', channel.u, channel.original_u), ('vOfs', channel.v, channel.original_v)) if value != baseline}
            changed |= merge_uv_values(elements, tuple(json.loads(channel.path)), original, values)
    return changed


def migrate_visual_uv_objects(owner, root):
    elements = visual_elements(root)
    markers = [obj for obj in owner.children if obj.get('hwek_vis_marker') == 'uvOffset']
    changed = False
    for marker in markers:
        original = json.loads(marker['hwek_vis_original_attrs'])
        values = {}
        for key, value in marker.items():
            if not key.startswith('vis_'):
                continue
            key = key[4:]
            if key in ('uOfs', 'vOfs') and float(value) == float(original.get(key, '0')):
                continue
            value = format(value, '.9g') if isinstance(value, float) else str(value)
            if value != original.get(key, ''):
                values[key] = value
        changed |= merge_uv_values(elements, tuple(json.loads(marker['hwek_vis_element_path'])), original, values)
    collection = owner.get('hwek_vis_uv_collection')
    owned = isinstance(collection, bpy.types.Collection) and (collection.get('hwek_vis_owner') == owner or
            (collection.get('hwek_vis_owner') is None and all(obj in markers for obj in collection.objects)))
    for marker in markers:
        for child in list(marker.children):
            world = child.matrix_world.copy()
            child.parent = owner
            child.matrix_world = world
        if bpy.context.view_layer.objects.active == marker:
            bpy.context.view_layer.objects.active = owner
            owner.select_set(True)
        bpy.data.objects.remove(marker, do_unlink=True)
    originals = json.loads(owner.get('hwek_vis_marker_originals', '{}'))
    paths = json.loads(owner.get('hwek_vis_marker_paths', '[]'))
    uv_paths = {path for path, element in elements.items() if element.tag == 'uvOffset'}
    paths = [path for path in paths if tuple(path) not in uv_paths and
             not {'channel', 'uOfs', 'vOfs'}.issubset(originals.get(json.dumps(path), {}))]
    owner['hwek_vis_marker_paths'] = json.dumps(paths)
    owner['hwek_vis_marker_originals'] = json.dumps({json.dumps(path): originals.get(json.dumps(path), {}) for path in paths})
    if owned and not collection.objects and not collection.children:
        bpy.data.collections.remove(collection)
    if 'hwek_vis_uv_collection' in owner:
        del owner['hwek_vis_uv_collection']
    return changed


def populate_visual_settings(owner, root=None):
    text = owner.get('hwek_vis_text')
    if root is None:
        root = read_visual_document(text.as_string().encode()).root
    changed = apply_visual_uv_edits(root, owner)
    changed |= migrate_visual_uv_objects(owner, root)
    if changed:
        text.clear()
        text.write(ET.tostring(root, encoding='unicode'))
    settings = owner.hwek_visual
    active_path = settings.uv_sets[min(settings.active_uv_set, len(settings.uv_sets) - 1)].path if settings.uv_sets else ''
    settings.uv_sets.clear()
    elements = visual_elements(root)
    names = json.loads(owner.get('hwek_vis_models', '[]'))
    groups = {}
    for path, element in elements.items():
        if element.tag != 'uvOffset' or not any(elements[path[:index]].tag == 'model' and
                elements[path[:index]].get('name', '') in names for index in range(len(path))):
            continue
        asset_path = path[:-1]
        if asset_path not in groups:
            group = settings.uv_sets.add()
            group.path = json.dumps(asset_path)
            branch = next((elements[path[:index]] for index in range(len(asset_path), 0, -1)
                           if elements[path[:index]].tag == 'logicdata'), None)
            group.name = (branch.get('name') or branch.get('value') or 'No Upgrade').replace('_', ' ').title() if branch is not None else 'Base'
            if branch is not None and not branch.attrib:
                group.name = (branch.text or '').strip() or 'Variant'
            groups[asset_path] = group
        channel = groups[asset_path].channels.add()
        channel.path = json.dumps(path)
        channel.channel = element.get('channel', '0')
        channel.original_attrs = json.dumps(element.attrib)
        channel.u = channel.original_u = float(element.get('uOfs', '0'))
        channel.v = channel.original_v = float(element.get('vOfs', '0'))
    for marker in owner.children:
        if marker.get('hwek_vis_marker') not in ('point', 'attach'):
            continue
        collection = visual_marker_collection(owner, marker['hwek_vis_marker'])
        if marker.name not in collection.objects:
            collection.objects.link(marker)
        for previous in list(marker.users_collection):
            if previous != collection and previous in owner.users_collection:
                previous.objects.unlink(marker)
    settings.active_uv_set = next((index for index, group in enumerate(settings.uv_sets) if group.path == active_path),
                                 next((index for index, group in enumerate(settings.uv_sets) if group.name == 'Base'), 0))
    update_point_display(settings, None)
    owner['hwek_visual_properties_ready'] = True
    owner['hwek_vis_uv_parameters_version'] = 1
