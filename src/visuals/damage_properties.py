"""Editable projections of native damage entries and mesh visibility previews."""
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from ..formats.native_xml import read_native_xml_document, xml_elements
from ..ugx.scene import ugx_authoring_mesh_objects
from ..ugx.naming import current_ugx_mesh_name
from .damage_preview import update_damage_debris
from .damage_format import model_damage_version, require_damage_version
from .damage_hw2 import ACTION_TYPES as HW2_ACTION_TYPES, WEAPON_TYPES, health_stage, preview_state

__all__ = ('HWEK_DamageField', 'HWEK_DamageAction', 'HWEK_DamageEvent', 'HWEK_DamageRegion', 'HWEK_DamageSettings',
           'populate_damage_settings', 'preview_damage', 'damage_document', 'write_damage_document', 'apply_damage_object_edits')


def damage_document(owner):
    require_damage_version(owner, model_damage_version(owner))
    text = owner.get('hwek_damage_text')
    return text, read_native_xml_document(text.as_string().encode()).root


def write_damage_document(owner, root):
    text = owner['hwek_damage_text']
    text.clear()
    text.write(ET.tostring(root, encoding='unicode'))
    owner['hwek_damage_projection_signature'] = hashlib.sha256(text.as_string().encode()).hexdigest()


def damage_part_name(obj, binding, native):
    name = current_ugx_mesh_name(obj)
    if binding:
        return name if name != binding else native
    # Older saved projections lack a binding snapshot. Keep native case aliases.
    return native if name.casefold() == native.casefold() else name


def apply_damage_object_edits(owner, root):
    text = owner.get('hwek_damage_text')
    if text is None or hashlib.sha256(text.as_string().encode()).hexdigest() != owner.get('hwek_damage_projection_signature'):
        return False
    elements = xml_elements(root)
    changed = False
    for region in owner.hwek_damage.regions:
        for event in region.events:
            for action in event.actions:
                if action.kind.casefold() not in ('swappart', 'throwpart', 'togglepart', 'hidepart'):
                    continue
                node = elements.get(tuple(json.loads(action.path)))
                if node is None or node.tag != 'action':
                    continue
                names = [name.strip() for name in (node.text or '').split(',')]
                original = list(names)
                if action.mesh is not None and action.kind.casefold() in ('swappart', 'throwpart', 'togglepart', 'hidepart'):
                    names[0] = damage_part_name(action.mesh, action.mesh_binding_name, names[0])
                if action.replacement is not None and action.kind.casefold() == 'swappart' and len(names) == 2:
                    names[1] = damage_part_name(action.replacement, action.replacement_binding_name, names[1])
                if names != original:
                    node.text = ','.join(names) or None
                    changed = True
    return changed


def edit_field(field, _context):
    owner = field.id_data
    if owner.hwek_damage.loading:
        return
    _, root = damage_document(owner)
    node = xml_elements(root)[tuple(json.loads(field.path))]
    value = format(field.number, '.9g') if field.format == 'NUMBER' else ('true' if field.enabled else 'false') if field.format == 'BOOL' else field.value
    if field.element_value:
        node.text = value
    else:
        node.set(field.name, value)
    write_damage_document(owner, root)
    preview_damage(owner)


def edit_action(action, _context):
    owner = action.id_data
    if owner.hwek_damage.loading:
        return
    _, root = damage_document(owner)
    node = xml_elements(root)[tuple(json.loads(action.path))]
    node.set('type', action.kind)
    if action.kind.casefold() == 'swappart':
        # Keep unresolved native names until a replacement is explicitly chosen.
        names = [name.strip() for name in (node.text or ',').split(',')]
        names += [''] * (2 - len(names))
        if action.mesh is not None:
            names[0] = current_ugx_mesh_name(action.mesh)
        if action.replacement is not None:
            names[1] = current_ugx_mesh_name(action.replacement)
        node.text = ','.join(names[:2])
    elif action.kind.casefold() in ('throwpart', 'togglepart', 'hidepart') and action.mesh is not None:
        node.text = current_ugx_mesh_name(action.mesh)
    else:
        node.text = action.target or None
    action.mesh_binding_name = current_ugx_mesh_name(action.mesh) if action.mesh is not None else ''
    action.replacement_binding_name = current_ugx_mesh_name(action.replacement) if action.replacement is not None else ''
    write_damage_document(owner, root)
    preview_damage(owner)


def edit_region(region, _context):
    owner = region.id_data
    if owner.hwek_damage.loading:
        return
    _, root = damage_document(owner)
    node = xml_elements(root)[tuple(json.loads(region.path))]
    if 'bone' in node.attrib or region.bone:
        node.set('bone', region.bone)
    write_damage_document(owner, root)
    preview_damage(owner)


def change_preview(item, _context):
    settings = item.id_data.hwek_damage
    if not settings.loading:
        if model_damage_version(item.id_data) == 6 and hasattr(item, 'events') and item.name == 'Health':
            count = len(item.events) - 1
            settings['hw2_health'] = 100. * (count - max(1, item.active_event)) / (count - 1) if count > 1 else 100.
        preview_damage(item.id_data)


def mesh_poll(item, obj):
    return obj.type == 'MESH' and obj in ugx_authoring_mesh_objects(item.id_data)


class HWEK_DamageField(bpy.types.PropertyGroup):
    path: StringProperty(options={'HIDDEN'})
    format: StringProperty(options={'HIDDEN'})
    element_value: BoolProperty(options={'HIDDEN'})
    label: StringProperty(options={'HIDDEN'})
    value: StringProperty(name='Value', update=edit_field)
    number: FloatProperty(name='Value', update=edit_field)
    enabled: BoolProperty(name='Enabled', update=edit_field)


DAMAGE_ACTION_TYPES = (
    ('swappart', 'Replace Part', ''), ('throwpart', 'Detach Part', ''), ('togglepart', 'Show / Hide Part', ''), ('hidepart', 'Hide Part', ''),
    ('attachparticle', 'Particle Effect', ''), ('playsound', 'Sound', ''), ('multiframeTextureIndex', 'Texture Frame', ''),
    ('CUSTOM', 'Custom', ''),
)


def damage_action_types(action, _context=None):
    return HW2_ACTION_TYPES if model_damage_version(action.id_data) == 6 else DAMAGE_ACTION_TYPES


def action_type_index(action):
    types = damage_action_types(action)
    return next((index for index, (kind, _, _) in enumerate(types) if kind.casefold() == action.kind.casefold()), len(types) - 1)


def set_action_type(action, index):
    types = damage_action_types(action)
    if types[index][0] != 'CUSTOM':
        action.kind = types[index][0]
        if model_damage_version(action.id_data) != 6:
            return
        settings = action.id_data.hwek_damage
        settings.loading = True
        try:
            _, root = damage_document(action.id_data)
            elements = xml_elements(root)
            node = elements[tuple(json.loads(action.path))]
            action.fields.clear()
            populate_action_fields(action, node, tuple(json.loads(action.path)), {child: path for path, child in elements.items()}, 6)
        finally:
            settings.loading = False


class HWEK_DamageAction(bpy.types.PropertyGroup):
    path: StringProperty(options={'HIDDEN'})
    kind: StringProperty(name='Native Type', update=edit_action)
    action_type: EnumProperty(name='Type', items=damage_action_types, get=action_type_index, set=set_action_type)
    target: StringProperty(name='Target', update=edit_action)
    mesh: PointerProperty(name='Part', type=bpy.types.Object, poll=mesh_poll, update=edit_action)
    replacement: PointerProperty(name='Replacement', type=bpy.types.Object, poll=mesh_poll, update=edit_action)
    mesh_binding_name: StringProperty(options={'HIDDEN'})
    replacement_binding_name: StringProperty(options={'HIDDEN'})
    fields: CollectionProperty(type=HWEK_DamageField)


class HWEK_DamageEvent(bpy.types.PropertyGroup):
    path: StringProperty(options={'HIDDEN'})
    actions: CollectionProperty(type=HWEK_DamageAction)
    active_action: IntProperty(name='Action', min=0)
    fields: CollectionProperty(type=HWEK_DamageField)


class HWEK_DamageRegion(bpy.types.PropertyGroup):
    path: StringProperty(options={'HIDDEN'})
    bone: StringProperty(name='Bone', update=edit_region)
    events: CollectionProperty(type=HWEK_DamageEvent)
    active_event: IntProperty(name='Stage', min=0, update=change_preview)
    fields: CollectionProperty(type=HWEK_DamageField)


def health_region(settings):
    return next((region for region in settings.regions if region.name == 'Health'), None)


def preview_health(settings):
    if model_damage_version(settings.id_data) == 6:
        return float(settings.get('hw2_health', 100.))
    region = health_region(settings)
    count = len(region.events) - 1 if region else 0
    return 100.0 * (count - max(1, region.active_event)) / (count - 1) if count > 1 else 100.0


def set_preview_health(settings, value):
    if model_damage_version(settings.id_data) == 6:
        settings['hw2_health'] = value
        region = health_region(settings)
        settings.loading = True
        try:
            if region is not None:
                region.active_event = health_stage(value, len(region.events) - 1)
        finally:
            settings.loading = False
        preview_damage(settings.id_data)
        return
    region = health_region(settings)
    count = len(region.events) - 1 if region else 0
    if count > 1:
        region.active_event = count - math.ceil(value * (count - 1) / 100.0 - 1e-9)


def reset_preview(settings):
    settings.loading = True
    for region in settings.regions:
        region.active_event = 0
    if model_damage_version(settings.id_data) == 6:
        settings['hw2_health'] = 100.
    settings.loading = False
    preview_damage(settings.id_data)


class HWEK_DamageSettings(bpy.types.PropertyGroup):
    loading: BoolProperty(options={'HIDDEN', 'SKIP_SAVE'})
    fields: CollectionProperty(type=HWEK_DamageField)
    regions: CollectionProperty(type=HWEK_DamageRegion)
    active_region: IntProperty(name='Region', min=0)
    region: StringProperty(name='Region',
        get=lambda self: self.regions[min(self.active_region, len(self.regions) - 1)].name if self.regions else '',
        set=lambda self, name: setattr(self, 'active_region', max(0, self.regions.find(name))))
    preview: BoolProperty(name='Preview Damage', default=True, update=change_preview)
    health: FloatProperty(name='Health', description='Preview the native health stages', subtype='PERCENTAGE', min=0, max=100,
        get=preview_health, set=set_preview_health)
    show_debris: BoolProperty(name='Show Detached Parts', description='Inspect detached parts in a separated view', update=change_preview)
    separation: FloatProperty(name='Separation', description='Viewport preview distance, does not change game physics',
        subtype='DISTANCE', min=0, default=1, soft_max=5, update=change_preview)
    hw2_weapon: EnumProperty(name='Weapon', description='Weapon damage category used by HW2 damage actions',
        items=WEAPON_TYPES, default='Medium', update=change_preview)
    hw2_timed: BoolProperty(name='Preview Timing', description='Preview delays within the selected HW2 stage', update=change_preview)
    hw2_time: FloatProperty(name='Event Time', description='Seconds since the selected stage began', min=0,
        subtype='TIME', update=change_preview)


def populate_fields(collection, node, path, excluded=()):
    for key, value in node.attrib.items():
        if key in excluded:
            continue
        field = collection.add()
        field.name, field.path, field.value = key, json.dumps(path), value
        if value.casefold() in ('true', 'false'):
            field.format, field.enabled = 'BOOL', value.casefold() == 'true'
        else:
            try:
                number = float(value)
            except ValueError:
                field.format = 'TEXT'
            else:
                field.format, field.number = 'NUMBER', number


def populate_action_fields(action, node, path, paths, version):
    populate_fields(action.fields, node, path, ('type',))
    if version != 6:
        return
    # Missing attributes remain implicit in the XML until edited. These are
    # the retail parser defaults, not replacements for imported values.
    defaults = {'delay': '0', 'weaponFilter': 'All', 'hpThreshold': '1'}
    missing = ET.Element('action', {key: value for key, value in defaults.items() if key not in node.attrib})
    populate_fields(action.fields, missing, path)
    for child in node.iter():
        if child is node or not isinstance(child.tag, str):
            continue
        start = len(action.fields)
        populate_fields(action.fields, child, paths[child])
        for field in list(action.fields)[start:]:
            field.label = child.tag + ' / ' + field.name


def damage_event_summary(node):
    counts = {}
    for action in node.findall('action'):
        kind = action.get('type', '').casefold()
        if kind == 'multiframetextureindex':
            label = 'Texture frame ' + action.get('toindex', '0')
        else:
            label = {'swappart': 'Replace part', 'throwpart': 'Detach part', 'togglepart': 'Visibility',
                     'hidepart': 'Hide part', 'attachparticle': 'Effect', 'playsound': 'Sound'}.get(kind, action.get('type', 'Action'))
        counts[label] = counts.get(label, 0) + 1
    return ' + '.join(label + (' x' + str(count) if count > 1 else '') for label, count in counts.items()) or 'No change'


def populate_damage_settings(owner):
    settings = owner.hwek_damage
    previous = {item.path: item.active_event for item in settings.regions}
    selected = settings.active_region
    first_load = not settings.regions
    settings.loading = True
    try:
        _, root = damage_document(owner)
        version = model_damage_version(owner)
        require_damage_version(owner, version)
        if apply_damage_object_edits(owner, root):
            write_damage_document(owner, root)
        settings.regions.clear()
        settings.fields.clear()
        elements = xml_elements(root)
        paths = {node: path for path, node in elements.items()}
        populate_fields(settings.fields, root, ())
        for index, child in enumerate(root):
            if isinstance(child.tag, str) and len(child) == 0 and (child.text or '').strip():
                field = settings.fields.add()
                field.name, field.path, field.element_value = child.tag, json.dumps((index,)), True
                field.value = child.text.strip()
                try:
                    field.number = float(field.value)
                except ValueError:
                    field.format = 'TEXT'
                else:
                    field.format = 'NUMBER'
                if version == 6 and field.value.casefold() in ('true', 'false'):
                    field.format, field.enabled = 'BOOL', field.value.casefold() == 'true'
        mesh_lookup = {}
        for obj in ugx_authoring_mesh_objects(owner):
            mesh_lookup.setdefault(current_ugx_mesh_name(obj).casefold(), obj)
        labels = {'percentagebased': 'Health', 'cryopercentagebased': 'Freeze', 'impactpoint': 'Impact',
                  'shatterdeathevent': 'Shatter Death', 'deathevent': 'Death'}
        for path, node in elements.items():
            if not isinstance(node.tag, str):
                continue
            tag = node.tag.casefold() if version == 6 else node.tag
            events = [child for child in node if child.tag == 'event']
            if not events and not tag.endswith('event') and tag not in ('percentagebased', 'cryopercentagebased', 'impactpoint'):
                continue
            if tag == 'event':
                continue
            region = settings.regions.add()
            region.path = json.dumps(path)
            region.name = labels.get(tag, node.tag) + (' / ' + node.get('name', node.get('bone', '')) if tag == 'impactpoint' else '')
            region.bone = node.get('bone', '')
            populate_fields(region.fields, node, path, ('bone',))
            intact = region.events.add()
            intact.name = 'Intact'
            for index, event_node in enumerate(events or ([node] if node.tag.endswith('event') else [])):
                event = region.events.add()
                event.name = str(index + 1) + '. ' + damage_event_summary(event_node) if events else region.name
                event.path = json.dumps(paths[event_node])
                populate_fields(event.fields, event_node, paths[event_node])
                for action_node in event_node:
                    if action_node.tag != 'action':
                        continue
                    action = event.actions.add()
                    action.path = json.dumps(paths[action_node])
                    action.kind, action.target = action_node.get('type', ''), action_node.text or ''
                    action.name = action.kind
                    names = [name.strip().casefold() for name in action.target.split(',')]
                    if action.kind.casefold() in ('swappart', 'throwpart', 'togglepart', 'hidepart'):
                        action.mesh = mesh_lookup.get(names[0])
                    if action.kind.casefold() == 'swappart' and len(names) == 2:
                        action.replacement = mesh_lookup.get(names[1])
                    action.mesh_binding_name = current_ugx_mesh_name(action.mesh) if action.mesh is not None else ''
                    action.replacement_binding_name = current_ugx_mesh_name(action.replacement) if action.replacement is not None else ''
                    populate_action_fields(action, action_node, paths[action_node], paths, version)
            region.active_event = min(previous.get(region.path, 0), len(region.events) - 1)
        settings.active_region = next((index for index, region in enumerate(settings.regions) if region.bone), 0) if first_load else min(selected, max(0, len(settings.regions) - 1))
    finally:
        settings.loading = False
    if version == 6:
        region = health_region(settings)
        count = len(region.events) - 1 if region is not None else 0
        if 'hw2_health' not in settings:
            settings['hw2_health'] = 100. * (count - max(1, region.active_event)) / (count - 1) if count > 1 else 100.
        elif region is not None and (region.active_event or settings.health < 100.):
            settings.loading = True
            try:
                region.active_event = health_stage(settings.health, count)
            finally:
                settings.loading = False
    owner['hwek_damage_projection_signature'] = hashlib.sha256(owner['hwek_damage_text'].as_string().encode()).hexdigest()
    owner['hwek_damage_projection_game'] = version
    preview_damage(owner)


def preview_damage(owner):
    settings = owner.hwek_damage
    _, root = damage_document(owner)
    apply_damage_object_edits(owner, root)
    meshes = [(current_ugx_mesh_name(obj).casefold(), obj) for obj in ugx_authoring_mesh_objects(owner)]
    visibility = {}
    for action in root.iter('action'):
        if action.get('type', '').casefold() == 'swappart':
            names = [name.strip().casefold() for name in (action.text or '').split(',')]
            if len(names) == 2:
                visibility.setdefault(names[0], True)
                visibility[names[1]] = False
    elements = xml_elements(root)
    if model_damage_version(owner) == 6:
        require_damage_version(owner, 6)
        visibility, detached = preview_state(
            root, [current_ugx_mesh_name(obj) for _, obj in meshes], settings.health,
            settings.regions if settings.preview else (), elements, settings.hw2_weapon,
            settings.hw2_timed, settings.hw2_time)
        for name, obj in meshes:
            if 'hwek_damage_original_hidden' not in obj:
                obj['hwek_damage_original_hidden'] = obj.hide_get()
                obj['hwek_damage_original_render_hidden'] = obj.hide_render
            obj.hide_set(bool(obj['hwek_damage_original_hidden']) or (settings.preview and not visibility.get(name, True)))
            obj.hide_render = bool(obj['hwek_damage_original_render_hidden']) or (settings.preview and not visibility.get(name, True))
        update_damage_debris(owner, meshes, detached, settings.preview and settings.show_debris, settings.separation)
        return
    if int(owner.get('hwek_ugx_cache_version', 0)) == 4:
        has_intact = any('intact' in current_ugx_mesh_name(obj) for _, obj in meshes)
        fractured = any(region.active_event > (1 if elements[tuple(json.loads(region.path))].tag == 'percentagebased' else 0)
                        for region in settings.regions if region.bone or elements[tuple(json.loads(region.path))].tag == 'percentagebased')
        for name, obj in meshes:
            native = current_ugx_mesh_name(obj)
            visible = 'intact' in native if has_intact and not fractured else not ('damage' in native or 'intact' in native)
            visibility[name] = visibility.get(name, True) and visible and 'optional' not in native and name != 'spartandriver'
    detached = set()
    if settings.preview:
        for region in settings.regions:
            for event in list(region.events)[1:region.active_event + 1]:
                for action in elements[tuple(json.loads(event.path))].findall('action'):
                    names = [name.strip().casefold() for name in (action.text or '').split(',')]
                    kind = action.get('type', '').casefold()
                    if kind == 'swappart' and len(names) == 2:
                        visibility[names[0]], visibility[names[1]] = False, True
                        detached.discard(names[1])
                    elif kind in ('throwpart', 'togglepart', 'hidepart'):
                        for name in names:
                            visibility[name] = kind == 'togglepart' and action.get('show', 'true').casefold() == 'true'
                            if kind == 'throwpart':
                                detached.add(name)
                            else:
                                detached.discard(name)
    for name, obj in meshes:
        if name not in visibility and 'hwek_damage_original_hidden' not in obj:
            continue
        if 'hwek_damage_original_hidden' not in obj:
            obj['hwek_damage_original_hidden'] = obj.hide_get()
            obj['hwek_damage_original_render_hidden'] = obj.hide_render
        hidden = bool(obj['hwek_damage_original_hidden']) or (settings.preview and not visibility.get(name, True))
        obj.hide_set(hidden)
        obj.hide_render = bool(obj['hwek_damage_original_render_hidden']) or (settings.preview and not visibility.get(name, True))
    update_damage_debris(owner, meshes, detached, settings.preview and settings.show_debris, settings.separation)
