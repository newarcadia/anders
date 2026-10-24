"""Retain and edit complete native damage templates beside their models."""
from pathlib import Path
import copy
import xml.etree.ElementTree as ET
import zlib
import bpy
from ..errors import UgxError
from ..formats.native_xml import read_native_xml_document, native_xml_text, encode_native_xml_document
from ..ugx.damage import hw2_mesh_binding_changes
from ..ugx.naming import sibling_damage_xmb
from ..ugx.read import parse_ugx
from ..ugx.scene import ugx_authoring_mesh_objects
from .paths import resolve_visual_resource, resource_key, visual_resource_name
from .damage_properties import populate_damage_settings, apply_damage_object_edits
from .damage_format import model_damage_version, source_damage_version, require_damage_version, damage_document_version

__all__ = ('damage_text', 'load_damage_document', 'load_associated_damage', 'update_damage_mesh_links', 'prepare_damage_export')


def damage_text(owner):
    text = owner.get('hwek_damage_text') if owner is not None else None
    return text if isinstance(text, bpy.types.Text) else None


def load_damage_document(owner, path):
    if owner is None:
        raise UgxError('Select a model or armature before loading damage.')
    document = read_native_xml_document(Path(path).read_bytes())
    if document.root.tag.lower() != 'damagetemplate':
        raise UgxError(f'{Path(path).name} is not a damage template.')
    version = model_damage_version(owner)
    if damage_text(owner) is not None and damage_document_version(owner) == version:
        raise UgxError('This model already has an editable damage document.')
    source_version = source_damage_version(document, path, version)
    require_damage_version(owner, version, source_version)
    text = bpy.data.texts.new(Path(path).name + ' XML')
    text.write(native_xml_text(document))
    text.use_fake_user = True
    text['hwek_damage_source_path'] = str(Path(path).absolute())
    text['hwek_damage_source'] = zlib.compress(document.source).hex()
    text['hwek_damage_game'] = source_version
    owner['hwek_damage_text'] = text
    owner['hwek_damage_event_path'] = ''
    settings = owner.hwek_damage
    settings.loading = True
    try:
        settings.regions.clear()
        if version == 6:
            settings['hw2_health'] = 100.
    finally:
        settings.loading = False
    populate_damage_settings(owner)
    return text


def load_associated_damage(owner, visual=None, visual_path='', names=()):
    if damage_text(owner) is not None:
        return []
    candidates, issues = [], []
    explicit_reference = False
    if visual is not None:
        for model in visual.findall('model'):
            if model.get('name', '') not in names:
                continue
            source = str(owner.get('hwek_ugx_source_path', ''))
            for asset in model.findall('./component/asset'):
                if asset.get('type', '').casefold() != 'model':
                    continue
                if source and resource_key(asset.findtext('file', '')) != resource_key(visual_resource_name(source)):
                    continue
                for file in asset.findall('damagefile'):
                    reference = (file.text or '').strip()
                    if not reference:
                        continue
                    explicit_reference = True
                    suffixes = ('.dmg.xmb', '.dmg', '.xml') if model_damage_version(owner) == 6 else ('.dmg', '.xml')
                    found = next((path for suffix in suffixes
                                  if (path := resolve_visual_resource(visual_path, reference, suffix)) is not None), None)
                    if found is not None:
                        candidates.append(found)
                    else:
                        issues.append('Missing damage template: ' + reference)
    if not candidates and not explicit_reference:
        source = str(owner.get('hwek_ugx_source_path', ''))
        path = sibling_damage_xmb(source, model_damage_version(owner))
        if path:
            expected = {Path(source).stem.casefold(), Path(source).stem.removeprefix('mesh_').casefold()}
            damage_name = Path(path).name.removesuffix('.xmb').removesuffix('.dmg').casefold()
            # An attached wheel or turret must not inherit another model's DMG
            # merely because both files share a folder.
            if visual is None or damage_name in expected:
                candidates.append(Path(path))
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) == 1:
        load_damage_document(owner, candidates[0])
    elif len(candidates) > 1:
        issues.append('Several damage templates reference this model. Use Load Damage to choose one.')
    return issues


def update_damage_mesh_links(root, renamed, additions, removed):
    parents = {child: parent for parent in root.iter() for child in parent}
    for element in list(root.iter()):
        is_mesh_text = element.tag == 'mesh' or (element.tag == 'action' and element.get('type', '').lower() in ('swappart', 'throwpart', 'togglepart', 'hidepart'))
        if is_mesh_text and element.text:
            original = [name.strip() for name in element.text.split(',')]
            if element.tag == 'mesh' and len(original) == 1 and original[0] not in removed:
                element.text = renamed.get(original[0], original[0])
                for name in additions.get(original[0], ()):
                    clone = copy.deepcopy(element)
                    clone.text = name
                    parents[element].append(clone)
                continue
            if element.get('type', '').lower() == 'swappart' and len(original) == 2:
                if any(name in removed for name in original):
                    parents[element].remove(element)
                    continue
                element.text = ','.join(renamed.get(name, name) for name in original)
                offset = list(parents[element]).index(element) + 1
                # A swap is a pair, not a mesh list. Split source parts hide
                # together, and split replacements become visible together.
                for index, name in enumerate(original):
                    for extra in additions.get(name, ()):
                        clone = copy.deepcopy(element)
                        clone.set('type', 'togglepart')
                        clone.set('show', 'true' if index else 'false')
                        clone.text = extra
                        parents[element].insert(offset, clone)
                        offset += 1
                continue
            names = []
            for name in original:
                if name not in removed:
                    names.append(renamed.get(name, name))
                    names.extend(additions.get(name, ()))
            if not names or (element.get('type') == 'swappart' and len(names) < 2):
                parents[element].remove(element)
                continue
            element.text = ','.join(dict.fromkeys(names))
        name = element.get('mesh')
        if name is not None:
            if name in removed:
                parents[element].remove(element)
                continue
            element.set('mesh', renamed.get(name, name))
            for extra in additions.get(name, ()):
                clone = copy.deepcopy(element)
                clone.set('mesh', extra)
                parents[element].append(clone)


def prepare_damage_export(owner, model_path, version):
    text = damage_text(owner)
    if text is None:
        return []
    document = read_native_xml_document(text.as_string().encode('utf-8'))
    if document.root.tag.lower() != 'damagetemplate':
        raise UgxError('Damage XML must have a damagetemplate root element.')
    require_damage_version(owner, version)
    apply_damage_object_edits(owner, document.root)
    source_path = str(owner.get('hwek_ugx_source_path', ''))
    if source_path and Path(source_path).is_file():
        model = parse_ugx(source_path)
        changes = hw2_mesh_binding_changes(model, ugx_authoring_mesh_objects(owner))
        if any(changes):
            update_damage_mesh_links(document.root, *changes)
    source = zlib.decompress(bytes.fromhex(text['hwek_damage_source'])) if text.get('hwek_damage_source') else b''
    xml = ET.tostring(document.root, encoding='unicode')
    payload = encode_native_xml_document(xml, source, version == 6)
    path = Path(model_path).with_suffix('.dmg.xmb' if version == 6 else '.dmg')
    return [(str(path), payload)]
