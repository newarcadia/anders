"""Prepare VIS documents and their editable animation dependencies."""
from pathlib import Path, PurePosixPath
import json
import xml.etree.ElementTree as ET
import zlib
import bpy
from ..errors import UgxError
from ..animation.export import prepare_uax_export
from ..formats.vis import encode_visual_document, read_visual_document
from .paths import find_visual_documents, visual_resource_name, visual_resource_roots
from .scene import apply_visual_document_edits, load_visual_document, visual_owner, visual_text
from .damage import damage_text, load_associated_damage, prepare_damage_export

__all__ = ('prepare_visual_export',)


def animation_destination(root, reference):
    path = PurePosixPath(reference.replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts or any(':' in part for part in path.parts):
        raise UgxError(f'VIS animation path must be relative to the game resource root: {reference}')
    target = root.joinpath(*path.parts)
    return target if target.suffix.casefold() == '.uax' else target.with_name(target.name + '.uax')


def animation_reference_key(reference):
    return reference.replace('\\', '/').casefold().removesuffix('.uax')


def animation_link_nodes(models, action, reference):
    bindings = json.loads(action.get('hwek_vis_animation_bindings', '[]'))
    selected = {(binding['model'], binding['state'], binding['exit_action'].casefold()) for binding in bindings}
    result = []
    for model in models:
        for animation in model.findall('anim'):
            identity = (model.get('name', ''), animation.get('type', ''), animation.get('exitAction', 'Loop').casefold())
            if selected and identity not in selected:
                continue
            for asset in animation.findall('asset'):
                node = asset.find('file')
                if asset.get('type', '').casefold() in ('anim', 'granny') and node is not None and animation_reference_key(node.text or '') == animation_reference_key(reference):
                    result.append(node)
    return result


def split_animation_reference(reference, action, reserved):
    path = PurePosixPath(reference.replace('\\', '/'))
    stem = path.name[:-4] if path.name.casefold().endswith('.uax') else path.name
    suffix = 'loop' if action.get('hwek_uax_looping') else 'held'
    extension = path.name[-4:] if path.name.casefold().endswith('.uax') else ''
    candidate = path.with_name(f'{stem}_{suffix}{extension}').as_posix()
    index = 2
    while animation_reference_key(candidate) in reserved:
        candidate = path.with_name(f'{stem}_{suffix}_{index}{extension}').as_posix()
        index += 1
    reserved.add(animation_reference_key(candidate))
    return candidate.replace('/', '\\') if '\\' in reference else candidate


def prepare_visual_export(obj, path, model_path=None, binary=None, animations=True):
    owner = visual_owner(obj)
    if owner is None:
        raise UgxError('Select a model or armature before exporting VIS.')
    path = Path(path).absolute()
    text = visual_text(owner)
    source_model = str(owner.get('hwek_ugx_source_path', ''))
    if text is None and source_model:
        documents, _ = find_visual_documents(source_model)
        if len(documents) == 1:
            load_visual_document(owner, documents[0], animations=False)
            text = visual_text(owner)
        else:
            load_associated_damage(owner)
    source = b''
    if text is not None:
        root = read_visual_document(text.as_string().encode('utf-8')).root
        source = zlib.decompress(bytes.fromhex(text['hwek_vis_source']))
        apply_visual_document_edits(root, owner)
    else:
        root = ET.Element('visual', defaultmodel='Default')
        component = ET.SubElement(ET.SubElement(root, 'model', name='Default'), 'component')
        asset = ET.SubElement(component, 'asset', type='Model')
        for channel in range(3):
            ET.SubElement(asset, 'uvOffset', channel=str(channel), uOfs='0', vOfs='0')
        ET.SubElement(asset, 'file').text = visual_resource_name(model_path or owner.get('hwek_ugx_source_path') or path.with_suffix('.ugx'))
    original_model = str(owner.get('hwek_vis_model_source', owner.get('hwek_ugx_source_path', '')))
    old_ref = visual_resource_name(original_model).replace('\\', '/').casefold() if original_model else ''
    if model_path is not None:
        for asset in root.findall('./model/component/asset'):
            file = asset.find('file')
            if asset.get('type', '').casefold() == 'model' and file is not None:
                reference = (file.text or '').replace('\\', '/').removesuffix('.ugx').casefold()
                if not source or reference == old_ref:
                    file.text = visual_resource_name(model_path)
    if binary is None:
        binary = int(owner.get('hwek_ugx_cache_version', owner.get('hwek_ugx_port_target', 4))) == 6
    damage_stem = path.name.removesuffix('.xmb').removesuffix('.vis')
    damage_model_path = model_path or path.with_name(damage_stem + '.ugx')
    damage = prepare_damage_export(owner, damage_model_path, 6 if binary else 4)
    if damage_text(owner) is not None:
        model_names = json.loads(owner.get('hwek_vis_models', '[]'))
        for model in root.findall('model'):
            if source and model.get('name', '') not in model_names:
                continue
            for asset in model.findall('./component/asset'):
                if asset.get('type', '').lower() != 'model':
                    continue
                current_ref = asset.findtext('file', '').replace('\\', '/').removesuffix('.ugx').casefold()
                target_ref = visual_resource_name(model_path or original_model or damage_model_path).replace('\\', '/').casefold()
                if current_ref != target_ref:
                    continue
                node = asset.find('damagefile')
                if node is None:
                    node = ET.SubElement(asset, 'damagefile')
                reference = visual_resource_name(damage_model_path)
                if (node.text or '').replace('\\', '/').casefold() != reference.replace('\\', '/').casefold():
                    node.text = reference
    if binary and not path.name.casefold().endswith('.xmb'):
        path = path.with_name(path.name + '.xmb')
    actions = [action for action in bpy.data.actions if action.get('hwek_vis_owner') == owner]
    active = owner.animation_data.action if owner.type == 'ARMATURE' and owner.animation_data else None
    if active is not None and active not in actions:
        actions.append(active)
    models = root.findall('model')
    names = json.loads(owner.get('hwek_vis_models', '[]'))
    target = next((model for model in models if model.get('name') in names), models[0] if models else None)
    if target is None:
        raise UgxError('VIS has no model element for animation links.')
    targets = [model for model in models if model.get('name') in names] or [target]
    references = {animation_reference_key(asset.findtext('file', '') or ''): asset.findtext('file', '')
                  for model in targets for asset in model.findall('./anim/asset') if asset.get('type', '').casefold() in ('anim', 'granny')}
    outputs = []
    roots = visual_resource_roots(model_path or path)
    output_root = roots[0] if roots else path.parent
    records = {}
    source_cache = {}
    for index, action in enumerate(actions):
        origin = str(action.get('hwek_uax_source_path', ''))
        reference = visual_resource_name(origin) if origin else action.name
        key = animation_reference_key(reference)
        if key in references:
            reference = references[key]
        elif not action.get('hwek_vis_unlinked') or str(action.get('vis_type', '')).strip():
            state = str(action.get('vis_type', action.name))
            animation = ET.SubElement(target, 'anim', type=state, exitAction=str(action.get('vis_exitAction', 'Loop')),
                                      tweenTime=str(action.get('vis_tweenTime', '10')), tweenToAnimation='')
            asset = ET.SubElement(animation, 'asset', type='Anim')
            ET.SubElement(asset, 'file').text = reference
            ET.SubElement(asset, 'weight').text = '1'
            references[key] = reference
        if not animations:
            continue
        payload, result = prepare_uax_export(owner, action, f'{index}.uax', source_cache=source_cache)
        records.setdefault(key, []).append({'action': action, 'reference': reference, 'payload': payload,
                                            'unchanged': result['samples'] == 0,
                                            'nodes': animation_link_nodes(targets, action, reference)})
    reserved = set(references) | set(records)
    used = {}
    for group in records.values():
        payloads = {}
        for record in group:
            payloads.setdefault(record['payload'], []).append(record)
        original = next((record['payload'] for record in group if record['unchanged']), group[0]['payload'])
        for payload, variants in payloads.items():
            reference = variants[0]['reference']
            if payload != original:
                nodes = {node for record in variants for node in record['nodes']}
                other_nodes = {node for record in group if record['payload'] != payload for node in record['nodes']}
                if not nodes or nodes.intersection(other_nodes):
                    raise UgxError(f'VIS animations have conflicting edits for {reference}. Assign each edited Action to separate VIS states before exporting.')
                reference = split_animation_reference(reference, variants[0]['action'], reserved)
                for node in nodes:
                    node.text = reference
            destination = animation_destination(output_root, reference)
            existing = used.get(str(destination).casefold())
            if existing is not None and existing != payload:
                raise UgxError(f'VIS animations export different edits to {destination}. Give their files separate paths.')
            if existing is None:
                outputs.append((str(destination), payload))
                used[str(destination).casefold()] = payload
    xml = ET.tostring(root, encoding='unicode')
    payload = encode_visual_document(xml, source, binary)
    read_visual_document(payload)
    return [(str(path), payload)] + damage + outputs
