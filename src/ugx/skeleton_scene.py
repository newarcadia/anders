"""Prepare imported armature count edits for the preservation exporter."""
from contextlib import contextmanager
import bpy
import copy
import json
import struct

from ..errors import UgxError
from ..formats.ecf import EcfFile, replace_ecf_chunk
from .bones import new_export_bone_matrix_rows, new_export_bone_name
from .data import Bone, source_vertex_section_ranges
from .granny import estimate_granny_lod_errors
from .mesh import mesh_int_attribute
from .metadata import bone_bound_property_values, section_entry_source_index, section_metadata_value, section_property_key, set_ugx_source_metadata, ugx_source_bytes_for_export
from .naming import section_mesh_indices
from .read import parse_ugx_data
from .scene import ugx_armature_for, ugx_export_object_for, ugx_lod_sibling_objects_for, ugx_ordered_armature_bones
from .skeleton_patch import patch_granny_skeleton_lod
from .skeleton_resize import resize_ugx_skeleton
from .skin import new_v4_uniform_rigid_bone, vertex_group_from_section_bone
from .transfer import refresh_export_bone_metadata

__all__ = ('prepared_imported_skeleton', 'patch_armature_skeleton_lod', 'set_skeleton_lod_mode')


def set_skeleton_lod_mode(armature, value):
    if value == -1:
        if 'hwek_ugx_granny_lod_type' in armature:
            del armature['hwek_ugx_granny_lod_type']
    else:
        armature['hwek_ugx_granny_lod_type'] = value


def patch_armature_skeleton_lod(data, model, armature, bone_indices=None):
    if armature is None:
        return data
    bones = ugx_ordered_armature_bones(armature)
    if 'hwek_ugx_granny_lod_type' not in armature and not any('hwek_ugx_granny_lod_error' in bone for bone in bones):
        return data
    if bone_indices is None:
        if len(bones) != len(model.bones):
            raise UgxError('Skeleton LOD editing needs the exported index of each armature bone.')
        bone_indices = {bone.name: index for index, bone in enumerate(bones)}
    native_errors = [bone.lod_error for bone in model.granny_info.bones]
    errors = list(native_errors)
    for bone in bones:
        if 'hwek_ugx_granny_lod_error' in bone:
            errors[bone_indices[bone.name]] = float(bone['hwek_ugx_granny_lod_error'])
    lod_type = int(armature.get('hwek_ugx_granny_lod_type', model.granny_info.skeleton_lod_type))
    if lod_type == model.granny_info.skeleton_lod_type and errors == native_errors:
        return data
    native = EcfFile.parse(bytes(data)).chunks[0x703].data
    patched = patch_granny_skeleton_lod(native, lod_type, errors)
    return data if patched is native else replace_ecf_chunk(bytearray(data), 0x703, patched)


def refresh_material_source_offsets(material, native):
    def locations(current, source, *, preserve_size=False):
        for key, value in source.items():
            if key not in current:
                continue
            if key == 'offset' or key.endswith('_offset') or key == 'capacity' or (key == 'value_byte_size' and not preserve_size):
                current[key] = value
            elif isinstance(current[key], dict) and isinstance(value, dict):
                locations(current[key], value, preserve_size=preserve_size)

    # Buffer and texture arrays can have pending structural edits. Refresh
    # source locations without replacing their authored members or values.
    material['hwek_ugx_name_offset'] = native.name_offset
    material['hwek_ugx_name_capacity'] = native.name_capacity
    material['hwek_ugx_name_entry_offset'] = native.name_entry_offset
    slots = json.loads(material.get('hwek_ugx_texture_slots', '[]'))
    native_slots = {int(slot.get('hogan_index', index)): slot for index, slot in enumerate(native.slots)}
    for index, slot in enumerate(slots):
        source = native_slots.get(int(slot.get('hogan_index', index)))
        if source is not None:
            locations(slot, source)
    material['hwek_ugx_texture_slots'] = json.dumps(slots)
    attrs = json.loads(material.get('hwek_ugx_ufx_attrs', '{}'))
    for name, info in attrs.items():
        if name in native.ufx_attrs:
            locations(info, native.ufx_attrs[name])
    material['hwek_ugx_ufx_attrs'] = json.dumps(attrs)
    buffers = json.loads(material.get('hwek_ugx_ufx_constant_buffers', '[]'))
    native_buffers = {buffer['name']: buffer for buffer in native.ufx_constant_buffers}
    for buffer in buffers:
        source = native_buffers.get(buffer['name'])
        if source is not None:
            # Pending replacement buffers can have different lengths from the
            # retained native record used during layout preparation.
            locations(buffer, source, preserve_size=True)
    material['hwek_ugx_ufx_constant_buffers'] = json.dumps(buffers)


def plain_property(value):
    if hasattr(value, 'to_dict'):
        return {key: plain_property(item) for key, item in value.items()}
    if hasattr(value, 'to_list'):
        return value.to_list()
    return value


def metadata_owners(objects, rig):
    owners = list(objects)
    if rig is not None:
        owners.extend([rig, *rig.data.bones])
    materials = {slot.material for obj in objects for slot in obj.material_slots if slot.material}
    owners.extend(materials)
    for material in materials:
        if material.use_nodes and material.node_tree:
            for node in material.node_tree.nodes:
                owners.append(node)
                if node.type == 'TEX_IMAGE' and node.image:
                    owners.append(node.image)
    return list(dict.fromkeys(owners))


def relocate_metadata(owners, before, after):
    ranges = [(chunk.offset, chunk.offset+chunk.size, after.chunks[key].offset-chunk.offset)
              for key, chunk in before.chunks.items() if key in after.chunks]

    def relocate(value, key=''):
        if isinstance(value, dict):
            return {name: relocate(item, name) for name, item in value.items()}
        if isinstance(value, list):
            return [relocate(item, key) for item in value]
        if isinstance(value, int) and value > 0 and (key == 'offset' or key.endswith('_offset')):
            for start, end, delta in ranges:
                if start <= value < end:
                    return value+delta
        return value

    for owner in owners:
        for key in list(owner.keys()):
            if not key.startswith('hwek_ugx_'):
                continue
            value = plain_property(owner[key])
            if isinstance(value, str) and value[:1] in ('[', '{'):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    continue
                changed = relocate(parsed)
                if changed != parsed:
                    owner[key] = json.dumps(changed)
            else:
                changed = relocate(value, key)
                if changed != value:
                    owner[key] = changed


def patch_current_rigid_bindings(payload, source_model, model, objects, bone_indices, source_indices):
    ranges = source_vertex_section_ranges(source_model)
    targets = {}
    remap = {}
    for index, source in enumerate(source_indices):
        if source is not None:
            remap.setdefault(source,index)
    for obj in objects:
        entries = json.loads(obj.get('hwek_ugx_sections', '[]'))
        source_vertices = mesh_int_attribute(obj.data, 'UGX_SourceVertex')
        for display, entry in enumerate(entries):
            index = section_entry_source_index(display, entry)
            section = model.sections[index]
            before = source_model.sections[index]
            requested = int(section_metadata_value(obj,display,entry,'rigid_bone_index',before.rigid_bone_index))
            explicit = requested != before.rigid_bone_index
            if explicit:
                old = vertex_group_from_section_bone(before,requested,len(source_model.bones))
                if old not in remap:
                    raise UgxError(f'Section {index} rigid bone edit references a removed or missing bone.')
                target = remap[old]
            elif not section.rigid_only:
                continue
            first, last = ranges[index]
            if source_vertices is not None:
                vertices = [vertex for vertex, source in enumerate(source_vertices) if first <= source < last]
            else:
                start = int(entry.get('vertex_start', 0))
                vertices = range(start, start+int(entry.get('vertex_count', 0)))
            if not explicit:
                target = new_v4_uniform_rigid_bone(obj, vertices, bone_indices)
            if target is None:
                continue
            if section.local_to_global_bones:
                target = section.local_to_global_bones.index(target)
            previous = targets.setdefault(index, target)
            if previous != target:
                raise UgxError(f'Section {index} has conflicting rigid bone assignments.')
    for index, target in targets.items():
        struct.pack_into('<i', payload, model.sections[index].rigid_bone_index_offset, target)
    return targets


@contextmanager
def prepared_imported_skeleton_data(obj):
    source = ugx_export_object_for(obj)
    rig = ugx_armature_for(source)
    if source is None or rig is None or not source.get('hwek_ugx_import_id') or rig.get('hwek_ugx_import_id') != source.get('hwek_ugx_import_id'):
        yield False
        return
    bpy_bones = list(rig.data.bones)
    source_count = int(source.get('hwek_ugx_bone_count', 0))
    identities = [int(bone.get('hwek_ugx_bone_index', -1)) for bone in bpy_bones]
    if len(bpy_bones) == source_count and sorted(identities) == list(range(source_count)):
        yield False
        return
    if not bpy_bones:
        raise UgxError('The imported armature has no bones. Keep at least one bone before export.')
    original, mode = ugx_source_bytes_for_export(source)
    if mode == 'rebased_source':
        raise UgxError('Source UGX changed before skeleton edits. Reimport it before changing its bone count.')
    model = parse_ugx_data(str(source.get('hwek_ugx_source_path')), original)
    indexed = []
    for ordinal, bone in enumerate(bpy_bones):
        identity = int(bone.get('hwek_ugx_bone_index', -1))
        if not 0 <= identity < len(model.bones):
            offset = int(bone.get('hwek_ugx_matrix_offset',0))
            candidates = [index for index, before in enumerate(model.bones) if offset and before.matrix_offset == offset]
            if len(candidates) != 1 and bone.get('hwek_ugx_source_matrix'):
                rows = json.loads(bone['hwek_ugx_source_matrix'])
                candidates = [index for index, before in enumerate(model.bones)
                              if before.name == bone.get('hwek_ugx_source_name') and before.matrix == rows]
            identity = candidates[0] if len(candidates) == 1 else None
            if identity is None and bone.get('hwek_ugx_source_matrix'):
                raise UgxError(f'Bone {bone.name!r}: its original native record cannot be identified. Restore its imported bone index before export.')
        copied = bone.name != bone.get('hwek_ugx_blender_name', bone.name)
        indexed.append((identity if identity is not None else len(model.bones), copied, ordinal, bone, identity))
    indexed.sort(key=lambda item: item[:3])
    ordered = [item[3] for item in indexed]
    source_indices = [item[4] for item in indexed]
    if source_indices == list(range(len(model.bones))):
        saved_indices = [(bone,bone.get('hwek_ugx_bone_index')) for bone in ordered]
        try:
            for index, bone in enumerate(ordered):
                bone['hwek_ugx_bone_index'] = index
            yield False
        except BaseException:
            for bone, value in saved_indices:
                if value is None:
                    del bone['hwek_ugx_bone_index']
                else:
                    bone['hwek_ugx_bone_index'] = value
            raise
        return
    by_name = {bone.name: index for index, bone in enumerate(ordered)}
    targets = []
    bounds_overrides = set()
    for index, (bone, identity) in enumerate(zip(ordered, source_indices)):
        target = copy.copy(model.bones[identity]) if identity is not None else Bone('', [], -1)
        target.name = new_export_bone_name(bone, index)
        target.matrix = new_export_bone_matrix_rows(rig, bone, index, ordered)
        target.parent = by_name[bone.parent.name] if bone.parent else -1
        for key in ('low', 'high'):
            value = bone_bound_property_values(bone, key, None)
            if value is not None:
                if identity is not None and tuple(value) != tuple(getattr(model.bones[identity], 'bounds_'+key)):
                    bounds_overrides.add((index, key))
                setattr(target, 'bounds_'+key, value)
        targets.append(target)
    errors = [float(bone.get('hwek_ugx_granny_lod_error',
                           model.granny_info.bones[identity].lod_error if identity is not None
                           else (-1.0 if model.granny_info.skeleton_lod_type else 0.0)))
              for bone, identity in zip(ordered, source_indices)]
    if not model.granny_info.skeleton_lod_type:
        errors = [0.0 if error < 0.0 else error for error in errors]
    if any(error < 0.0 for error in errors):
        errors = estimate_granny_lod_errors(targets, errors)
    payload = resize_ugx_skeleton(original, model, targets, source_indices, errors, removed_bone_fallback=0)
    prepared = parse_ugx_data(model.path, bytes(payload))
    objects = ugx_lod_sibling_objects_for(source)
    rigid_targets = patch_current_rigid_bindings(payload, model, prepared, objects, by_name, source_indices)
    if rigid_targets:
        prepared = parse_ugx_data(model.path, bytes(payload))
    owners = metadata_owners(objects, rig)
    saved = [(owner, {key: plain_property(owner[key]) for key in owner.keys() if key.startswith('hwek_ugx_')})
             for owner in owners]
    try:
        relocate_metadata(owners, EcfFile.parse(original), EcfFile.parse(bytes(payload)))
        for index, bone in enumerate(ordered):
            bone['hwek_ugx_bone_index'] = index
        refresh_export_bone_metadata([rig], prepared)
        for item in objects+[rig]:
            set_ugx_source_metadata(item, model.path, bytes(payload))
            item['hwek_ugx_bone_count'] = len(targets)
            # The old mesh fingerprint belongs to the original bone table.
            # Its hash mismatch forces every current weight through the writer.
        for item in objects:
            entries = json.loads(item.get('hwek_ugx_sections', '[]'))
            for display, entry in enumerate(entries):
                index = section_entry_source_index(display, entry)
                target = prepared.sections[index].rigid_bone_index
                entry['rigid_bone_index'] = target
                item[section_property_key(index, 'rigid_bone_index')] = target
                entry['accessory_first_bone'] = prepared.sections[index].accessory_first_bone
                entry['accessory_bone_count'] = prepared.sections[index].accessory_bone_count
            item['hwek_ugx_sections'] = json.dumps(entries)
            item['hwek_ugx_bone_bounds'] = json.dumps([
                dict(index=index, name=bone.name, low=bone.bounds_low, high=bone.bounds_high,
                     low_offset=bone.bounds_low_offset, high_offset=bone.bounds_high_offset, big_endian=False)
                for index, bone in enumerate(prepared.bones)])
        yield {'bounds_overrides': bounds_overrides, 'section_mesh_indices': section_mesh_indices(model)}
    except BaseException:
        for owner, properties in saved:
            for key in list(owner.keys()):
                if key.startswith('hwek_ugx_') and key not in properties:
                    del owner[key]
            for key, value in properties.items():
                owner[key] = value
        raise


@contextmanager
def prepared_imported_skeleton(obj):
    rig = ugx_armature_for(ugx_export_object_for(obj))
    edit_mode = rig is not None and rig.mode == 'EDIT'
    active = bpy.context.view_layer.objects.active
    if edit_mode:
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.update()
    try:
        with prepared_imported_skeleton_data(obj) as prepared:
            yield prepared
    finally:
        if edit_mode:
            bpy.context.view_layer.objects.active = rig
            rig.select_set(True)
            bpy.ops.object.mode_set(mode='EDIT')
            if active is not None and active != rig:
                bpy.context.view_layer.objects.active = active
