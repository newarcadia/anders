"""Prepare painted native colour layouts before preservation export."""
from contextlib import contextmanager
import json
from ..formats.ecf import EcfFile
from ..materials.context import ugx_material_index_for_slot
from ..materials.hw2 import hw2_material_spec_from_blender, hw2_material_spec_for_authoring, read_hw2_vertex_color_inputs
from ..materials.properties import ufx_attr_property_name
from ..materials.patch import patch_hw2_shader_fields
from .color_layout import rewrite_section_colors
from .data import source_vertex_section_ranges
from .metadata import build_section_entry, section_entry_source_index, set_ugx_source_metadata, ugx_source_bytes_for_export
from .mesh import mesh_color_attribute, mesh_colors_by_loop, mesh_int_attribute
from .read import parse_ugx_data
from .scene import ugx_armature_for, ugx_export_object_for, ugx_lod_sibling_objects_for, ugx_used_material_slots
from .skeleton_scene import metadata_owners, plain_property, relocate_metadata, refresh_material_source_offsets
from .snapshots import imported_mesh_changes
from .vertices import changed, encode_vertex_color, layout_for

__all__ = ('prepared_vertex_color_edits',)


def shader_inputs_changed(material):
    attrs = json.loads(material.get('hwek_ugx_ufx_attrs', '{}'))
    for pass_index in range(4):
        for field in ('shaderPermutationName', 'shaderPermutationHash'):
            name = field+str(pass_index)
            key = ufx_attr_property_name(name)
            if key in material and str(material[key]) != str(attrs.get(name, {}).get('value', '')):
                return True
    return False


@contextmanager
def prepared_vertex_color_edits(obj):
    source = ugx_export_object_for(obj)
    if source is None:
        yield False
        return
    objects = ugx_lod_sibling_objects_for(source)
    materials = {material for item in objects for _, material in ugx_used_material_slots(item) if material is not None}
    if all(imported_mesh_changes(item) is not None for item in objects) and not any(shader_inputs_changed(mat) for mat in materials):
        yield False
        return
    if not any(mesh_color_attribute(item.data) is not None for item in objects) and not any(shader_inputs_changed(mat) for mat in materials):
        yield False
        return
    original, mode = ugx_source_bytes_for_export(source)
    if mode == 'rebased_source':
        yield False
        return
    model = parse_ugx_data(str(source.get('hwek_ugx_source_path')), original)
    ranges = source_vertex_section_ranges(model)
    groups = []
    native_materials = {}
    for item in objects:
        sections = mesh_int_attribute(item.data, 'UGX_Section')
        sources = mesh_int_attribute(item.data, 'UGX_SourceVertex')
        if sections is None or sources is None:
            continue
        colors = mesh_colors_by_loop(item.data)
        grouped = {}
        for polygon in item.data.polygons:
            index = sections[polygon.index]
            if not 0 <= index < len(model.sections):
                continue
            slot = polygon.material_index
            material = item.material_slots[slot].material if slot < len(item.material_slots) else None
            if material is not None:
                native_materials.setdefault(ugx_material_index_for_slot(item, slot, model), material)
            grouped.setdefault((index, material), []).extend(polygon.loop_indices)
        for (index, material), loops in grouped.items():
            section = model.sections[index]
            native = layout_for(section.layout, 'D')
            values = [colors[loop] for loop in loops if loop in colors]
            edited = native is None and bool(values)
            first, last = ranges[index]
            for loop in loops:
                vertex = item.data.loops[loop].vertex_index
                identity = sources[vertex]
                baseline = section.vertices[identity-first].color if first <= identity < last else (1.,1.,1.,1.)
                if loop in colors and changed(colors[loop], baseline):
                    edited = True
            groups.append((index, material, values, edited))
    # A shared material's vertex contract applies to every section, including
    # hidden LODs and sections with no painted colour attribute.
    represented = {index for index, _, _, _ in groups}
    groups.extend((index, native_materials.get(section.material_index), [], False)
                  for index, section in enumerate(model.sections) if index not in represented)
    schemas, plans, specs = {}, {}, {}
    if model.cache_version == 6:
        for index, material, _, edited in groups:
            if not edited or material is None:
                continue
            section = model.sections[index]
            skinned = not section.rigid_only and layout_for(section.layout, 'I') is not None
            spec = hw2_material_spec_for_authoring(material, skinned, True, promote_color=True)
            before = hw2_material_spec_from_blender(material, skinned)
            specs[material] = spec
            if spec['shader_names'] != before['shader_names'] or spec['shader_hashes'] != before['shader_hashes']:
                edits = plans.setdefault(material, {})
                for pass_index in range(4):
                    edits[ufx_attr_property_name('shaderPermutationName'+str(pass_index))] = spec['shader_names'][pass_index]
                    edits[ufx_attr_property_name('shaderPermutationHash'+str(pass_index))] = str(spec['shader_hashes'][pass_index])
                edits[ufx_attr_property_name('skinned')] = spec['skinned']
    for index, material, values, edited in groups:
        section = model.sections[index]
        native = layout_for(section.layout, 'D')
        skinned = not section.rigid_only and layout_for(section.layout, 'I') is not None
        prefix, required = False, False
        if model.cache_version == 6:
            if native is not None and material not in plans and (material is None or not shader_inputs_changed(material)):
                required, prefix = True, skinned and native.offset == 20
            else:
                required, prefix = read_hw2_vertex_color_inputs(material, skinned, edited, spec=specs.get(material))
        kind = native.type_id if native is not None else 5 if model.cache_version == 4 else 9
        if model.cache_version == 4 and any(component < 0 or component > 1 for value in values for component in value):
            kind = 4
        if edited:
            for value in values:
                encode_vertex_color(value, kind)
        move = model.cache_version == 6 and native is not None and skinned and required and (native.offset == 20) != prefix
        if (native is None and (edited or required)) or (native is not None and kind != native.type_id) or move:
            previous = schemas.get(index)
            if previous is not None and model.cache_version == 4 and previous[0] == 4:
                kind = 4
            schemas[index] = (kind, prefix)
    if not schemas and not plans:
        yield False
        return
    payload = rewrite_section_colors(original, model, schemas)
    if model.cache_version == 6:
        targets = {}
        for index, material in native_materials.items():
            spec = specs.get(material) or hw2_material_spec_from_blender(material, False)
            fields = {'skinned': spec['skinned']}
            for pass_index in range(4):
                fields['shaderPermutationName'+str(pass_index)] = spec['shader_names'][pass_index]
                fields['shaderPermutationHash'+str(pass_index)] = spec['shader_hashes'][pass_index]
            targets[index] = fields
        payload = patch_hw2_shader_fields(payload, targets)
    prepared = parse_ugx_data(model.path, bytes(payload))
    rig = ugx_armature_for(source)
    owners = metadata_owners(objects, rig)
    saved = [(owner, {key: plain_property(owner[key]) for key in owner.keys() if key.startswith('hwek_ugx_')}) for owner in owners]
    try:
        relocate_metadata(owners, EcfFile.parse(original), EcfFile.parse(bytes(payload)))
        for material, edits in plans.items():
            for key, value in edits.items():
                material[key] = value
        if model.cache_version == 6:
            for index, material in native_materials.items():
                refresh_material_source_offsets(material, prepared.materials[index])
        for item in objects:
            entries = json.loads(item.get('hwek_ugx_sections', '[]'))
            for display, entry in enumerate(entries):
                index = section_entry_source_index(display, entry)
                entry.update(build_section_entry(prepared.sections[index], index, entry.get('vertex_start', 0), entry.get('vertex_count', 0),
                                                entry.get('face_start', 0), entry.get('face_count', 0)))
            item['hwek_ugx_sections'] = json.dumps(entries)
            set_ugx_source_metadata(item, model.path, bytes(payload))
        yield True
    except BaseException:
        for owner, properties in saved:
            for key in list(owner.keys()):
                if key.startswith('hwek_ugx_') and key not in properties:
                    del owner[key]
            for key, value in properties.items():
                owner[key] = value
        raise
