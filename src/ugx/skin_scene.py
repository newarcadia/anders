"""Prepare painted rigid parts for native section and shader export."""
from contextlib import contextmanager
import json

from ..errors import UgxError
from ..formats.ecf import EcfFile
from ..materials.context import ugx_material_index_for_slot
from ..materials.properties import ufx_attr_property_name
from ..materials.skinning import material_skinned_shader_plan
from ..materials.patch import patch_hw2_shader_fields
from .data import source_vertex_section_ranges
from .metadata import build_section_entry, section_entry_source_index, section_property_key, set_ugx_source_metadata, ugx_source_bytes_for_export
from .mesh import mesh_int_attribute, vertex_group_bone_indices
from .naming import section_mesh_indices
from .read import parse_ugx_data
from .scene import ugx_armature_for, ugx_export_object_for, ugx_lod_sibling_objects_for, ugx_used_material_slots
from .scene_validation import validate_ugx_export_meshes
from .skeleton_scene import metadata_owners, plain_property, relocate_metadata, refresh_material_source_offsets
from .skin import encode_skin, new_v6_rigid_fallback_bone, vertex_group_from_section_bone
from .skin_layout import rewrite_rigid_section_skin
from .snapshots import imported_mesh_changes
from .vertices import layout_for

__all__ = ('prepared_rigid_skin_edits',)


def source_rigid_weights(model, section):
    bone = vertex_group_from_section_bone(section, section.rigid_bone_index, len(model.bones))
    return {bone if bone is not None else 0: 1.}


def painted_rigid_weights(objects, rig, model):
    ranges = source_vertex_section_ranges(model)
    assignments = {}
    required = {}
    influences = {}
    for obj in objects:
        sections = mesh_int_attribute(obj.data, 'UGX_Section')
        sources = mesh_int_attribute(obj.data, 'UGX_SourceVertex')
        if sections is None or sources is None:
            continue
        groups = vertex_group_bone_indices(obj, rig)
        fallback = new_v6_rigid_fallback_bone(
            obj, {bone.name: int(bone.get('hwek_ugx_bone_index', index))
                  for index, bone in enumerate(rig.data.bones)}, default_bone_index=0,
        ) if model.cache_version == 6 else None
        members = {}
        for polygon in obj.data.polygons:
            section = sections[polygon.index]
            if not 0 <= section < len(model.sections):
                continue
            native = model.sections[section]
            if not native.rigid_only and layout_for(native.layout, 'I') is not None:
                continue
            members.setdefault(section, set()).update(polygon.vertices)
        for index, vertices in members.items():
            native = model.sections[index]
            baseline = source_rigid_weights(model, native)
            target = assignments.setdefault(index, [dict(baseline) for _ in native.vertices])
            first, last = ranges[index]
            seen = set()
            for vertex in vertices:
                current = {}
                for group in obj.data.vertices[vertex].groups:
                    bone = groups[group.group]
                    if bone >= 0 and group.weight > 0.00001:
                        current[bone] = current.get(bone, 0.)+float(group.weight)
                if not current:
                    current = {fallback: 1.} if fallback is not None else dict(baseline)
                indices, weights = encode_skin(current)
                current = {bone: weight/255. for bone, weight in zip(indices, weights) if weight}
                required.setdefault(index, set()).update(current)
                influences[index] = max(influences.get(index,0),len(current))
                source = sources[vertex]
                if first <= source < last and source not in seen:
                    target[source-first] = current
                    seen.add(source)
    changed = {}
    for index, weights in assignments.items():
        baseline = source_rigid_weights(model, model.sections[index])
        if required[index] != set(baseline) or any(value != baseline for value in weights):
            if len(required[index]) == 1:
                weights = [{next(iter(required[index])): 1.} for _ in weights]
            changed[index] = weights
    return changed, required, influences


@contextmanager
def prepared_rigid_skin_edits(obj):
    source = ugx_export_object_for(obj)
    rig = ugx_armature_for(source)
    if source is None or rig is None:
        yield False
        return
    objects = ugx_lod_sibling_objects_for(source)
    if all(imported_mesh_changes(item) is not None for item in objects):
        yield False
        return
    original, mode = ugx_source_bytes_for_export(source)
    if mode == 'rebased_source':
        yield False
        return
    model = parse_ugx_data(str(source.get('hwek_ugx_source_path')), original)
    validate_ugx_export_meshes(objects, model)
    weights, required, influences = painted_rigid_weights(objects, rig, model)
    if not weights:
        yield False
        return
    promoted = {index for index in weights if len(required[index]) > 1}
    plans = {}
    targets = {}
    materials = {}
    color = set()
    if model.cache_version == 6 and promoted:
        material_indices = {model.sections[index].material_index for index in promoted}
        materials = {}
        for item in objects:
            for slot, material in ugx_used_material_slots(item):
                if material is not None:
                    index = ugx_material_index_for_slot(item,slot,model)
                    if index in material_indices:
                        materials.setdefault(index, set()).add(material)
        if set(materials) != material_indices:
            raise UgxError('A weighted section has no editable Halo Wars 2 material. Assign its native material before export.')
        for index, candidates in materials.items():
            for material in candidates:
                edits, has_color = material_skinned_shader_plan(material, model.path)
                plans[material] = edits
                targets[index] = {name: edits[ufx_attr_property_name(name)] for name in
                                  ['skinned']+[field+str(pass_index) for pass_index in range(4)
                                               for field in ('shaderPermutationName', 'shaderPermutationHash')]}
                for section, native in enumerate(model.sections):
                    if native.material_index != index or (not native.rigid_only and layout_for(native.layout,'I') is not None):
                        continue
                    promoted.add(section)
                    weights.setdefault(section, [source_rigid_weights(model,native) for _ in native.vertices])
                    if has_color:
                        color.add(section)
    payload = rewrite_rigid_section_skin(original, model, weights, force_skinned=promoted,
                                         color_sections=color, required_bones=required, required_influences=influences)
    payload = patch_hw2_shader_fields(payload, targets)
    prepared = parse_ugx_data(model.path, bytes(payload))
    owners = metadata_owners(objects,rig)
    saved = [(owner,{key: plain_property(owner[key]) for key in owner.keys() if key.startswith('hwek_ugx_')})
             for owner in owners]
    try:
        relocate_metadata(owners,EcfFile.parse(original),EcfFile.parse(bytes(payload)))
        for material, edits in plans.items():
            for key, value in edits.items():
                material[key] = value
        for index in targets:
            for material in materials[index]:
                refresh_material_source_offsets(material, prepared.materials[index])
        for item in objects:
            entries = json.loads(item.get('hwek_ugx_sections','[]'))
            for display, entry in enumerate(entries):
                index = section_entry_source_index(display,entry)
                native = prepared.sections[index]
                entry.update(build_section_entry(native,index,entry.get('vertex_start',0),entry.get('vertex_count',0),
                                                  entry.get('face_start',0),entry.get('face_count',0)))
                if index in weights:
                    for key in ('max_bones','rigid_bone_index','rigid_only'):
                        item[section_property_key(index,key)] = int(getattr(native,key))
            item['hwek_ugx_sections'] = json.dumps(entries)
            set_ugx_source_metadata(item,model.path,bytes(payload))
        yield {'bounds_overrides': set(), 'section_mesh_indices': section_mesh_indices(model)}
    except BaseException:
        for owner, properties in saved:
            for key in list(owner.keys()):
                if key.startswith('hwek_ugx_') and key not in properties:
                    del owner[key]
            for key, value in properties.items():
                owner[key] = value
        raise
