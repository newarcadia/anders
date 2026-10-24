"""Prepare authored and imported Blender models for native UGX export."""

from typing import Dict, List, Optional, Tuple
import bpy
import copy
import json
import traceback
import numpy as np
from ..errors import UgxError
from ..formats.binary import patch_cstring
from ..formats.ecf import (
    CACHED_DATA_CHUNK,
    EcfFile,
    INDEX_BUFFER_CHUNK,
    MATERIAL_CHUNK,
    VERTEX_BUFFER_CHUNK,
    build_ecf_file,
    ecf_chunk_records,
    refresh_changed_ecf_adlers,
    replace_ecf_chunk,
    validate_game_ready_ugx_ecf_layout,
)
from ..formats.granny import GrannyInfo, GrannyModelInfo, repair_granny_model_binding_overlap, safe_grx_name
from ..materials.context import material_node_texture_path, material_texture_slot_current_value
from ..materials.data import (
    V4_DEFAULT_DIFFUSE_TEXTURE,
    V4_MATERIAL_MAP_USAGES,
    build_v4_material_bdt,
    build_v6_material_bdt_from_specs,
    v4_material_usage,
)
from ..materials.discovery import discover_ugx_texture_dependencies
from ..materials.hw1 import (
    v4_material_attrs_for_mesh,
    v4_material_attrs_from_blender,
    v4_material_names_for_mesh,
    v4_material_slots_for_mesh,
    v4_material_slots_from_blender,
)
from ..materials.hw2 import build_authored_v6_material_bdt, hw2_material_spec_for_authoring, v6_material_specs_for_model, v6_material_table_needs_rebuild
from ..materials.native import load_native_material, retain_native_materials
from ..materials.origin import extend_v6_material_model_for_objects
from ..materials.tree import rebuild_material_tree
from ..materials.patch import (
    add_bdt_replacement,
    patch_material_bdt_strings,
    patch_material_ufx_fields,
    patch_material_v4_fields,
    refresh_material_bdt_crc,
)
from ..materials.properties import ufx_attr_property_name, v4_attr_property_name
from ..materials.slots import (
    apply_v4_source_material_context_for_diffuse_edits,
    sync_ugx_materials_from_texture_props,
)
from ..materials.sync import sync_ugx_export_material_nodes
from ..textures.export import write_ugx_with_textures
from ..textures.paths import disabled_texture_path
from .granny import authored_granny_meshes, build_granny_info_chunk, granny_bones_for_model
from .skeleton_patch import patch_granny_skeleton
from .mesh_names import patch_granny_mesh_names, patch_granny_model_name
from .model_placement import restore_ugx_model_placement, patch_armature_model_placement
from .naming import current_ugx_mesh_name
from .skeleton_order import canonicalize_ugx_skeleton
from .skeleton_scene import prepared_imported_skeleton, patch_armature_skeleton_lod
from .skin_scene import prepared_rigid_skin_edits
from .shape_scene import prepared_shape_key_edits
from .color_scene import prepared_vertex_color_edits
from .skeleton_resize import refresh_resized_bone_bounds
from .mesh_bounds import patch_edited_mesh_bounds
from .tree_bounds import patch_tree_bounds
from .naming import section_mesh_indices
from .mesh import mesh_has_any_split_uv, mesh_has_split_colors, collect_vertex_color_materials
from .metadata import ugx_source_bytes_for_export
from .patch import (
    mesh_positions_changed,
    patch_bone_bounds,
    patch_bone_matrices,
    patch_cached_bone_names,
    patch_cached_bounds,
    patch_mesh_vertices,
    patch_position_edits,
    patch_section_cached_metadata,
    patch_section_materials,
    rebuild_topology_buffers,
    same_topology,
    ugx_mesh_splits_material_sections,
)
from .proxies import (
    build_hw2_structural_export_proxy,
    build_lod_export_proxy,
    build_material_export_proxy,
    hw2_structure_changed,
    remove_export_proxy,
    remove_temp_objects,
)
from .read import parse_ugx_data
from .basis_layout import extend_hw1_tangents
from .scene import ugx_used_material_slots, ugx_armature_for, ugx_export_object_for, ugx_lod_sibling_objects_for, ugx_ordered_armature_bones, ugx_authoring_mesh_objects
from .scene_validation import validate_ugx_export_meshes
from .metadata import lod_section_indices
from .snapshots import imported_mesh_changes, mesh_game_positions, remember_imported_mesh
from .transfer import (
    compatible_rebased_source,
    rebase_legacy_structural_export_objects,
    rebase_structural_export_objects,
    refresh_export_object_metadata,
    refresh_exported_mesh_basis,
    refresh_added_hw1_tangent_attributes,
)
from .validation import (
    validate_game_ready_ugx_model,
    validate_granny_model_contract,
    validate_v4_instanced_index_buffer,
    validate_v6_instanced_index_buffer,
)
from .uv_export import extend_hw1_uv_sets, prepared_hw1_uv_mappings
from .vertices import layout_for
from .write_v4 import build_new_v4_authoring_chunks
from .write_v6 import build_new_v6_authoring_chunks

__all__ = ('export_new_v4_ugx', 'export_new_v6_ugx', 'export_ugx_copy')

def _export_new_v4_ugx(
    obj: bpy.types.Object,
    path: str,
    axis_forward: str = "Z",
    axis_up: str = "Y",
    export_loaded_textures: bool = False,
    export_visual: bool = False,
):
    discover_ugx_texture_dependencies(obj, path)
    cached, vertex_buffer, index_buffer, materials, mesh_names, mesh_section_indices = build_new_v4_authoring_chunks(
        obj,
        axis_forward,
        axis_up,
    )
    material_names = []
    material_slots = []
    material_attrs = []
    native_materials = [load_native_material(material, 4) for material in materials]
    for index, mat in enumerate(materials):
        if mat is None:
            material_names.append(f"material_{index}")
        else:
            value = mat.get("hwek_ugx_export_name", mat.get("hwek_ugx_source_name", mat.name))
            if value == mat.get("hwek_ugx_source_name", "") and mat.name != mat.get("hwek_ugx_blender_name", mat.name):
                value = mat.name
            material_names.append(str(value or f"material_{index}"))
        native = native_materials[index]
        if native is not None and material_names[-1] == native.name:
            material_names[-1] = native.native_name
        slots = v4_material_slots_from_blender(mat)
        if not slots:
            slots = [{
                "usage": "diffuse",
                "path": V4_DEFAULT_DIFFUSE_TEXTURE,
                "channel": 0,
                "flags": 7,
            }]
        material_slots.append(slots)
        material_attrs.append(v4_material_attrs_from_blender(mat))
    material_chunk = build_v4_material_bdt(
        material_names,
        material_slots,
        material_attrs,
        native_materials=native_materials,
    )
    material_chunk = retain_native_materials(material_chunk, native_materials)
    probe = build_ecf_file([
        (CACHED_DATA_CHUNK, cached),
        (VERTEX_BUFFER_CHUNK, vertex_buffer),
        (INDEX_BUFFER_CHUNK, index_buffer),
        (MATERIAL_CHUNK, material_chunk),
    ])
    # HaloWarsDocs/010Editor/UGX.bt identifies chunk 0x703 as a 64-bit
    # granny_file_info. Granny.bt defines the complete skeleton, bone, mesh
    # binding, and model graph, so a new v4 export must synthesize that graph
    # from the Blender armature instead of borrowing a retail UGX.
    probe_model = parse_ugx_data(path, probe)
    # The model name addresses UAX groups independently of its skeleton name.
    # Preserve imported identities when rebuilding a model from its armature.
    armature = ugx_armature_for(obj)
    model_name = safe_grx_name(
        probe_model.bones[0].name if probe_model.bones else "GrannyRootBone",
        "GrannyRootBone",
    )
    if armature is not None and 'hwek_ugx_granny_model_name' in armature:
        model_name = str(armature['hwek_ugx_granny_model_name'])
    restore_ugx_model_placement(armature)
    probe_model.granny_info.models = [GrannyModelInfo(model_name, list(mesh_names))]
    probe_model.granny_info.meshes = authored_granny_meshes(probe_model, mesh_names, mesh_section_indices)
    grx_chunk = build_granny_info_chunk(probe_model, obj.name)
    payload = build_ecf_file([
        (0x703, grx_chunk),
        (CACHED_DATA_CHUNK, cached),
        (VERTEX_BUFFER_CHUNK, vertex_buffer),
        (INDEX_BUFFER_CHUNK, index_buffer),
        (MATERIAL_CHUNK, material_chunk),
    ])
    validate_game_ready_ugx_ecf_layout(payload)
    model = parse_ugx_data(path, payload)
    payload = patch_armature_model_placement(payload, model, armature)
    if payload is not model.original_bytes:
        model = parse_ugx_data(path, bytes(payload))
    patched_lod = patch_armature_skeleton_lod(payload, model, armature)
    if patched_lod is not payload:
        payload = patched_lod
        model = parse_ugx_data(path, bytes(payload))
        payload, _ = canonicalize_ugx_skeleton(payload, model)
        model = parse_ugx_data(path, bytes(payload))
    validate_game_ready_ugx_model(model)
    generated_ecf = EcfFile.parse(payload)
    validate_v4_instanced_index_buffer(
        model,
        generated_ecf.chunks[CACHED_DATA_CHUNK].data,
        generated_ecf.chunks[INDEX_BUFFER_CHUNK].data,
    )
    # Legacy retail v4 files contain several valid accessory/Granny variants,
    # so the exact generated graph contract belongs here, at the creation
    # boundary, rather than in the general validator used by copy exports.
    validate_granny_model_contract(
        model,
        "HWDE v4",
        require_v4_generated_transforms=True,
    )
    expected_names = [safe_grx_name(name, "") for name in mesh_names]
    actual_names = model.granny_info.models[0].mesh_names if model.granny_info.models else []
    if actual_names != expected_names:
        raise UgxError("New v4 UGX export did not preserve its armature mesh bindings.")
    if model.granny_info.models[0].name != model_name:
        raise UgxError("New v4 UGX did not preserve its animation model name.")
    return write_ugx_with_textures(obj, path, payload, 4, export_loaded_textures, export_visual)


def _export_new_v6_ugx(obj: bpy.types.Object, path: str, export_loaded_textures: bool = False, export_visual: bool = False):
    discover_ugx_texture_dependencies(obj, path)
    armature = ugx_armature_for(obj)
    restore_ugx_model_placement(armature)
    cached, vertex_buffer, index_buffer, materials, skinned, mesh_names, granny_metadata = build_new_v6_authoring_chunks(obj)
    colored = collect_vertex_color_materials(ugx_authoring_mesh_objects(obj))
    material_chunk = build_authored_v6_material_bdt(materials,
        [hw2_material_spec_for_authoring(material, skinned, material in colored, adapt_skin=True) for material in materials])
    probe = build_ecf_file([
        (CACHED_DATA_CHUNK, cached),
        (VERTEX_BUFFER_CHUNK, vertex_buffer),
        (INDEX_BUFFER_CHUNK, index_buffer),
        (MATERIAL_CHUNK, material_chunk),
    ])
    probe_model = parse_ugx_data(path, probe)
    model_name = str(granny_metadata["model_name"])
    generated_granny_bones = granny_bones_for_model(probe_model)
    for index, granny_bone in enumerate(generated_granny_bones):
        if index < len(granny_metadata["lod_errors"]):
            granny_bone.lod_error = float(granny_metadata["lod_errors"][index])
    probe_model.granny_info = GrannyInfo(
        from_file_name=str(granny_metadata.get("from_file_name", "gr2ugx")),
        skeleton_name=str(granny_metadata.get("skeleton_name", generated_granny_bones[0].name)),
        skeleton_lod_type=int(granny_metadata.get("skeleton_lod_type", 1)),
        bones=generated_granny_bones,
        meshes=[],
        models=[GrannyModelInfo(model_name, list(mesh_names))],
        parsed=True,
    )
    grx_chunk = build_granny_info_chunk(probe_model, obj.name)
    payload = build_ecf_file([
        (0x703, grx_chunk),
        (CACHED_DATA_CHUNK, cached),
        (VERTEX_BUFFER_CHUNK, vertex_buffer),
        (INDEX_BUFFER_CHUNK, index_buffer),
        (MATERIAL_CHUNK, material_chunk),
    ])
    validate_game_ready_ugx_ecf_layout(payload)
    model = parse_ugx_data(path, payload)
    if model.cache_version != 6:
        raise UgxError("New v6 UGX export did not reparse as v6.")
    payload = patch_armature_model_placement(payload, model, armature)
    if payload is not model.original_bytes:
        model = parse_ugx_data(path, bytes(payload))
    patched_lod = patch_armature_skeleton_lod(payload, model, armature, granny_metadata['bone_indices'])
    if patched_lod is not payload:
        payload = patched_lod
        model = parse_ugx_data(path, bytes(payload))
        payload, _ = canonicalize_ugx_skeleton(payload, model)
        model = parse_ugx_data(path, bytes(payload))
    validate_game_ready_ugx_model(model)
    generated_ecf = EcfFile.parse(payload)
    validate_v6_instanced_index_buffer(
        model,
        generated_ecf.chunks[CACHED_DATA_CHUNK].data,
        generated_ecf.chunks[INDEX_BUFFER_CHUNK].data,
    )
    expected_names = [safe_grx_name(name, "") for name in mesh_names]
    actual_names = model.granny_info.models[0].mesh_names if model.granny_info.models else []
    if actual_names != expected_names:
        raise UgxError("New v6 UGX export did not preserve its armature mesh bindings.")
    if model.granny_info.models[0].name != model_name:
        raise UgxError("New v6 UGX did not preserve its animation model name.")
    return write_ugx_with_textures(obj, path, payload, 6, export_loaded_textures, export_visual)


def _export_ugx_copy(obj: bpy.types.Object, path: str, export_loaded_textures: bool = False, export_visual: bool = False, skeleton_resized=False):
    discover_ugx_texture_dependencies(obj, path)
    obj = ugx_export_object_for(obj)
    if obj is None:
        raise UgxError("Selected object does not have UGX source metadata.")
    restore_edit_mode = getattr(obj, "mode", "") == "EDIT"
    if restore_edit_mode:
        try:
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception as exc:
            raise UgxError(f"Could not synchronize Edit Mode data before UGX export: {exc}") from exc
    bpy.context.view_layer.update()
    source_path = obj.get("hwek_ugx_source_path")
    metadata_objects = ugx_lod_sibling_objects_for(obj)
    original, source_mode = ugx_source_bytes_for_export(obj)
    data = bytearray(original)
    model = parse_ugx_data(source_path, bytes(original))
    if source_mode == "rebased_source":
        compatible, reason = compatible_rebased_source(obj, model)
        if not compatible:
            raise UgxError(f"Source UGX changed since import and no matching import snapshot is available ({reason}). Reimport the UGX, then export again.")
    if any(item.get("hwek_ugx_reimport_required") for item in metadata_objects):
        metadata_objects = rebase_legacy_structural_export_objects(
            metadata_objects,
            model,
            str(source_path),
            bytes(original),
        )
        obj = metadata_objects[0] if obj not in metadata_objects else obj
    proxy: Optional[bpy.types.Object] = None
    temp_objects: List[bpy.types.Object] = []
    structural_object_pairs: List[Tuple[bpy.types.Object, bpy.types.Object]] = []
    export_obj = obj
    lod_objects = ugx_lod_sibling_objects_for(obj)
    sync_ugx_export_material_nodes(lod_objects, model.cache_version)
    structural_change = model.cache_version == 6 and hw2_structure_changed(model, lod_objects)
    changes = [imported_mesh_changes(item) for item in lod_objects]
    direct_meshes = source_mode != 'rebased_source' and not structural_change and all(change is not None for change in changes)
    if direct_meshes and model.cache_version == 4:
        direct_meshes = all(layout_for(model.sections[index].layout, 'T' + name[-1]) is not None
                            for item in lod_objects for index in lod_section_indices(item)
                            for name in ('UV0', 'UV1', 'UV2') if item.data.uv_layers.get(name) is not None)
    validate_ugx_export_meshes(lod_objects, model, unchanged_buffers=direct_meshes)
    structural_model = model
    material_model = model
    if structural_change:
        proxy, structural_model, temp_objects, structural_object_pairs = build_hw2_structural_export_proxy(model, lod_objects)
        export_obj = proxy
        material_model = structural_model
    else:
        if model.cache_version == 6:
            material_model = copy.copy(model)
            material_model.materials = list(model.materials)
            extend_v6_material_model_for_objects(material_model, lod_objects)
        if direct_meshes:
            proxy = build_material_export_proxy(material_model, lod_objects)
            export_obj = proxy
        elif len(lod_objects) > 1:
            proxy = build_lod_export_proxy(material_model, lod_objects)
            export_obj = proxy
    try:
        bone_order = list(range(len(model.bones)))
        topology_unchanged = direct_meshes or (not structural_change and same_topology(model, export_obj.data) and not mesh_has_any_split_uv(export_obj.data, model.cache_version) and not mesh_has_split_colors(export_obj.data) and not (model.cache_version == 6 and ugx_mesh_splits_material_sections(export_obj)))
        retain_source_basis = any(item.data.shape_keys for item in lod_objects) or (model.cache_version == 4 and not topology_unchanged)
        geometry_changed = any(changes) if direct_meshes else (not topology_unchanged) or mesh_positions_changed(model, export_obj)
        bounds_positions = None
        if direct_meshes:
            positions = []
            for item, moved in zip(lod_objects, changes):
                local_model = copy.copy(material_model)
                local_model.sections = [model.sections[index] for index in lod_section_indices(item)]
                if moved:
                    positions.append(patch_position_edits(data, local_model, item))
                elif geometry_changed:
                    positions.append(mesh_game_positions(item))
                patch_section_materials(data, local_model, item)
                patch_section_cached_metadata(data, local_model, item)
            if positions:
                bounds_positions = np.concatenate(positions)
        elif topology_unchanged:
            patch_mesh_vertices(data, model, export_obj)
            patch_section_materials(data, material_model, export_obj)
        if not structural_change:
            if not direct_meshes:
                patch_section_cached_metadata(data, model, export_obj)
            patch_cached_bounds(data, model, export_obj, topology_unchanged, geometry_changed, bounds_positions)
        sync_ugx_materials_from_texture_props(export_obj)
        rebuild_v6_material_bdt = model.cache_version == 6 and (
            len(material_model.materials) != len(model.materials)
            or v6_material_table_needs_rebuild(export_obj, material_model)
        )

        armature = ugx_armature_for(export_obj)
        armature_bones = ugx_ordered_armature_bones(armature)
        restore_ugx_model_placement(armature, original)
        patch_bone_matrices(data, model, armature)
        patch_bone_bounds(data, model, export_obj)
        material_bdt_replacements: Dict[int, str] = {}
        material_bdt_entry_replacements: Dict[int, str] = {}
        rebuild_v4_material_bdt = False
        data = patch_cached_bone_names(data, model, armature_bones)
        apply_v4_source_material_context_for_diffuse_edits(export_obj)

        material_record = next((record for record in ecf_chunk_records(bytes(data)) if record.id == MATERIAL_CHUNK), None)
        material_name_targets: Dict[int, List[Tuple[bpy.types.Material, str, str, int]]] = {}
        for mat in ([] if rebuild_v6_material_bdt else [mat for _, mat in ugx_used_material_slots(export_obj) if mat is not None]):
            if mat.get("hwek_ugx_name_capacity", 0):
                value = mat.get("hwek_ugx_export_name", mat.get("hwek_ugx_source_name", mat.name))
                if value == mat.get("hwek_ugx_source_name", "") and mat.name != mat.get("hwek_ugx_blender_name", mat.name):
                    value = mat.name
                source = str(mat.get("hwek_ugx_source_name", ""))
                offset = int(mat.get("hwek_ugx_name_offset", 0))
                if material_record and material_record.offset <= offset < material_record.offset + material_record.size:
                    material_name_targets.setdefault(offset, []).append((mat, str(value), source, int(mat.get("hwek_ugx_name_entry_offset", 0))))
                else:
                    patch_cstring(data, offset, int(mat.get("hwek_ugx_name_capacity", 0)), str(value), f"Material {mat.name} name")

        for offset, items in material_name_targets.items():
            targets = {target for _, target, _, _ in items}
            source = items[0][2]
            if len(targets) == 1:
                target = next(iter(targets))
                if target != source:
                    add_bdt_replacement(material_bdt_replacements, offset, target, f"Material {items[0][0].name} name")
                continue
            for mat, target, source, entry_offset in items:
                if target == source:
                    continue
                if not entry_offset:
                    raise UgxError(f"Material {mat.name} name is not independently patchable.")
                previous = material_bdt_entry_replacements.get(entry_offset)
                if previous is not None and previous != target:
                    raise UgxError(f"Material {mat.name} name entry received conflicting edits.")
                material_bdt_entry_replacements[entry_offset] = target

        for mat in ([] if rebuild_v6_material_bdt else [mat for _, mat in ugx_used_material_slots(export_obj) if mat is not None]):
            for slot_index, slot in enumerate(json.loads(mat.get("hwek_ugx_texture_slots", "[]"))):
                usage = slot.get("usage", "").lower()
                if not usage:
                    continue
                source_texture_path = slot.get("path", "")
                slot_key = f"hwek_ugx_texture_slot_{slot_index}"
                value = material_texture_slot_current_value(mat, slot_index, slot)
                field_changed = slot_key in mat and str(mat.get(slot_key, source_texture_path)).strip() != str(mat.get(slot_key + "_synced", source_texture_path)).strip()
                node_value = material_node_texture_path(mat, usage, source_path, source_texture_path, slot_index)
                if not field_changed and node_value and node_value != source_texture_path:
                    value = node_value
                value = str(value).strip()
                if not source_texture_path and disabled_texture_path(value):
                    continue
                if value != str(slot.get("path", "")):
                    offset = int(slot.get("offset", 0))
                    if not offset:
                        if model.cache_version == 4 and v4_material_usage(usage) in V4_MATERIAL_MAP_USAGES and value and not disabled_texture_path(value):
                            rebuild_v4_material_bdt = True
                            continue
                        raise UgxError(f"Material {mat.name} {usage} texture is not patchable.")
                    add_bdt_replacement(material_bdt_replacements, offset, value, f"Material {mat.name} {usage} texture")
            for info in json.loads(mat.get("hwek_ugx_ufx_attrs", "{}")).values():
                if str(info.get("kind", "")) != "string":
                    continue
                name = str(info.get("name", ""))
                prop = ufx_attr_property_name(name)
                if prop not in mat:
                    continue
                value = str(mat.get(prop, info.get("value", "")))
                if value == str(info.get("value", "")):
                    continue
                offset = int(info.get("file_offset", 0))
                if not offset:
                    raise UgxError(f"Material {mat.name} {name} is not patchable.")
                add_bdt_replacement(material_bdt_replacements, offset, value, f"Material {mat.name} {name}")
            for info in json.loads(mat.get("hwek_ugx_v4_attrs", "{}")).values():
                if str(info.get("kind", "")) != "string":
                    continue
                name = str(info.get("name", ""))
                prop = v4_attr_property_name(name)
                if prop not in mat:
                    continue
                value = str(mat.get(prop, info.get("value", "")))
                if value == str(info.get("value", "")):
                    continue
                offset = int(info.get("file_offset", 0))
                if not offset:
                    raise UgxError(f"Material {mat.name} {name} is not patchable.")
                add_bdt_replacement(material_bdt_replacements, offset, value, f"Material {mat.name} {name}")

        if rebuild_v6_material_bdt:
            material_chunk = build_v6_material_bdt_from_specs(v6_material_specs_for_model(export_obj, material_model))
            material_chunk = rebuild_material_tree(material_record.data,material_chunk,material_model.materials)
            data = replace_ecf_chunk(data, MATERIAL_CHUNK, material_chunk)
        elif rebuild_v4_material_bdt:
            material_count = len(model.materials)
            material_chunk = build_v4_material_bdt(
                v4_material_names_for_mesh(export_obj.data, material_count),
                v4_material_slots_for_mesh(export_obj.data, material_count),
                v4_material_attrs_for_mesh(export_obj.data, material_count),
                preserve_schema=True,
            )
            material_chunk = rebuild_material_tree(material_record.data,material_chunk,model.materials)
            data = replace_ecf_chunk(data, MATERIAL_CHUNK, material_chunk)
        else:
            v4_binary_changed = patch_material_v4_fields(data, export_obj)
            ufx_binary_changed = patch_material_ufx_fields(data, export_obj)
            data = patch_material_bdt_strings(data, material_bdt_replacements, material_bdt_entry_replacements)
            if v4_binary_changed or ufx_binary_changed:
                refresh_material_bdt_crc(data)
        if model.cache_version == 4 and not direct_meshes:
            material_model = parse_ugx_data(source_path, bytes(data))
            before_layout = material_model
            data, material_model = extend_hw1_uv_sets(data, material_model, export_obj.data)
            if topology_unchanged and material_model is not before_layout:
                patch_mesh_vertices(data, material_model, export_obj)
        bounds_source = model
        if not structural_change and (geometry_changed or not direct_meshes):
            bounds_source = parse_ugx_data(source_path, bytes(data), reuse_geometry_from=model)
        if structural_change:
            patched_model = parse_ugx_data(source_path, bytes(data))
            structural_model.bones = patched_model.bones
            structural_model.materials = patched_model.materials
            data = rebuild_topology_buffers(data, structural_model, export_obj, True, preserve_source_basis=True)
        elif not topology_unchanged:
            data = rebuild_topology_buffers(data, material_model, export_obj)
        skeleton_target = parse_ugx_data(source_path, bytes(data), reuse_geometry_from=model)
        skeleton_changed = any(a.name != b.name or a.parent != b.parent or a.matrix != b.matrix
                               for a, b in zip(model.bones, skeleton_target.bones))
        skeleton_changed |= any(a.name != b.name or a.parent != b.parent or a.inverse_world != b.matrix
                                for a, b in zip(skeleton_target.granny_info.bones, skeleton_target.bones))
        if skeleton_changed or skeleton_resized:
            if geometry_changed or not topology_unchanged or skeleton_resized:
                skeleton_target = parse_ugx_data(source_path, bytes(data))
            if skeleton_resized:
                refresh_resized_bone_bounds(data, skeleton_target, skeleton_resized['bounds_overrides'])
            identity_model = copy.copy(skeleton_target)
            identity_model.bones = [copy.copy(bone) for bone in model.bones]
            for bone, native in zip(identity_model.bones, model.granny_info.bones):
                bone.name = native.name
            indices = skeleton_resized['section_mesh_indices'] if skeleton_resized else section_mesh_indices(identity_model)
            names = skeleton_target.granny_info.models[0].mesh_names
            bindings = authored_granny_meshes(skeleton_target, names,
                       [[index for index, mesh in enumerate(indices) if mesh == slot] for slot in range(len(names))])
            source_grx = EcfFile.parse(bytes(data)).chunks[0x703].data
            patched_grx = patch_granny_skeleton(source_grx, model.bones, skeleton_target.bones,
                                                [mesh.bone_bindings for mesh in bindings], update_bindings=skeleton_resized)
            data = replace_ecf_chunk(data, 0x703, patched_grx)
        canonical_model = parse_ugx_data(source_path, bytes(data))
        patched_lod = patch_armature_skeleton_lod(data, canonical_model, armature)
        if patched_lod is not data:
            data = patched_lod
            canonical_model = parse_ugx_data(source_path, bytes(data))
        if not structural_change and (geometry_changed or not direct_meshes):
            indices = skeleton_resized['section_mesh_indices'] if skeleton_resized else section_mesh_indices(model)
            data, bounds_changed = patch_edited_mesh_bounds(data, model, canonical_model, indices, bounds_source)
            if bounds_changed:
                canonical_model = parse_ugx_data(source_path, bytes(data))
        data = patch_tree_bounds(data, model, canonical_model)
        data, added_tangents = extend_hw1_tangents(data, canonical_model)
        if added_tangents:
            canonical_model = parse_ugx_data(source_path, bytes(data))
        data, bone_order = canonicalize_ugx_skeleton(data, canonical_model)
        grx_chunk = EcfFile.parse(bytes(data)).chunks.get(0x703)
        if grx_chunk is not None:
            repaired_grx = repair_granny_model_binding_overlap(grx_chunk.data)
            if repaired_grx is not grx_chunk.data:
                data = replace_ecf_chunk(data, 0x703, repaired_grx)
        reordered_bones = bone_order != list(range(len(bone_order)))
        if model.cache_version == 4 and model.granny_info.models:
            renamed = {}
            names_by_mesh = {}
            for item in lod_objects:
                index = int(item.get('hwek_ugx_mesh_index', -1))
                names_by_mesh.setdefault(index, set()).add(current_ugx_mesh_name(item))
            for index, names in names_by_mesh.items():
                if 0 <= index < len(model.granny_info.models[0].mesh_names) and len(names) == 1:
                    name = next(iter(names))
                    if name != model.granny_info.models[0].mesh_names[index]:
                        renamed[index] = name
            if renamed:
                native = EcfFile.parse(bytes(data)).chunks[0x703].data
                data = replace_ecf_chunk(data, 0x703, patch_granny_mesh_names(native, renamed))
        armature = ugx_armature_for(obj)
        if armature is not None and 'hwek_ugx_granny_model_name' in armature:
            name = safe_grx_name(str(armature['hwek_ugx_granny_model_name']), '')
            if model.granny_info.models and name != model.granny_info.models[0].name:
                native = EcfFile.parse(bytes(data)).chunks[0x703].data
                data = replace_ecf_chunk(data, 0x703, patch_granny_model_name(native, name))
        data = patch_armature_model_placement(data, canonical_model, armature)
        updated_model = parse_ugx_data(source_path, bytes(data), reuse_geometry_from=None if reordered_bones or skeleton_resized or added_tangents else model)
        validate_game_ready_ugx_model(updated_model, model)
        updated_ecf = EcfFile.parse(bytes(data))
        instance_validator = validate_v4_instanced_index_buffer if updated_model.cache_version == 4 else validate_v6_instanced_index_buffer
        instance_validator(
            updated_model,
            updated_ecf.chunks[CACHED_DATA_CHUNK].data,
            updated_ecf.chunks[INDEX_BUFFER_CHUNK].data,
        )

        refresh_changed_ecf_adlers(data, original)
        texture_outputs = write_ugx_with_textures(obj, path, data, updated_model.cache_version, export_loaded_textures, export_visual)
        # Shape keys bake derived geometry. V4 topology rebuilds reorder output
        # vertices without remapping editable source IDs. Both retain the source
        # template, artist basis and bone identities for subsequent edits.
        if reordered_bones and not retain_source_basis:
            for new, old in enumerate(bone_order):
                armature_bones[old]['hwek_ugx_bone_index'] = new
        if structural_change and not retain_source_basis:
            metadata_objects = rebase_structural_export_objects(
                structural_object_pairs,
                updated_model,
                path,
                bytes(data),
            )
        elif not retain_source_basis:
            refresh_export_object_metadata(metadata_objects, updated_model, path, bytes(data))
            if added_tangents:
                refresh_added_hw1_tangent_attributes(metadata_objects, updated_model, added_tangents)
            if direct_meshes and not geometry_changed:
                for item in lod_objects:
                    remember_imported_mesh(item, unchanged_buffers=True)
            elif topology_unchanged:
                for item in lod_objects:
                    refresh_exported_mesh_basis(item, updated_model)
        return texture_outputs
    finally:
        remove_export_proxy(proxy)
        remove_temp_objects(temp_objects)
        if restore_edit_mode and obj.name in bpy.data.objects:
            try:
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)
                bpy.ops.object.mode_set(mode="EDIT")
            except Exception:
                traceback.print_exc()


def export_new_v4_ugx(obj, path, axis_forward="Z", axis_up="Y", export_loaded_textures=False, export_visual=False):
    discover_ugx_texture_dependencies(obj, path)
    with prepared_shape_key_edits(obj), prepared_vertex_color_edits(obj), prepared_hw1_uv_mappings(obj):
        return _export_new_v4_ugx(obj, path, axis_forward, axis_up, export_loaded_textures, export_visual)


def export_new_v6_ugx(obj, path, export_loaded_textures=False, export_visual=False):
    with prepared_shape_key_edits(obj), prepared_vertex_color_edits(obj):
        return _export_new_v6_ugx(obj, path, export_loaded_textures, export_visual)


def export_ugx_copy(obj, path, export_loaded_textures=False, export_visual=False):
    with (
        prepared_shape_key_edits(obj),
        prepared_imported_skeleton(obj) as skeleton,
        prepared_vertex_color_edits(obj),
        prepared_rigid_skin_edits(obj) as skin,
    ):
        prepared = skeleton or skin
        source = ugx_export_object_for(obj)
        if source is not None and int(source.get("hwek_ugx_cache_version", 0)) == 4:
            discover_ugx_texture_dependencies(source, path)
            with prepared_hw1_uv_mappings(source, imported=True):
                return _export_ugx_copy(source, path, export_loaded_textures, export_visual, prepared)
        return _export_ugx_copy(obj, path, export_loaded_textures, export_visual, prepared)
