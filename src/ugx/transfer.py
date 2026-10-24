"""Adopt donor meshes and rebase metadata after structural exports."""

from typing import Dict, Iterable, List, Tuple
import bpy
import copy
import json
from mathutils import Vector
from ..errors import UgxError
from ..formats.binary import sha256_bytes
from ..formats.granny import granny_mesh_for_model_index
from ..formats.matrix import mat4_mul
from ..materials.context import ugx_material_index_for_slot
from ..materials.data import editable_ugx_texture_slot
from ..materials.native import remember_native_material
from ..materials.properties import (
    ensure_ufx_shader_hash_string_props,
    is_ufx_shader_hash_name,
    set_ufx_buffer_material_properties,
    ufx_attr_property_name,
    ufx_idprop_value,
    ufx_shader_hash_text_from_info,
    v4_attr_property_name,
)
from .data import Accessory, Section, UgxModel, is_no_rigid_bone_index, source_vertex_section_ranges, ugx_game_name
from .bone_data import bone_display_matrices
from .transforms import bone_rest_display_rows, bone_head_export_position, blender_matrix_to_game_rows
from .model_space import ugx_model_matrix
from .mesh import mesh_int_attribute, object_matrix_is_identity
from .patch import same_topology
from .snapshots import remember_imported_mesh
from .metadata import (
    build_section_entry,
    cache_ugx_source_snapshot,
    lod_section_indices,
    section_entry_source_index,
    set_section_metadata_properties,
    set_bone_bound_properties,
    set_ugx_source_metadata,
    ugx_source_bytes_for_export,
)
from .naming import current_ugx_mesh_name, section_lod_key, section_mesh_indices
from .skeleton_patch import inverse_bind_rows, transform_bone_binding_bounds
from .read import parse_ugx_data
from .scene import ugx_armature_for, ugx_export_object_for, ugx_lod_sibling_objects_for, ugx_ordered_armature_bones
from .vertices import layout_for
from .coordinates import game_to_blender_vec3

__all__ = ('compatible_rebased_source', 'set_ugx_transplant_source_metadata', 'ugx_transplant_source_bytes', 'ugx_transplant_model', 'json_int_map', 'ugx_transplant_section_map', 'ugx_transplant_material_map', 'ugx_transplant_bone_map', 'ugx_transplant_mesh_index', 'remap_transplanted_granny_mesh', 'bone_name_index_map', 'transplanted_section_required_bones', 'remap_transplanted_v6_section', 'target_accessory_for_donor', 'remap_object_vertex_groups_to_target', 'adopt_ugx_mesh_into_target', 'refresh_export_source_metadata', 'refresh_export_material_metadata', 'refresh_export_bone_metadata', 'refresh_export_object_metadata', 'refresh_added_hw1_tangent_attributes', 'clear_ugx_transplant_metadata', 'copy_mesh_int_attribute', 'rebase_structural_export_objects', 'rebase_legacy_structural_export_objects')

def compatible_rebased_source(obj: bpy.types.Object, model: UgxModel) -> Tuple[bool, str]:
    checks = (
        ("version", int(obj.get("hwek_ugx_cache_version", model.cache_version)), model.cache_version),
        ("sections", int(obj.get("hwek_ugx_section_count", len(model.sections))), len(model.sections)),
        ("materials", int(obj.get("hwek_ugx_material_count", len(model.materials))), len(model.materials)),
        ("bones", int(obj.get("hwek_ugx_bone_count", len(model.bones))), len(model.bones)),
    )
    mismatches = [f"{name} {expected}!={actual}" for name, expected, actual in checks if expected != actual]
    return (not mismatches, ", ".join(mismatches))


def set_ugx_transplant_source_metadata(obj: bpy.types.Object, source_path: str, data: bytes):
    digest = sha256_bytes(data)
    obj["hwek_ugx_transplant_source_path"] = source_path
    obj["hwek_ugx_transplant_source_sha256"] = digest
    try:
        snapshot_path = cache_ugx_source_snapshot(source_path, data, digest)
    except OSError:
        snapshot_path = ""
    if snapshot_path:
        obj["hwek_ugx_transplant_source_cache_path"] = snapshot_path
        obj["hwek_ugx_transplant_source_cache_sha256"] = digest


def ugx_transplant_source_bytes(obj: bpy.types.Object) -> bytes:
    source_path = str(obj.get("hwek_ugx_transplant_source_path", ""))
    expected_hash = str(obj.get("hwek_ugx_transplant_source_sha256", ""))
    candidates = (
        (source_path, "donor source"),
        (str(obj.get("hwek_ugx_transplant_source_cache_path", "")), "donor snapshot"),
    )
    for candidate, _ in candidates:
        if not candidate:
            continue
        try:
            with open(candidate, "rb") as handle:
                data = handle.read()
        except OSError:
            continue
        if not expected_hash or sha256_bytes(data) == expected_hash:
            return data
    raise UgxError(
        f"The donor UGX for {obj.name} changed or is unavailable. Reimport the donor and adopt the mesh again."
    )


def ugx_transplant_model(obj: bpy.types.Object) -> UgxModel:
    source_path = str(obj.get("hwek_ugx_transplant_source_path", ""))
    if not source_path:
        raise UgxError(f"{obj.name} has no donor UGX metadata.")
    model = parse_ugx_data(source_path, ugx_transplant_source_bytes(obj))
    if model.cache_version != 6:
        raise UgxError("Only Halo Wars 2 v6 meshes can be adopted between UGX files.")
    return model


def json_int_map(obj: bpy.types.Object, key: str, left: str, right: str) -> Dict[int, int]:
    try:
        values = json.loads(obj.get(key, "[]"))
    except Exception as exc:
        raise UgxError(f"{obj.name} has invalid {key} metadata.") from exc
    result: Dict[int, int] = {}
    for value in values:
        result[int(value[left])] = int(value[right])
    return result


def ugx_transplant_section_map(obj: bpy.types.Object) -> Dict[int, int]:
    return json_int_map(obj, "hwek_ugx_transplant_section_map", "virtual", "donor")


def ugx_transplant_material_map(obj: bpy.types.Object) -> Dict[int, int]:
    return json_int_map(obj, "hwek_ugx_transplant_material_map", "donor", "target")


def ugx_transplant_bone_map(obj: bpy.types.Object) -> List[int]:
    try:
        return [int(value) for value in json.loads(obj.get("hwek_ugx_transplant_bone_map", "[]"))]
    except Exception as exc:
        raise UgxError(f"{obj.name} has invalid transplanted bone metadata.") from exc


def ugx_transplant_mesh_index(obj: bpy.types.Object, donor: UgxModel) -> int:
    if obj.get("hwek_ugx_transplant_source_path"):
        sections = list(ugx_transplant_section_map(obj).values())
        identity = obj.get("hwek_ugx_transplant_mesh_index")
    else:
        sections = lod_section_indices(obj)
        identity = obj.get("hwek_ugx_mesh_index")
    indices = section_mesh_indices(donor)
    if not sections or any(index < 0 or index >= len(indices) for index in sections):
        raise UgxError(f"{obj.name} has invalid donor section identities.")
    owners = {indices[index] for index in sections}
    if len(owners) != 1:
        raise UgxError(f"{obj.name} combines sections from multiple donor native meshes.")
    inferred = next(iter(owners))
    if identity is None:
        identity = inferred
    if int(identity) != inferred:
        raise UgxError(f"{obj.name} donor mesh identity disagrees with its source sections.")
    if donor.granny_info.parsed and granny_mesh_for_model_index(donor.granny_info, int(identity)) is None:
        raise UgxError(f"{obj.name} references a missing donor native mesh {identity}.")
    return int(identity)


def remap_transplanted_granny_mesh(mesh, donor: UgxModel, target: UgxModel, bone_map: List[int], name: str):
    result = copy.deepcopy(mesh)
    result.name = name
    source_names = {}
    target_names = {}
    for index, bone in enumerate(donor.bones):
        source_names.setdefault(bone.name, index)
    for index, bone in enumerate(target.bones):
        target_names.setdefault(bone.name, index)
    for binding in result.bone_bindings:
        source = source_names.get(binding.bone_name)
        target_index = bone_map[source] if source is not None and source < len(bone_map) else -1
        if not 0 <= target_index < len(target.bones):
            raise UgxError(f"Donor native mesh {mesh.name!r} binds bone {binding.bone_name!r}, which is absent from the target skeleton.")
        binding.bone_name = target.bones[target_index].name
        # Native bindings resolve a repeated name to its first bone frame.
        before = donor.bones[source].matrix
        after = target.bones[target_names[binding.bone_name]].matrix
        if before != after:
            delta = mat4_mul(inverse_bind_rows(before), after)
            bounds = transform_bone_binding_bounds(tuple(binding.bounds_min) + tuple(binding.bounds_max), delta)
            binding.bounds_min, binding.bounds_max = tuple(bounds[:3]), tuple(bounds[3:])
    return result


def bone_name_index_map(donor: UgxModel, target: UgxModel) -> List[int]:
    targets: Dict[str, List[int]] = {}
    for index, bone in enumerate(target.bones):
        targets.setdefault(bone.name.lower(), []).append(index)
    used: Dict[str, int] = {}
    result = []
    for bone in donor.bones:
        key = bone.name.lower()
        occurrence = used.get(key, 0)
        candidates = targets.get(key, [])
        result.append(candidates[occurrence] if occurrence < len(candidates) else -1)
        used[key] = occurrence + 1
    return result


def transplanted_section_required_bones(section: Section) -> set:
    required = set(section.local_to_global_bones)
    if not section.local_to_global_bones:
        for vertex in section.vertices:
            required.update(
                int(index)
                for index, weight in zip(vertex.bone_indices, vertex.bone_weights)
                if weight > 0.00001
            )
        if not is_no_rigid_bone_index(section.rigid_bone_index):
            required.add(int(section.rigid_bone_index))
    return required


def remap_transplanted_v6_section(section: Section, bone_map: List[int]) -> Section:
    remapped = copy.deepcopy(section)
    required = transplanted_section_required_bones(remapped)
    missing = sorted(index for index in required if index < 0 or index >= len(bone_map) or bone_map[index] < 0)
    if missing:
        raise UgxError(f"Donor section uses bone(s) absent from the target skeleton: {missing[:8]}")

    if remapped.local_to_global_bones:
        remapped.local_to_global_bones = [bone_map[index] for index in remapped.local_to_global_bones]
        return remapped

    if not is_no_rigid_bone_index(remapped.rigid_bone_index):
        remapped.rigid_bone_index = bone_map[remapped.rigid_bone_index]
    index_layout = layout_for(remapped.layout, "I")
    raw = bytearray(remapped.source_vertex_data)
    vertices = []
    for vertex_index, source_vertex in enumerate(remapped.vertices):
        vertex = copy.deepcopy(source_vertex)
        mapped_indices = tuple(
            bone_map[index] if weight > 0.00001 else int(index)
            for index, weight in zip(vertex.bone_indices, vertex.bone_weights)
        )
        if any(weight > 0.00001 for weight in vertex.bone_weights):
            if index_layout is None or index_layout.type_id != 6 or not raw:
                raise UgxError("The donor skin index layout cannot be remapped safely.")
            offset = vertex_index * remapped.vertex_stride + index_layout.offset
            if offset + 4 > len(raw) or any(index < 0 or index > 255 for index in mapped_indices):
                raise UgxError("The remapped donor skin indices do not fit the source vertex layout.")
            raw[offset:offset + 4] = bytes(mapped_indices)
        vertex.bone_indices = mapped_indices
        vertices.append(vertex)
    remapped.vertices = vertices
    remapped.source_vertex_data = bytes(raw)
    return remapped


def target_accessory_for_donor(accessory: Accessory, bone_map: List[int], bone_count: int) -> Accessory:
    donor_indices = range(max(0, accessory.first_bone), max(0, accessory.first_bone + accessory.bone_count))
    mapped = [bone_map[index] for index in donor_indices if 0 <= index < len(bone_map) and bone_map[index] >= 0]
    if not mapped:
        return Accessory(0, max(1, bone_count))
    first = min(mapped)
    return Accessory(first, max(mapped) - first + 1)


def remap_object_vertex_groups_to_target(
    obj: bpy.types.Object,
    donor: UgxModel,
    target: UgxModel,
    bone_map: List[int],
    target_object: bpy.types.Object,
):
    weights: List[List[Tuple[int, float]]] = []
    for vertex in obj.data.vertices:
        mapped = []
        for group in vertex.groups:
            if group.weight <= 0.00001:
                continue
            if group.group >= len(bone_map) or bone_map[group.group] < 0:
                donor_name = donor.bones[group.group].name if group.group < len(donor.bones) else str(group.group)
                raise UgxError(f"Donor vertex weights use bone {donor_name!r}, which is absent from the target skeleton.")
            mapped.append((bone_map[group.group], float(group.weight)))
        weights.append(mapped)

    for group in list(obj.vertex_groups):
        obj.vertex_groups.remove(group)
    target_group_names = [group.name for group in target_object.vertex_groups]
    if len(target_group_names) != len(target.bones):
        target_group_names = [bone.name for bone in target.bones]
    for name in target_group_names:
        obj.vertex_groups.new(name=name)
    for vertex_index, groups in enumerate(weights):
        for group_index, weight in groups:
            obj.vertex_groups[group_index].add([vertex_index], weight, "REPLACE")


def adopt_ugx_mesh_into_target(target_obj: bpy.types.Object, donor_obj: bpy.types.Object) -> bpy.types.Object:
    target_obj = ugx_export_object_for(target_obj)
    donor_obj = ugx_export_object_for(donor_obj)
    if target_obj is None or donor_obj is None or target_obj.type != "MESH" or donor_obj.type != "MESH":
        raise UgxError("Select one imported donor mesh and make an imported target UGX mesh active.")
    if target_obj == donor_obj:
        raise UgxError("The donor mesh and target UGX must be different objects.")
    if str(target_obj.get("hwek_ugx_import_id", "")) == str(donor_obj.get("hwek_ugx_import_id", "")):
        raise UgxError("The selected mesh already belongs to the active UGX.")

    target_bytes, target_mode = ugx_source_bytes_for_export(target_obj)
    donor_bytes, donor_mode = ugx_source_bytes_for_export(donor_obj)
    target_model = parse_ugx_data(str(target_obj.get("hwek_ugx_source_path", "")), target_bytes)
    donor_model = parse_ugx_data(str(donor_obj.get("hwek_ugx_source_path", "")), donor_bytes)
    if target_model.cache_version != 6 or donor_model.cache_version != 6:
        raise UgxError("Cross-file mesh adoption currently supports Halo Wars 2 v6 UGXs only.")
    if target_mode == "rebased_source" or donor_mode == "rebased_source":
        raise UgxError("A UGX source changed since import. Reimport both files before adopting a mesh.")

    donor_sections = lod_section_indices(donor_obj)
    if not donor_sections:
        raise UgxError("The donor object has no UGX sections.")
    if any(index < 0 or index >= len(donor_model.sections) for index in donor_sections):
        raise UgxError("The donor object's section metadata does not match its source UGX.")

    donor_mesh_index = ugx_transplant_mesh_index(donor_obj, donor_model)
    bone_map = bone_name_index_map(donor_model, target_model)
    for section_index in donor_sections:
        remap_transplanted_v6_section(donor_model.sections[section_index], bone_map)

    target_objects = ugx_lod_sibling_objects_for(target_obj)
    used_sections = set(range(len(target_model.sections)))
    used_meshes = set(range(len(target_model.accessories)))
    used_materials = set(range(len(target_model.materials)))
    for item in target_objects:
        used_sections.update(lod_section_indices(item))
        used_meshes.add(int(item.get("hwek_ugx_mesh_index", -1)))
        if item.get("hwek_ugx_transplant_material_map"):
            used_materials.update(ugx_transplant_material_map(item).values())

    next_section = max(used_sections, default=-1) + 1
    virtual_sections = list(range(next_section, next_section + len(donor_sections)))
    section_map = dict(zip(virtual_sections, donor_sections))
    mesh_index = max(used_meshes, default=-1) + 1

    donor_material_indices = []
    for slot_index, material in enumerate(donor_obj.data.materials):
        if material is None:
            donor_material_indices.append(-1)
            continue
        donor_index = ugx_material_index_for_slot(donor_obj, slot_index, donor_model)
        if donor_index < 0 or donor_index >= len(donor_model.materials):
            raise UgxError(f"Donor material slot {slot_index} maps outside its source UGX.")
        donor_material_indices.append(donor_index)
    material_map: Dict[int, int] = {}
    next_material = max(used_materials, default=-1) + 1
    for donor_index in donor_material_indices:
        if donor_index >= 0 and donor_index not in material_map:
            material_map[donor_index] = next_material
            next_material += 1

    adopted = donor_obj.copy()
    adopted.data = donor_obj.data.copy()
    collection = target_obj.users_collection[0] if target_obj.users_collection else bpy.context.scene.collection
    collection.objects.link(adopted)
    desired_mesh_name = current_ugx_mesh_name(donor_obj)
    adopted.name = desired_mesh_name
    adopted["hwek_ugx_mesh_name"] = desired_mesh_name
    adopted["hwek_ugx_mesh_import_name"] = adopted.name
    adopted["hwek_ugx_mesh_addition_anchor"] = current_ugx_mesh_name(target_obj)

    for slot_index, material in enumerate(list(adopted.data.materials)):
        if material is not None:
            adopted.data.materials[slot_index] = material.copy()
    adopted["hwek_ugx_material_slot_indices"] = json.dumps([
        material_map.get(index, 0) for index in donor_material_indices
    ])

    section_attribute = adopted.data.attributes.get("UGX_Section") if hasattr(adopted.data, "attributes") else None
    if section_attribute is None or len(section_attribute.data) < len(adopted.data.polygons):
        raise UgxError("The donor mesh is missing its UGX_Section face attribute.")
    donor_to_virtual = {donor: virtual for virtual, donor in section_map.items()}
    for item in section_attribute.data:
        source_index = int(item.value)
        if source_index not in donor_to_virtual:
            raise UgxError("The donor mesh contains faces outside its declared UGX sections.")
        item.value = donor_to_virtual[source_index]

    old_entries = json.loads(donor_obj.get("hwek_ugx_sections", "[]"))
    old_by_source = {
        section_entry_source_index(index, entry): entry
        for index, entry in enumerate(old_entries)
    }
    entries = []
    for virtual_index, donor_index in section_map.items():
        source = donor_model.sections[donor_index]
        old = old_by_source.get(donor_index, {})
        entries.append(build_section_entry(
            source,
            virtual_index,
            int(old.get("vertex_start", 0)),
            int(old.get("vertex_count", len(source.vertices))),
            int(old.get("face_start", 0)),
            int(old.get("face_count", len(source.indices) // 3)),
        ))

    set_ugx_transplant_source_metadata(adopted, donor_model.path, donor_bytes)
    set_ugx_source_metadata(adopted, target_model.path, target_bytes)
    adopted["hwek_ugx_cache_version"] = 6
    adopted["hwek_ugx_game"] = ugx_game_name(6)
    adopted["hwek_ugx_import_id"] = str(target_obj.get("hwek_ugx_import_id", ""))
    adopted["hwek_ugx_section_count"] = len(target_model.sections)
    adopted["hwek_ugx_bone_count"] = len(target_model.bones)
    adopted["hwek_ugx_accessory_count"] = len(target_model.accessories) + 1
    adopted["hwek_ugx_material_count"] = next_material
    adopted["hwek_ugx_lod_index"] = int(donor_obj.get("hwek_ugx_lod_index", 0))
    adopted["hwek_ugx_lod_min_distance"] = float(donor_obj.get("hwek_ugx_lod_min_distance", 0.0))
    adopted["hwek_ugx_lod_max_distance"] = float(donor_obj.get("hwek_ugx_lod_max_distance", 0.0))
    adopted["hwek_ugx_transplant_mesh_index"] = donor_mesh_index
    adopted["hwek_ugx_mesh_index"] = mesh_index
    adopted["hwek_ugx_lod_section_indices"] = json.dumps(virtual_sections)
    adopted["hwek_ugx_sections"] = json.dumps(entries)
    adopted["hwek_ugx_transplant_section_map"] = json.dumps([
        {"virtual": virtual, "donor": donor} for virtual, donor in section_map.items()
    ])
    adopted["hwek_ugx_transplant_material_map"] = json.dumps([
        {"donor": donor, "target": target} for donor, target in material_map.items()
    ])
    adopted["hwek_ugx_transplant_bone_map"] = json.dumps(bone_map)
    set_section_metadata_properties(adopted, entries)

    remap_object_vertex_groups_to_target(adopted, donor_model, target_model, bone_map, target_obj)
    for modifier in list(adopted.modifiers):
        if modifier.type == "ARMATURE":
            adopted.modifiers.remove(modifier)
    target_armature = ugx_armature_for(target_obj)
    if target_armature:
        modifier = adopted.modifiers.new("UGX Armature", "ARMATURE")
        modifier.object = target_armature

    all_target_objects = target_objects + [adopted]
    for item in all_target_objects:
        item["hwek_ugx_lod_object_count"] = len(all_target_objects)
    bpy.ops.object.select_all(action="DESELECT")
    adopted.select_set(True)
    bpy.context.view_layer.objects.active = adopted
    return adopted


def refresh_export_source_metadata(objects: Iterable[bpy.types.Object], source_path: str, data: bytes):
    digest = sha256_bytes(data)
    for item in objects:
        if item and item.type == "MESH" and item.get("hwek_ugx_source_path"):
            set_ugx_source_metadata(item, source_path, data, digest=digest)


def refresh_export_material_metadata(obj: bpy.types.Object, model: UgxModel, refreshed=None):
    for slot_index, mat in enumerate(obj.data.materials):
        if mat is None:
            continue
        material_index = ugx_material_index_for_slot(obj, slot_index, model)
        if material_index < 0 or material_index >= len(model.materials):
            continue
        key = (mat, material_index)
        if refreshed is not None:
            if key in refreshed:
                continue
            refreshed.add(key)
        source = model.materials[material_index]
        remember_native_material(mat, source)
        mat["hwek_ugx_material_index"] = material_index
        mat['hwek_ugx_source_path']=str(obj.get('hwek_ugx_source_path',model.path))
        mat['hwek_ugx_has_hogan_material']=source.has_hogan_material
        mat["hwek_ugx_source_name"] = source.name
        mat["hwek_ugx_export_name"] = source.name
        mat["hwek_ugx_name_offset"] = source.name_offset
        mat["hwek_ugx_name_capacity"] = source.name_capacity
        mat["hwek_ugx_name_entry_offset"] = source.name_entry_offset
        mat["hwek_ugx_texture_slots"] = json.dumps(source.slots)
        for slot_index, slot in enumerate(source.slots):
            usage = str(slot.get("usage", "")).lower()
            path = str(slot.get("path", ""))
            key = f"hwek_ugx_texture_slot_{slot_index}"
            if editable_ugx_texture_slot(slot, model.cache_version):
                mat[key] = path
                mat[key + "_synced"] = path
                if usage:
                    mat[f"hwek_ugx_texture_{usage}"] = path
                    mat[f"hwek_ugx_texture_{usage}_source"] = path
            else:
                for suffix in ("", "_synced", "_auto_context", "_auto_context_value"):
                    prop = key + suffix
                    if prop in mat:
                        del mat[prop]
        if source.v4_attrs:
            mat["hwek_ugx_v4_attrs"] = json.dumps(source.v4_attrs)
            for name, info in source.v4_attrs.items():
                value = ufx_idprop_value(info)
                prop = v4_attr_property_name(name)
                mat[prop] = value
                mat[prop + "_source"] = value
        if source.has_hogan_material:
            mat['hwek_ugx_ufx_constant_buffers']=json.dumps(source.ufx_constant_buffers)
        if source.ufx_attrs or source.has_hogan_material:
            mat["hwek_ugx_ufx_attrs"] = json.dumps(source.ufx_attrs)
            for name, info in source.ufx_attrs.items():
                value = ufx_shader_hash_text_from_info(info) if is_ufx_shader_hash_name(name) else ufx_idprop_value(info)
                mat[ufx_attr_property_name(name)] = value
            ensure_ufx_shader_hash_string_props(mat, source.ufx_attrs)
        if source.ufx_constant_buffers:
            mat["hwek_ugx_ufx_constant_buffers"] = json.dumps(source.ufx_constant_buffers)
            for index, buffer in enumerate(source.ufx_constant_buffers):
                set_ufx_buffer_material_properties(mat, index, buffer)
        if mat.use_nodes and mat.node_tree:
            for node in mat.node_tree.nodes:
                if node.type != "TEX_IMAGE":
                    continue
                slot_index = int(node.get("hwek_ugx_slot_index", -1))
                if 0 <= slot_index < len(source.slots):
                    path = str(source.slots[slot_index].get("path", ""))
                    node["hwek_ugx_texture_path"] = path
                    image = getattr(node, "image", None)
                    if image is not None:
                        image["hwek_ugx_texture_path"] = path


def refresh_export_bone_metadata(armatures, model):
    mode, worlds = bone_display_matrices(model)
    for rig in armatures:
        animation = rig.animation_data
        animated = animation is not None and (animation.action is not None or animation.nla_tracks)
        for index, bone in enumerate(ugx_ordered_armature_bones(rig)):
            if index >= len(model.bones):
                break
            source = model.bones[index]
            bone['hwek_ugx_source_name'] = source.name
            bone['hwek_ugx_export_name'] = source.name
            bone['hwek_ugx_blender_name'] = bone.name
            bone['hwek_ugx_matrix_offset'] = source.matrix_offset
            bone['hwek_ugx_source_matrix'] = json.dumps(source.matrix)
            bone['hwek_ugx_export_matrix'] = json.dumps(source.matrix)
            bone['hwek_ugx_source_display_matrix'] = json.dumps(worlds[index])
            bone['hwek_ugx_import_display_matrix'] = json.dumps(bone_rest_display_rows(rig, bone))
            bone['hwek_ugx_source_head_local'] = bone_head_export_position(rig, bone)
            bone['hwek_ugx_skeleton_transform_mode'] = mode
            set_bone_bound_properties(bone, source)
            pose = rig.pose.bones.get(bone.name)
            if pose is not None and not animated:
                bone['hwek_ugx_source_pose_basis'] = json.dumps([list(row) for row in pose.matrix_basis.to_3x3()])
                bone['hwek_ugx_import_pose_display_matrix'] = json.dumps(
                    blender_matrix_to_game_rows(ugx_model_matrix(rig) @ pose.matrix))
            if index < len(model.granny_info.bones):
                native = model.granny_info.bones[index]
                transform = native.transform
                bone['hwek_ugx_granny_local_transform'] = json.dumps([transform.flags,transform.position,transform.orientation,transform.scale_shear])
                bone['hwek_ugx_granny_lod_error'] = native.lod_error


def refresh_added_hw1_tangent_attributes(objects, model, sections):
    ranges = source_vertex_section_ranges(model)
    vertices = {ranges[index][0]+local: vertex for index in sections
                for local, vertex in enumerate(model.sections[index].vertices)}
    for obj in objects:
        source = mesh_int_attribute(obj.data, 'UGX_SourceVertex')
        tangents = obj.data.attributes.get('UGX_Tangent')
        signs = obj.data.attributes.get('UGX_BitangentSign')
        if tangents is None or signs is None:
            continue
        inverse = ugx_model_matrix(obj).to_3x3().inverted_safe()
        for local, index in enumerate(source):
            vertex = vertices.get(index)
            if vertex is None:
                continue
            tangents.data[local].vector = inverse @ Vector(game_to_blender_vec3(vertex.tangent))
            signs.data[local].value = vertex.bitangent_sign


def refresh_export_object_metadata(objects: Iterable[bpy.types.Object], model: UgxModel, source_path: str, data: bytes):
    objects = list(objects)
    refresh_export_source_metadata(objects, source_path, data)
    armatures = {rig for obj in objects if (rig := ugx_armature_for(obj)) is not None}
    refresh_export_bone_metadata(armatures, model)
    refreshed_materials = set()
    for obj in objects:
        if obj is None or obj.type != "MESH":
            continue
        section_indices = json.loads(obj.get("hwek_ugx_lod_section_indices", "[]"))
        if not section_indices:
            section_indices = list(range(len(model.sections)))
        old_entries = json.loads(obj.get("hwek_ugx_sections", "[]"))
        old_by_source = {
            section_entry_source_index(index, entry): entry
            for index, entry in enumerate(old_entries)
        }
        section_attr = mesh_int_attribute(obj.data, "UGX_Section")
        faces_by_section = {}
        if section_attr is not None and len(section_attr) >= len(obj.data.polygons):
            for face_index, section_index in enumerate(section_attr):
                if section_index not in faces_by_section:
                    faces_by_section[section_index] = [face_index, 0]
                faces_by_section[section_index][1] += 1
        entries = []
        for source_section_index in section_indices:
            if source_section_index < 0 or source_section_index >= len(model.sections):
                continue
            section = model.sections[source_section_index]
            old = old_by_source.get(source_section_index, {})
            vertex_start = int(old.get("vertex_start", 0))
            vertex_count = int(old.get("vertex_count", len(section.vertices)))
            face_start = int(old.get("face_start", 0))
            if section_attr is not None and len(section_attr) >= len(obj.data.polygons):
                face_start, face_count = faces_by_section.get(source_section_index, (face_start, 0))
            else:
                face_count = int(old.get("face_count", len(section.indices) // 3))
            entries.append(build_section_entry(section, source_section_index, vertex_start, vertex_count, face_start, face_count))
        if entries:
            obj["hwek_ugx_sections"] = json.dumps(entries)
            set_section_metadata_properties(obj, entries)
        obj["hwek_ugx_section_count"] = len(model.sections)
        obj["hwek_ugx_bone_count"] = len(model.bones)
        obj['hwek_ugx_bone_patches'] = json.dumps([
            dict(name=bone.name, offset=bone.name_offset, capacity=bone.name_capacity) for bone in model.bones])
        obj['hwek_ugx_bone_bounds'] = json.dumps([
            dict(index=index, name=bone.name, low=bone.bounds_low, high=bone.bounds_high,
                 low_offset=bone.bounds_low_offset, high_offset=bone.bounds_high_offset, big_endian=bone.big_endian)
            for index, bone in enumerate(model.bones)])
        obj["hwek_ugx_accessory_count"] = len(model.accessories)
        obj["hwek_ugx_material_count"] = len(model.materials)
        refresh_export_material_metadata(obj, model, refreshed_materials)


def clear_ugx_transplant_metadata(obj: bpy.types.Object):
    for key in list(obj.keys()):
        if str(key).startswith("hwek_ugx_transplant_"):
            del obj[key]


def copy_mesh_int_attribute(source: bpy.types.Mesh, target: bpy.types.Mesh, name: str):
    source_attr = source.attributes.get(name) if hasattr(source, "attributes") else None
    target_attr = target.attributes.get(name) if hasattr(target, "attributes") else None
    if source_attr is None or target_attr is None or len(source_attr.data) != len(target_attr.data):
        raise UgxError(f"Structural UGX export could not refresh the {name} mesh attribute.")
    for source_item, target_item in zip(source_attr.data, target_attr.data):
        target_item.value = int(source_item.value)


def refresh_exported_mesh_basis(obj: bpy.types.Object, model: UgxModel):
    sections = lod_section_indices(obj)
    if not sections or any(index < 0 or index >= len(model.sections) for index in sections):
        return
    ranges = source_vertex_section_ranges(model)
    expected = [index for section in sections for index in range(*ranges[section])]
    if mesh_int_attribute(obj.data, 'UGX_SourceVertex') != expected:
        return
    local = copy.copy(model)
    local.sections = [model.sections[index] for index in sections]
    if not same_topology(local, obj.data):
        return
    vertices = [vertex for section in local.sections for vertex in section.vertices]
    linear = ugx_model_matrix(obj).to_3x3()
    inverse = linear.inverted()
    normal_to_local = linear.transposed()
    identity = object_matrix_is_identity(obj)
    for name, field, transform in (('UGX_Normal','normal',normal_to_local), ('UGX_Tangent','tangent',inverse)):
        attr = obj.data.attributes.get(name)
        if attr is None:
            continue
        for index, vertex in enumerate(vertices):
            value = Vector(game_to_blender_vec3(getattr(vertex,field)))
            if not identity:
                value = transform @ value
            if not identity and value.length_squared:
                value.normalize()
            attr.data[index].vector = value
    signs = obj.data.attributes.get('UGX_BitangentSign')
    if signs is not None:
        for index, vertex in enumerate(vertices):
            signs.data[index].value = vertex.bitangent_sign
    normals = obj.data.attributes.get('UGX_Normal')
    if normals is not None:
        obj.data.normals_split_custom_set_from_vertices([tuple(item.vector) for item in normals.data])
        obj.data.update()
    remember_imported_mesh(obj)


def rebase_structural_export_objects(
    object_pairs: List[Tuple[bpy.types.Object, bpy.types.Object]],
    model: UgxModel,
    source_path: str,
    data: bytes,
) -> List[bpy.types.Object]:
    objects = []
    for obj, remapped in object_pairs:
        if obj is None or remapped is None or obj.name not in bpy.data.objects:
            continue
        copy_mesh_int_attribute(remapped.data, obj.data, "UGX_Section")
        copy_mesh_int_attribute(remapped.data, obj.data, "UGX_SourceVertex")
        obj["hwek_ugx_lod_section_indices"] = str(remapped.get("hwek_ugx_lod_section_indices", "[]"))
        obj["hwek_ugx_mesh_index"] = int(remapped.get("hwek_ugx_mesh_index", -1))
        obj["hwek_ugx_mesh_name"] = str(remapped.get("hwek_ugx_mesh_name", current_ugx_mesh_name(obj)))
        clear_ugx_transplant_metadata(obj)
        if "hwek_ugx_reimport_required" in obj:
            del obj["hwek_ugx_reimport_required"]
        objects.append(obj)
    for obj in objects:
        obj["hwek_ugx_lod_object_count"] = len(objects)
    refresh_export_object_metadata(objects, model, source_path, data)
    for obj in objects:
        refresh_exported_mesh_basis(obj, model)
    return objects


def rebase_legacy_structural_export_objects(
    objects: List[bpy.types.Object],
    model: UgxModel,
    source_path: str,
    data: bytes,
) -> List[bpy.types.Object]:
    mesh_names = model.granny_info.models[0].mesh_names if model.granny_info.models else []
    section_ranges = source_vertex_section_ranges(model)
    rebased = []
    for obj in objects:
        if not obj.get("hwek_ugx_reimport_required"):
            continue
        old_sections = lod_section_indices(obj)
        if not old_sections:
            continue
        mesh_name = current_ugx_mesh_name(obj)
        old_mesh_index = int(obj.get("hwek_ugx_mesh_index", -1))
        if 0 <= old_mesh_index < len(mesh_names) and mesh_names[old_mesh_index].lower() == mesh_name.lower():
            mesh_index = old_mesh_index
        else:
            matches = [index for index, name in enumerate(mesh_names) if name.lower() == mesh_name.lower()]
            if len(matches) != 1:
                raise UgxError(f"Could not automatically match rebuilt mesh {mesh_name!r}; reimport the last exported UGX once.")
            mesh_index = matches[0]
        lod_key = (
            float(obj.get("hwek_ugx_lod_min_distance", 0.0)),
            float(obj.get("hwek_ugx_lod_max_distance", 0.0)),
        )
        new_sections = [
            index for index, section in enumerate(model.sections)
            if section.accessory_index == mesh_index
            and all(abs(left - right) <= 0.0001 for left, right in zip(section_lod_key(section), lod_key))
        ]
        if len(new_sections) != len(old_sections):
            new_sections = [index for index, section in enumerate(model.sections) if section.accessory_index == mesh_index]
        if len(new_sections) != len(old_sections):
            raise UgxError(
                f"Could not automatically refresh {mesh_name!r}: expected {len(old_sections)} section(s), found {len(new_sections)}. "
                "Reimport the last exported UGX once."
            )
        section_remap = dict(zip(old_sections, new_sections))
        section_attr = obj.data.attributes.get("UGX_Section") if hasattr(obj.data, "attributes") else None
        source_attr = obj.data.attributes.get("UGX_SourceVertex") if hasattr(obj.data, "attributes") else None
        if section_attr is None or source_attr is None:
            raise UgxError(f"Could not automatically refresh {mesh_name!r}; its UGX mesh attributes are missing.")
        for item in section_attr.data:
            item.value = section_remap.get(int(item.value), -1)
        for item in source_attr.data:
            item.value = -1
        try:
            entries = json.loads(obj.get("hwek_ugx_sections", "[]"))
        except Exception:
            entries = []
        old_entries = {
            section_entry_source_index(index, entry): entry
            for index, entry in enumerate(entries)
        }
        for old_index, new_index in section_remap.items():
            entry = old_entries.get(old_index, {})
            vertex_start = int(entry.get("vertex_start", 0))
            vertex_count = min(
                int(entry.get("vertex_count", 0)),
                len(model.sections[new_index].vertices),
                max(0, len(obj.data.vertices) - vertex_start),
            )
            new_start, _ = section_ranges[new_index]
            for local_index in range(vertex_count):
                source_attr.data[vertex_start + local_index].value = new_start + local_index
        obj["hwek_ugx_lod_section_indices"] = json.dumps(new_sections)
        obj["hwek_ugx_mesh_index"] = mesh_index
        obj["hwek_ugx_mesh_name"] = mesh_names[mesh_index] if mesh_index < len(mesh_names) else mesh_name
        clear_ugx_transplant_metadata(obj)
        del obj["hwek_ugx_reimport_required"]
        rebased.append(obj)
    if not rebased:
        return objects
    for obj in rebased:
        obj["hwek_ugx_lod_object_count"] = len(rebased)
    refresh_export_object_metadata(rebased, model, source_path, data)
    return rebased
