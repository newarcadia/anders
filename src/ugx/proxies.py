"""Own temporary export meshes for material, LOD and topology changes."""

from mathutils import Vector
from typing import Any, Dict, Iterable, List, Optional, Tuple
import bpy
from .model_space import ugx_model_matrix
import copy
import json
import os
from ..errors import UgxError
from ..formats.granny import GrannyModelInfo, granny_mesh_for_model_index
from ..materials.context import parsed_ufx_metadata_signature, ugx_material_index_for_slot
from ..materials.origin import extend_v6_material_model_for_objects
from .data import Accessory, UV_SEMANTICS, UgxModel, source_vertex_section_ranges
from .mesh import mesh_color_attribute, mesh_colors_by_loop, mesh_int_attribute, mesh_uv_data, mesh_uv_value, mesh_vertex_export_co, object_matrix_is_identity
from .metadata import (
    apply_common_import_metadata,
    build_section_entry,
    editable_section_metadata_keys,
    lod_section_indices,
    section_entry_source_index,
    section_property_key,
    set_section_metadata_properties,
)
from .naming import current_ugx_mesh_name, granny_mesh_name_for_accessory
from .scene import ugx_used_material_slots, ugx_armature_for, ugx_ordered_armature_bones
from .transfer import (
    remap_transplanted_v6_section,
    remap_transplanted_granny_mesh,
    ugx_transplant_mesh_index,
    target_accessory_for_donor,
    ugx_transplant_bone_map,
    ugx_transplant_material_map,
    ugx_transplant_model,
    ugx_transplant_section_map,
)

__all__ = ('object_space_to_proxy_point', 'object_space_to_proxy_direction', 'point_vector_attribute', 'build_lod_export_proxy', 'build_material_export_proxy', 'hw2_structure_changed', 'build_hw2_structural_export_proxy', 'remove_export_proxy', 'remove_temp_objects')


def build_material_export_proxy(model, objects):
    """Collect material slots and section properties without copying mesh buffers."""
    mesh = bpy.data.meshes.new(objects[0].name + '_export_materials')
    proxy = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.scene.collection.objects.link(proxy)
    materials, entries = {}, {}
    for obj in objects:
        for slot, mat in ugx_used_material_slots(obj):
            if mat is not None:
                materials.setdefault(ugx_material_index_for_slot(obj, slot, model), mat)
        for index, entry in enumerate(json.loads(obj.get('hwek_ugx_sections', '[]'))):
            section = section_entry_source_index(index, entry)
            entries[section] = dict(entry)
            for key in editable_section_metadata_keys(entry):
                entries[section][key] = obj.get(section_property_key(section, key), entry.get(key))
    # Material routines enumerate used slots through faces. One triangle per
    # slot keeps that contract while the original meshes supply all geometry.
    mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)] * len(materials))
    for slot, source in enumerate(sorted(materials)):
        mesh.materials.append(materials[source])
        mesh.polygons[slot].material_index = slot
    apply_common_import_metadata(proxy, model, str(objects[0].get('hwek_ugx_import_id', '')), len(objects))
    proxy['hwek_ugx_material_slot_indices'] = json.dumps(sorted(materials))
    proxy['hwek_ugx_sections'] = json.dumps([entries[index] for index in sorted(entries)])
    armature = ugx_armature_for(objects[0])
    if armature is not None:
        proxy.modifiers.new('UGX Armature', 'ARMATURE').object = armature
    return proxy

def object_space_to_proxy_point(obj: bpy.types.Object, co) -> Tuple[float, float, float]:
    value = ugx_model_matrix(obj) @ Vector(co)
    return (float(value.x), float(value.y), float(value.z))


def object_space_to_proxy_direction(obj: bpy.types.Object, direction) -> Tuple[float, float, float]:
    if object_matrix_is_identity(obj):
        return tuple(float(component) for component in direction)
    value = ugx_model_matrix(obj).to_3x3() @ Vector(direction)
    if value.length_squared > 0.0:
        value.normalize()
    return (float(value.x), float(value.y), float(value.z))


def point_vector_attribute(mesh: bpy.types.Mesh, name: str) -> Optional[List[Tuple[float, float, float]]]:
    if not hasattr(mesh, "attributes"):
        return None
    attr = mesh.attributes.get(name)
    if attr is None:
        return None
    return [tuple(float(component) for component in getattr(item, "vector", (0.0, 0.0, 0.0))[:3]) for item in attr.data]


def build_lod_export_proxy(model: UgxModel, objects: List[bpy.types.Object]) -> bpy.types.Object:
    if not objects:
        raise UgxError("LOD export found no mesh objects.")
    section_owner: Dict[int, bpy.types.Object] = {}
    for obj in objects:
        for section_index in lod_section_indices(obj):
            if section_index in section_owner:
                raise UgxError(f"UGX section {section_index} exists in multiple LOD objects.")
            section_owner[section_index] = obj
    missing = [index for index in range(len(model.sections)) if index not in section_owner]
    if missing:
        raise UgxError(f"LOD export is missing UGX section(s): {missing[:8]}")

    positions: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    face_materials: List[int] = []
    face_sections: List[int] = []
    source_vertices: List[int] = []
    normal_values: List[Tuple[float, float, float]] = []
    tangent_values: List[Tuple[float, float, float]] = []
    vertex_weights: List[List[Tuple[str, float]]] = []
    color_faces = []
    has_colors = any(mesh_color_attribute(item.data) is not None for item in objects)
    object_colors = {item: mesh_colors_by_loop(item.data) for item in objects} if has_colors else {}
    uv_faces: Dict[str, List[List[Tuple[float, float]]]] = {layer_name: [] for _, layer_name, _ in UV_SEMANTICS}
    section_entries: List[Dict[str, Any]] = []
    section_ranges = source_vertex_section_ranges(model)
    materials_by_source_index: Dict[int, bpy.types.Material] = {}
    has_normal_attribute = any(
        item.data.attributes.get("UGX_Normal") is not None
        for item in objects
        if hasattr(item.data, "attributes")
    )
    has_tangent_attribute = any(
        item.data.attributes.get("UGX_Tangent") is not None
        for item in objects
        if hasattr(item.data, "attributes")
    )
    for obj in objects:
        for slot_index, material in ugx_used_material_slots(obj):
            if material is None:
                continue
            source_index = ugx_material_index_for_slot(obj, slot_index, model)
            materials_by_source_index.setdefault(source_index, material)

    mesh_snapshots = {}
    for obj in objects:
        mesh = obj.data
        section_attr = mesh_int_attribute(mesh, "UGX_Section")
        source_attr = mesh_int_attribute(mesh, "UGX_SourceVertex")
        if section_attr is None or source_attr is None:
            raise UgxError("LOD export needs UGX_Section face and UGX_SourceVertex point attributes on every LOD mesh.")
        polygons_by_section = {}
        for polygon in mesh.polygons:
            polygons_by_section.setdefault(section_attr[polygon.index], []).append(polygon)
        mesh_snapshots[obj] = (
            source_attr, polygons_by_section,
            point_vector_attribute(mesh, "UGX_Normal"), point_vector_attribute(mesh, "UGX_Tangent"),
            {name: mesh_uv_data(mesh, mesh.uv_layers.get(name)) for _, name, _ in UV_SEMANTICS},
            object_matrix_is_identity(obj),
        )

    for section_index, section in enumerate(model.sections):
        obj = section_owner[section_index]
        mesh = obj.data
        source_attr, polygons_by_section, normals, tangents, uv_layers, identity_transform = mesh_snapshots[obj]
        source_start, source_end = section_ranges[section_index]
        polygons = polygons_by_section.get(section_index, [])
        used_vertices = {vertex_index for poly in polygons for vertex_index in poly.vertices}
        original_vertices = [
            (source_attr[vertex_index], vertex_index)
            for vertex_index in range(min(len(source_attr), len(mesh.vertices)))
            if source_start <= source_attr[vertex_index] < source_end
        ]
        original_vertices.sort(key=lambda item: (item[0], item[1]))
        ordered_vertices = [vertex_index for _, vertex_index in original_vertices]
        ordered_set = set(ordered_vertices)
        ordered_vertices.extend(vertex_index for vertex_index in sorted(used_vertices) if vertex_index not in ordered_set)

        vertex_start = len(positions)
        face_start = len(faces)
        vertex_map: Dict[int, int] = {}
        for vertex_index in ordered_vertices:
            vertex_map[vertex_index] = len(positions)
            vertex = mesh.vertices[vertex_index]
            positions.append(object_space_to_proxy_point(obj, mesh_vertex_export_co(obj, vertex_index)))
            source_vertices.append(source_attr[vertex_index] if vertex_index < len(source_attr) else -1)
            if has_normal_attribute:
                if normals and vertex_index < len(normals):
                    normal_values.append(normals[vertex_index] if identity_transform else object_space_to_proxy_direction(obj, normals[vertex_index]))
                else:
                    normal_values.append(object_space_to_proxy_direction(obj, vertex.normal))
            if has_tangent_attribute:
                if tangents and vertex_index < len(tangents):
                    tangent_values.append(tangents[vertex_index] if identity_transform else object_space_to_proxy_direction(obj, tangents[vertex_index]))
                else:
                    tangent_values.append((1.0, 0.0, 0.0))
            vertex_weights.append([
                (obj.vertex_groups[group.group].name, float(group.weight))
                for group in vertex.groups
                if group.weight > 0.00001 and 0 <= group.group < len(obj.vertex_groups)
            ])

        for poly in polygons:
            if len(poly.vertices) != 3:
                raise UgxError("UGX LOD export currently supports triangles only.")
            faces.append(tuple(vertex_map[vertex_index] for vertex_index in poly.vertices))
            face_materials.append(ugx_material_index_for_slot(obj, int(poly.material_index), model))
            face_sections.append(section_index)
            for _, layer_name, _ in UV_SEMANTICS:
                uv_data, uv_property = uv_layers[layer_name]
                if uv_data is None:
                    uv_faces[layer_name].append([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)])
                else:
                    uv_faces[layer_name].append([mesh_uv_value(uv_data, uv_property, loop_index) for loop_index in poly.loop_indices])
            if has_colors:
                color_faces.append([object_colors[obj].get(loop, (1., 1., 1., 1.)) for loop in poly.loop_indices])

        section_entries.append(build_section_entry(section, section_index, vertex_start, len(ordered_vertices), face_start, len(polygons)))

    mesh = bpy.data.meshes.new(objects[0].name + "_export_proxy_mesh")
    mesh.from_pydata(positions, [], faces)
    mesh.update()
    proxy = bpy.data.objects.new(objects[0].name + "_export_proxy", mesh)
    bpy.context.scene.collection.objects.link(proxy)
    source_to_proxy_material = {}
    for source_index in sorted(materials_by_source_index):
        source_to_proxy_material[source_index] = len(proxy.data.materials)
        proxy.data.materials.append(materials_by_source_index[source_index])
    proxy["hwek_ugx_material_slot_indices"] = json.dumps(sorted(materials_by_source_index))
    for poly, source_material_index in zip(mesh.polygons, face_materials):
        if proxy.data.materials:
            poly.material_index = source_to_proxy_material.get(source_material_index, 0)

    if face_sections and hasattr(mesh, "attributes"):
        section_attribute = mesh.attributes.new("UGX_Section", "INT", "FACE")
        for index, section_index in enumerate(face_sections):
            section_attribute.data[index].value = section_index
    if source_vertices and hasattr(mesh, "attributes"):
        source_attribute = mesh.attributes.new("UGX_SourceVertex", "INT", "POINT")
        for index, source_index in enumerate(source_vertices):
            source_attribute.data[index].value = source_index
    if normal_values and hasattr(mesh, "attributes"):
        for poly in mesh.polygons:
            poly.use_smooth = True
        normal_attribute = mesh.attributes.new("UGX_Normal", "FLOAT_VECTOR", "POINT")
        for index, value in enumerate(normal_values):
            normal_attribute.data[index].vector = value
        try:
            mesh.normals_split_custom_set_from_vertices(normal_values)
            mesh.update()
        except Exception:
            pass
    if tangent_values and hasattr(mesh, "attributes"):
        tangent_attribute = mesh.attributes.new("UGX_Tangent", "FLOAT_VECTOR", "POINT")
        for index, value in enumerate(tangent_values):
            tangent_attribute.data[index].vector = value

    for _, layer_name, _ in UV_SEMANTICS:
        if not any(item.data.uv_layers.get(layer_name) for item in objects if item.data.uv_layers):
            continue
        layer = mesh.uv_layers.new(name=layer_name)
        uv_attribute = mesh.attributes.get(layer.name) if hasattr(mesh, "attributes") else None
        for poly, face_uvs in zip(mesh.polygons, uv_faces[layer_name]):
            for loop_index, uv in zip(poly.loop_indices, face_uvs):
                if uv_attribute is not None:
                    uv_attribute.data[loop_index].vector = uv
                else:
                    layer.data[loop_index].uv = uv

    if has_colors:
        attribute = mesh.color_attributes.new(name='Color', type='FLOAT_COLOR', domain='CORNER')
        attribute.data.foreach_set('color', [component for face in color_faces for value in face for component in value])

    armature = ugx_armature_for(objects[0])
    proxy_group_names: List[str] = []
    seen_group_names = set()
    if armature is not None:
        for bone in ugx_ordered_armature_bones(armature):
            if bone.name not in seen_group_names:
                proxy_group_names.append(bone.name)
                seen_group_names.add(bone.name)
    for source_obj in objects:
        for group in source_obj.vertex_groups:
            if group.name not in seen_group_names:
                proxy_group_names.append(group.name)
                seen_group_names.add(group.name)
    for group_name in proxy_group_names:
        proxy.vertex_groups.new(name=group_name)
    for vertex_index, groups in enumerate(vertex_weights):
        for group_name, weight in groups:
            group = proxy.vertex_groups.get(group_name)
            if group is not None:
                group.add([vertex_index], weight, "ADD")

    if armature:
        modifier = proxy.modifiers.new("UGX Armature", "ARMATURE")
        modifier.object = armature

    apply_common_import_metadata(proxy, model, str(objects[0].get("hwek_ugx_import_id", "")), len(objects))
    proxy["hwek_ugx_sections"] = json.dumps(section_entries)
    set_section_metadata_properties(proxy, section_entries)
    for section_index, owner in section_owner.items():
        entry = next((item for item in section_entries if section_entry_source_index(section_index, item) == section_index), {})
        for key in editable_section_metadata_keys(entry):
            prop_key = section_property_key(section_index, key)
            if prop_key in owner:
                proxy[prop_key] = owner[prop_key]
    return proxy


def hw2_structure_changed(model: UgxModel, objects: List[bpy.types.Object]) -> bool:
    owners: Dict[int, int] = {}
    for obj in objects:
        for section_index in lod_section_indices(obj):
            owners[section_index] = owners.get(section_index, 0) + 1
        mesh_index = int(obj.get("hwek_ugx_mesh_index", -1))
        source_name = granny_mesh_name_for_accessory(model, mesh_index, "")
        if mesh_index >= 0 and current_ugx_mesh_name(obj) != source_name:
            return True
    return any(owners.get(index, 0) != 1 for index in range(len(model.sections)))


def build_hw2_structural_export_proxy(
    model: UgxModel,
    objects: List[bpy.types.Object],
) -> Tuple[bpy.types.Object, UgxModel, List[bpy.types.Object], List[Tuple[bpy.types.Object, bpy.types.Object]]]:
    structural = copy.deepcopy(model)
    structural.sections = []
    structural.accessories = []
    mesh_names = []
    mesh_indices = []
    temp_objects = []
    object_pairs = []
    ordered = sorted(objects, key=lambda item: (
        int(item.get("hwek_ugx_lod_index", 0)),
        int(item.get("hwek_ugx_mesh_index", -1)),
        item.name,
    ))

    try:
        for obj in ordered:
            source_indices = lod_section_indices(obj)
            section_attr = mesh_int_attribute(obj.data, "UGX_Section")
            if section_attr is None:
                raise UgxError(f"{obj.name} has no UGX_Section attribute; duplicate an imported damage object to create a new part.")
            transplant = bool(obj.get("hwek_ugx_transplant_source_path"))
            donor_model = ugx_transplant_model(obj) if transplant else model
            section_map = ugx_transplant_section_map(obj) if transplant else {index: index for index in source_indices}
            material_map = ugx_transplant_material_map(obj) if transplant else {}
            bone_map = ugx_transplant_bone_map(obj) if transplant else list(range(len(model.bones)))
            if transplant and len(bone_map) != len(donor_model.bones):
                raise UgxError(f"{obj.name} has an incomplete donor-to-target bone map.")
            source_ranges = source_vertex_section_ranges(donor_model)
            used_sources = {
                virtual_index
                for virtual_index in source_indices
                if virtual_index in section_map
                and 0 <= section_map[virtual_index] < len(donor_model.sections)
                and any(section_attr[poly.index] == virtual_index for poly in obj.data.polygons)
            }
            if not used_sources:
                continue

            if transplant:
                for donor_index, target_index in sorted(material_map.items(), key=lambda item: item[1]):
                    if donor_index < 0 or donor_index >= len(donor_model.materials):
                        raise UgxError(f"{obj.name} maps an invalid donor material {donor_index}.")
                    if target_index < len(model.materials):
                        raise UgxError(f"{obj.name} maps a donor material over retail material {target_index}.")
                    if target_index > len(structural.materials):
                        raise UgxError(f"{obj.name} leaves a gap in the adopted material table at {target_index}.")
                    source_material = copy.deepcopy(donor_model.materials[donor_index])
                    if target_index == len(structural.materials):
                        structural.materials.append(source_material)
                    elif parsed_ufx_metadata_signature(structural.materials[target_index]) != parsed_ufx_metadata_signature(source_material):
                        raise UgxError(f"Adopted UGX material {target_index} has conflicting donor definitions.")

            accessory_index = len(structural.accessories)
            if transplant:
                donor_accessories = {
                    donor_model.sections[section_map[index]].accessory_index
                    for index in used_sources
                }
                if len(donor_accessories) != 1:
                    raise UgxError(f"{obj.name} combines sections from multiple donor mesh bindings.")
                donor_accessory_index = next(iter(donor_accessories))
                donor_accessory = (
                    donor_model.accessories[donor_accessory_index]
                    if 0 <= donor_accessory_index < len(donor_model.accessories)
                    else Accessory(0, max(1, len(donor_model.bones)))
                )
                source_accessory = target_accessory_for_donor(donor_accessory, bone_map, len(model.bones))
            else:
                source_mesh_index = int(obj.get("hwek_ugx_mesh_index", -1))
                source_accessory = model.accessories[source_mesh_index] if 0 <= source_mesh_index < len(model.accessories) else Accessory(0, max(1, len(model.bones)))
            structural.accessories.append(copy.deepcopy(source_accessory))
            mesh_name = current_ugx_mesh_name(obj)
            mesh_names.append(mesh_name)
            source_model_index = int(obj.get("hwek_ugx_mesh_index", -1))
            source_file_indices = model.granny_info.models[0].mesh_indices if model.granny_info.models else []
            source_file_index = source_file_indices[source_model_index] if 0 <= source_model_index < len(source_file_indices) else -1
            mesh_indices.append(source_file_index)
            if transplant and donor_model.granny_info.parsed:
                donor_index = ugx_transplant_mesh_index(obj, donor_model)
                donor_mesh = granny_mesh_for_model_index(donor_model.granny_info, donor_index)
                adopted_mesh = remap_transplanted_granny_mesh(donor_mesh, donor_model, model, bone_map, mesh_name)
                mesh_indices[-1] = len(structural.granny_info.meshes)
                structural.granny_info.meshes.append(adopted_mesh)
            remap = {}
            range_map: Dict[int, Tuple[int, int]] = {}
            for virtual_index in source_indices:
                if virtual_index not in used_sources:
                    continue
                donor_section_index = section_map[virtual_index]
                section = (
                    remap_transplanted_v6_section(donor_model.sections[donor_section_index], bone_map)
                    if transplant
                    else copy.deepcopy(model.sections[donor_section_index])
                )
                if transplant:
                    if section.material_index not in material_map:
                        raise UgxError(f"{obj.name} section material {section.material_index} was not adopted.")
                    section.material_index = material_map[section.material_index]
                section.accessory_index = accessory_index
                section.accessory_first_bone = source_accessory.first_bone
                section.accessory_bone_count = source_accessory.bone_count
                remap[virtual_index] = len(structural.sections)
                range_map[virtual_index] = source_ranges[donor_section_index]
                structural.sections.append(section)

            temp = obj.copy()
            temp.data = obj.data.copy()
            bpy.context.scene.collection.objects.link(temp)
            temp.name = f"{obj.name}_ugx_structural_temp"
            temp.hide_set(True)
            temp['hwek_ugx_export_enabled'] = False
            temp["hwek_ugx_lod_section_indices"] = json.dumps(list(remap.values()))
            temp["hwek_ugx_mesh_index"] = accessory_index
            temp["hwek_ugx_mesh_name"] = mesh_names[-1]
            temp_objects.append(temp)
            object_pairs.append((obj, temp))
            temp_section_attr = temp.data.attributes.get("UGX_Section")
            for item in temp_section_attr.data:
                item.value = remap.get(int(item.value), -1)

            new_ranges = source_vertex_section_ranges(structural)
            source_vertex_attr = temp.data.attributes.get("UGX_SourceVertex")
            if source_vertex_attr:
                for item in source_vertex_attr.data:
                    old_value = int(item.value)
                    for source_index, new_index in remap.items():
                        old_start, old_end = range_map[source_index]
                        if old_start <= old_value < old_end:
                            new_start, _ = new_ranges[new_index]
                            item.value = new_start + (old_value - old_start)
                            break

        if not structural.sections:
            raise UgxError("Removing every UGX mesh is not supported.")
        if structural.granny_info.models:
            structural.granny_info.models[0].mesh_names = mesh_names
            structural.granny_info.models[0].mesh_indices = mesh_indices
        else:
            structural.granny_info.models = [GrannyModelInfo(os.path.splitext(os.path.basename(model.path))[0], mesh_names, mesh_indices=mesh_indices)]
        extend_v6_material_model_for_objects(structural, temp_objects)
        proxy = build_lod_export_proxy(structural, temp_objects)
        return proxy, structural, temp_objects, object_pairs
    except BaseException:
        remove_temp_objects(temp_objects)
        raise


def remove_export_proxy(proxy: Optional[bpy.types.Object]):
    if proxy is None:
        return
    mesh = proxy.data
    bpy.data.objects.remove(proxy, do_unlink=True)
    if mesh and mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def remove_temp_objects(objects: Iterable[bpy.types.Object]):
    for obj in list(objects):
        if obj and obj.name in bpy.data.objects:
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh and mesh.users == 0:
                bpy.data.meshes.remove(mesh)
