"""Create Blender armatures and meshes from native model records."""

from typing import Any, Dict, List, Optional, Tuple
import bpy
import hashlib
import json
import os
import numpy as np
from ..errors import UgxError
from ..materials.assignment import make_material
from ..materials.player_color import initialize_ugx_player_color
from .bone_data import bone_display_matrices
from .coordinates import game_to_blender_uv_for_version, game_to_blender_vec3, reverse_triangle_winding
from .data import UgxModel, source_vertex_section_ranges, ugx_game_name
from .model_placement import restore_ugx_model_placement
from .metadata import (
    apply_common_import_metadata,
    build_section_entry,
    set_bone_bound_properties,
    set_section_metadata_properties,
)
from .naming import (
    granny_mesh_name_for_accessory,
    imported_mesh_object_name,
    section_mesh_indices,
    section_lod_key,
)
from .read import parse_ugx
from .scene import ugx_ordered_armature_bones
from .skin import vertex_group_from_section_bone, vertex_group_from_source_bone
from .snapshots import mesh_buffer, remember_imported_mesh
from .transforms import bone_rest_display_rows, game_rows_to_blender_matrix
from .vertices import layout_for, tangent_layout_for

__all__ = ('create_armature', 'create_imported_mesh_object', 'create_empty_imported_mesh_object', 'import_ugx')

def create_armature(model: UgxModel, collection) -> Optional[bpy.types.Object]:
    if not model.bones:
        return None
    armature = bpy.data.armatures.new(os.path.basename(model.path) + "_armature")
    arm_obj = bpy.data.objects.new(armature.name, armature)
    collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = []
    created_bone_names: List[str] = []
    skeleton_mode, display_matrices = bone_display_matrices(model)
    for index, bone in enumerate(model.bones):
        edit_bone = armature.edit_bones.new(bone.name or f"bone_{index}")
        # UGX stores bone transforms and parents, but no Blender tail or
        # connection flags. A child position must not change the rest axes.
        edit_bone.tail = (0.0, 0.0, 0.25)
        edit_bone.use_connect = False
        edit_bones.append(edit_bone)
        created_bone_names.append(edit_bone.name)
    for index, bone in enumerate(model.bones):
        if 0 <= bone.parent < len(edit_bones):
            edit_bones[index].parent = edit_bones[bone.parent]
    for index, edit_bone in enumerate(edit_bones):
        edit_bone.matrix = game_rows_to_blender_matrix(display_matrices[index])
        edit_bone.length = 0.25
    bpy.ops.object.mode_set(mode="OBJECT")
    arm_obj["hwek_ugx_skeleton_transform_mode"] = skeleton_mode
    arm_obj["hwek_ugx_bone_display_mode"] = "source_frame"
    if model.granny_info.parsed and model.granny_info.models:
        arm_obj["hwek_ugx_granny_model_name"] = model.granny_info.models[0].name
        restore_ugx_model_placement(arm_obj, model.original_bytes)
    if model.granny_info.parsed:
        # Keep Granny's skeleton policy on the armature as well as in the UGX
        # source snapshot. New meshes bound to an imported armature can then
        # preserve animation-compatible bone order and LOD data.
        arm_obj["hwek_ugx_granny_skeleton_name"] = model.granny_info.skeleton_name
        arm_obj["hwek_ugx_granny_lod_type"] = int(model.granny_info.skeleton_lod_type)
        arm_obj["hwek_ugx_granny_from_file_name"] = model.granny_info.from_file_name
    for index, bone in enumerate(model.bones):
        if index >= len(created_bone_names):
            break
        data_bone = armature.bones.get(created_bone_names[index])
        if data_bone is None:
            continue
        data_bone["hwek_ugx_bone_index"] = index
        data_bone["hwek_ugx_source_name"] = bone.name
        data_bone["hwek_ugx_blender_name"] = data_bone.name
        data_bone["hwek_ugx_export_name"] = bone.name
        data_bone["hwek_ugx_matrix_offset"] = bone.matrix_offset
        data_bone["hwek_ugx_source_matrix"] = json.dumps(bone.matrix)
        data_bone["hwek_ugx_export_matrix"] = json.dumps(bone.matrix)
        data_bone["hwek_ugx_skeleton_transform_mode"] = skeleton_mode
        data_bone["hwek_ugx_source_display_matrix"] = json.dumps(display_matrices[index])
        data_bone["hwek_ugx_import_display_matrix"] = json.dumps(bone_rest_display_rows(arm_obj, data_bone))
        data_bone["hwek_ugx_source_head_local"] = [float(data_bone.head_local.x), float(data_bone.head_local.y), float(data_bone.head_local.z)]
        if index < len(model.granny_info.bones):
            granny_bone = model.granny_info.bones[index]
            if granny_bone.name == bone.name and granny_bone.parent == bone.parent:
                data_bone["hwek_ugx_granny_lod_error"] = float(granny_bone.lod_error)
                transform = granny_bone.transform
                data_bone['hwek_ugx_granny_local_transform'] = json.dumps(
                    [transform.flags, transform.position, transform.orientation, transform.scale_shear])
        set_bone_bound_properties(data_bone, bone)
    arm_obj.select_set(False)
    return arm_obj


def create_imported_mesh_object(
    model: UgxModel,
    collection: bpy.types.Collection,
    name: str,
    section_indices: List[int],
    materials: List[bpy.types.Material],
    arm_obj: Optional[bpy.types.Object],
    import_id: str,
    lod_index: int,
    lod_key: Tuple[float, float],
    total_lods: int,
    mesh_index: int,
    mesh_name: str,
    total_objects: int,
    has_tangents: bool,
    has_uv1: bool,
    has_uv2: bool,
) -> bpy.types.Object:
    positions: List[Tuple[float, float, float]] = []
    normals: List[Tuple[float, float, float]] = []
    tangents: List[Tuple[float, float, float]] = []
    bitangent_signs: List[float] = []
    colors = []
    has_colors = any(layout_for(model.sections[index].layout, 'D') is not None for index in section_indices)
    uvs: List[Tuple[float, float]] = []
    uv1s: List[Tuple[float, float]] = []
    uv2s: List[Tuple[float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    face_materials: List[int] = []
    face_sections: List[int] = []
    source_vertices: List[int] = []
    section_vertex_offsets: Dict[int, int] = {}
    section_face_offsets: Dict[int, int] = {}
    section_face_counts: Dict[int, int] = {}
    section_entries: List[Dict[str, Any]] = []
    section_ranges = source_vertex_section_ranges(model)

    for section_index in section_indices:
        section = model.sections[section_index]
        section_start, _ = section_ranges[section_index]
        vertex_start = len(positions)
        face_start = len(faces)
        section_vertex_offsets[section_index] = vertex_start
        section_face_offsets[section_index] = face_start
        for local_index, vertex in enumerate(section.vertices):
            positions.append(game_to_blender_vec3(vertex.pos))
            normals.append(game_to_blender_vec3(vertex.normal))
            if has_colors:
                colors.append(vertex.color)
            if has_tangents:
                tangents.append(game_to_blender_vec3(vertex.tangent))
                bitangent_signs.append(float(vertex.bitangent_sign))
            uvs.append(game_to_blender_uv_for_version(vertex.uv0, model.cache_version))
            if has_uv1:
                uv1s.append(game_to_blender_uv_for_version(vertex.uv1, model.cache_version))
            if has_uv2:
                uv2s.append(game_to_blender_uv_for_version(vertex.uv2, model.cache_version))
            source_vertices.append(section_start + local_index)
        base = vertex_start
        for i in range(0, len(section.indices) - 2, 3):
            a, b, c = section.indices[i], section.indices[i + 1], section.indices[i + 2]
            if a != b and b != c and c != a:
                faces.append(reverse_triangle_winding((base + a, base + b, base + c)))
                face_materials.append(max(0, section.material_index))
                face_sections.append(section_index)
        face_count = len(faces) - face_start
        section_face_counts[section_index] = face_count
        section_entries.append(build_section_entry(section, section_index, vertex_start, len(section.vertices), face_start, face_count))

    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(positions, [], faces)
    mesh.update()
    if colors:
        color = mesh.color_attributes.new(name='Color', type='FLOAT_COLOR', domain='POINT')
        color.data.foreach_set('color', np.asarray(colors, dtype=np.float32).ravel())
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)

    if model.cache_version == 6:
        source_material_indices = []
        for source_index in face_materials:
            source_index = max(0, min(int(source_index), len(materials) - 1)) if materials else 0
            if source_index not in source_material_indices:
                source_material_indices.append(source_index)
    else:
        source_material_indices = list(range(len(materials)))
    source_to_local_material = {}
    for source_index in source_material_indices:
        if 0 <= source_index < len(materials):
            source_to_local_material[source_index] = len(obj.data.materials)
            obj.data.materials.append(materials[source_index])
    obj["hwek_ugx_material_slot_indices"] = json.dumps(source_material_indices)
    if obj.data.materials:
        mesh.polygons.foreach_set('material_index', [source_to_local_material.get(index, 0) for index in face_materials])

    if face_sections and hasattr(mesh, "attributes"):
        try:
            section_attr = mesh.attributes.new("UGX_Section", "INT", "FACE")
            section_attr.data.foreach_set('value', face_sections)
        except Exception:
            obj["hwek_ugx_section_attr_note"] = "Face section attribute API unavailable in this Blender build."

    if positions and hasattr(mesh, "attributes"):
        try:
            source_attr = mesh.attributes.new("UGX_SourceVertex", "INT", "POINT")
            source_attr.data.foreach_set('value', source_vertices)
        except Exception:
            obj["hwek_ugx_source_vertex_attr_note"] = "Point source-vertex attribute API unavailable in this Blender build."

    for layer_name, layer_uvs in (("UV0", uvs), ("UV1", uv1s), ("UV2", uv2s)):
        if layer_uvs and mesh.polygons:
            layer = mesh.uv_layers.new(name=layer_name)
            uv_attribute = mesh.attributes.get(layer.name) if hasattr(mesh, "attributes") else None
            corners = mesh_buffer(mesh.loops, 'vertex_index', dtype=np.int32)
            values = np.asarray(layer_uvs, dtype=np.float32)[corners].ravel()
            if uv_attribute is not None:
                uv_attribute.data.foreach_set('vector', values)
            else:
                layer.data.foreach_set('uv', values)

    if normals and len(normals) == len(positions):
        # UGX carries authored per-vertex normals.  Blender leaves faces created
        # by from_pydata() flat-shaded, which makes the viewport ignore those
        # normals even after installing them as custom split normals.
        mesh.polygons.foreach_set('use_smooth', np.ones(len(mesh.polygons), dtype=np.bool_))
        try:
            normal_attr = mesh.attributes.new("UGX_Normal", "FLOAT_VECTOR", "POINT")
            normal_attr.data.foreach_set('vector', np.asarray(normals, dtype=np.float32).ravel())
        except Exception:
            obj["hwek_ugx_normal_attr_note"] = "Point normal attribute API unavailable in this Blender build."
        try:
            mesh.normals_split_custom_set_from_vertices(normals)
            mesh.update()
        except Exception:
            obj["hwek_ugx_normal_note"] = "Custom normal API unavailable in this Blender build."

    if tangents and len(tangents) == len(positions) and hasattr(mesh, "attributes"):
        try:
            tangent_attr = mesh.attributes.new("UGX_Tangent", "FLOAT_VECTOR", "POINT")
            tangent_attr.data.foreach_set('vector', np.asarray(tangents, dtype=np.float32).ravel())
        except Exception:
            obj["hwek_ugx_tangent_attr_note"] = "Point tangent attribute API unavailable in this Blender build."
        try:
            sign_attr = mesh.attributes.new("UGX_BitangentSign", "FLOAT", "POINT")
            sign_attr.data.foreach_set('value', bitangent_signs)
        except Exception:
            obj["hwek_ugx_bitangent_sign_attr_note"] = "Point bitangent-sign attribute API unavailable in this Blender build."

    apply_common_import_metadata(obj, model, import_id, total_objects)
    obj["hwek_ugx_lod_index"] = lod_index
    obj["hwek_ugx_mesh_index"] = mesh_index
    obj["hwek_ugx_mesh_name"] = mesh_name
    obj["hwek_ugx_mesh_import_name"] = obj.name
    if model.cache_version == 6:
        obj["hwek_ugx_lod_min_distance"] = lod_key[0]
        obj["hwek_ugx_lod_max_distance"] = lod_key[1]
    obj["hwek_ugx_lod_section_indices"] = json.dumps(section_indices)
    obj["hwek_ugx_sections"] = json.dumps(section_entries)
    set_section_metadata_properties(obj, section_entries)

    if arm_obj:
        for bone in ugx_ordered_armature_bones(arm_obj):
            obj.vertex_groups.new(name=bone.name)
        for section_index in section_indices:
            section = model.sections[section_index]
            base = section_vertex_offsets[section_index]
            has_skin_weights = any(any(weight > 0.0 for weight in vertex.bone_weights) for vertex in section.vertices)
            rigid_group = vertex_group_from_section_bone(section, section.rigid_bone_index, len(obj.vertex_groups))
            if rigid_group is not None and (section.rigid_only or not has_skin_weights):
                group = obj.vertex_groups[rigid_group]
                group.add(list(range(base, base + len(section.vertices))), 1.0, "ADD")
                continue
            for vertex_index, vertex in enumerate(section.vertices):
                for local_bone, weight in zip(vertex.bone_indices, vertex.bone_weights):
                    if weight <= 0.0:
                        continue
                    bone_index = vertex_group_from_source_bone(section, local_bone, len(obj.vertex_groups))
                    if bone_index is not None:
                        obj.vertex_groups[bone_index].add([base + vertex_index], weight, "ADD")
        modifier = obj.modifiers.new("UGX Armature", "ARMATURE")
        modifier.object = arm_obj

    remember_imported_mesh(obj)
    return obj


def create_empty_imported_mesh_object(
    model: UgxModel,
    collection: bpy.types.Collection,
    name: str,
    materials: List[bpy.types.Material],
    arm_obj: Optional[bpy.types.Object],
    import_id: str,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name + "_empty_mesh")
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    for material in materials:
        obj.data.materials.append(material)
    obj["hwek_ugx_material_slot_indices"] = json.dumps(list(range(len(materials))))
    apply_common_import_metadata(obj, model, import_id, 1)
    obj["hwek_ugx_empty_import"] = True
    obj["hwek_ugx_lod_index"] = 0
    obj["hwek_ugx_lod_min_distance"] = 0.0
    obj["hwek_ugx_lod_max_distance"] = 0.0
    obj["hwek_ugx_lod_section_indices"] = "[]"
    obj["hwek_ugx_sections"] = "[]"
    if arm_obj:
        modifier = obj.modifiers.new("UGX Armature", "ARMATURE")
        modifier.object = arm_obj
    return obj


def import_ugx(path: str, expected_cache_version: Optional[int] = None) -> bpy.types.Object:
    model = parse_ugx(path)
    if expected_cache_version is not None and model.cache_version != expected_cache_version:
        raise UgxError(f"Expected {ugx_game_name(expected_cache_version)} UGX v{expected_cache_version}, got {ugx_game_name(model.cache_version)} v{model.cache_version}.")
    name = os.path.splitext(os.path.basename(path))[0]
    collection = bpy.data.collections.new(name + "_ugx")
    bpy.context.scene.collection.children.link(collection)

    has_tangents = any(tangent_layout_for(section.layout) is not None for section in model.sections)
    has_uv1 = any(layout_for(section.layout, "T1") is not None for section in model.sections)
    has_uv2 = any(layout_for(section.layout, "T2") is not None for section in model.sections)
    materials = [make_material(material, index, model.path, model.cache_version) for index, material in enumerate(model.materials)]
    arm_obj = create_armature(model, collection)
    lod_groups: Dict[Tuple[float, float], Dict[int, List[int]]] = {}
    mesh_indices = section_mesh_indices(model)
    for section_index, section in enumerate(model.sections):
        key = section_lod_key(section) if model.cache_version == 6 else (0.0, 0.0)
        mesh_index = mesh_indices[section_index]
        lod_groups.setdefault(key, {}).setdefault(mesh_index, []).append(section_index)
    ordered_lods = sorted(lod_groups.items(), key=lambda item: (item[0][0], item[0][1]))
    object_groups = [
        (lod_index, lod_key, mesh_index, section_indices)
        for lod_index, (lod_key, mesh_groups) in enumerate(ordered_lods)
        for mesh_index, section_indices in sorted(mesh_groups.items())
    ]
    import_id = hashlib.sha1(f"{os.path.abspath(model.path)}|{os.urandom(8).hex()}".encode("utf-8")).hexdigest()
    # Shared source identity and the cached source snapshot belong to the
    # skeleton/export set.  Mesh objects still keep their per-section and
    # per-accessory metadata because those values are genuinely mesh-specific,
    # but a newly authored mesh only needs to bind to this armature to join a
    # fresh HW2 export.
    if arm_obj is not None:
        apply_common_import_metadata(arm_obj, model, import_id, max(1, len(object_groups)))
        arm_obj["hwek_ugx_metadata_owner"] = "ARMATURE"
    objects = [
        create_imported_mesh_object(
            model,
            collection,
            imported_mesh_object_name(
                name,
                granny_mesh_name_for_accessory(model, mesh_index, name),
                mesh_index,
                lod_index,
                lod_key,
                len(ordered_lods),
            ),
            section_indices,
            materials,
            arm_obj,
            import_id,
            lod_index,
            lod_key,
            len(ordered_lods),
            mesh_index,
            granny_mesh_name_for_accessory(model, mesh_index, name),
            len(object_groups),
            has_tangents,
            has_uv1,
            has_uv2,
        )
        for lod_index, lod_key, mesh_index, section_indices in object_groups
    ]
    if not objects:
        objects.append(create_empty_imported_mesh_object(model, collection, name, materials, arm_obj, import_id))

    initialize_ugx_player_color(arm_obj or objects[0])

    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    if objects:
        bpy.context.view_layer.objects.active = objects[0]
    return objects[0]
