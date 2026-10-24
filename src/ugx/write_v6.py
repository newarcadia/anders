"""Build HW2 model chunks and cached section records."""

from typing import Any, Dict, List, Optional, Tuple
import bpy
import math
import struct
from ..errors import UgxError
from ..formats.binary import PACKED_NULL_OFFSET_64, align4, align_data, packed_array_header
from ..formats.granny import safe_grx_name
from ..formats.matrix import uax_matrix4_identity
from ..materials.hw2 import read_hw2_vertex_color_inputs
from ..materials.sync import sync_ugx_export_material_nodes
from .bones import new_v6_export_bones
from .coordinates import blender_to_game_uv_for_version, reverse_triangle_winding
from .data import CACHED_SIGNATURE_HW2, UGX_NO_RIGID_BONE, V6_MAX_SECTION_VERTICES, v6_instance_settings
from .mesh import (
    mesh_calculated_basis_by_loop_world,
    collect_vertex_color_materials,
    mesh_colors_by_loop,
    mesh_uv_data,
    mesh_uv_value,
    object_normal_game_vector,
    object_vertex_game_position,
)
from .naming import current_ugx_mesh_name
from .vertices import encode_vertex_color
from .scene import ugx_armature_for, ugx_authoring_mesh_objects
from .scene_validation import synchronize_ugx_export_meshes, validate_ugx_export_meshes
from .skin import mesh_has_v4_skin, new_v6_rigid_fallback_bone, new_v6_vertex_skin
from .vertex_authoring import encoded_new_v6_skinned_vertex, encoded_new_v6_vertex, tangent_for_game_normal

__all__ = ('new_v6_corner_normal', 'new_v6_accessory_name', 'new_v6_bone_bounds', 'build_new_v6_authoring_chunks', 'build_new_v6_cached_data')

def new_v6_corner_normal(obj: bpy.types.Object, loop_index: int, vertex_index: int, polygon_index: int) -> Tuple[float, float, float]:
    mesh = obj.data
    try:
        value = tuple(float(component) for component in mesh.corner_normals[loop_index].vector)
    except Exception:
        value = tuple(float(component) for component in mesh.polygons[polygon_index].normal)
    return object_normal_game_vector(obj, value)


def new_v6_accessory_name(obj: bpy.types.Object) -> str:
    return safe_grx_name(current_ugx_mesh_name(obj), "")


def new_v6_bone_bounds(
    objects: List[bpy.types.Object],
    bones: List[Dict[str, Any]],
    bone_name_to_index: Dict[str, int],
    skinned: bool,
    rigid_fallbacks: Dict[bpy.types.Object, Optional[int]],
) -> List[Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]]:
    """Reproduce BUGXGeomInstancer::findBoneBounds for fresh v6 data."""
    lows: List[Optional[List[float]]] = [None] * len(bones)
    highs: List[Optional[List[float]]] = [None] * len(bones)

    def expand(bone_index: int, position: Tuple[float, float, float]):
        if not (0 <= bone_index < len(bones)):
            return
        if lows[bone_index] is None:
            lows[bone_index] = [float(value) for value in position]
            highs[bone_index] = [float(value) for value in position]
            return
        for axis in range(3):
            lows[bone_index][axis] = min(lows[bone_index][axis], float(position[axis]))
            highs[bone_index][axis] = max(highs[bone_index][axis], float(position[axis]))

    for obj in objects:
        mesh = obj.data
        mesh.calc_loop_triangles()
        rendered_vertices = sorted({int(vertex_index) for tri in mesh.loop_triangles for vertex_index in tri.vertices})
        for vertex_index in rendered_vertices:
            position = object_vertex_game_position(obj, vertex_index)
            if not skinned:
                expand(0, position)
                continue
            raw_indices, raw_weights, _ = new_v6_vertex_skin(
                obj,
                vertex_index,
                bone_name_to_index,
                rigid_fallbacks.get(obj),
            )
            for encoded_index, encoded_weight in zip(raw_indices, raw_weights):
                if (float(encoded_weight) / 255.0) < (1.0 / 64.0):
                    continue
                expand(int(encoded_index), position)

    return [
        (tuple(low), tuple(highs[index])) if low is not None and highs[index] is not None else None
        for index, low in enumerate(lows)
    ]


def build_new_v6_authoring_chunks(
    context_obj: bpy.types.Object,
) -> Tuple[bytes, bytes, bytes, List[Optional[bpy.types.Material]], bool, List[str], Dict[str, Any]]:
    synchronize_ugx_export_meshes(context_obj)
    objects = ugx_authoring_mesh_objects(context_obj)
    if not objects:
        raise UgxError("New HW2 UGX export needs a mesh, or an armature with at least one bound mesh.")
    validate_ugx_export_meshes(objects, cache_version=6)
    sync_ugx_export_material_nodes(objects, 6)
    bpy.context.view_layer.update()

    armatures = {ugx_armature_for(item) for item in objects if ugx_armature_for(item) is not None}
    if len(armatures) > 1:
        raise UgxError("All meshes in a new HW2 UGX export must use the same armature.")
    bones, bone_name_to_index, granny_metadata = new_v6_export_bones(objects[0])
    # HW2 reserves cached bone zero for the model's Granny root, independently
    # of animated bones such as bone_root and of unused vertex-group names.
    rigid_fallbacks = {item: new_v6_rigid_fallback_bone(item, bone_name_to_index, default_bone_index=0)
                       for item in objects}
    skinned_objects = {
        item: mesh_has_v4_skin(item, bone_name_to_index) or rigid_fallbacks[item] is not None
        for item in objects
    }
    skinned = bool(armatures) or any(mesh_has_v4_skin(item, bone_name_to_index) for item in objects)
    if armatures and not skinned:
        raise UgxError("Skinned HW2 UGX export needs vertex groups named after armature bones with non-zero weights.")
    missing_skin = [item.name for item in objects if skinned and not skinned_objects[item]]
    if missing_skin:
        names = ", ".join(missing_skin[:8])
        suffix = "..." if len(missing_skin) > 8 else ""
        raise UgxError(f"Every mesh in a skinned HW2 export needs armature bone weights. Missing: {names}{suffix}")

    colored_materials = collect_vertex_color_materials(objects)
    for item in objects:
        for slot in {polygon.material_index for polygon in item.data.polygons}:
            material = item.material_slots[slot].material if slot < len(item.material_slots) else None
            if read_hw2_vertex_color_inputs(material, skinned)[0]:
                colored_materials.add(material)
    color_prefixes = {material: read_hw2_vertex_color_inputs(material, skinned, True)[1] for material in colored_materials}

    materials: List[Optional[bpy.types.Material]] = []
    material_indices: Dict[Tuple[str, int], int] = {}

    def global_material_index(material: Optional[bpy.types.Material]) -> int:
        key = ("material", int(material.as_pointer())) if material is not None else ("none", 0)
        if key not in material_indices:
            material_indices[key] = len(materials)
            materials.append(material)
        return material_indices[key]

    accessory_names: List[str] = []
    accessory_by_name: Dict[str, int] = {}
    sections: List[Dict[str, Any]] = []
    all_positions: List[Tuple[float, float, float]] = []

    for obj in objects:
        mesh = obj.data
        mesh.calc_loop_triangles()
        if not mesh.loop_triangles:
            continue
        accessory_name = new_v6_accessory_name(obj)
        name_key = accessory_name.lower()
        accessory_index = accessory_by_name.get(name_key)
        if accessory_index is None:
            accessory_index = len(accessory_names)
            accessory_by_name[name_key] = accessory_index
            accessory_names.append(accessory_name)
        uv_layer = mesh.uv_layers.active
        uv_layer_name = str(uv_layer.name) if uv_layer else ""
        basis_by_loop = mesh_calculated_basis_by_loop_world(obj, uv_layer_name) if uv_layer_name else {}
        # calc_tangents() can replace Blender's backing corner-layer storage and
        # invalidate the RNA layer handle, so reacquire both after it completes.
        uv_layer = mesh.uv_layers.get(uv_layer_name) if uv_layer_name else None
        uv_data, uv_property = mesh_uv_data(mesh, uv_layer)
        colors = mesh_colors_by_loop(mesh)
        local_sections: Dict[int, Dict[str, Any]] = {}
        for tri in mesh.loop_triangles:
            local_material_index = max(0, int(mesh.polygons[tri.polygon_index].material_index))
            material = obj.material_slots[local_material_index].material if local_material_index < len(obj.material_slots) else None
            material_index = global_material_index(material)
            section = local_sections.get(material_index)
            # The native multiplier is a uint16 power of two. A triangle-
            # aligned 32,766-vertex ceiling keeps the largest value at 32,768.
            if section is None or section["vertex_count"] + 3 > V6_MAX_SECTION_VERTICES:
                section = {
                    "material_index": material_index,
                    "colored": material in colored_materials,
                    "color_prefix": color_prefixes.get(material, False),
                    "accessory_index": accessory_index,
                    "vertices": bytearray(),
                    "indices": [],
                    "vertex_count": 0,
                    "max_influences": 1,
                    "used_bones": set(),
                    "lod_min_distance": float(obj.get("hwek_ugx_lod_min_distance", 0.0)),
                    "lod_max_distance": float(obj.get("hwek_ugx_lod_max_distance", 3.4028234663852886e38)),
                    "sorting_bias": float(obj.get("hwek_ugx_sorting_bias", 0.0)),
                }
                sections.append(section)
                local_sections[material_index] = section
            blender_corners = list(zip(tri.loops, tri.vertices))
            for loop_index, vertex_index in reverse_triangle_winding(blender_corners):
                position = object_vertex_game_position(obj, vertex_index)
                all_positions.append(position)
                uv = blender_to_game_uv_for_version(mesh_uv_value(uv_data, uv_property, loop_index), 6) if uv_layer else (0.0, 0.0)
                basis = basis_by_loop.get(loop_index)
                if basis is None:
                    game_normal = new_v6_corner_normal(obj, loop_index, vertex_index, tri.polygon_index)
                    game_tangent = tangent_for_game_normal(game_normal)
                    bitangent_sign = 1.0
                else:
                    game_normal, game_tangent, bitangent_sign = basis
                if skinned:
                    raw_indices, raw_weights, influence_count = new_v6_vertex_skin(
                        obj,
                        vertex_index,
                        bone_name_to_index,
                        rigid_fallbacks[obj],
                    )
                    section["max_influences"] = max(section["max_influences"], influence_count)
                    for encoded_index, encoded_weight in zip(raw_indices, raw_weights):
                        if encoded_weight > 0:
                            section["used_bones"].add(int(encoded_index))
                    encoded = bytearray(encoded_new_v6_skinned_vertex(
                        position,
                        game_normal,
                        uv,
                        raw_indices,
                        raw_weights,
                        game_tangent,
                        bitangent_sign,
                    ))
                else:
                    section["used_bones"].add(0)
                    encoded = bytearray(encoded_new_v6_vertex(
                        position,
                        game_normal,
                        uv,
                        game_tangent,
                        bitangent_sign,
                    ))
                if section['colored']:
                    color = encode_vertex_color(colors.get(loop_index, (1., 1., 1., 1.)), 9)
                    insertion = 20 if section['color_prefix'] or not skinned else len(encoded)
                    encoded[insertion:insertion] = color
                section['vertices'].extend(encoded)
                section["indices"].append(section["vertex_count"])
                section["vertex_count"] += 1

    if not all_positions:
        raise UgxError("New HW2 UGX export produced no vertices.")
    if not materials:
        materials.append(None)

    vertex_payload = bytearray()
    section_infos: List[Dict[str, Any]] = []
    for section in sections:
        vertex_offset = len(vertex_payload)
        vertex_payload.extend(section["vertices"])
        section_infos.append({
            "material_index": section["material_index"],
            "colored": section['colored'],
            "accessory_index": section["accessory_index"],
            "vertex_offset": vertex_offset,
            "vertex_size": len(section["vertices"]),
            "vertex_count": section["vertex_count"],
            "tri_count": len(section["indices"]) // 3,
            # Retail v6 assets advertise one here even when vertices carry
            # multiple normalized weights; Hogan selects skinning separately.
            "max_bones": 1,
            "used_bones": sorted(section["used_bones"]),
            "lod_min_distance": section["lod_min_distance"],
            "lod_max_distance": section["lod_max_distance"],
            "sorting_bias": section["sorting_bias"],
        })

    max_instances, instance_index_multiplier = v6_instance_settings(section_infos)
    index_payload = bytearray()
    for section, info in zip(sections, section_infos):
        info["index_start"] = len(index_payload) // 2
        for instance_index in range(max_instances):
            instance_offset = instance_index_multiplier * instance_index
            for index in section["indices"]:
                index_payload.extend(struct.pack("<H", int(index) + instance_offset))

    accessory_sections = [
        [index for index, info in enumerate(section_infos) if info["accessory_index"] == accessory_index]
        for accessory_index in range(len(accessory_names))
    ]
    bone_bounds = new_v6_bone_bounds(objects, bones, bone_name_to_index, skinned, rigid_fallbacks)
    cached_payload = build_new_v6_cached_data(
        section_infos,
        all_positions,
        bones,
        skinned,
        accessory_sections=accessory_sections,
        bone_bounds=bone_bounds,
    )
    return cached_payload, bytes(vertex_payload), bytes(index_payload), materials, skinned, accessory_names, granny_metadata


def build_new_v6_cached_data(
    section_infos: List[Dict],
    positions: List[Tuple[float, float, float]],
    bones: Optional[List[Dict[str, Any]]] = None,
    skinned: bool = False,
    accessory_sections: Optional[List[List[int]]] = None,
    bone_bounds: Optional[List[Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]]] = None,
) -> bytes:
    bones = bones or [{
        "name": "GrannyRootBone",
        "parent": -1,
        "matrix": uax_matrix4_identity(),
    }]
    accessory_sections = accessory_sections or [list(range(len(section_infos)))]
    if not accessory_sections or any(not indices for indices in accessory_sections):
        raise UgxError("Every HW2 UGX accessory needs at least one section.")
    covered_sections = sorted(index for indices in accessory_sections for index in indices)
    if covered_sections != list(range(len(section_infos))):
        raise UgxError("HW2 UGX accessory section lists must cover every section exactly once.")
    mins = tuple(min(position[axis] for position in positions) for axis in range(3))
    maxs = tuple(max(position[axis] for position in positions) for axis in range(3))
    center = tuple((mins[axis] + maxs[axis]) * 0.5 for axis in range(3))
    radius = math.sqrt(sum((maxs[axis] - center[axis]) ** 2 for axis in range(3)))
    max_instances, instance_index_multiplier = v6_instance_settings(section_infos)

    # UGX.bt reads the discriminator before it switches the cached payload to
    # little-endian.  HW2 therefore stores this one value as bytes 06 00 34 C2,
    # followed by ordinary little-endian v6 fields.
    data = bytearray(struct.pack(">I", CACHED_SIGNATURE_HW2))
    data.extend(struct.pack(
        "<I10f",
        0,
        center[0], center[1], center[2],
        radius,
        mins[0], mins[1], mins[2],
        maxs[0], maxs[1], maxs[2],
    ))
    data.extend(struct.pack(
        "<hHhBBBBhI",
        max_instances,
        instance_index_multiplier,
        0x7FFF,
        0 if skinned else 1,
        1,
        1 if skinned else 0,
        0 if skinned else 1,
        0,
        0,
    ))
    headers_offset = len(data)
    data.extend(b"\0" * (16 * 6))
    bone_name_offsets = []
    for index, bone in enumerate(bones):
        align_data(data, 2)
        bone_name_offsets.append(len(data))
        data.extend(safe_grx_name(str(bone.get("name", "")), "").encode("utf-8") + b"\0")
    align_data(data, 8)

    sections_offset = len(data)
    vertex_stride = 28 if skinned else 20
    for info in section_infos:
        record = bytearray()
        record.extend(struct.pack(
            "<iiiiiiiiii",
            info["material_index"],
            int(info.get("accessory_index", 0)),
            max(1, int(info.get("max_bones", 1))),
            UGX_NO_RIGID_BONE if skinned else 0,
            info["index_start"],
            info["tri_count"],
            info["vertex_offset"],
            info["vertex_size"],
            vertex_stride + (4 if info.get('colored') else 0),
            info["vertex_count"],
        ))
        record.extend(struct.pack(
            "<Bxxx3f",
            0 if skinned else 1,
            float(info.get("lod_min_distance", 0.0)),
            float(info.get("lod_max_distance", 3.4028234663852886e38)),
            float(info.get("sorting_bias", 0.0)),
        ))
        record.extend(packed_array_header(0, PACKED_NULL_OFFSET_64))
        if len(record) != 72:
            raise UgxError("Internal v6 section record size mismatch.")
        data.extend(record)
    align_data(data, 8)

    bones_offset = len(data)
    for index, bone in enumerate(bones):
        rows = bone.get("matrix", uax_matrix4_identity())
        flat = [float(rows[row][column]) for row in range(4) for column in range(4)]
        data.extend(struct.pack("<Q16fiI", bone_name_offsets[index], *flat, int(bone.get("parent", -1)), 0))
    align_data(data, 4)

    accessory_section_offsets = []
    for indices in accessory_sections:
        accessory_section_offsets.append(len(data))
        for index in indices:
            data.extend(struct.pack("<i", index))
    valid_accessories_offset = len(data)
    for index in range(len(accessory_sections)):
        data.extend(struct.pack("<i", index))
    align_data(data, 8)

    accessories_offset = len(data)
    for index, indices in enumerate(accessory_sections):
        used_bones = sorted({
            int(bone_index)
            for section_index in indices
            for bone_index in section_infos[section_index].get(
                "used_bones",
                range(len(bones)) if skinned else (0,),
            )
            if 0 <= int(bone_index) < len(bones)
        })
        first_bone = used_bones[0] if used_bones else 0
        bone_count = used_bones[-1] - first_bone + 1 if used_bones else 0
        data.extend(struct.pack("<ii", first_bone, bone_count))
        data.extend(packed_array_header(len(indices), accessory_section_offsets[index]))
    align4(data)

    bone_bounds_low_offset = len(data)
    if bone_bounds is None:
        resolved_bone_bounds: List[Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]] = [None] * len(bones)
        if skinned:
            resolved_bone_bounds = [(mins, maxs) for _ in bones]
        elif bones:
            resolved_bone_bounds[0] = (mins, maxs)
    else:
        resolved_bone_bounds = list(bone_bounds)
    if len(resolved_bone_bounds) < len(bones):
        resolved_bone_bounds.extend([None] * (len(bones) - len(resolved_bone_bounds)))
    for bounds in resolved_bone_bounds[:len(bones)]:
        low = bounds[0] if bounds is not None else (1.0e37, 1.0e37, 1.0e37)
        data.extend(struct.pack("<3f", *low))
    bone_bounds_high_offset = len(data)
    for bounds in resolved_bone_bounds[:len(bones)]:
        high = bounds[1] if bounds is not None else (-1.0e37, -1.0e37, -1.0e37)
        data.extend(struct.pack("<3f", *high))

    headers = b"".join((
        packed_array_header(len(section_infos), sections_offset),
        packed_array_header(len(bones), bones_offset),
        packed_array_header(len(accessory_sections), accessories_offset),
        packed_array_header(len(accessory_sections), valid_accessories_offset),
        packed_array_header(len(bones), bone_bounds_low_offset),
        packed_array_header(len(bones), bone_bounds_high_offset),
    ))
    data[headers_offset:headers_offset + len(headers)] = headers
    return bytes(data)
