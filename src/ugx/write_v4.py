"""Build HWDE model chunks and cached section records."""

from typing import Any, Dict, List, Optional, Tuple
import bpy
import math
import struct
from ..errors import UgxError
from ..formats.binary import PACKED_NULL_OFFSET_64, align4, align_data, packed_array_header
from ..formats.granny import safe_grx_name
from ..formats.matrix import uax_matrix4_identity
from ..materials.sync import sync_ugx_export_material_nodes
from ..materials.bindings import ugx_material_node_bindings
from .bones import new_v4_export_bones, normalize_unused_v4_damage_root
from .coordinates import blender_to_game_uv, reverse_triangle_winding
from .data import (
    CACHED_SIGNATURE_HW1_PC_BYTES,
    UGX_MAX_SECTION_TRIANGLES,
    UGX_NO_RIGID_BONE,
    V4_MAX_SECTION_VERTICES,
    V4_MAX_SKELETON_BONES,
    v4_instance_settings,
)
from .mesh import (
    mesh_calculated_basis_by_loop_world,
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
from .skin import mesh_has_v4_skin, new_v4_uniform_rigid_bone, new_v4_vertex_skin, new_v6_rigid_fallback_bone
from .transforms import ugx_export_axis_correction_rows
from .vertex_authoring import encoded_new_v4_skinned_vertex, encoded_new_v4_vertex, tangent_for_game_normal

__all__ = ('unique_v4_accessory_names', 'new_v4_bone_bounds', 'build_new_v4_authoring_chunks', 'build_new_v4_cached_data')

def unique_v4_accessory_names(objects: List[bpy.types.Object]) -> List[str]:
    """Give every authored object its own stable Granny/accessory mesh name.

    Duplicating an imported mesh in Blender also duplicates its UGX metadata,
    including hwek_ugx_mesh_name.  Treating that stale name as an identity
    collapses all of the duplicates into one accessory and drops their model
    bindings.  Object names are Blender-unique, so use them to disambiguate
    copied authoring meshes while preserving the original name where possible.
    """
    result: List[str] = []
    used: set = set()
    for index, obj in enumerate(objects):
        preferred = safe_grx_name(current_ugx_mesh_name(obj), "")
        candidate = preferred
        if candidate.lower() in used:
            candidate = safe_grx_name(obj.name, f"mesh_{index}")
        stem = candidate
        suffix = 1
        while candidate.lower() in used:
            candidate = safe_grx_name(f"{stem}_{suffix}", f"mesh_{index}_{suffix}")
            suffix += 1
        used.add(candidate.lower())
        result.append(candidate)
    return result


def new_v4_bone_bounds(
    objects: List[bpy.types.Object],
    bones: List[Dict[str, Any]],
    bone_name_to_index: Dict[str, int],
    section_skinned: bool,
    object_rigid_bones: Dict[bpy.types.Object, Optional[int]],
    rigid_fallbacks: Dict[bpy.types.Object, Optional[int]],
) -> List[Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]]:
    """Reproduce BUGXGeomInstancer::findBoneBounds for fresh v4 data."""
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
            if not section_skinned:
                expand(object_rigid_bones.get(obj) if object_rigid_bones.get(obj) is not None else 0, position)
                continue
            # Bounds must follow the exact top-four, normalized, quantized
            # influences that are serialized, not Blender's unbounded source
            # group list. Ensemble applies the 1/64 cutoff to those Univert
            # weights in BUGXGeomInstancer::findBoneBounds.
            raw_indices, raw_weights, _ = new_v4_vertex_skin(
                obj,
                vertex_index,
                bone_name_to_index,
                rigid_fallbacks.get(obj),
            )
            for encoded_index, encoded_weight in zip(raw_indices, raw_weights):
                if encoded_index == 0 or (float(encoded_weight) / 255.0) < (1.0 / 64.0):
                    continue
                expand(int(encoded_index) - 1, position)

    return [
        (tuple(low), tuple(highs[index])) if low is not None and highs[index] is not None else None
        for index, low in enumerate(lows)
    ]


def build_new_v4_authoring_chunks(
    context_obj: bpy.types.Object,
    axis_forward: str = "Z",
    axis_up: str = "Y",
) -> Tuple[bytes, bytes, bytes, List[Optional[bpy.types.Material]], List[str], List[List[int]]]:
    synchronize_ugx_export_meshes(context_obj)
    objects = ugx_authoring_mesh_objects(context_obj)
    if not objects:
        raise UgxError("New HWDE v4 UGX export needs a mesh, or an armature with at least one bound mesh.")
    validate_ugx_export_meshes(objects)
    sync_ugx_export_material_nodes(objects, 4)
    bpy.context.view_layer.update()

    armatures = {ugx_armature_for(item) for item in objects if ugx_armature_for(item) is not None}
    if len(armatures) > 1:
        raise UgxError("All meshes in a new HWDE v4 UGX export must use the same armature.")
    axis_rows = ugx_export_axis_correction_rows(axis_forward, axis_up)
    bones, bone_name_to_index = new_v4_export_bones(objects[0], axis_rows)
    if len(bones) > V4_MAX_SKELETON_BONES:
        raise UgxError(f"HWDE v4 UGX supports up to {V4_MAX_SKELETON_BONES} skeleton bones, including GrannyRootBone.")
    normalize_unused_v4_damage_root(objects, bones, bone_name_to_index)
    rigid_fallbacks = {item: new_v6_rigid_fallback_bone(item, bone_name_to_index) for item in objects}
    has_skin = {item: mesh_has_v4_skin(item, bone_name_to_index) for item in objects}
    authored_as_boned = bool(armatures) or any(has_skin.values())

    object_rigid_bones: Dict[bpy.types.Object, Optional[int]] = {}
    for obj in objects:
        mesh = obj.data
        mesh.calc_loop_triangles()
        rendered_vertices = [int(vertex_index) for tri in mesh.loop_triangles for vertex_index in tri.vertices]
        rigid_bone = new_v4_uniform_rigid_bone(obj, rendered_vertices, bone_name_to_index) if has_skin[obj] else None
        if rigid_bone is None and not has_skin[obj] and authored_as_boned:
            rigid_bone = rigid_fallbacks[obj]
        object_rigid_bones[obj] = rigid_bone

    rigid_values = [value for value in object_rigid_bones.values() if value is not None]
    all_objects_rigid = authored_as_boned and len(rigid_values) == len(objects) and len(set(rigid_values)) == 1
    rigid_bone_index = rigid_values[0] if all_objects_rigid else 0
    section_skinned = authored_as_boned and not all_objects_rigid

    missing_skin = [
        item.name
        for item in objects
        if section_skinned and not has_skin[item] and rigid_fallbacks[item] is None
    ]
    if missing_skin:
        names = ", ".join(missing_skin[:8])
        suffix = "..." if len(missing_skin) > 8 else ""
        raise UgxError(f"Every mesh in a skinned HWDE v4 export needs armature bone weights. Missing: {names}{suffix}")

    materials: List[Optional[bpy.types.Material]] = []
    material_indices: Dict[Tuple[str, int], int] = {}

    def global_material_index(material: Optional[bpy.types.Material]) -> int:
        key = ("material", int(material.as_pointer())) if material is not None else ("none", 0)
        if key not in material_indices:
            material_indices[key] = len(materials)
            materials.append(material)
        return material_indices[key]

    mesh_names = unique_v4_accessory_names(objects)
    sections: List[Dict[str, Any]] = []
    all_positions: List[Tuple[float, float, float]] = []

    for mesh_index, obj in enumerate(objects):
        mesh = obj.data
        mesh.calc_loop_triangles()
        if not mesh.loop_triangles:
            continue
        uv_layer = mesh.uv_layers.get("UV0") or mesh.uv_layers.active
        uv_data, uv_property = mesh_uv_data(mesh, uv_layer)
        uvs_by_loop = {
            int(loop.index): mesh_uv_value(uv_data, uv_property, loop.index)
            for loop in mesh.loops
        } if uv_layer else {}
        tangent_uv = mesh.uv_layers.get('_HW1_Source_Tangent_UV') or next((layer for layer in mesh.uv_layers if layer.active_render), uv_layer)
        basis_by_loop = mesh_calculated_basis_by_loop_world(obj, tangent_uv.name) if tangent_uv else {}
        colors = mesh_colors_by_loop(mesh)
        color_type = (4 if any(component < 0 or component > 1 for value in colors.values() for component in value) else 5) if colors else 0
        extra_layers = [index for index in (1, 2) if mesh.uv_layers.get(f"UV{index}") is not None]
        extra_uvs = {}
        for index in extra_layers:
            data, prop = mesh_uv_data(mesh, mesh.uv_layers[f"UV{index}"])
            extra_uvs[index] = [mesh_uv_value(data, prop, loop.index) for loop in mesh.loops]
        used_slots = {int(poly.material_index) for poly in mesh.polygons}
        for index in used_slots:
            mat = obj.material_slots[index].material if index < len(obj.material_slots) else None
            if mat is None or not mat.get("hwek_ugx_native_material"):
                continue
            for binding in ugx_material_node_bindings(mat, 4, allow_missing=True):
                name = f"UV{binding['channel']}"
                if name != 'UV0' and mesh.uv_layers.get(name) is None:
                    raise UgxError(f"Mesh {obj.name}: material {mat.name} needs missing UV map {name}.")
        local_sections: Dict[int, Dict[str, Any]] = {}
        for tri in mesh.loop_triangles:
            local_material_index = max(0, int(mesh.polygons[tri.polygon_index].material_index))
            material = obj.material_slots[local_material_index].material if local_material_index < len(obj.material_slots) else None
            material_index = global_material_index(material)
            section = local_sections.get(material_index)
            # Source BSection caps sections at 8,192 triangles. The authored
            # path expands triangle corners, so its 16-bit instance multiplier
            # reaches the stricter vertex ceiling first.
            if (
                section is None
                or section["vertex_count"] + 3 > V4_MAX_SECTION_VERTICES
                or len(section["indices"]) // 3 + 1 > UGX_MAX_SECTION_TRIANGLES
            ):
                section = {
                    "material_index": material_index,
                    "extra_uvs": extra_layers,
                    "color_type": color_type,
                    "accessory_index": mesh_index,
                    "authoring_mesh_index": mesh_index,
                    "vertices": bytearray(),
                    "indices": [],
                    "vertex_count": 0,
                    "max_influences": 1,
                    "used_bones": set(),
                }
                sections.append(section)
                local_sections[material_index] = section
            polygon_normal = tuple(float(component) for component in mesh.polygons[tri.polygon_index].normal)
            fallback_normal = object_normal_game_vector(obj, polygon_normal)
            blender_corners = list(zip(tri.loops, tri.vertices))
            for loop_index, vertex_index in reverse_triangle_winding(blender_corners):
                position = object_vertex_game_position(obj, vertex_index)
                all_positions.append(position)
                uv = blender_to_game_uv(uvs_by_loop[loop_index]) if uv_layer else (0.0, 0.0)
                basis = basis_by_loop.get(loop_index)
                if basis is None:
                    game_normal = fallback_normal
                    game_tangent = tangent_for_game_normal(game_normal)
                    bitangent_sign = 1.0
                else:
                    game_normal, game_tangent, bitangent_sign = basis
                if section_skinned:
                    raw_indices, raw_weights, influence_count = new_v4_vertex_skin(
                        obj,
                        vertex_index,
                        bone_name_to_index,
                        rigid_fallbacks[obj],
                    )
                    section["max_influences"] = max(section["max_influences"], influence_count)
                    for encoded_index, encoded_weight in zip(raw_indices, raw_weights):
                        if encoded_index > 0 and encoded_weight > 0:
                            section["used_bones"].add(int(encoded_index) - 1)
                    section["vertices"].extend(encoded_new_v4_skinned_vertex(
                        position, game_normal, uv, raw_indices, raw_weights, game_tangent, bitangent_sign
                    ))
                else:
                    section["used_bones"].add(
                        object_rigid_bones.get(obj)
                        if object_rigid_bones.get(obj) is not None
                        else 0
                    )
                    section["vertices"].extend(encoded_new_v4_vertex(
                        position, game_normal, uv, game_tangent, bitangent_sign
                    ))
                for index in extra_layers:
                    section["vertices"].extend(struct.pack('<2e', *blender_to_game_uv(extra_uvs[index][loop_index])))
                if color_type:
                    section["vertices"].extend(encode_vertex_color(colors[loop_index], color_type))
                section["indices"].append(section["vertex_count"])
                section["vertex_count"] += 1

    if not all_positions:
        raise UgxError("New HWDE v4 UGX export produced no vertices.")
    if not materials:
        materials.append(None)

    vertex_payload = bytearray()
    section_infos: List[Dict[str, Any]] = []
    for section in sections:
        vertex_offset = len(vertex_payload)
        vertex_payload.extend(section["vertices"])
        section_infos.append({
            "material_index": section["material_index"],
            "extra_uvs": section["extra_uvs"],
            "color_type": section['color_type'],
            "accessory_index": section["accessory_index"],
            "authoring_mesh_index": section["authoring_mesh_index"],
            "vertex_offset": vertex_offset,
            "vertex_size": len(section["vertices"]),
            "vertex_count": section["vertex_count"],
            "tri_count": len(section["indices"]) // 3,
            "max_bones": max(1, min(4, int(section["max_influences"]))),
            "used_bones": sorted(section["used_bones"]),
        })

    max_instances, instance_index_multiplier = v4_instance_settings(section_infos)
    index_payload = bytearray()
    for section, info in zip(sections, section_infos):
        info["index_start"] = len(index_payload) // 2
        for instance_index in range(max_instances):
            instance_offset = instance_index_multiplier * instance_index
            for index in section["indices"]:
                index_payload.extend(struct.pack("<H", index + instance_offset))

    mesh_section_indices = [
        [index for index, info in enumerate(section_infos) if info["authoring_mesh_index"] == mesh_index]
        for mesh_index in range(len(mesh_names))
    ]
    accessory_sections = [
        [index for index, info in enumerate(section_infos) if info["authoring_mesh_index"] == mesh_index]
        for mesh_index in range(len(mesh_names))
    ]
    bone_bounds = new_v4_bone_bounds(
        objects,
        bones,
        bone_name_to_index,
        section_skinned,
        object_rigid_bones,
        rigid_fallbacks,
    )
    cached_payload = build_new_v4_cached_data(
        section_infos,
        all_positions,
        bones,
        section_skinned,
        rigid_bone_index=rigid_bone_index,
        accessory_sections=accessory_sections,
        bone_bounds=bone_bounds,
    )
    return cached_payload, bytes(vertex_payload), bytes(index_payload), materials, mesh_names, mesh_section_indices


def build_new_v4_cached_data(
    section_infos: List[Dict],
    positions: List[Tuple[float, float, float]],
    bones: Optional[List[Dict[str, Any]]] = None,
    skinned: bool = False,
    rigid_bone_index: int = 0,
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
        raise UgxError("Every HWDE v4 UGX accessory needs at least one section.")
    covered_sections = sorted(index for indices in accessory_sections for index in indices)
    if covered_sections != list(range(len(section_infos))):
        raise UgxError("HWDE v4 UGX accessory section lists must cover every section exactly once.")
    mins = tuple(min(position[axis] for position in positions) for axis in range(3))
    maxs = tuple(max(position[axis] for position in positions) for axis in range(3))
    center = tuple((mins[axis] + maxs[axis]) * 0.5 for axis in range(3))
    radius = math.sqrt(sum((maxs[axis] - center[axis]) ** 2 for axis in range(3)))
    max_instances, instance_index_multiplier = v4_instance_settings(section_infos)

    # HWDE retail files store the v4 PC discriminator as bytes 04 00 34 C2.
    # Packing CACHED_SIGNATURE_HW1_LE here reverses those bytes and makes the
    # runtime enter the Xbox/big-endian relocation path.
    data = bytearray(CACHED_SIGNATURE_HW1_PC_BYTES)
    data.extend(struct.pack(
        "<i10fhhhBBBBh",
        int(rigid_bone_index),
        center[0], center[1], center[2],
        radius,
        mins[0], mins[1], mins[2],
        maxs[0], maxs[1], maxs[2],
        max_instances,
        instance_index_multiplier,
        0x7FFF,
        0 if skinned else 1,
        1,
        1 if skinned else 0,
        0 if skinned else 1,
        0,
    ))
    data.extend(b"\0" * 4)
    headers_offset = len(data)
    data.extend(b"\0" * (16 * 6))
    align_data(data, 2)
    pack_order_offset = len(data)
    data.extend(b"PNA0ST0\0" if skinned else b"PNA0T0\0")
    pack_orders = {((), False): pack_order_offset}
    for info in section_infos:
        extra = tuple(info.get('extra_uvs', ()))
        key = (extra, bool(info.get('color_type')))
        if key not in pack_orders:
            pack_orders[key] = len(data)
            order = ('PNA0ST0' if skinned else 'PNA0T0') + ''.join('T' + str(index) for index in extra) + ('D' if key[1] else '')
            data.extend(order.encode('ascii') + b'\0')
    decl_order_offset = PACKED_NULL_OFFSET_64
    bone_name_offsets = []
    for index, bone in enumerate(bones):
        align_data(data, 2)
        bone_name_offsets.append(len(data))
        data.extend(safe_grx_name(str(bone.get("name", "")), "").encode("utf-8") + b"\0")
    align_data(data, 8)

    sections_offset = len(data)
    # These are the UnivertPacker defaults present in every shipped DE v4
    # section. PackOrder selects the members in use; A0 is mTangent (index 3),
    # so clearing unused defaults makes the runtime derive the wrong stride.
    types = [17, 3, 16, 3, 3, 16, 16, 16, 16, 16, 16, 16, 16, 6, 9, 5, 7]
    vertex_stride = 44 if skinned else 36
    for info in section_infos:
        record = bytearray()
        record.extend(struct.pack(
            "<iiiiiiiiii",
            info["material_index"],
            int(info.get("accessory_index", 0)),
            max(1, int(info.get("max_bones", 1))),
            UGX_NO_RIGID_BONE if skinned else int(rigid_bone_index),
            info["index_start"],
            info["tri_count"],
            info["vertex_offset"],
            info["vertex_size"],
            vertex_stride + 4 * len(info.get("extra_uvs", ())) + (16 if info.get('color_type') == 4 else 4 if info.get('color_type') else 0),
            info["vertex_count"],
        ))
        record.extend(packed_array_header(0, PACKED_NULL_OFFSET_64))
        record.extend(struct.pack("<QQ", pack_orders[(tuple(info.get("extra_uvs", ())), bool(info.get('color_type'))) ], decl_order_offset))
        section_types = list(types)
        section_types[15] = info.get('color_type') or 5
        record.extend(struct.pack("<17I", *section_types))
        # Native x64 UnivertPacker rounds its 84 logical bytes to 88, placing
        # four bytes of alignment here. BSection::mRigidOnly then lives at
        # +144, followed by three explicit and four ABI tail-padding bytes.
        record.extend(b"\0" * 4)
        record.extend(struct.pack("<B", 0 if skinned else 1))
        record.extend(b"\0" * 7)
        if len(record) != 152:
            raise UgxError("Internal v4 section record size mismatch.")
        data.extend(record)
    align_data(data, 8)

    bones_offset = len(data)
    for index, bone in enumerate(bones):
        rows = bone.get("matrix", uax_matrix4_identity())
        flat = [float(rows[row][column]) for row in range(4) for column in range(4)]
        data.extend(struct.pack("<Q16fiI", bone_name_offsets[index], *flat, int(bone.get("parent", -1)), 0))
    align4(data)

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
        if len(accessory_sections) == 1:
            first_bone = 0
            bone_count = max(1, len(bones))
        else:
            used_bones = sorted({
                int(bone_index)
                for section_index in indices
                for bone_index in section_infos[section_index].get("used_bones", ())
                if 0 <= int(bone_index) < len(bones)
            })
            first_bone = used_bones[0] if used_bones else 0
            bone_count = used_bones[-1] - first_bone + 1 if used_bones else 0
        data.extend(struct.pack("<ii", first_bone, bone_count))
        data.extend(packed_array_header(len(indices), accessory_section_offsets[index]))
    align4(data)

    bone_bounds_low_offset = len(data)
    if bone_bounds is None:
        resolved_bone_bounds = [None] * len(bones)
        if skinned:
            # Direct low-level callers cannot supply per-vertex influence data;
            # keep their bounds conservative. The authored export path always
            # supplies source-faithful per-bone bounds.
            resolved_bone_bounds = [(mins, maxs) for _ in bones]
        elif bones:
            target = rigid_bone_index if 0 <= rigid_bone_index < len(bones) else 0
            resolved_bone_bounds[target] = (mins, maxs)
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
