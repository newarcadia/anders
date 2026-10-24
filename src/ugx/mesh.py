"""Read Blender mesh attributes in native export coordinates."""

from mathutils import Vector
from typing import Any, Dict, Iterable, List, Optional, Tuple
import bpy
from .model_space import ugx_model_matrix
import math
from array import array
from ..errors import UgxError
from .coordinates import blender_to_game_uv_for_version, blender_to_game_vec3
from .data import Section, UV_SEMANTICS, UgxModel
from .skin import encoded_bone_from_vertex_group
from .vertex_authoring import orthonormal_tangent_tuple
from .vertices import changed, normalized_tuple, section_position_changed

__all__ = ('collect_vertex_color_materials', 'mesh_color_attribute', 'mesh_colors_by_vertex', 'mesh_colors_by_loop', 'mesh_has_split_colors', 'mesh_uv_data', 'mesh_uv_value', 'mesh_uv_by_vertex', 'mesh_uv_by_loop', 'mesh_has_split_uv', 'mesh_has_any_split_uv', 'mesh_vector_attribute_by_vertex', 'mesh_float_attribute_by_vertex', 'mesh_bitangent_signs_by_vertex', 'mesh_shape_key_layers', 'mesh_vertex_export_co', 'object_vertex_game_position', 'object_matrix_is_identity', 'object_normal_game_vector', 'mesh_normals_by_vertex_world', 'object_direction_game_vector', 'blender_bitangent_sign_to_game', 'mesh_calculated_basis_by_loop_world', 'mesh_tangents_by_vertex_world', 'mesh_calculated_normals_by_vertex_world', 'mesh_calculated_tangents_by_vertex_world', 'export_vector_for_edited_position', 'mesh_connected_vertices', 'mesh_position_changed_vertices', 'vertex_group_bone_indices', 'mesh_skin_map', 'mesh_int_attribute')

def mesh_uv_data(mesh: bpy.types.Mesh, layer: Any) -> Tuple[Any, str]:
    if layer is None:
        return None, ""
    attribute = mesh.attributes.get(layer.name) if hasattr(mesh, "attributes") else None
    if attribute is not None and attribute.domain == "CORNER" and attribute.data_type == "FLOAT2":
        return attribute.data, "vector"
    return layer.data, "uv"


def mesh_uv_value(data: Any, property_name: str, loop_index: int) -> Tuple[float, float]:
    value = getattr(data[loop_index], property_name)
    return tuple(float(component) for component in value)


def mesh_uv_by_vertex(mesh: bpy.types.Mesh, layer_name: str, cache_version: int = 4) -> Dict[int, Tuple[float, float]]:
    layer = mesh.uv_layers.get(layer_name) if mesh.uv_layers else None
    if layer is None:
        return {}
    uv_data, uv_property = mesh_uv_data(mesh, layer)
    result: Dict[int, Tuple[float, float]] = {}
    for loop in mesh.loops:
        uv = blender_to_game_uv_for_version(mesh_uv_value(uv_data, uv_property, loop.index), cache_version)
        previous = result.get(loop.vertex_index)
        if previous is not None and changed(previous, uv):
            raise UgxError(f"UGX stores {layer_name} per vertex; split per-face {layer_name} edits need a full vertex/index rebuild.")
        result[loop.vertex_index] = uv
    return result


def mesh_uv_by_loop(mesh: bpy.types.Mesh, layer_name: str, cache_version: int = 4) -> Dict[int, Tuple[float, float]]:
    layer = mesh.uv_layers.get(layer_name) if mesh.uv_layers else None
    if layer is None:
        return {}
    uv_data, uv_property = mesh_uv_data(mesh, layer)
    return {loop.index: blender_to_game_uv_for_version(mesh_uv_value(uv_data, uv_property, loop.index), cache_version) for loop in mesh.loops}


def mesh_has_split_uv(mesh: bpy.types.Mesh, layer_name: str, cache_version: int = 4) -> bool:
    layer = mesh.uv_layers.get(layer_name) if mesh.uv_layers else None
    if layer is None:
        return False
    uv_data, uv_property = mesh_uv_data(mesh, layer)
    seen: Dict[int, Tuple[float, float]] = {}
    for loop in mesh.loops:
        uv = blender_to_game_uv_for_version(mesh_uv_value(uv_data, uv_property, loop.index), cache_version)
        previous = seen.get(loop.vertex_index)
        if previous is not None and changed(previous, uv):
            return True
        seen[loop.vertex_index] = uv
    return False


def mesh_has_any_split_uv(mesh: bpy.types.Mesh, cache_version: int = 4) -> bool:
    return any(mesh_has_split_uv(mesh, layer_name, cache_version) for _, layer_name, _ in UV_SEMANTICS)


def mesh_color_attribute(mesh):
    if not mesh.color_attributes:
        return None
    active = mesh.color_attributes.active_color
    if active is not None:
        return active
    render = mesh.color_attributes.render_color_index
    return mesh.color_attributes[render if 0 <= render < len(mesh.color_attributes) else 0]


def collect_vertex_color_materials(objects):
    materials = set()
    for obj in objects:
        if mesh_color_attribute(obj.data) is not None:
            for slot in {polygon.material_index for polygon in obj.data.polygons}:
                materials.add(obj.material_slots[slot].material if slot < len(obj.material_slots) else None)
    return materials


def mesh_colors_by_vertex(mesh):
    attribute = mesh_color_attribute(mesh)
    if attribute is None:
        return {}
    if attribute.domain == 'POINT':
        return {index: tuple(point.color) for index, point in enumerate(attribute.data)}
    result = {}
    for loop in mesh.loops:
        result.setdefault(loop.vertex_index, tuple(attribute.data[loop.index].color))
    return result


def mesh_colors_by_loop(mesh):
    attribute = mesh_color_attribute(mesh)
    if attribute is None:
        return {}
    if attribute.domain == 'CORNER':
        return {index: tuple(point.color) for index, point in enumerate(attribute.data)}
    return {loop.index: tuple(attribute.data[loop.vertex_index].color) for loop in mesh.loops}


def mesh_has_split_colors(mesh):
    attribute = mesh_color_attribute(mesh)
    if attribute is None or attribute.domain == 'POINT':
        return False
    seen = {}
    for loop in mesh.loops:
        value = tuple(attribute.data[loop.index].color)
        previous = seen.setdefault(loop.vertex_index, value)
        if changed(previous, value):
            return True
    return False


def mesh_vector_attribute_by_vertex(mesh: bpy.types.Mesh, name: str) -> Dict[int, Tuple[float, float, float]]:
    attr = mesh.attributes.get(name) if hasattr(mesh, "attributes") else None
    if attr is None:
        return {}
    result: Dict[int, Tuple[float, float, float]] = {}
    for index in range(min(len(mesh.vertices), len(attr.data))):
        result[index] = tuple(float(v) for v in attr.data[index].vector)
    return result


def mesh_float_attribute_by_vertex(mesh: bpy.types.Mesh, name: str) -> Dict[int, float]:
    attr = mesh.attributes.get(name) if hasattr(mesh, "attributes") else None
    if attr is None:
        return {}
    return {
        index: float(getattr(attr.data[index], "value", 0.0))
        for index in range(min(len(mesh.vertices), len(attr.data)))
    }


def mesh_bitangent_signs_by_vertex(mesh: bpy.types.Mesh) -> Dict[int, float]:
    return {
        index: (-1.0 if value < 0.0 else 1.0)
        for index, value in mesh_float_attribute_by_vertex(mesh, "UGX_BitangentSign").items()
    }


def mesh_shape_key_layers(obj: bpy.types.Object):
    keys = obj.data.shape_keys
    if not keys or not keys.key_blocks:
        return
    basis = keys.key_blocks[0]
    for key in keys.key_blocks[1:]:
        if key.mute or abs(key.value) <= 0.000001:
            continue
        group = obj.vertex_groups.get(key.vertex_group) if key.vertex_group else None
        reference = key.relative_key if keys.use_relative else basis
        yield key, reference, group.index if group is not None else -1


def mesh_vertex_export_co(obj: bpy.types.Object, vertex_index: int):
    vertex = obj.data.vertices[vertex_index]
    co = vertex.co.copy()
    for key, reference, group in mesh_shape_key_layers(obj):
        value = float(key.value)
        if group >= 0:
            value *= next((entry.weight for entry in vertex.groups if entry.group == group), 0.)
        co += (key.data[vertex_index].co - reference.data[vertex_index].co) * value
    return co


def object_vertex_game_position(obj: bpy.types.Object, vertex_index: int) -> Tuple[float, float, float]:
    co = mesh_vertex_export_co(obj, vertex_index)
    if object_matrix_is_identity(obj):
        return blender_to_game_vec3(tuple(float(component) for component in co))
    co = ugx_model_matrix(obj) @ co
    return blender_to_game_vec3(tuple(float(component) for component in co))


def object_matrix_is_identity(obj: bpy.types.Object) -> bool:
    matrix = ugx_model_matrix(obj)
    for row in range(4):
        for column in range(4):
            expected = 1.0 if row == column else 0.0
            if abs(float(matrix[row][column]) - expected) > 0.0000001:
                return False
    return True


def object_normal_game_vector(obj: bpy.types.Object, value: Tuple[float, float, float]) -> Tuple[float, float, float]:
    vector = ugx_model_matrix(obj).to_3x3().inverted().transposed() @ Vector(value)
    if vector.length_squared > 0.0:
        vector.normalize()
    return blender_to_game_vec3(tuple(float(component) for component in vector))


def mesh_normals_by_vertex_world(obj: bpy.types.Object) -> Dict[int, Tuple[float, float, float]]:
    normals = mesh_vector_attribute_by_vertex(obj.data, "UGX_Normal")
    if object_matrix_is_identity(obj):
        return {index: blender_to_game_vec3(value) for index, value in normals.items()}
    return {index: object_normal_game_vector(obj, value) for index, value in normals.items()}


def object_direction_game_vector(obj: bpy.types.Object, value: Tuple[float, float, float]) -> Tuple[float, float, float]:
    vector = ugx_model_matrix(obj).to_3x3() @ Vector(value)
    if vector.length_squared > 0.0:
        vector.normalize()
    return blender_to_game_vec3(tuple(float(component) for component in vector))


def blender_bitangent_sign_to_game(obj: bpy.types.Object, value: float) -> float:
    object_determinant = float(ugx_model_matrix(obj).to_3x3().determinant())
    reflection_sign = -1.0 if object_determinant >= 0.0 else 1.0
    return reflection_sign * (-1.0 if float(value) < 0.0 else 1.0)


def mesh_calculated_basis_by_loop_world(
    obj: bpy.types.Object,
    uv_layer: str = "UV0",
) -> Dict[int, Tuple[Tuple[float, float, float], Tuple[float, float, float], float]]:
    mesh = obj.data
    if uv_layer == 'UV0' and mesh.uv_layers.get('_HW1_Source_Tangent_UV') is not None:
        uv_layer = '_HW1_Source_Tangent_UV'
    if not mesh.uv_layers or uv_layer not in mesh.uv_layers:
        return {}
    try:
        mesh.calc_tangents(uvmap=uv_layer)
    except Exception:
        return {}
    result = {}
    for loop in mesh.loops:
        try:
            normal = object_normal_game_vector(obj, tuple(float(component) for component in loop.normal))
            tangent = object_direction_game_vector(obj, tuple(float(component) for component in loop.tangent))
            sign = blender_bitangent_sign_to_game(obj, float(loop.bitangent_sign))
        except Exception:
            continue
        if not all(math.isfinite(component) for component in (*normal, *tangent, sign)):
            continue
        normal = normalized_tuple(normal, (0.0, 1.0, 0.0))
        result[int(loop.index)] = (normal, orthonormal_tangent_tuple(normal, tangent), sign)
    return result


def mesh_tangents_by_vertex_world(obj: bpy.types.Object) -> Dict[int, Tuple[float, float, float]]:
    tangents = mesh_vector_attribute_by_vertex(obj.data, "UGX_Tangent")
    if object_matrix_is_identity(obj):
        return {index: blender_to_game_vec3(value) for index, value in tangents.items()}
    return {index: object_direction_game_vector(obj, value) for index, value in tangents.items()}


def mesh_calculated_normals_by_vertex_world(obj: bpy.types.Object) -> Dict[int, Tuple[float, float, float]]:
    mesh = obj.data
    mesh.update()
    accum: Dict[int, Vector] = {}
    for poly in mesh.polygons:
        vertices = list(poly.vertices)
        if len(vertices) < 3:
            continue
        p0 = mesh_vertex_export_co(obj, vertices[0])
        for index in range(1, len(vertices) - 1):
            p1 = mesh_vertex_export_co(obj, vertices[index])
            p2 = mesh_vertex_export_co(obj, vertices[index + 1])
            normal = (p1 - p0).cross(p2 - p0)
            if normal.length_squared <= 0.0000001:
                continue
            for vertex_index in (vertices[0], vertices[index], vertices[index + 1]):
                accum[vertex_index] = accum.get(vertex_index, Vector((0.0, 0.0, 0.0))) + normal
    result: Dict[int, Tuple[float, float, float]] = {}
    for vertex in mesh.vertices:
        normal = accum.get(vertex.index)
        if normal is None or normal.length_squared <= 0.0000001:
            normal = Vector(tuple(float(component) for component in vertex.normal))
        else:
            normal.normalize()
        value = tuple(float(component) for component in normal)
        if object_matrix_is_identity(obj):
            result[vertex.index] = blender_to_game_vec3(value)
        else:
            result[vertex.index] = object_normal_game_vector(obj, value)
    return result


def mesh_calculated_tangents_by_vertex_world(obj: bpy.types.Object, uv_layer: str = "UV0") -> Dict[int, Tuple[float, float, float]]:
    mesh = obj.data
    if uv_layer == 'UV0' and mesh.uv_layers.get('_HW1_Source_Tangent_UV') is not None:
        uv_layer = '_HW1_Source_Tangent_UV'
    if not mesh.uv_layers or uv_layer not in mesh.uv_layers:
        return {}
    uv_data, uv_property = mesh_uv_data(mesh, mesh.uv_layers[uv_layer])
    accum: Dict[int, Vector] = {}
    for poly in mesh.polygons:
        if len(poly.loop_indices) < 3:
            continue
        poly_vertices = list(poly.vertices)
        poly_loops = list(poly.loop_indices)
        p0 = mesh_vertex_export_co(obj, poly_vertices[0])
        uv0 = Vector(mesh_uv_value(uv_data, uv_property, poly_loops[0]))
        for index in range(1, len(poly_vertices) - 1):
            p1 = mesh_vertex_export_co(obj, poly_vertices[index])
            p2 = mesh_vertex_export_co(obj, poly_vertices[index + 1])
            uv1 = Vector(mesh_uv_value(uv_data, uv_property, poly_loops[index]))
            uv2 = Vector(mesh_uv_value(uv_data, uv_property, poly_loops[index + 1]))
            edge1 = p1 - p0
            edge2 = p2 - p0
            duv1 = uv1 - uv0
            duv2 = uv2 - uv0
            denom = duv1.x * duv2.y - duv2.x * duv1.y
            if abs(float(denom)) <= 0.0000001:
                continue
            tangent = (edge1 * duv2.y - edge2 * duv1.y) / denom
            if tangent.length_squared <= 0.0000001:
                continue
            for vertex_index in (poly_vertices[0], poly_vertices[index], poly_vertices[index + 1]):
                accum[vertex_index] = accum.get(vertex_index, Vector((0.0, 0.0, 0.0))) + tangent
    result: Dict[int, Tuple[float, float, float]] = {}
    for vertex_index, tangent in accum.items():
        if tangent.length_squared <= 0.0000001:
            continue
        tangent.normalize()
        value = tuple(float(component) for component in tangent)
        if object_matrix_is_identity(obj):
            result[vertex_index] = blender_to_game_vec3(value)
        else:
            result[vertex_index] = object_direction_game_vector(obj, value)
    return result


def export_vector_for_edited_position(
    vertex_index: int,
    source_value: Tuple[float, float, float],
    attr_values: Dict[int, Tuple[float, float, float]],
    calculated_values: Dict[int, Tuple[float, float, float]],
    refresh_shading: bool,
) -> Optional[Tuple[float, float, float]]:
    current = attr_values.get(vertex_index)
    if refresh_shading and (current is None or not changed(current, source_value)):
        calculated = calculated_values.get(vertex_index)
        if calculated is not None:
            return calculated
    return current


def mesh_connected_vertices(mesh: bpy.types.Mesh, vertices: Iterable[int]) -> set:
    changed_vertices = set(vertices)
    if not changed_vertices:
        return set()
    result = set(changed_vertices)
    for poly in mesh.polygons:
        poly_vertices = set(int(index) for index in poly.vertices)
        if poly_vertices & changed_vertices:
            result.update(poly_vertices)
    return result


def mesh_position_changed_vertices(model: UgxModel, obj: bpy.types.Object) -> set:
    result = set()
    vertex_base = 0
    for section in model.sections:
        for local_index, source_vertex in enumerate(section.vertices):
            vertex_index = vertex_base + local_index
            if section_position_changed(section, local_index, object_vertex_game_position(obj, vertex_index)):
                result.add(vertex_index)
        vertex_base += len(section.vertices)
    return result


def vertex_group_bone_indices(obj, armature):
    if armature is None:
        return list(range(len(obj.vertex_groups)))
    indices = {bone.name: int(bone.get('hwek_ugx_bone_index', index))
               for index, bone in enumerate(armature.data.bones)}
    return [indices.get(group.name, -1) for group in obj.vertex_groups]


def mesh_skin_map(obj: bpy.types.Object, section: Section, vertex_index: int,
                  bone_indices_by_group=None, bone_count=None) -> Dict[int, float]:
    if bone_indices_by_group is None:
        from .scene import ugx_armature_for
        armature = ugx_armature_for(obj)
        bone_indices_by_group = vertex_group_bone_indices(obj, armature)
        if armature is not None and bone_count is None:
            bone_count = len(armature.data.bones)
    vertex = obj.data.vertices[vertex_index]
    result: Dict[int, float] = {}
    for group in vertex.groups:
        if group.weight <= 0.00001:
            continue
        bone_index = bone_indices_by_group[group.group]
        local_bone = encoded_bone_from_vertex_group(section, bone_index,
                                                   len(obj.vertex_groups) if bone_count is None else bone_count)
        if local_bone is None:
            if section.cache_version == 6 and bone_index >= 0:
                raise UgxError('The HW2 section bone palette does not contain a weighted bone. Rebuild the mesh before exporting these weights.')
            continue
        if 0 <= local_bone <= 255:
            result[local_bone] = result.get(local_bone, 0.0) + float(group.weight)
    if not result and section.cache_version == 6:
        fallback = 0
        requested = str(obj.get('hwek_ugx_rigid_bone_name', '')).strip() or str(obj.parent_bone).strip()
        if requested:
            from .scene import ugx_armature_for
            armature = ugx_armature_for(obj)
            bone = armature.data.bones.get(requested) if armature is not None else None
            if bone is not None:
                fallback = int(bone['hwek_ugx_bone_index']) if 'hwek_ugx_bone_index' in bone else armature.data.bones.find(requested)
        root = encoded_bone_from_vertex_group(section, fallback, bone_count if bone_count is not None else len(obj.vertex_groups))
        if root is None:
            raise UgxError('The HW2 section bone palette does not contain the unweighted vertex attachment. Rebuild the mesh before exporting unweighted vertices.')
        result[root] = 1.0
    return result


def mesh_int_attribute(mesh: bpy.types.Mesh, name: str) -> Optional[List[int]]:
    if not hasattr(mesh, "attributes"):
        return None
    attr = mesh.attributes.get(name)
    if attr is None:
        return None
    values = array('i', [0]) * len(attr.data)
    attr.data.foreach_get('value', values)
    return values.tolist()
