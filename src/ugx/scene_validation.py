"""Synchronize model edits and reject unsupported export inputs."""

from typing import List, Optional
import bpy
import json
import math
from ..errors import UgxError
from .data import UgxModel
from .mesh import mesh_int_attribute, mesh_uv_data, mesh_uv_value, object_vertex_game_position
from .metadata import section_entry_source_index, section_metadata_value
from .scene import ugx_armature_for

__all__ = ('synchronize_ugx_export_meshes', 'validate_ugx_export_meshes')

def synchronize_ugx_export_meshes(obj: Optional[bpy.types.Object]):
    armature = ugx_armature_for(obj)
    objects = [
        item for item in bpy.data.objects
        if item.type == "MESH" and ugx_armature_for(item) == armature
    ] if armature is not None else ([obj] if obj is not None and obj.type == "MESH" else [])
    for item in objects:
        if item.mode == "EDIT":
            item.update_from_editmode()
    bpy.context.view_layer.update()


def validate_ugx_export_meshes(objects: List[bpy.types.Object], source_model: Optional[UgxModel] = None,
                             unchanged_buffers=False, *, cache_version=None):
    source_vertices = [vertex for section in source_model.sections for vertex in section.vertices] if source_model is not None else []
    for obj in objects:
        entries = json.loads(obj.get("hwek_ugx_sections", "[]"))
        for index, entry in enumerate(entries):
            section_index = section_entry_source_index(index, entry)
            max_bones = int(section_metadata_value(obj, index, entry, "max_bones", 0))
            if not 0 <= max_bones <= 4:
                raise UgxError(f"Mesh {obj.name!r} section {section_index} exceeds the four-bone vertex influence limit.")
            accessory_index = int(section_metadata_value(obj, index, entry, "accessory_index", 0))
            accessory_count = int(obj.get("hwek_ugx_accessory_count", 0))
            if accessory_index < 0 or (accessory_count > 0 and accessory_index >= accessory_count):
                raise UgxError(f"Mesh {obj.name!r} section {section_index} references missing accessory {accessory_index}.")
            for key in ("lod_min_distance", "lod_max_distance", "sorting_bias"):
                if not math.isfinite(float(section_metadata_value(obj, index, entry, key, 0.0))):
                    raise UgxError(f"Mesh {obj.name!r} section {section_index} {key} is not finite.")
        mesh = obj.data
        if unchanged_buffers:
            # The exact imported buffers, including weights and UVs, matched.
            # Bone renames still need the normal group-to-bone validation.
            armature = ugx_armature_for(obj)
            if armature is None or all(group.name in armature.data.bones for group in obj.vertex_groups):
                continue
        source_indices = mesh_int_attribute(mesh, "UGX_SourceVertex") if source_model is not None else None
        mesh.calc_loop_triangles()
        rendered_vertices = {int(index) for tri in mesh.loop_triangles for index in tri.vertices}
        rendered_loops = {int(index) for tri in mesh.loop_triangles for index in tri.loops}
        for poly in mesh.polygons:
            material_index = int(poly.material_index)
            if material_index < 0 or material_index >= max(1, len(mesh.materials)):
                raise UgxError(f"Mesh {obj.name!r} face {poly.index} references missing material slot {material_index}.")
        for vertex_index in rendered_vertices:
            if not all(math.isfinite(value) for value in object_vertex_game_position(obj, vertex_index)):
                raise UgxError(f"Mesh {obj.name!r} vertex {vertex_index} has a non-finite position.")
        for layer in mesh.uv_layers:
            uv_data, uv_property = mesh_uv_data(mesh, layer)
            for loop_index in rendered_loops:
                if not all(math.isfinite(value) for value in mesh_uv_value(uv_data, uv_property, loop_index)):
                    raise UgxError(f"Mesh {obj.name!r} UV layer {layer.name!r} corner {loop_index} has a non-finite UV.")
        armature = ugx_armature_for(obj)
        if armature is None:
            continue
        version = cache_version
        if version is None:
            version = source_model.cache_version if source_model is not None else int(obj.get('hwek_ugx_cache_version', 0))
        bone_names = {bone.name for bone in armature.data.bones}
        weighted_mesh = any(group.name in bone_names for group in obj.vertex_groups)
        explicit_rigid_bone = str(obj.get("hwek_ugx_rigid_bone_name", "")).strip() or str(obj.parent_bone).strip()
        if explicit_rigid_bone and explicit_rigid_bone not in bone_names:
            raise UgxError(f"Mesh {obj.name!r} references missing rigid bone {explicit_rigid_bone!r}.")
        for vertex_index in rendered_vertices:
            influences = 0
            unknown_bones = []
            for group in mesh.vertices[vertex_index].groups:
                if not math.isfinite(group.weight) or group.weight < 0.0:
                    raise UgxError(f"Mesh {obj.name!r} vertex {vertex_index} has an invalid bone weight.")
                if group.weight <= 0.00001:
                    continue
                if group.group >= len(obj.vertex_groups):
                    raise UgxError(f"Mesh {obj.name!r} vertex {vertex_index} references a missing vertex group.")
                name = obj.vertex_groups[group.group].name
                if name in bone_names:
                    influences += 1
                else:
                    unknown_bones.append(name)
            if influences > 4:
                raise UgxError(f"Mesh {obj.name!r} vertex {vertex_index} exceeds the four-bone weight limit.")
            if influences == 0 and not explicit_rigid_bone and (weighted_mesh or unknown_bones):
                # Unbound groups do not deform vertices in Blender. HW2 export
                # binds those vertices to the reserved model root instead.
                if version == 6:
                    continue
                source_index = int(source_indices[vertex_index]) if source_indices is not None and vertex_index < len(source_indices) else -1
                if not unknown_bones and 0 <= source_index < len(source_vertices) and not any(weight > 0.00001 for weight in source_vertices[source_index].bone_weights):
                    continue
                raise UgxError(f"Mesh {obj.name!r} vertex {vertex_index} has no weights for armature {armature.name!r}.")
