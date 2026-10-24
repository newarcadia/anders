"""Resolve model ownership, bound meshes and export selections."""

from typing import List, Optional, Tuple
import bpy
import os
from .mesh import mesh_int_attribute
from .naming import current_ugx_mesh_name

__all__ = ('ugx_armature_for', 'ugx_bound_mesh_objects', 'ugx_authoring_mesh_objects', 'ugx_used_material_slots', 'ugx_mesh_has_imported_topology', 'ugx_ordered_armature_bones', 'ugx_export_object_for', 'ugx_material_context_object', 'ugx_lod_sibling_objects_for')

def ugx_used_material_slots(obj: bpy.types.Object) -> List[Tuple[int, Optional[bpy.types.Material]]]:
    if obj is None or obj.type != "MESH":
        return []
    used = {int(polygon.material_index) for polygon in obj.data.polygons}
    return [(index, obj.material_slots[index].material) for index in sorted(used)
            if 0 <= index < len(obj.material_slots)]

def ugx_armature_for(obj: Optional[bpy.types.Object]) -> Optional[bpy.types.Object]:
    if obj is None:
        return None
    if obj.type == "ARMATURE":
        return obj
    for modifier in getattr(obj, "modifiers", []):
        if modifier.type == "ARMATURE" and modifier.object:
            return modifier.object
    parent = getattr(obj, "parent", None)
    if parent is not None and parent.type == "ARMATURE":
        return parent
    return None


def ugx_bound_mesh_objects(armature: Optional[bpy.types.Object]) -> List[bpy.types.Object]:
    if armature is None or armature.type != "ARMATURE":
        return []
    objects = [
        candidate
        for candidate in bpy.data.objects
        if candidate.type == "MESH"
        and len(candidate.data.polygons) > 0
        and bool(candidate.get("hwek_ugx_export_enabled", True))
        and ugx_armature_for(candidate) == armature
    ]
    objects.sort(key=lambda item: (
        int(item.get("hwek_ugx_lod_index", 0)),
        int(item.get("hwek_ugx_mesh_index", 0x7FFFFFFF)),
        current_ugx_mesh_name(item).lower(),
        item.name.lower(),
    ))
    return objects


def ugx_authoring_mesh_objects(obj: Optional[bpy.types.Object]) -> List[bpy.types.Object]:
    if obj is None:
        return []
    armature = ugx_armature_for(obj)
    if armature is not None:
        return ugx_bound_mesh_objects(armature)
    if obj.type == "MESH" and len(obj.data.polygons) > 0 and bool(obj.get("hwek_ugx_export_enabled", True)):
        return [obj]
    return []


def ugx_mesh_has_imported_topology(obj: bpy.types.Object) -> bool:
    if obj.type != "MESH" or not obj.get("hwek_ugx_source_path"):
        return False
    return (
        mesh_int_attribute(obj.data, "UGX_Section") is not None
        and mesh_int_attribute(obj.data, "UGX_SourceVertex") is not None
        and "hwek_ugx_lod_section_indices" in obj
    )


def ugx_ordered_armature_bones(armature: Optional[bpy.types.Object]) -> List[bpy.types.Bone]:
    bones = list(armature.data.bones) if armature else []
    indexed = []
    for bone in bones:
        if "hwek_ugx_bone_index" not in bone:
            return bones
        indexed.append((int(bone["hwek_ugx_bone_index"]), bone))
    indices = [index for index, _ in indexed]
    if sorted(indices) != list(range(len(indices))):
        return bones
    return [bone for _, bone in sorted(indexed, key=lambda item: item[0])]


def ugx_export_object_for(obj: Optional[bpy.types.Object]) -> Optional[bpy.types.Object]:
    if obj is None:
        return None
    if obj.type == "MESH" and obj.get("hwek_ugx_source_path"):
        return obj
    armature = ugx_armature_for(obj)
    if armature is not None:
        candidates = [
            candidate
            for candidate in ugx_bound_mesh_objects(armature)
            if candidate.get("hwek_ugx_source_path")
        ]
        if candidates:
            return candidates[0]
        if armature.get("hwek_ugx_source_path"):
            return armature
    return None


def ugx_material_context_object(obj: Optional[bpy.types.Object]) -> Optional[bpy.types.Object]:
    source_obj = ugx_export_object_for(obj)
    if source_obj is not None:
        return source_obj
    if obj is not None and obj.type == "MESH":
        return obj
    return None


def ugx_lod_sibling_objects_for(obj: bpy.types.Object) -> List[bpy.types.Object]:
    import_id = str(obj.get("hwek_ugx_import_id", ""))
    source_path = str(obj.get("hwek_ugx_source_path", ""))
    if not import_id or not source_path or int(obj.get("hwek_ugx_lod_object_count", 1)) <= 1:
        return [obj]
    source_key = os.path.normcase(os.path.abspath(source_path))
    siblings = [
        candidate
        for candidate in bpy.data.objects
        if candidate.type == "MESH"
        and str(candidate.get("hwek_ugx_import_id", "")) == import_id
        and os.path.normcase(os.path.abspath(str(candidate.get("hwek_ugx_source_path", "")))) == source_key
    ]
    if not siblings:
        return [obj]
    siblings.sort(key=lambda item: (
        int(item.get("hwek_ugx_lod_index", 0)),
        int(item.get("hwek_ugx_mesh_index", -1)),
        item.name,
    ))
    return siblings
