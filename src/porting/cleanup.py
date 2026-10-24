"""Remove model provenance within the selected object scope."""

from typing import Dict, Iterable, List
import bpy
from ..animation.scene import uax_target_armature

__all__ = ('clear_ugx_id_properties', 'ugx_portable_object_scope', 'material_used_outside_object_scope', 'image_used_outside_material_scope', 'remove_ugx_data_for_port', 'object_scope_has_ugx_data')

_PORTABLE_OBJECT_PROPERTIES = {
    'hwek_ugx_player_color', 'hwek_ugx_mesh_name', 'hwek_ugx_mesh_import_name',
    'hwek_ugx_lod_min_distance', 'hwek_ugx_lod_max_distance', 'hwek_ugx_sorting_bias',
}
_PORTABLE_BONE_PROPERTIES = {
    'hwek_ugx_source_name', 'hwek_ugx_blender_name', 'hwek_ugx_export_name',
}


def clear_ugx_id_properties(datablock, preserve=()) -> int:
    if datablock is None or not hasattr(datablock, "keys"):
        return 0
    removed = 0
    for key in list(datablock.keys()):
        if str(key).startswith("hwek_ugx_") and key not in preserve:
            try:
                del datablock[key]
                removed += 1
            except (KeyError, TypeError):
                pass
    return removed


def ugx_portable_object_scope(objects: Iterable[bpy.types.Object]) -> List[bpy.types.Object]:
    """Expand selected meshes/armatures to the complete bound model set."""
    scope = {obj for obj in objects if obj is not None}
    import_ids = {str(obj.get("hwek_ugx_import_id", "")) for obj in scope if obj.get("hwek_ugx_import_id")}
    armatures = {uax_target_armature(obj) for obj in scope}
    armatures.discard(None)
    scope.update(armatures)
    for candidate in bpy.data.objects:
        if candidate.get('hwek_vis_marker') and candidate.parent in armatures:
            scope.add(candidate)
            continue
        if str(candidate.get("hwek_ugx_import_id", "")) in import_ids and import_ids:
            scope.add(candidate)
            continue
        if getattr(candidate, "type", "") != "MESH":
            continue
        if getattr(candidate, "parent", None) in armatures:
            scope.add(candidate)
            continue
        if any(
            getattr(modifier, "type", "") == "ARMATURE" and getattr(modifier, "object", None) in armatures
            for modifier in getattr(candidate, "modifiers", [])
        ):
            scope.add(candidate)
    return sorted(scope, key=lambda item: item.name.lower())


def material_used_outside_object_scope(material: bpy.types.Material, scope: set) -> bool:
    for obj in bpy.data.objects:
        if obj in scope or getattr(obj, "type", "") != "MESH":
            continue
        if any(slot.material is material for slot in getattr(obj, "material_slots", [])):
            return True
    return False


def image_used_outside_material_scope(image: bpy.types.Image, materials: set) -> bool:
    for material in bpy.data.materials:
        if material in materials or not getattr(material, "use_nodes", False) or material.node_tree is None:
            continue
        if any(getattr(node, "image", None) is image for node in material.node_tree.nodes):
            return True
    return False


def remove_ugx_data_for_port(objects: Iterable[bpy.types.Object]) -> Dict[str, int]:
    """Detach source-game UGX contracts while preserving editable model data."""
    scoped_objects = list(objects)
    scope_set = set(scoped_objects)
    stats = {"objects": len(scoped_objects), "properties": 0, "attributes": 0, "materials": 0, "images": 0}

    # Materials, image encodings and shader parameters are authored data.
    # Removing model provenance must not detach their independent native records.
    shared_data = {other.data for other in bpy.data.objects
                   if other not in scope_set and getattr(other, 'data', None) is not None}
    data_copies = {}
    for obj in scoped_objects:
        data = getattr(obj, "data", None)
        if data not in shared_data:
            continue
        if data not in data_copies:
            data_copies[data] = data.copy()
        obj.data = data_copies[data]

    for obj in scoped_objects:
        # VIS and damage links use native part and bone names, which may differ
        # from Blender's shortened or automatically suffixed display names.
        stats["properties"] += clear_ugx_id_properties(obj, _PORTABLE_OBJECT_PROPERTIES)
        data = getattr(obj, "data", None)
        stats["properties"] += clear_ugx_id_properties(data)
        if getattr(obj, "type", "") == "ARMATURE" and data is not None:
            for bone in getattr(data, "bones", []):
                stats["properties"] += clear_ugx_id_properties(bone, _PORTABLE_BONE_PROPERTIES)
            for pose_bone in getattr(getattr(obj, "pose", None), "bones", []) or []:
                stats["properties"] += clear_ugx_id_properties(pose_bone)
        if getattr(obj, "type", "") == "MESH" and data is not None and hasattr(data, "attributes"):
            attribute_names = [attribute.name for attribute in data.attributes if attribute.name.startswith("UGX_")]
            for name in attribute_names:
                data.attributes.remove(data.attributes[name])
                stats["attributes"] += 1

    return stats


def object_scope_has_ugx_data(objects: Iterable[bpy.types.Object]) -> bool:
    for obj in objects:
        datablocks = [obj, getattr(obj, "data", None)]
        if any(block is not None and any(str(key).startswith("hwek_ugx_") and key not in _PORTABLE_OBJECT_PROPERTIES for key in block.keys()) for block in datablocks):
            return True
        if getattr(obj, "type", "") == "MESH" and hasattr(obj.data, "attributes"):
            if any(attribute.name.startswith("UGX_") for attribute in obj.data.attributes):
                return True
    return False
