"""Extend HW2 material tables for newly assigned Blender slots."""

from typing import Dict, Iterable, List
import bpy
import json
from ..errors import UgxError
from ..ugx.data import UgxModel
from ..ugx.scene import ugx_armature_for, ugx_used_material_slots
from .context import material_origin_matches_model
from .hw2 import parsed_v6_material_from_blender

__all__ = ('extend_v6_material_model_for_objects',)

def extend_v6_material_model_for_objects(model: UgxModel, objects: Iterable[bpy.types.Object]):
    """Assign stable table indices to new Blender slots and extend the v6 BDT model."""
    if model.cache_version != 6:
        return
    def parsed_material(mat, obj):
        parsed=parsed_v6_material_from_blender(mat,bool(ugx_armature_for(obj)))
        index=int(mat.get("hwek_ugx_material_index",-1))
        if 0<=index<len(model.materials) and material_origin_matches_model(obj,mat,model,index):
            parsed.bdt_data=model.materials[index].bdt_data
            parsed.bdt_index=model.materials[index].bdt_index
            parsed.has_texture_array=model.materials[index].has_texture_array
            parsed.source_texture_paths=model.materials[index].source_texture_paths
            source = model.materials[index]
            parsed.native_name=source.native_name if parsed.name == source.name else parsed.name
        return parsed

    new_material_indices: Dict[int, int] = {}
    target_material_keys: Dict[int, int] = {}
    for obj in objects:
        if obj is None or obj.type != "MESH":
            continue
        try:
            mapping = [int(value) for value in json.loads(obj.get("hwek_ugx_material_slot_indices", "[]"))]
        except Exception:
            mapping = []
        updated: List[int] = []
        used_slots = dict(ugx_used_material_slots(obj))
        for slot_index in range(len(obj.material_slots)):
            if slot_index not in used_slots:
                updated.append(mapping[slot_index] if slot_index < len(mapping) else 0)
                continue
            mat = used_slots[slot_index]
            if mat is None:
                updated.append(mapping[slot_index] if slot_index < len(mapping) else 0)
                continue

            material_key = int(mat.as_pointer())
            if slot_index < len(mapping):
                target_index = int(mapping[slot_index])
                if target_index < 0:
                    raise UgxError(f"Material slot {slot_index} has a negative UGX material index.")
                if target_index > len(model.materials):
                    raise UgxError(
                        f"Material slot {slot_index} leaves a gap before UGX material {target_index}."
                    )
                if target_index == len(model.materials):
                    model.materials.append(
                        parsed_material(mat,obj)
                    )
                elif not material_origin_matches_model(obj,mat,model,target_index):
                    model.materials[target_index]=parsed_material(mat,obj)
                previous_key = target_material_keys.get(target_index)
                if previous_key is not None and previous_key != material_key:
                    if not mat.get("hwek_ugx_native_material"):
                        raise UgxError(f"Multiple Blender materials map to new UGX material {target_index}.")
                    target_index = new_material_indices.get(material_key, -1)
                    if target_index < 0:
                        target_index = len(model.materials)
                        model.materials.append(parsed_material(mat,obj))
                        new_material_indices[material_key] = target_index
                target_material_keys[target_index] = material_key
                updated.append(target_index)
                continue

            source_index = int(mat.get("hwek_ugx_material_index", -1))
            if (
                0 <= source_index < len(model.materials)
                and material_origin_matches_model(obj, mat, model, source_index)
            ):
                previous_key = target_material_keys.get(source_index)
                if previous_key is None or previous_key == material_key:
                    target_material_keys[source_index] = material_key
                    updated.append(source_index)
                    continue

            target_index = new_material_indices.get(material_key, -1)
            if target_index < 0:
                target_index = len(model.materials)
                model.materials.append(
                    parsed_material(mat,obj)
                )
                new_material_indices[material_key] = target_index
                target_material_keys[target_index] = material_key
            updated.append(target_index)
        obj["hwek_ugx_material_slot_indices"] = json.dumps(updated)
