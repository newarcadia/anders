"""Resolve native mesh names, accessory names and LOD labels."""

from typing import List, Tuple
import bpy
import math
import os
import re
from ..formats.granny import granny_mesh_for_model_index, safe_grx_name
from .data import Section, UgxModel
from .skin import vertex_group_from_section_bone, vertex_group_from_source_bone

__all__ = ('section_lod_key', 'lod_value_label', 'lod_object_name', 'granny_mesh_name_for_accessory', 'section_accessory_index', 'section_mesh_indices', 'imported_mesh_object_name', 'sibling_damage_xmb', 'current_ugx_mesh_name')

def section_lod_key(section: Section) -> Tuple[float, float]:
    return (float(section.lod_min_distance), float(section.lod_max_distance))


def lod_value_label(value: float) -> str:
    if not math.isfinite(value) or value > 3.0e38:
        return "inf"
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_]+", "_", text)


def lod_object_name(base_name: str, lod_index: int, lod_key: Tuple[float, float], total_lods: int) -> str:
    if total_lods <= 1:
        return base_name
    return f"{base_name}_LOD{lod_index}_{lod_value_label(lod_key[0])}_{lod_value_label(lod_key[1])}"


def granny_mesh_name_for_accessory(model: UgxModel, accessory_index: int, base_name: str = "") -> str:
    if accessory_index >= 0 and model.granny_info.models:
        mesh_names = model.granny_info.models[0].mesh_names
        if accessory_index < len(mesh_names):
            return safe_grx_name(mesh_names[accessory_index], "")
    if accessory_index >= 0 and accessory_index < len(model.granny_info.meshes):
        return safe_grx_name(model.granny_info.meshes[accessory_index].name, "")
    return base_name or (f"mesh_{accessory_index}" if accessory_index >= 0 else "mesh")


def section_accessory_index(model: UgxModel, section_index: int) -> int:
    direct = int(model.sections[section_index].accessory_index)
    memberships = [
        accessory_index
        for accessory_index, accessory in enumerate(model.accessories)
        if section_index in accessory.section_indices
    ]
    if direct in memberships:
        return direct
    if len(memberships) == 1:
        return memberships[0]
    return direct


def section_mesh_indices(model: UgxModel) -> List[int]:
    indices = [section_accessory_index(model, index) for index in range(len(model.sections))]
    if model.cache_version != 4 or not model.granny_info.models:
        return indices
    names = model.granny_info.models[0].mesh_names
    if len(names) <= len(model.accessories):
        return indices
    # HW1 can flatten several Granny meshes into one visibility accessory.
    # Recover their identity only when every section has an unambiguous binding.
    bindings = []
    for index in range(len(names)):
        mesh = granny_mesh_for_model_index(model.granny_info, index)
        bindings.append({binding.bone_name for binding in mesh.bone_bindings} if mesh else set())
    resolved = []
    for section in model.sections:
        required = set()
        if section.rigid_only:
            bone_index = vertex_group_from_section_bone(section, section.rigid_bone_index, len(model.bones))
            if bone_index is not None:
                required.add(model.bones[bone_index].name)
        else:
            for vertex in section.vertices:
                for encoded, weight in zip(vertex.bone_indices, vertex.bone_weights):
                    if weight <= 0.00001:
                        continue
                    bone_index = vertex_group_from_source_bone(section, encoded, len(model.bones))
                    if bone_index is not None:
                        required.add(model.bones[bone_index].name)
        candidates = [index for index, bound in enumerate(bindings)
                      if required and required.issubset(bound)]
        if len(candidates) != 1:
            return indices
        resolved.append(candidates[0])
    return resolved


def imported_mesh_object_name(
    base_name: str,
    mesh_name: str,
    mesh_index: int,
    lod_index: int,
    lod_key: Tuple[float, float],
    total_lods: int,
) -> str:
    if mesh_index < 0:
        name = base_name
    else:
        name = mesh_name or f"{base_name}_mesh_{mesh_index}"
    return lod_object_name(name, lod_index, lod_key, total_lods)


def sibling_damage_xmb(ugx_path: str, cache_version=None) -> str:
    folder = os.path.dirname(os.path.abspath(ugx_path))
    if not os.path.isdir(folder):
        return ""
    if cache_version is None:
        from ..textures.paths import art_root_for, data_root_for
        cache_version = 4 if art_root_for(ugx_path)[0] else 6 if data_root_for(ugx_path) else 0
    xmb_candidates = sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith(".dmg.xmb") and cache_version != 4
    )
    ugx_stem = os.path.splitext(os.path.basename(ugx_path))[0].lower()
    preferred_stems = [ugx_stem]
    if ugx_stem.startswith("mesh_"):
        preferred_stems.append(ugx_stem[5:])
    for stem in preferred_stems:
        exact = [path for path in xmb_candidates if os.path.basename(path).lower() == f"{stem}.dmg.xmb"]
        if len(exact) == 1:
            return exact[0]
    if len(xmb_candidates) == 1 and cache_version != 6:
        return xmb_candidates[0]
    candidates = sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith(".dmg")
    )
    for stem in preferred_stems:
        exact = [path for path in candidates if os.path.basename(path).lower() == f"{stem}.dmg"]
        if len(exact) == 1:
            return exact[0]
    return candidates[0] if len(candidates) == 1 and cache_version != 6 else ""


def current_ugx_mesh_name(obj: bpy.types.Object) -> str:
    source = str(obj.get("hwek_ugx_mesh_name", obj.name))
    imported = str(obj.get("hwek_ugx_mesh_import_name", source))
    return obj.name if obj.name != imported else source
