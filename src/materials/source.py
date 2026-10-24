"""Find retained HW1 material settings for replacement textures."""

from __future__ import annotations
from typing import List, Optional, Dict, Tuple
import os
from ..textures.context import ugx_texture_resource_key
from ..textures.discovery import resolve_texture_path
from ..ugx.read import parse_ugx
from .data import Material, v4_material_usage

V4_SOURCE_MATERIAL_LOOKUP_CACHE: Dict[Tuple[str, str, str], Optional["Material"]] = {}




__all__ = ('V4_SOURCE_MATERIAL_LOOKUP_CACHE', 'v4_texture_source_ugx_candidates', 'find_v4_source_material_for_texture')

def v4_texture_source_ugx_candidates(texture_file: str) -> List[str]:
    folder = os.path.dirname(os.path.abspath(texture_file))
    if not os.path.isdir(folder):
        return []
    texture_stem = os.path.splitext(os.path.basename(texture_file))[0].lower()
    root = texture_stem
    for suffix in ("_df", "_nm", "_xf", "_sp", "_gl", "_op", "_em", "_ef", "_ao", "_rm", "_en", "_ds", "_hl", "_ml"):
        if root.endswith(suffix):
            root = root[:-len(suffix)]
            break
    candidates = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith(".ugx")
    ]
    candidates.sort(key=lambda item: (os.path.splitext(os.path.basename(item))[0].lower() != root, os.path.basename(item).lower()))
    return candidates


def find_v4_source_material_for_texture(source_path: str, texture_path: str, usage: str) -> Optional[Material]:
    usage = v4_material_usage(usage)
    target_key = ugx_texture_resource_key(texture_path)
    resolved = resolve_texture_path(source_path, texture_path)
    if not resolved or not target_key:
        return None
    cache_key = (os.path.normcase(os.path.abspath(resolved)), target_key, usage)
    if cache_key in V4_SOURCE_MATERIAL_LOOKUP_CACHE:
        return V4_SOURCE_MATERIAL_LOOKUP_CACHE[cache_key]
    result: Optional[Material] = None
    for candidate in v4_texture_source_ugx_candidates(resolved):
        try:
            model = parse_ugx(candidate)
        except Exception:
            continue
        if model.cache_version != 4:
            continue
        for material in model.materials:
            for slot in material.slots:
                if v4_material_usage(slot.get("usage", "")) != usage:
                    continue
                if ugx_texture_resource_key(str(slot.get("path", ""))) == target_key:
                    result = material
                    break
            if result is not None:
                break
        if result is not None:
            break
    V4_SOURCE_MATERIAL_LOOKUP_CACHE[cache_key] = result
    return result
