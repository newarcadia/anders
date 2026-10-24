"""Retain texture provenance and determine native slot semantics."""

from typing import Optional
import bpy
import os
from .paths import (
    art_root_for,
    data_root_for,
    data_texture_resource_stem,
    relative_texture_stem,
    selected_texture_art_stem,
)

__all__ = ('NON_COLOR_TEXTURE_USAGES', 'texture_usage_is_non_color', 'IMAGE_TEXTURE_EXTENSIONS', 'IMAGE_TEXTURE_FILTER', 'ugx_texture_context_path', 'ensure_ugx_texture_context_from_image', 'ugx_texture_path_from_image', 'replacement_texture_path_for_image', 'ugx_texture_resource_key')

NON_COLOR_TEXTURE_USAGES = {
    "normal", "gloss", "roughness", "opacity", "packedmask", "specular", "ao", "effect",
    "envmask", "xform", "emxform", "distortion", "highlight", "modulate",
}


def texture_usage_is_non_color(usage, cache_version=0):
    if cache_version == 4:
        return usage.lower() not in {'diffuse', 'albedo', 'gloss', 'specular', 'opacity', 'xform', 'emxform', 'effect', 'envmask'}
    return usage.lower() in NON_COLOR_TEXTURE_USAGES


IMAGE_TEXTURE_EXTENSIONS = (".ddx", ".dds", ".xpr") + tuple(sorted(bpy.path.extensions_image - {".dds"}))


IMAGE_TEXTURE_FILTER = ";".join("*" + ext for ext in IMAGE_TEXTURE_EXTENSIONS)


def ugx_texture_context_path(obj: Optional[bpy.types.Object]) -> str:
    if obj is None:
        return ""
    context_path = str(obj.get("hwek_ugx_texture_context_path", ""))
    return context_path or str(obj.get("hwek_ugx_source_path", ""))


def ensure_ugx_texture_context_from_image(obj: Optional[bpy.types.Object], image: Optional[bpy.types.Image]):
    if obj is None or image is None:
        return
    current = ugx_texture_context_path(obj)
    if current and art_root_for(current)[0]:
        return
    source = str(image.get("hwek_ugx_ddx_path", "")) or getattr(image, "filepath_raw", "") or getattr(image, "filepath", "")
    if source and art_root_for(source)[0]:
        obj["hwek_ugx_texture_context_path"] = os.path.abspath(source)


def ugx_texture_path_from_image(source_path: str, original_texture_path: str, image: Optional[bpy.types.Image]) -> str:
    if not image:
        return ""
    value = image.get("hwek_ugx_texture_path")
    if value:
        value = str(value)
        root, extension = os.path.splitext(value)
        return root if extension.lower() in IMAGE_TEXTURE_EXTENSIONS else value
    raw_path = getattr(image, "filepath_raw", "") or getattr(image, "filepath", "")
    if not raw_path:
        return ""
    file_path = os.path.abspath(bpy.path.abspath(raw_path))
    if os.path.splitext(file_path)[1].lower() not in IMAGE_TEXTURE_EXTENSIONS:
        return ""
    original = original_texture_path.replace("/", "\\")
    art_root, _ = art_root_for(source_path)
    relative = relative_texture_stem(file_path, art_root)
    if relative:
        return ("\\" if original.startswith("\\") else "") + relative
    data_root = data_root_for(source_path)
    source_data_relative = relative_texture_stem(file_path, data_root)
    if source_data_relative:
        return ("\\" if original.startswith("\\") else "") + data_texture_resource_stem(source_data_relative)
    selected_relative = selected_texture_art_stem(file_path)
    if selected_relative:
        return ("\\" if original.startswith("\\") else "") + selected_relative
    selected_data_root = data_root_for(file_path)
    selected_data_relative = relative_texture_stem(file_path, selected_data_root)
    if selected_data_relative:
        return ("\\" if original.startswith("\\") else "") + data_texture_resource_stem(selected_data_relative)
    source_relative = relative_texture_stem(file_path, os.path.dirname(os.path.abspath(source_path)))
    if source_relative:
        original_dir = os.path.dirname(original)
        return (original_dir + "\\" if original_dir else "") + os.path.basename(source_relative)
    return ""


def replacement_texture_path_for_image(source_path: str, original_texture_path: str, image: bpy.types.Image) -> str:
    value = ugx_texture_path_from_image(source_path, original_texture_path, image)
    if value:
        return value
    source = str(image.get("hwek_ugx_ddx_path", "")) or getattr(image, "filepath_raw", "") or getattr(image, "filepath", "")
    stem = os.path.splitext(os.path.basename(source.replace("\\", "/")))[0]
    if not stem:
        return ""
    original = original_texture_path.replace("/", "\\")
    original_dir = os.path.dirname(original)
    return (original_dir + "\\" if original_dir else "") + stem


def ugx_texture_resource_key(path: str) -> str:
    value = str(path or "").replace("/", "\\").strip().lower()
    root, ext = os.path.splitext(value)
    if ext.lower() in IMAGE_TEXTURE_EXTENSIONS:
        value = root
    return value.lstrip("\\")
