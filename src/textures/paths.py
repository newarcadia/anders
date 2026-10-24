"""Resolve native asset roots and texture resource paths."""

from __future__ import annotations
from typing import Any, Dict, Tuple
import os

__all__ = ('art_root_for', 'data_root_for', 'sibling_art_root_for_data_root', 'default_hw2_dump_data_root', 'disabled_texture_path', 'relative_texture_stem', 'data_texture_resource_stem', 'selected_texture_art_stem', 'texture_path_hint_for_slot')

def art_root_for(source_path: str) -> Tuple[str, str]:
    current = os.path.dirname(os.path.abspath(source_path))
    while True:
        if os.path.basename(current).lower() == "art":
            return current, os.path.dirname(current)
        parent = os.path.dirname(current)
        if parent == current:
            return "", ""
        current = parent


def data_root_for(source_path: str) -> str:
    current = os.path.dirname(os.path.abspath(source_path))
    while True:
        if os.path.basename(current).lower() == "data":
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return ""
        current = parent


def sibling_art_root_for_data_root(data_root: str) -> str:
    if not data_root:
        return ""
    candidate = os.path.join(os.path.dirname(os.path.abspath(data_root)), "art")
    return candidate if os.path.isdir(candidate) else ""


def default_hw2_dump_data_root() -> str:
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if not local_app_data:
        return ""
    return os.path.join(local_app_data, "Packages", "Microsoft.HoganThreshold_8wekyb3d8bbwe", "TempState", "DUMP", "data")


def disabled_texture_path(texture_path: str) -> bool:
    return texture_path.strip().lower() in ("", "0", "null", "none")


def relative_texture_stem(path: str, root: str) -> str:
    if not root:
        return ""
    abs_path = os.path.abspath(path)
    abs_root = os.path.abspath(root)
    try:
        if os.path.commonpath([os.path.normcase(abs_path), os.path.normcase(abs_root)]) != os.path.normcase(abs_root):
            return ""
    except ValueError:
        return ""
    return os.path.splitext(os.path.relpath(abs_path, abs_root))[0].replace(os.sep, "\\")


def data_texture_resource_stem(relative: str) -> str:
    normalized = relative.replace("/", "\\").lstrip("\\")
    prefix = "textures\\"
    if normalized.lower().startswith(prefix):
        return normalized[len(prefix):]
    return normalized


def selected_texture_art_stem(path: str) -> str:
    art_root, _ = art_root_for(path)
    if art_root:
        return relative_texture_stem(path, art_root)
    return ""


def texture_path_hint_for_slot(slot: Dict[str, Any]) -> str:
    path = str(slot.get("path", ""))
    if path:
        return path
    return str(slot.get("path_prefix", ""))
