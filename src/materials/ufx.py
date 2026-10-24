"""Resolve UFX shader files and select material permutations."""

from typing import Any, Optional, Tuple
import bpy
import os
from ..errors import UgxError
from ..formats.ufx import ufx_shader_header_hash
from ..textures.paths import data_root_for, default_hw2_dump_data_root
from ..ui.preferences import configured_ufx_shader_directory
from .data import parse_ufx_shader_hash
from .properties import ufx_attr_property_name, ufx_shader_file_property_name

__all__ = ('ufx_shader_path_for_name', 'ufx_permutation_status', 'material_ufx_permutation_status', 'set_material_ufx_shader_pass_from_file')

def ufx_shader_path_for_name(source_path: str, permutation_name: str) -> str:
    name = str(permutation_name or "").strip().lower()
    if not name:
        return ""
    if not name.endswith(".ufx"):
        name += ".ufx"
    roots = []
    data_root = data_root_for(source_path) if source_path else ""
    if data_root:
        roots.append(data_root)
    configured_root = configured_ufx_shader_directory()
    if configured_root:
        roots.append(configured_root)
    default_root = default_hw2_dump_data_root()
    if default_root:
        roots.append(default_root)
    seen = set()
    for root in roots:
        key = os.path.normcase(os.path.abspath(root))
        if key in seen:
            continue
        seen.add(key)
        for relative in (name, os.path.join("shaders", "dx12", "asset_shaders", name)):
            candidate = os.path.join(root, relative)
            if os.path.isfile(candidate):
                return candidate
    return ""


def ufx_permutation_status(source_path: str, name: str, expected_hash: Any) -> Tuple[str, str]:
    if not name:
        return "empty", "QUESTION"
    path = ufx_shader_path_for_name(source_path, name)
    if not path:
        return "missing UFX", "ERROR"
    header_hash = ufx_shader_header_hash(path)
    if header_hash is None:
        return "invalid UFX", "ERROR"
    if expected_hash in ("", None):
        expected = None
    else:
        try:
            expected = parse_ufx_shader_hash(expected_hash)
        except UgxError:
            return "invalid hash", "ERROR"
    if expected is not None and header_hash != expected:
        return "hash mismatch", "ERROR"
    return "UFX ok", "CHECKMARK"


def material_ufx_permutation_status(
    source_path: str,
    mat: Optional[bpy.types.Material],
    index: int,
    name: str,
    expected_hash: Any,
) -> Tuple[str, str]:
    status, icon = ufx_permutation_status(source_path, name, expected_hash)
    if status != "missing UFX" or mat is None:
        return status, icon
    selected_path = str(mat.get(ufx_shader_file_property_name(index), "")).strip()
    if not selected_path or not os.path.isfile(selected_path):
        return status, icon
    selected_name = os.path.splitext(os.path.basename(selected_path))[0]
    if selected_name.lower() != str(name or "").strip().lower():
        return status, icon
    header_hash = ufx_shader_header_hash(selected_path)
    if header_hash is None:
        return "invalid selected UFX", "ERROR"
    if expected_hash in ("", None):
        expected = None
    else:
        try:
            expected = parse_ufx_shader_hash(expected_hash)
        except UgxError:
            return "invalid hash", "ERROR"
    if expected is not None and header_hash != expected:
        return "hash mismatch", "ERROR"
    return "selected UFX; deploy it", "INFO"


def set_material_ufx_shader_pass_from_file(
    mat: bpy.types.Material,
    index: int,
    path: str,
) -> Tuple[str, int]:
    if mat is None:
        raise UgxError("Select a Halo Wars 2 UGX material first.")
    if index < 0 or index > 3:
        raise UgxError(f"Halo Wars 2 UFX pass {index} is outside the supported range 0-3.")
    absolute_path = os.path.abspath(path)
    if not os.path.isfile(absolute_path):
        raise UgxError(f"UFX file does not exist: {absolute_path}")
    header_hash = ufx_shader_header_hash(absolute_path)
    if header_hash is None:
        raise UgxError("The selected file is not a Halo Wars 2 UFXS shader or has an incomplete header.")
    permutation_name = os.path.splitext(os.path.basename(absolute_path))[0]
    if not permutation_name:
        raise UgxError("The selected UFX has no permutation name.")
    mat[ufx_attr_property_name(f"shaderPermutationName{index}")] = permutation_name
    mat[ufx_attr_property_name(f"shaderPermutationHash{index}")] = str(header_hash)
    mat[ufx_shader_file_property_name(index)] = absolute_path
    mat.update_tag()
    return permutation_name, header_hash
