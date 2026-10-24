"""Find texture files across mod folders and extracted game archives."""

from __future__ import annotations
from typing import Dict, List, Tuple
import os
import re
from ..materials.data import hogan_texture_usage, texture_usage_from_suffix, v4_material_usage
from .context import IMAGE_TEXTURE_EXTENSIONS
from .paths import art_root_for, data_root_for, sibling_art_root_for_data_root
from ..assets.paths import asset_resource_roots

__all__ = ('TEXTURE_LOOKUP_CACHE', 'texture_path_variants', 'relaxed_texture_lookup_key', 'texture_lookup_for', 'best_texture_lookup_match', 'resolve_texture_from_data_index', 'resolve_texture_path', 'texture_file_with_matching_case', 'discovered_texture_usage', 'texture_set_stem', 'texture_set_usage_candidates')

TEXTURE_LOOKUP_CACHE: Dict[str, Tuple[Dict[str, List[str]], Dict[str, List[str]]]] = {}


def texture_path_variants(path: str) -> List[str]:
    normalized = path.replace("\\", "/").lstrip("/")
    if not normalized:
        return []
    root, ext = os.path.splitext(normalized)
    return [normalized] if ext else [root + extension for extension in IMAGE_TEXTURE_EXTENSIONS]


def relaxed_texture_lookup_key(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path.replace("\\", "/")))[0].lower()
    return re.sub(r"_+(?=\[[^\]]+\]$)", "", stem)


def texture_lookup_for(data_root: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    key = os.path.normcase(os.path.abspath(data_root))
    cached = TEXTURE_LOOKUP_CACHE.get(key)
    if cached is not None:
        return cached
    exact: Dict[str, List[str]] = {}
    relaxed: Dict[str, List[str]] = {}
    textures_root = os.path.join(data_root, "textures")
    if os.path.isdir(textures_root):
        for base, _, names in os.walk(textures_root):
            for name in names:
                if not name.lower().endswith(IMAGE_TEXTURE_EXTENSIONS):
                    continue
                path = os.path.join(base, name)
                stem = os.path.splitext(name)[0].lower()
                exact.setdefault(stem, []).append(path)
                relaxed.setdefault(relaxed_texture_lookup_key(name), []).append(path)
    for values in list(exact.values()) + list(relaxed.values()):
        values.sort(key=lambda value: value.lower())
    cached = (exact, relaxed)
    TEXTURE_LOOKUP_CACHE[key] = cached
    return cached


def best_texture_lookup_match(candidates: List[str], texture_path: str) -> str:
    candidates = [candidate for candidate in candidates if os.path.isfile(candidate)]
    if not candidates:
        return ""
    hint_parts = {
        part
        for part in texture_path.replace("/", "\\").lower().split("\\")
        if part and part not in {"tests", "test", "placeholder_swatches", "thom_armer", "francisco_alcalde"}
    }
    return min(
        candidates,
        key=lambda candidate: (
            -sum(1 for part in hint_parts if part in candidate.lower()),
            len(candidate),
            candidate.lower(),
        ),
    )


def resolve_texture_from_data_index(data_root: str, texture_path: str) -> str:
    exact, relaxed = texture_lookup_for(data_root)
    stem = os.path.splitext(os.path.basename(texture_path.replace("\\", "/")))[0].lower()
    candidates = exact.get(stem, [])
    if candidates:
        return best_texture_lookup_match(candidates, texture_path)
    key = relaxed_texture_lookup_key(texture_path)
    candidates = relaxed.get(key, [])
    if candidates:
        return best_texture_lookup_match(candidates, texture_path)
    return ""


def resolve_texture_path(source_path: str, texture_path: str) -> str:
    art_root, scenario_root = art_root_for(source_path)
    data_root = data_root_for(source_path)
    sibling_art_root = sibling_art_root_for_data_root(data_root)
    source_dir = os.path.dirname(os.path.abspath(source_path))
    candidates: List[str] = []
    for variant in texture_path_variants(texture_path):
        native = variant.replace("/", os.sep)
        if os.path.isabs(native):
            candidates.append(native)
        candidates.append(os.path.join(source_dir, native))
        candidates.append(os.path.join(source_dir, os.path.basename(native)))
        if scenario_root:
            candidates.append(os.path.join(scenario_root, native))
        if art_root:
            candidates.append(os.path.join(art_root, native[4:] if native.lower().startswith("art" + os.sep) else native))
        if data_root:
            data_relative = native[5:] if native.lower().startswith("data" + os.sep) else native
            candidates.append(os.path.join(data_root, data_relative))
            if not data_relative.lower().startswith("textures" + os.sep):
                candidates.append(os.path.join(data_root, "textures", data_relative))
        if sibling_art_root:
            candidates.append(os.path.join(sibling_art_root, native.lstrip("\\/")))
    seen = set()
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(candidate))
        if key in seen:
            continue
        seen.add(key)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    for root in asset_resource_roots(source_path):
        for variant in texture_path_variants(texture_path):
            relative = variant
            prefix = root.name.casefold() + '/'
            if relative.casefold().startswith(prefix):
                relative = relative[len(prefix):]
            relatives = [relative]
            if root.name.casefold() == 'data' and not relative.casefold().startswith('textures/'):
                relatives.append('textures/' + relative)
            for relative in relatives:
                candidate = texture_file_with_matching_case(str(root), relative)
                if candidate:
                    return candidate
    if os.path.isdir(source_dir):
        names = {name.lower(): name for name in os.listdir(source_dir)}
        for variant in texture_path_variants(texture_path):
            name = names.get(os.path.basename(variant).lower())
            if name and os.path.isfile(os.path.join(source_dir, name)):
                return os.path.join(source_dir, name)
    if data_root:
        resolved = resolve_texture_from_data_index(data_root, texture_path)
        if resolved:
            return os.path.abspath(resolved)
    if art_root:
        shared_root = os.path.dirname(scenario_root)
        for archive in ("scenarioshared", "root"):
            root = os.path.join(shared_root, archive, "art")
            if not os.path.isdir(root) or os.path.abspath(root) == os.path.abspath(art_root):
                continue
            for variant in texture_path_variants(texture_path):
                relative = variant.replace("\\", "/").lstrip("/")
                if relative.lower().startswith("art/"):
                    relative = relative[4:]
                candidate = texture_file_with_matching_case(root, relative)
                if candidate:
                    return candidate
    return ""


def texture_file_with_matching_case(root: str, relative: str) -> str:
    path = root
    for part in relative.replace("\\", "/").split("/"):
        if not part or part in (".", ".."):
            return ""
        candidate = os.path.join(path, part)
        if not os.path.exists(candidate):
            if not os.path.isdir(path):
                return ""
            matches = [name for name in os.listdir(path) if name.casefold() == part.casefold()]
            if len(matches) != 1:
                return ""
            candidate = os.path.join(path, matches[0])
        path = candidate
    return os.path.abspath(path) if os.path.isfile(path) else ""


def discovered_texture_usage(path: str) -> str:
    usage = texture_usage_from_suffix(path) or hogan_texture_usage(path)
    if usage and usage != "texture":
        return v4_material_usage(usage)
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    for suffix, usage in (("albedo", "diffuse"), ("diffuse", "diffuse"), ("basecolor", "diffuse"),
                          ("normal", "normal"), ("specular", "specular"), ("emissive", "emissive"),
                          ("packedmask", "packedmask"), ("masks", "packedmask")):
        if stem == suffix or stem.endswith(("_" + suffix, "-" + suffix)):
            return usage
    return ""


def texture_set_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path.replace("\\", "/")))[0].lower()
    return re.sub(r"(?:[_-](?:df|nm|sp|em|ef|ao|op|gl|rm|xf|en|ds|hl|ml|albedo|diffuse|basecolor|normal|specular|emissive|packedmask|masks)|_?\[(?:al|nm|sp|em|pm)\])$", "", stem)


def texture_set_usage_candidates(directory: str) -> Dict[str, str]:
    candidates: Dict[str, str] = {}
    if not os.path.isdir(directory):
        return candidates
    for name in sorted(os.listdir(directory), key=str.lower):
        if not name.lower().endswith(IMAGE_TEXTURE_EXTENSIONS):
            continue
        usage = v4_material_usage(texture_usage_from_suffix(name) or hogan_texture_usage(name))
        if usage == "texture":
            continue
        if not usage:
            continue
        candidates.setdefault(usage, os.path.join(directory, name))
    return candidates
