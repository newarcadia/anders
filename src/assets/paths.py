"""Resolve mod files before inherited extracted game files."""
from pathlib import Path
from ..textures.paths import art_root_for, data_root_for
from ..ui.preferences import configured_asset_directory

__all__ = ('asset_resource_roots',)


def extracted_art_roots(folder):
    folder = Path(folder)
    if folder.name.casefold() == 'art':
        return [folder]
    roots = [folder / 'art']
    archives = ('root_update', 'root', 'scenarioshared')
    roots.extend(folder / archive / 'art' for archive in archives)
    if folder.is_dir():
        roots.extend(item / 'art' for item in sorted(folder.iterdir()) if item.is_dir() and item.name not in archives)
    return roots


def asset_resource_roots(path):
    art, _ = art_root_for(str(path))
    data = data_root_for(str(path))
    roots = [Path(root) for root in (art, data) if root]
    version = 4 if art else 6 if data else 0
    if art:
        roots.extend(extracted_art_roots(Path(art).parent.parent))
    folder = configured_asset_directory(version)
    if folder:
        if version == 4:
            roots.extend(extracted_art_roots(folder))
        else:
            base = Path(folder)
            roots.append(base if base.name.casefold() == 'data' else base / 'data')
    return list(dict.fromkeys(root for root in roots if root.is_dir()))
