"""Resolve VIS model and animation paths across extracted game archives."""
from pathlib import Path
from ..formats.vis import read_visual_document
from ..assets.paths import asset_resource_roots

__all__ = ('visual_resource_roots', 'visual_resource_name', 'resolve_visual_resource', 'find_visual_documents', 'visual_model_names', 'visual_animation_references')


def visual_resource_roots(path):
    return asset_resource_roots(path)


def visual_resource_name(path):
    path = Path(path).absolute()
    for root in visual_resource_roots(path):
        if path.is_relative_to(root):
            return path.relative_to(root).with_suffix('').as_posix().replace('/', '\\')
    return path.stem


def resource_key(value):
    return str(value).replace('\\', '/').strip('/').removesuffix('.ugx').casefold()


def path_with_case(path):
    if path.is_file():
        return path
    current = Path(path.anchor)
    for part in path.parts[1:]:
        direct = current / part
        if direct.exists():
            current = direct
            continue
        if not current.is_dir():
            return None
        matches = [item for item in current.iterdir() if item.name.casefold() == part.casefold()]
        if len(matches) != 1:
            return None
        current = matches[0]
    return current if current.is_file() else None


def resolve_visual_resource(source, reference, extension):
    relative = str(reference).strip().replace('\\', '/').lstrip('/')
    if not relative:
        return None
    if not relative.casefold().endswith(extension.casefold()):
        relative += extension
    candidates = [Path(source).absolute().parent / relative]
    for root in visual_resource_roots(source):
        prefix = root.name.casefold() + '/'
        candidates.append(root / (relative[len(prefix):] if relative.casefold().startswith(prefix) else relative))
    # Bare filenames are local references, but full resource paths never fall
    # back to a same-named file from a different asset directory.
    for path in dict.fromkeys(candidates):
        found = path_with_case(path)
        if found is not None:
            return found
    return None


def visual_model_names(root, ugx_path):
    expected = resource_key(visual_resource_name(ugx_path))
    names = []
    for model in root.findall('model'):
        for asset in model.findall('./component/asset'):
            if asset.get('type', '').casefold() == 'model' and resource_key(asset.findtext('file', '')) == expected:
                names.append(model.get('name', ''))
                break
    return names


def find_visual_documents(ugx_path):
    source = Path(ugx_path).absolute()
    directories = [source.parent]
    for root in visual_resource_roots(source):
        for base in visual_resource_roots(source):
            if source.is_relative_to(base):
                directories.append(root / source.relative_to(base).parent)
    matches, failures = [], []
    for directory in dict.fromkeys(directories):
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if not path.name.casefold().endswith(('.vis', '.vis.xmb')):
                continue
            try:
                document = read_visual_document(path.read_bytes())
                if visual_model_names(document.root, source):
                    matches.append(path)
            except (OSError, ValueError, RuntimeError) as exc:
                failures.append(f'{path.name}: {exc}')
        if matches:
            break
    return matches, failures


def visual_animation_references(root, names):
    references = []
    for model in root.findall('model'):
        if model.get('name', '') not in names:
            continue
        for animation in model.findall('anim'):
            for asset in animation.findall('asset'):
                if asset.get('type', '').casefold() in ('anim', 'granny'):
                    reference = asset.findtext('file', '').strip()
                    if reference:
                        references.append((reference, animation.get('type', ''), animation.get('exitAction', 'Loop'), model.get('name', '')))
    return references
