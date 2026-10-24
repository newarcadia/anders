"""Build a deterministic Anders extension archive and its release metadata."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import tomllib
import zipfile


def extension_files(source):
    manifest = tomllib.loads((source / 'blender_manifest.toml').read_text())
    files = {source / 'blender_manifest.toml'}
    for entry in manifest['build']['paths']:
        path = source / entry
        if path.is_dir():
            files.update(path.rglob('*.py'))
        elif path.is_file():
            files.add(path)
        else:
            raise FileNotFoundError(f'Extension build path is missing: {entry}')
    for path in files:
        if not path.resolve().is_relative_to(source.resolve()):
            raise ValueError(f'Extension build path leaves its source directory: {path}')
    return manifest, sorted(files, key=lambda path: path.relative_to(source).as_posix())


def validate_version(source, manifest):
    metadata = ast.parse((source / 'src/metadata.py').read_text())
    value = next(node.value for node in metadata.body
                 if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'bl_info' for target in node.targets))
    version = ast.literal_eval(value)['version']
    if '.'.join(map(str, version)) != manifest['version']:
        raise ValueError('Manifest and Blender compatibility metadata versions differ')


def package(source, output):
    source, output = source.resolve(), output.resolve()
    manifest, files = extension_files(source)
    validate_version(source, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='anders-package-') as temporary:
        archive_path = Path(temporary) / output.name
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in files:
                info = zipfile.ZipInfo(path.relative_to(source).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, path.read_bytes(), compresslevel=9)
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise ValueError('Extension archive failed its integrity check')
        payload = archive_path.read_bytes()
        metadata = {'id': manifest['id'], 'version': manifest['version'], 'archive_size': len(payload),
                    'archive_hash': 'sha256:' + hashlib.sha256(payload).hexdigest(), 'files': len(files)}
        archive_path.replace(output)
    output.with_suffix(output.suffix + '.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    manifest = tomllib.loads((args.source / 'blender_manifest.toml').read_text())
    output = args.output or args.source / 'out' / f"{manifest['id']}-{manifest['version']}.zip"
    print(json.dumps(package(args.source, output), indent=2))


if __name__ == '__main__':
    main()
