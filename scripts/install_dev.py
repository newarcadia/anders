"""Copy the addon into a dedicated Blender development extension repository."""
import argparse
from pathlib import Path
import shutil
import tempfile

from package import extension_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True, help='Directory of a separate local development extension repository')
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    manifest, files = extension_files(source)
    repository = args.repository.expanduser().resolve()
    if repository == source or repository.is_relative_to(source):
        parser.error('The development repository must be outside the source checkout')
    repository.mkdir(parents=True, exist_ok=True)
    destination = repository / manifest['id']
    resolved = destination.resolve()
    if resolved == source or source.is_relative_to(resolved) or resolved.is_relative_to(source):
        parser.error('The development copy must not contain or replace the source checkout')
    with tempfile.TemporaryDirectory(dir=repository, prefix='anders-dev-') as temporary:
        staged = Path(temporary) / manifest['id']
        for path in files:
            target = staged / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        # Keep the previous development install recoverable until copying succeeds.
        backup = Path(temporary) / 'previous'
        if destination.exists():
            destination.rename(backup)
        try:
            staged.rename(destination)
        except OSError:
            if backup.exists():
                backup.rename(destination)
            raise
    print(f'Development addon: {destination}')


if __name__ == '__main__':
    main()
