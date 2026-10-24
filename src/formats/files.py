"""Stage asset exports and restore previous files if replacement fails."""

from __future__ import annotations
from typing import List, Tuple
import os
import shutil
import tempfile
from ..errors import UgxError

__all__ = ('write_ugx_export_files',)


def file_matches_payload(path, payload):
    try:
        handle = open(path, 'rb')
    except FileNotFoundError:
        return False
    with handle:
        if os.fstat(handle.fileno()).st_size != len(payload):
            return False
        view = memoryview(payload)
        for offset in range(0, len(payload), 1024 * 1024):
            chunk = view[offset:offset + 1024 * 1024]
            if handle.read(len(chunk)) != chunk:
                return False
        return True


def write_ugx_export_files(outputs: List[Tuple[str, bytes]]):
    destinations = {}
    for path, payload in outputs:
        path = os.path.abspath(path)
        key = os.path.normcase(path).casefold()
        previous = destinations.get(key)
        if previous is not None and previous[1] != payload:
            raise UgxError(f'Export contains different data for {path}. Give these resources separate paths.')
        if previous is None:
            destinations[key] = (path, payload)
    staged = []
    replaced = []
    retain_backups = False
    try:
        for path, payload in destinations.values():
            if file_matches_payload(path, payload):
                continue
            directory = os.path.dirname(path)
            os.makedirs(directory, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".ugx-export-", dir=directory)
            staged.append([path, temporary, None])
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
            if os.path.isfile(path):
                fd, backup = tempfile.mkstemp(prefix=".ugx-backup-", dir=directory)
                os.close(fd)
                staged[-1][2] = backup
                os.remove(backup)
                try:
                    # Both names share the old bytes until the export commits.
                    os.link(path, backup)
                except OSError:
                    shutil.copy2(path, backup)
                shutil.copymode(path, temporary)
        # Stage every file before replacing any member of the export.
        for item in staged:
            os.replace(item[1], item[0])
            replaced.append(item)
    except Exception as exc:
        failures = []
        for path, _, backup in reversed(replaced):
            try:
                if backup is None:
                    os.remove(path)
                else:
                    os.replace(backup, path)
            except OSError as restore_error:
                failures.append(f"{path}: {restore_error}; backup: {backup or '<new file>'}")
        if failures:
            retain_backups = True
            raise UgxError(f"Export failed ({exc}); restoring previous files failed: " + "; ".join(failures)) from exc
        raise
    finally:
        for _, temporary, backup in staged:
            for path in (temporary, None if retain_backups else backup):
                if path is not None and os.path.isfile(path):
                    os.remove(path)
