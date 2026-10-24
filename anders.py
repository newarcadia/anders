"""Compatibility entry point for scripts using the Anders package."""
from pathlib import Path

# File-based callers must keep the package directory beside this entry point.
if not __package__:
    __package__ = __name__
    __path__ = [str(Path(__file__).resolve().parent)]

from .src.metadata import bl_info
from .src.registration import register, unregister
from .src.compatibility import lookup


def __getattr__(name):
    return lookup(name)


__all__ = ("bl_info", "register", "unregister")


if __name__ == "__main__":
    register()
