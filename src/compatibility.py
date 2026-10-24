"""Temporary access to the former single-module API."""
from importlib import import_module
from .lifecycle import MODULE_ORDER


def lookup(name):
    if name == "os":
        import os
        return os
    if name == "np":
        from .textures.compression import np
        return np
    package = __package__
    for module_name in MODULE_ORDER:
        module = import_module("." + module_name, package)
        if name in module.__all__:
            return getattr(module, name)
    raise AttributeError(f"Anders has no attribute {name!r}")
