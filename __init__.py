"""Blender extension entry point for Anders."""
from importlib import import_module, reload
import sys

from .src.lifecycle import MODULE_ORDER


_restore_registration = globals().get("_registered", False)
if "_registered" in locals():
    if _registered:
        import_module(".src.registration", __name__).unregister(keep_node_types=True)
        _registered = False
    for _name in MODULE_ORDER:
        _module = sys.modules.get(__name__ + ".src." + _name)
        if _module is not None:
            reload(_module)

from .src.metadata import bl_info

_registered = False


def register():
    global _registered
    registration = import_module(".src.registration", __name__)
    registration.register()
    _registered = True


def unregister():
    global _registered
    if not _registered:
        return
    registration = sys.modules.get(__name__ + ".src.registration")
    if registration is not None:
        registration.unregister()
    _registered = False


def __getattr__(name):
    from .src.compatibility import lookup
    return lookup(name)


__all__ = ("bl_info", "register", "unregister")

if _restore_registration:
    register()
