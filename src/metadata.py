"""Blender compatibility metadata and extension package identity."""

from __future__ import annotations

__all__ = ('bl_info',)

bl_info = {
    "name": "Anders: UGX Tools for Blender",
    "author": "New Arcadia Dev Team (dev.kuru)",
    "version": (2026, 10, 2301),
    "blender": (4, 2, 0),
    "location": "File > Import-Export",
    "description": "Convert FBX to UGX v4/v6 and edit Halo Wars models, materials and animations",
    "category": "Import-Export",
}


ADDON_PACKAGE = __package__.removesuffix(".src")
