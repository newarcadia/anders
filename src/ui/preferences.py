"""Blender preferences for extracted assets and toolkit locations."""

from bpy.props import BoolProperty, StringProperty
import bpy
import os
from ..metadata import ADDON_PACKAGE

__all__ = ('HWEK_AddonPreferences', 'configured_ufx_shader_directory', 'configured_asset_directory')

class HWEK_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_PACKAGE

    hw1_asset_directory: StringProperty(
        name="Halo Wars Asset Folder",
        description="Extracted game files used by mods, containing art or extracted archive folders",
        subtype="DIR_PATH",
    )

    hw2_asset_directory: StringProperty(
        name="Halo Wars 2 Asset Folder",
        description="Extracted game files used by mods, containing data, or the data folder itself; includes textures and shaders",
        subtype="DIR_PATH",
    )

    # Read saved shader-only preferences from older installations.
    shader_directory: StringProperty(subtype="DIR_PATH", options={'HIDDEN'})

    toolkit_directory: StringProperty(name="Halo Wars Toolkit Folder", subtype="DIR_PATH",
        description="Folder containing Cutter, Forge and cutter_blender_terrain.zip")
    cutter_launcher: StringProperty(name="Cutter Launcher", subtype="FILE_PATH",
        description="Optional executable or Linux launcher used for native terrain creation and export")
    forge_launcher: StringProperty(name="Forge Launcher", subtype="FILE_PATH",
        description="Optional executable or Linux launcher used for project object import")
    project_file: StringProperty(name="Halo Wars Project", subtype="FILE_PATH")
    scenario_file: StringProperty(name="Cutter Scenario",
        description="Optional scenario path inside the project; leave empty to create a new scenario for Blender terrain")
    enable_terrain: BoolProperty(name="Enable Terrain Authoring on Startup", default=False)

    def draw(self, context):
        self.layout.prop(self, "hw1_asset_directory")
        self.layout.prop(self, "hw2_asset_directory")
        self.layout.prop(self, "toolkit_directory")
        self.layout.prop(self, "cutter_launcher")
        self.layout.prop(self, "forge_launcher")
        self.layout.prop(self, "project_file")
        self.layout.prop(self, "scenario_file")
        self.layout.prop(self, "enable_terrain")
        self.layout.operator('hwek.enable_terrain_tools')
        from ..toolkit import _terrain_error
        if _terrain_error:
            self.layout.label(text=_terrain_error, icon='ERROR')


def configured_ufx_shader_directory() -> str:
    directory = configured_asset_directory(6)
    if directory:
        directory = os.path.normpath(directory)
        return directory if os.path.basename(directory).casefold() == 'data' else os.path.join(directory, 'data')
    addon = bpy.context.preferences.addons.get(ADDON_PACKAGE)
    directory = getattr(addon.preferences, "shader_directory", "") if addon else ""
    return bpy.path.abspath(directory) if directory else ""


def configured_asset_directory(cache_version: int) -> str:
    addon = bpy.context.preferences.addons.get(ADDON_PACKAGE)
    field = {4: 'hw1_asset_directory', 6: 'hw2_asset_directory'}.get(cache_version)
    directory = getattr(addon.preferences, field, "") if addon and field else ""
    return bpy.path.abspath(directory) if directory else ""
