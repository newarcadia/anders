"""Connect Blender authoring to the installed Halo Wars toolkit."""

import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import bpy
from .metadata import ADDON_PACKAGE
from .errors import UgxError

__all__ = ('HWEK_OT_enable_terrain_tools', 'enable_terrain_tools', 'disable_terrain_tools',
           'terrain_enabled', 'run_tool', 'tool_command', 'add_models_to_project', 'toolkit_preferences',
           'restore_terrain_tools', 'launch_tool')

_terrain_modules = []
_bridge_path = None
_terrain_error = ''


def toolkit_preferences():
    addon = bpy.context.preferences.addons.get(ADDON_PACKAGE)
    if addon is None:
        raise UgxError('Enable Anders in Blender preferences before using toolkit integration.')
    return addon.preferences


def toolkit_file(name):
    preferences = toolkit_preferences()
    directory = Path(bpy.path.abspath(preferences.toolkit_directory)) if preferences.toolkit_directory else None
    if directory is None or not directory.is_dir():
        raise UgxError('Choose the Halo Wars toolkit folder in Anders preferences.')
    for candidate in (directory / name, directory / name.replace('.exe', ''), directory / (Path(name).stem + '.sh')):
        if candidate.is_file():
            return candidate
    raise UgxError(f'{name} was not found in the selected toolkit folder.')


def tool_command(name, arguments):
    preferences = toolkit_preferences()
    override = preferences.cutter_launcher if name == 'cutter.exe' else preferences.forge_launcher
    executable = Path(bpy.path.abspath(override)) if override else toolkit_file(name)
    if not executable.is_file():
        raise UgxError(f'Tool launcher does not exist: {executable}')
    if executable.suffix.lower() == '.py':
        python = shutil.which('python3') or shutil.which('python')
        if not python:
            raise UgxError('Python was not found for the selected toolkit launcher.')
        command = [python, str(executable)]
        if executable.name == 'run-hwek.py':
            command.append(Path(name).stem)
        command.append('--')
    elif os.name != 'nt' and executable.suffix.lower() == '.exe':
        raise UgxError('Select a Linux toolkit launcher in Anders preferences to run the Windows toolkit.')
    else:
        command = [str(executable)]
    # Proton launchers translate absolute paths themselves; Wine drive Z also
    # accepts these paths when called from native Blender.
    return command + [str(value) for value in arguments], executable.parent


def run_tool(name, arguments):
    command, directory = tool_command(name, arguments)
    result = subprocess.run(command, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding='utf-8', errors='replace',
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        details = result.stdout.strip()[-3000:]
        raise UgxError(f'{Path(command[0]).name} failed ({result.returncode}): {details}')
    return result.stdout


def launch_tool(name, arguments):
    command, directory = tool_command(name, arguments)
    return subprocess.Popen(command, cwd=directory)


def add_models_to_project(paths):
    preferences = toolkit_preferences()
    project = bpy.path.abspath(preferences.project_file)
    if not project or not Path(project).is_file():
        raise UgxError('Select a Forge project in Anders preferences.')
    completed = []
    for path in paths:
        try:
            arguments = ['--project', project, '--createobject', Path(path).stem, '--ugx', path]
            visual = Path(path).with_suffix('.vis')
            if visual.is_file():
                arguments += ['--vis', str(visual)]
            run_tool('forge.exe', arguments)
            completed.append(Path(path).stem)
        except Exception as exc:
            previous = ', '.join(completed)
            raise UgxError(f'UGX files were exported. Project import failed for {Path(path).stem}. '
                           f'Objects already added: {previous or "none"}. {exc}') from exc


def terrain_enabled():
    return bool(_terrain_modules)


def restore_terrain_tools():
    global _terrain_error
    try:
        preferences = toolkit_preferences()
        if preferences.enable_terrain:
            enable_terrain_tools()
    except Exception as exc:
        _terrain_error = str(exc)
        print(f'Anders terrain authoring could not start: {exc}')


def enable_terrain_tools():
    global _bridge_path, _terrain_error
    if _terrain_modules:
        return
    archive = toolkit_file('cutter_blender_terrain.zip')
    _bridge_path = str(archive)
    sys.path.insert(0, _bridge_path)
    try:
        importlib.invalidate_caches()
        package = importlib.import_module('cutter_blender_terrain')
        package.anders = importlib.import_module(ADDON_PACKAGE)
        for name in ('terrain', 'terrain_fbx', 'authoring'):
            module = importlib.import_module('cutter_blender_terrain.' + name)
            module.register()
            _terrain_modules.append(module)
        _terrain_error = ''
    except Exception:
        disable_terrain_tools()
        raise


def disable_terrain_tools():
    global _bridge_path
    if bpy.app.timers.is_registered(restore_terrain_tools):
        bpy.app.timers.unregister(restore_terrain_tools)
    while _terrain_modules:
        _terrain_modules.pop().unregister()
    if _bridge_path in sys.path:
        sys.path.remove(_bridge_path)
    _bridge_path = None
    for name in list(sys.modules):
        if name == 'cutter_blender_terrain' or name.startswith('cutter_blender_terrain.'):
            del sys.modules[name]


class HWEK_OT_enable_terrain_tools(bpy.types.Operator):
    bl_idname = 'hwek.enable_terrain_tools'
    bl_label = 'Enable Terrain Authoring'

    def execute(self, context):
        try:
            enable_terrain_tools()
            toolkit_preferences().enable_terrain = True
            self.report({'INFO'}, 'Terrain authoring enabled in the Anders sidebar.')
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
