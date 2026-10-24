"""Discover texture dependencies without replacing authored connections."""

import bpy
import json
import os
from ..errors import UgxError
from ..textures.context import (
    IMAGE_TEXTURE_EXTENSIONS,
    texture_usage_is_non_color,
    replacement_texture_path_for_image,
)
from ..textures.discovery import discovered_texture_usage, texture_set_stem, resolve_texture_path
from ..textures.images import image_from_ddx
from ..textures.paths import disabled_texture_path
from ..ugx.scene import ugx_authoring_mesh_objects, ugx_used_material_slots
from .bindings import ugx_image_node_from_socket
from .context import is_ugx_material, ugx_material_cache_version
from .data import hw_shader_socket_name, v4_material_usage
from .sockets import active_ugx_material_shader
from .textures import link_material_texture
from .uv import arrange_ugx_material_nodes, connect_hw1_texture_coordinates

__all__ = ('discover_material_textures', 'discover_ugx_texture_dependencies')

def _discover_hw1_normal_texture(mat, shader, slots, source_path):
    socket = shader.inputs.get('Normal (nm)')
    if socket is None or socket.is_linked:
        return 0
    diffuse = next((slot for slot in slots if v4_material_usage(slot.get('usage', '')) == 'diffuse'), None)
    normal = next((slot for slot in slots if v4_material_usage(slot.get('usage', '')) == 'normal'), None)
    if diffuse is None or disabled_texture_path(str(diffuse.get('path', ''))) or (normal is not None and normal.get('path')):
        return 0
    diffuse_path = str(diffuse['path']).replace('\\', '/')
    companion = os.path.join(os.path.dirname(diffuse_path), texture_set_stem(diffuse_path) + '_nm')
    resolved = resolve_texture_path(source_path, companion)
    if not resolved:
        return 0
    if normal is None:
        normal = dict(usage='normal', path='')
        slots.append(normal)
    for name in ('channel', 'flags', 'uvwvel', 'uv_scroll'):
        if name in diffuse:
            normal[name] = diffuse[name]
    index = slots.index(normal)
    image = image_from_ddx(resolved, True, pack_preview=False)
    target = companion.replace('/', '\\')
    image['hwek_ugx_texture_path'] = target
    link_material_texture(mat, 'normal', image, index, target, f"UV{int(normal.get('channel', 0))}",
                          int(normal.get('channel', 0)), int(normal.get('flags', 0)),
                          str(normal.get('uvwvel', '')), shader_node=shader, cache_version=4)
    mat['hwek_ugx_texture_slots'] = json.dumps(slots)
    mat[f'hwek_ugx_texture_slot_{index}'] = mat[f'hwek_ugx_texture_slot_{index}_synced'] = target
    mat['hwek_ugx_texture_normal'] = target
    mat['hwek_ugx_texture_normal_file'] = resolved
    return 1


def discover_material_textures(mat: bpy.types.Material, directory: str, source_path: str = "", include_hw1_normals: bool = True) -> int:
    if not directory or not os.path.isdir(directory) or not is_ugx_material(mat):
        return 0
    version = ugx_material_cache_version(mat)
    shader = active_ugx_material_shader(mat, version)
    files = [os.path.join(directory, name) for name in sorted(os.listdir(directory), key=str.lower)
             if name.lower().endswith(IMAGE_TEXTURE_EXTENSIONS) and os.path.isfile(os.path.join(directory, name))]
    slots = json.loads(mat.get("hwek_ugx_texture_slots", "[]"))
    hints = {os.path.splitext(os.path.basename(str(value).replace("\\", "/")))[0].lower()
             for value in (source_path, mat.name, mat.get("hwek_ugx_source_name", "")) if value}
    changed = _discover_hw1_normal_texture(mat, shader, slots, source_path) if version == 4 and include_hw1_normals else 0
    for index, slot in enumerate(slots):
        original = str(slot.get("path", ""))
        if mat.get("hwek_ugx_source_path") and disabled_texture_path(original):
            continue
        usage = v4_material_usage(slot.get("usage", ""))
        socket_name = hw_shader_socket_name(usage, version)
        socket = shader.inputs.get(socket_name) if socket_name else None
        if socket is None:
            continue
        try:
            node = ugx_image_node_from_socket(socket, mat, allow_missing=True) if socket.is_linked else None
        except UgxError:
            # Discovery leaves unsupported user graphs for export validation.
            continue
        if node is not None and node.image is not None and not node.image.get("hwek_ugx_missing_texture"):
            continue
        # Preserve incomplete or procedural user connections too.
        if socket.is_linked and (node is None or not node.image or not node.image.get("hwek_ugx_missing_texture")):
            continue
        original = str(slot.get("path", ""))
        stem = os.path.splitext(os.path.basename(original.replace("\\", "/")))[0].lower()
        exact = [path for path in files if os.path.splitext(os.path.basename(path))[0].lower() == stem] if stem else []
        matches = exact
        if not matches:
            candidates = [path for path in files if discovered_texture_usage(path) == usage]
            candidate_hints = {texture_set_stem(original)} if mat.get("hwek_ugx_source_path") and not disabled_texture_path(original) else hints
            preferred = [path for path in candidates if texture_set_stem(path) in candidate_hints or texture_set_stem(path) == texture_set_stem(original)]
            matches = preferred or (candidates if disabled_texture_path(original) else [])
        # Prefer native files when multiple formats hold the same texture.
        stems = {os.path.splitext(os.path.basename(path))[0].lower() for path in matches}
        if len(stems) != 1:
            continue
        path = min(matches, key=lambda path: (IMAGE_TEXTURE_EXTENSIONS.index(os.path.splitext(path)[1].lower()), path.lower()))
        image = image_from_ddx(path, texture_usage_is_non_color(usage, version))
        target = replacement_texture_path_for_image(source_path, original, image)
        image["hwek_ugx_texture_path"] = target
        if node is None:
            link_material_texture(mat, usage, image, index, target, f"UV{int(slot.get('channel', 0))}",
                                  int(slot.get("channel", 0)), int(slot.get("flags", 0)), shader_node=shader, cache_version=version)
        else:
            node.image = image
            node["hwek_ugx_texture_path"] = target
            node["hwek_ugx_ddx_path"] = str(image.get("hwek_ugx_ddx_path", ""))
        key = f"hwek_ugx_texture_slot_{index}"
        mat[key] = mat[key + "_synced"] = target
        mat[f"hwek_ugx_texture_{usage}"] = target
        if f"hwek_ugx_texture_{usage}_missing" in mat:
            del mat[f"hwek_ugx_texture_{usage}_missing"]
        changed += 1
    if changed:
        arrange_ugx_material_nodes(mat, version)
        if version == 4:
            connect_hw1_texture_coordinates(mat)
        mat.update_tag()
    return changed


def discover_ugx_texture_dependencies(obj: bpy.types.Object, ugx_path: str) -> int:
    seen = set()
    changed = 0
    for mesh in ugx_authoring_mesh_objects(obj) or [obj]:
        for _, mat in ugx_used_material_slots(mesh):
            if mat is None or mat.as_pointer() in seen or not is_ugx_material(mat):
                continue
            seen.add(mat.as_pointer())
            changed += discover_material_textures(mat, os.path.dirname(os.path.abspath(ugx_path)), ugx_path, include_hw1_normals=False)
    return changed
