"""Prepare texture dependencies and write paired model exports."""

from typing import List, Optional, Tuple
import bpy
import os
from ..errors import UgxError
from ..formats.files import write_ugx_export_files
from ..materials.bindings import ugx_material_node_bindings
from ..materials.context import find_ugx_material, ugx_material_cache_version
from ..ugx.read import parse_ugx
from ..ugx.scene import ugx_authoring_mesh_objects, ugx_material_context_object, ugx_used_material_slots
from ..visuals.export import prepare_visual_export
from .context import ugx_texture_context_path
from .images import can_rebuild_packed_dds, dds_image_bytes
from .paths import data_root_for

__all__ = ('active_ugx_texture_image', 'ugx_loaded_texture_images', 'packed_ugx_texture_recoveries', 'texture_export_relative_path', 'prepare_loaded_ugx_texture_exports', 'write_ugx_with_textures', 'export_loaded_ugx_textures', 'ugx_texture_export_root', 'export_loaded_ugx_texture_dependencies')

def active_ugx_texture_image(obj: Optional[bpy.types.Object]) -> Optional[bpy.types.Image]:
    obj = ugx_material_context_object(obj)
    mat = find_ugx_material(obj)
    if obj is None or mat is None or not mat.use_nodes or not mat.node_tree:
        return None
    nodes = mat.node_tree.nodes
    active = getattr(nodes, "active", None)
    if active and active.type == "TEX_IMAGE":
        image = getattr(active, "image", None)
        if image and min(image.size) > 0:
            return image
    for node in nodes:
        image = getattr(node, "image", None)
        if node.type == "TEX_IMAGE" and image and min(image.size) > 0:
            return image
    return None


def ugx_loaded_texture_images(obj: Optional[bpy.types.Object]) -> List[bpy.types.Image]:
    objects = ugx_authoring_mesh_objects(obj)
    if not objects and obj is not None and obj.type == "MESH":
        objects = [obj]
    images = []
    seen = set()
    materials = set()
    for mesh_obj in objects:
        for _, mat in ugx_used_material_slots(mesh_obj):
            if mat is None or not mat.use_nodes or not mat.node_tree:
                continue
            version = ugx_material_cache_version(mat, mesh_obj)
            source = ugx_texture_context_path(mesh_obj)
            key = (mat.as_pointer(), version, source)
            if key in materials:
                continue
            materials.add(key)
            if mat.get("hwek_ugx_native_material"):
                candidates = [binding["image"] for binding in ugx_material_node_bindings(
                    mat, version, source,
                )]
            else:
                candidates = [node.image for node in mat.node_tree.nodes if node.type == "TEX_IMAGE" and node.image and node.image.get("hwek_ugx_ddx_path")]
            for image in candidates:
                if image.as_pointer() not in seen:
                    seen.add(image.as_pointer())
                    images.append(image)
    return images


def packed_ugx_texture_recoveries(obj: bpy.types.Object) -> set:
    two_dimensional = {'diffuse', 'normal', 'gloss', 'packedmask', 'specular', 'emissive',
                       'xform', 'ao', 'opacity', 'envmask', 'emxform'}
    allowed, excluded = set(), set()
    materials = set()
    for mesh in ugx_authoring_mesh_objects(obj):
        for _, mat in ugx_used_material_slots(mesh):
            if mat is None or not mat.get('hwek_ugx_native_material'):
                continue
            version = ugx_material_cache_version(mat, mesh)
            source = ugx_texture_context_path(mesh)
            key = (mat.as_pointer(), version, source)
            if key in materials:
                continue
            materials.add(key)
            for binding in ugx_material_node_bindings(mat, version, source):
                image = binding['image']
                if binding['slot']['usage'] in two_dimensional:
                    allowed.add(image)
                else:
                    excluded.add(image)
    return {image for image in allowed - excluded if image.get('hwek_ugx_ddx_path')
            and not os.path.isfile(bpy.path.abspath(str(image['hwek_ugx_ddx_path'])))
            and can_rebuild_packed_dds(image)}


def texture_export_relative_path(image: bpy.types.Image) -> str:
    texture_path = str(image.get("hwek_ugx_texture_path", "")).replace("\\", "/").replace("/", os.sep).lstrip("\\/")
    source = str(image.get("hwek_ugx_ddx_path", ""))
    if texture_path:
        parts = [part for part in texture_path.split(os.sep) if part and part not in (".", "..") and not part.endswith(":")]
        if parts:
            relative = os.path.join(*parts)
            if os.path.splitext(relative)[1].lower() == ".xpr":
                return relative
            if not os.path.splitext(relative)[1]:
                relative += ".ddx"
            return os.path.splitext(relative)[0] + ".ddx"
    source = source or image.get("hwek_ugx_image_path") or image.filepath_raw or image.name
    return os.path.splitext(os.path.basename(str(source)))[0] + ".ddx"


def prepare_loaded_ugx_texture_exports(obj: bpy.types.Object, directory: str) -> List[Tuple[str, bytes]]:
    textures = {}
    source_cache = {}
    encoded_cache = {}
    recoveries = packed_ugx_texture_recoveries(obj)
    for image in ugx_loaded_texture_images(obj):
        if image.get("hwek_ugx_missing_texture"):
            continue
        output_path = os.path.join(directory, texture_export_relative_path(image))
        # Game resource paths are case-insensitive, including exports on Linux.
        key = os.path.normcase(os.path.abspath(output_path)).casefold()
        encoded = dds_image_bytes(image, allow_packed_preview=image in recoveries,
                                  source_cache=source_cache, encoded_cache=encoded_cache)
        previous = textures.get(key)
        if previous is not None:
            if previous[1] != encoded:
                raise UgxError(
                    f"Textures {previous[2]} and {image.name} export different data to {output_path}. "
                    "Give them separate texture paths or use the same image in both materials."
                )
            continue
        textures[key] = (output_path, encoded, image.name)
    return [(path, encoded) for path, encoded, _ in textures.values()]


def write_ugx_with_textures(obj, path, payload, cache_version, export_loaded_textures, export_visual=False):
    textures = prepare_loaded_ugx_texture_exports(obj, ugx_texture_export_root(path, cache_version)) if export_loaded_textures else []
    visual = prepare_visual_export(obj, os.path.splitext(path)[0] + '.vis', model_path=path, binary=cache_version == 6) if export_visual else []
    write_ugx_export_files([(path, bytes(payload))] + textures + visual)
    return [texture_path for texture_path, _ in textures]


def export_loaded_ugx_textures(obj: bpy.types.Object, directory: str) -> List[str]:
    outputs = prepare_loaded_ugx_texture_exports(obj, directory)
    write_ugx_export_files(outputs)
    return [path for path, _ in outputs]


def ugx_texture_export_root(ugx_path: str, cache_version: int = 4) -> str:
    """Return the asset root used by leading-backslash UGX texture paths."""
    directory = os.path.dirname(os.path.abspath(ugx_path))
    if cache_version == 6:
        data_root = data_root_for(ugx_path)
        return os.path.join(data_root, "textures") if data_root else directory
    current = directory
    while True:
        if os.path.basename(current).lower() == "art":
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return directory
        current = parent


def export_loaded_ugx_texture_dependencies(obj: bpy.types.Object, ugx_path: str) -> List[str]:
    # UGX paths such as \hook\bonus\house_01\house_01_df are rooted at the
    # scenario/mod art directory, not beside the model's UGX file. Exporting
    # them beside the model silently loses the material when that UGX moves
    # between scenario archives.
    cache_version = int(obj.get("hwek_ugx_cache_version", 0))
    if cache_version not in (4, 6):
        cache_version = parse_ugx(ugx_path).cache_version
    return export_loaded_ugx_textures(obj, ugx_texture_export_root(ugx_path, cache_version))
