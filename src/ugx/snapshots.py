"""Compare imported mesh buffers without constructing combined LOD meshes."""
import hashlib
import json
import struct
import numpy as np
from .mesh import mesh_uv_data, mesh_color_attribute, mesh_shape_key_layers
from .model_space import ugx_model_matrix

__all__ = ('mesh_buffer', 'mesh_game_positions', 'remember_imported_mesh', 'imported_mesh_changes')


def mesh_buffer(collection, property_name, width=1, dtype=np.float32):
    values = np.empty(len(collection) * width, dtype=dtype)
    collection.foreach_get(property_name, values)
    return values.reshape((-1, width)) if width > 1 else values


def mesh_game_positions(obj):
    mesh = obj.data
    positions = mesh_buffer(mesh.vertices, 'co', 3).astype(np.float64)
    for key, reference, group in mesh_shape_key_layers(obj):
        delta = mesh_buffer(key.data, 'co', 3) - mesh_buffer(reference.data, 'co', 3)
        if group >= 0:
            weights = np.fromiter((next((entry.weight for entry in vertex.groups if entry.group == group), 0.)
                                   for vertex in mesh.vertices), dtype=np.float64, count=len(mesh.vertices))
            delta = delta * weights[:, None]
        positions += delta * key.value
    matrix = np.asarray(ugx_model_matrix(obj), dtype=np.float64)
    positions = positions @ matrix[:3, :3].T + matrix[:3, 3]
    return positions[:, (0, 2, 1)] * (-1, 1, -1)


def mesh_state(obj):
    mesh = obj.data
    digest = hashlib.sha256()
    digest.update(struct.pack('<QQQ', len(mesh.vertices), len(mesh.loops), len(mesh.polygons)))
    for collection, name, width, dtype in (
        (mesh.loops, 'vertex_index', 1, np.int32),
        (mesh.polygons, 'loop_start', 1, np.int32),
        (mesh.polygons, 'loop_total', 1, np.int32),
        (mesh.polygons, 'material_index', 1, np.int32),
        (mesh.polygons, 'use_smooth', 1, np.bool_),
    ):
        digest.update(mesh_buffer(collection, name, width, dtype))
    for name, prop, width, dtype in (
        ('UGX_SourceVertex', 'value', 1, np.int32), ('UGX_Section', 'value', 1, np.int32),
        ('UGX_Normal', 'vector', 3, np.float32), ('UGX_Tangent', 'vector', 3, np.float32),
        ('UGX_BitangentSign', 'value', 1, np.float32),
    ):
        attribute = mesh.attributes.get(name)
        digest.update(repr((name, attribute.domain if attribute else None)).encode())
        if attribute is not None:
            digest.update(mesh_buffer(attribute.data, prop, width, dtype))
    for layer in mesh.uv_layers:
        digest.update(layer.name.encode())
        data, prop = mesh_uv_data(mesh, layer)
        digest.update(mesh_buffer(data, prop, 2))
    color = mesh_color_attribute(mesh)
    if color is not None:
        digest.update(repr((color.name, color.domain, color.data_type)).encode())
        digest.update(mesh_buffer(color.data, 'color', 4))
    digest.update(repr(tuple(tuple(row) for row in ugx_model_matrix(obj))).encode())
    digest.update(repr([(group.index, group.name) for group in obj.vertex_groups]).encode())
    if obj.vertex_groups:
        for vertex in mesh.vertices:
            digest.update(struct.pack('<I', len(vertex.groups)))
            for group in vertex.groups:
                digest.update(struct.pack('<If', group.group, group.weight))
    entries = json.loads(obj.get('hwek_ugx_sections', '[]'))
    digest.update(repr([(entry.get('source_section_index'), entry.get('vertex_start'), entry.get('vertex_count'),
                         entry.get('face_start'), entry.get('face_count')) for entry in entries]).encode())
    digest.update(str(obj.get('hwek_ugx_lod_section_indices', '')).encode())
    positions = mesh_buffer(mesh.vertices, 'co', 3)
    return digest.hexdigest(), hashlib.sha256(positions).hexdigest()


def remember_imported_mesh(obj, unchanged_buffers=False):
    if obj.data.shape_keys:
        return
    if unchanged_buffers:
        _, state, _ = json.loads(obj['hwek_ugx_mesh_snapshot'])
        positions = hashlib.sha256(mesh_buffer(obj.data.vertices, 'co', 3)).hexdigest()
    else:
        state, positions = mesh_state(obj)
    obj['hwek_ugx_mesh_snapshot'] = json.dumps((obj.get('hwek_ugx_source_sha256'), state, positions))


def imported_mesh_changes(obj):
    saved = obj.get('hwek_ugx_mesh_snapshot')
    if not saved or obj.data.shape_keys:
        return None
    source, saved_state, saved_positions = json.loads(saved)
    if source != obj.get('hwek_ugx_source_sha256'):
        return None
    state, positions = mesh_state(obj)
    if state != saved_state:
        return None
    return positions != saved_positions
