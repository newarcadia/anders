"""Prepare temporary HW1 UV sets for material mapping nodes."""
from contextlib import contextmanager
import bpy
import struct
from mathutils import Matrix, Vector
from ..errors import UgxError
from ..materials.bindings import ugx_material_node_bindings
from ..materials.data import v4_material_slot_flags
from ..materials.mapping import hw1_texture_mapping
from ..materials.parameters import retarget_ugx_parameter_drivers
from ..formats.ecf import CACHED_DATA_CHUNK, VERTEX_BUFFER_CHUNK, ecf_chunk_records, replace_ecf_chunk
from .mesh import mesh_uv_data, mesh_uv_value
from .read import parse_ugx_data
from .vertices import layout_for
from .scene import ugx_authoring_mesh_objects, ugx_export_object_for, ugx_lod_sibling_objects_for

__all__ = ('prepared_hw1_uv_mappings', 'extend_hw1_uv_sets')


def mapping_key(layer, matrix):
    return layer, tuple(float(matrix[row][column]) for row in range(2) for column in (0, 1, 3))


def material_mapping_plan(mat):
    mappings = []
    for binding in ugx_material_node_bindings(mat, 4, allow_missing=True):
        layer, matrix, velocity, coordinates = hw1_texture_mapping(binding['node'], mat)
        mappings.append((binding, layer, matrix, velocity, coordinates))
    if not any(matrix != Matrix.Identity(4) for _, _, matrix, _, _ in mappings):
        return None
    channels = {}
    for _, layer, matrix, _, _ in mappings:
        if matrix == Matrix.Identity(4):
            channels[mapping_key(layer, matrix)] = int(layer[-1])
    for binding, layer, matrix, _, _ in mappings:
        key = mapping_key(layer, matrix)
        if key in channels:
            continue
        available = [channel for channel in range(3) if channel not in channels.values()]
        if not available:
            raise UgxError(f'Material {mat.name}: texture mappings need more than three UV sets. Bake a texture input first.')
        preferred = int(layer[-1])
        channels[key] = preferred if preferred in available else available[0]
    return mappings, channels


def plain_property(value):
    if hasattr(value, 'to_dict'):
        return value.to_dict()
    if hasattr(value, 'to_list'):
        return value.to_list()
    return value


@contextmanager
def prepared_hw1_uv_mappings(obj, imported=False):
    objects = ugx_lod_sibling_objects_for(ugx_export_object_for(obj)) if imported else ugx_authoring_mesh_objects(obj)
    for item in objects:
        if item.mode == 'EDIT':
            item.update_from_editmode()
    plans = {}
    for item in objects:
        used = {int(poly.material_index) for poly in item.data.polygons}
        for index in used:
            mat = item.material_slots[index].material if index < len(item.material_slots) else None
            if mat is not None and mat not in plans:
                plans[mat] = material_mapping_plan(mat)
    plans = {mat: plan for mat, plan in plans.items() if plan is not None}
    if not plans:
        yield
        return
    meshes, materials, originals = [], {}, []
    groups_before = set(bpy.data.node_groups)
    edit_objects = [item for item in objects if item.mode == 'EDIT']
    active = bpy.context.view_layer.objects.active
    selected = list(bpy.context.selected_objects)
    if edit_objects:
        bpy.ops.object.mode_set(mode='OBJECT')
    try:
        for mat, (mappings, channels) in plans.items():
            copy = mat.copy()
            materials[mat] = copy
            retarget_ugx_parameter_drivers(copy)
            copy['hwek_ugx_export_name'] = mat.get('hwek_ugx_export_name', mat.name)
            copy['hwek_ugx_blender_name'] = copy.name
            for binding, layer, matrix, velocity, coordinates in mappings:
                texture = copy.node_tree.nodes[binding['node'].name]
                uv = copy.node_tree.nodes.new('ShaderNodeUVMap')
                uv.uv_map = 'UV' + str(channels[mapping_key(layer, matrix)])
                animation = copy.node_tree.nodes.new('ShaderNodeAndersHW1UV')
                copy.node_tree.links.new(uv.outputs['UV'], animation.inputs['Vector'])
                copy.node_tree.links.new(animation.outputs['UV'], texture.inputs['Vector'])
                for name, value in zip(('Speed U', 'Speed V', 'Speed W'), velocity):
                    animation.inputs[name].default_value = value
                flags = int(coordinates.inputs['Map Flags'].default_value) if coordinates is not None else v4_material_slot_flags(binding['slot']['usage'], binding['slot'])
                animation.inputs['Map Flags'].default_value = flags
        for item in objects:
            if not any(slot.material in plans for slot in item.material_slots):
                continue
            original = item.data
            slots = [(slot.link, slot.material) for slot in item.material_slots]
            properties = {key: plain_property(value) for key, value in item.items()}
            mesh = original.copy()
            meshes.append(mesh)
            originals.append((item, original, slots, properties))
            source_uvs = {}
            for name in ('UV0', 'UV1', 'UV2'):
                layer = original.uv_layers.get(name)
                if layer is None and name == 'UV0':
                    layer = original.uv_layers.active
                if layer is not None:
                    data, prop = mesh_uv_data(original, layer)
                    source_uvs[name] = [mesh_uv_value(data, prop, loop.index) for loop in original.loops]
            tangent_uv = next((layer for layer in original.uv_layers if layer.active_render), original.uv_layers.active)
            if tangent_uv is not None:
                layer = mesh.uv_layers.new(name='_HW1_Source_Tangent_UV')
                target_data, target_property = mesh_uv_data(mesh, layer)
                source_data, source_property = mesh_uv_data(original, tangent_uv)
                for loop in original.loops:
                    setattr(target_data[loop.index], target_property, mesh_uv_value(source_data, source_property, loop.index))
            for poly in original.polygons:
                mat = slots[poly.material_index][1] if poly.material_index < len(slots) else None
                if mat not in plans:
                    continue
                mappings, channels = plans[mat]
                for _, source, matrix, _, _ in mappings:
                    if source not in source_uvs:
                        raise UgxError(f'Mesh {item.name}: material {mat.name} needs missing UV map {source}.')
                    name = 'UV' + str(channels[mapping_key(source, matrix)])
                    layer = mesh.uv_layers.get(name) or mesh.uv_layers.new(name=name)
                    data, prop = mesh_uv_data(mesh, layer)
                    for loop_index in poly.loop_indices:
                        u, v = source_uvs[source][loop_index]
                        transformed = matrix @ Vector((u, v, 0, 1))
                        setattr(data[loop_index], prop, transformed[:2])
            item.data = mesh
            for slot, (_, mat) in zip(item.material_slots, slots):
                if mat in materials:
                    slot.material = materials[mat]
        yield
    finally:
        for item, original, slots, properties in reversed(originals):
            item.data = original
            for slot, (link, mat) in zip(item.material_slots, slots):
                slot.link = link
                if link == 'OBJECT':
                    slot.material = mat
            for key in list(item.keys()):
                if key not in properties:
                    del item[key]
            for key, value in properties.items():
                item[key] = value
        for mesh in meshes:
            bpy.data.meshes.remove(mesh)
        for mat in materials.values():
            bpy.data.materials.remove(mat)
        for group in set(bpy.data.node_groups) - groups_before:
            # Blender 5 can retain a user count after removing a custom group's
            # temporary material, even when no data block references it.
            if group.users == 0 or (not group.use_fake_user and not bpy.data.user_map(subset={group}).get(group)):
                bpy.data.node_groups.remove(group)
        if edit_objects:
            for item in bpy.context.selected_objects:
                item.select_set(False)
            for item in edit_objects:
                item.select_set(True)
            bpy.context.view_layer.objects.active = active if active in edit_objects else edit_objects[0]
            bpy.ops.object.mode_set(mode='EDIT')
            for item in selected:
                item.select_set(True)
            bpy.context.view_layer.objects.active = active


def extend_hw1_uv_sets(data, model, mesh):
    if model.cache_version != 4:
        return data, model
    additions = [[index for index in range(3) if mesh.uv_layers.get(f"UV{index}") is not None
                  and layout_for(section.layout, f"T{index}") is None] for section in model.sections]
    if not any(additions):
        return data, model
    records = ecf_chunk_records(bytes(data))
    cached_record = next(record for record in records if record.id == CACHED_DATA_CHUNK)
    vertices = next(record for record in records if record.id == VERTEX_BUFFER_CHUNK)
    cached, payload = bytearray(cached_record.data), bytearray()
    for section, extra in zip(model.sections, additions):
        offset, stride = len(payload), section.vertex_stride + 4 * len(extra)
        for index in range(section.vertex_count):
            start = section.vertex_offset + index * section.vertex_stride
            payload.extend(vertices.data[start:start + section.vertex_stride])
            payload.extend(bytes(4 * len(extra)))
        record = section.section_record_file_offset - cached_record.offset
        endian = '>' if section.big_endian else '<'
        struct.pack_into(endian + '3i', cached, record + 24, offset, section.vertex_count * stride, stride)
        if extra:
            is64 = section.rigid_only_offset - section.section_record_file_offset >= 140
            order = (section.decl_order or section.pack_order) + ''.join('T' + str(index) for index in extra)
            pointer = len(cached)
            cached.extend(order.encode('ascii') + b'\0')
            pointer_field = record + (56 if is64 else 48)
            struct.pack_into(endian + ('Q' if is64 else 'I'), cached, pointer_field, pointer)
            if section.decl_order:
                struct.pack_into(endian + ('Q' if is64 else 'I'), cached, pointer_field + (8 if is64 else 4), pointer)
            types = record + (72 if is64 else 56)
            for index in extra:
                struct.pack_into(endian + 'I', cached, types + (5 + index) * 4, 16)
    used_end = max(section.vertex_offset + section.vertex_buffer_size for section in model.sections)
    payload.extend(vertices.data[used_end:])
    data = replace_ecf_chunk(data, CACHED_DATA_CHUNK, bytes(cached))
    data = replace_ecf_chunk(data, VERTEX_BUFFER_CHUNK, bytes(payload))
    return data, parse_ugx_data(model.path, bytes(data))
