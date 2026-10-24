"""Refresh retained mesh bindings and culling bounds after geometry edits."""
import struct

from ..errors import UgxError
from ..formats.ecf import EcfFile, replace_ecf_chunk
from .granny import authored_granny_meshes
from .skeleton_order import bone_headers
from .skeleton_patch import patch_granny_skeleton
from .skeleton_resize import refresh_resized_bone_bounds
from .skin import vertex_group_from_section_bone, vertex_group_from_source_bone

__all__ = ('patch_edited_mesh_bounds',)


def vertex_influences(model, section, vertex):
    if section.rigid_only:
        bone = vertex_group_from_section_bone(section, section.rigid_bone_index, len(model.bones))
        return {bone if bone is not None else 0: 1.} if model.bones else {}
    result = {}
    for encoded, weight in zip(vertex.bone_indices, vertex.bone_weights):
        bone = vertex_group_from_source_bone(section, encoded, len(model.bones))
        if bone is not None and weight > 0:
            result[bone] = result.get(bone, 0.) + weight
    return result


def patch_edited_mesh_bounds(data, source, target, section_mesh_indices, bounds_source=None):
    if not target.bones or len(source.sections) != len(target.sections):
        return data, False
    affected_bones = set()
    mesh_bones = {}
    edited_sections = set()
    for index, (before, after) in enumerate(zip(source.sections, target.sections)):
        different_count = len(before.vertices) != len(after.vertices)
        for vertex_index in range(max(len(before.vertices), len(after.vertices))):
            old = before.vertices[vertex_index] if vertex_index < len(before.vertices) else None
            new = after.vertices[vertex_index] if vertex_index < len(after.vertices) else None
            old_skin = vertex_influences(source, before, old) if old is not None else {}
            new_skin = vertex_influences(target, after, new) if new is not None else {}
            if not different_count and old.pos == new.pos and old_skin == new_skin:
                continue
            edited_sections.add(index)
            bones = set(old_skin) | set(new_skin)
            affected_bones.update(bones)
            mesh = section_mesh_indices[index]
            mesh_bones.setdefault(mesh, set()).update(bones)
    if not edited_sections:
        return data, False
    if any(bone.big_endian for bone in target.bones):
        raise UgxError('Editing native mesh bounds requires a PC v4 or v6 UGX.')

    # Explicit bound edits remain authoritative. Other untouched bones keep
    # their native padding rather than being tightened by an unrelated edit.
    bounds_source = source if bounds_source is None else bounds_source
    overrides = {(index, key) for index in range(len(target.bones)) for key in ('low', 'high')
                 if index not in affected_bones or
                 getattr(bounds_source.bones[index], 'bounds_' + key) != getattr(source.bones[index], 'bounds_' + key)}
    for index, key in overrides:
        offset = getattr(target.bones[index], 'bounds_' + key + '_offset')
        if offset:
            struct.pack_into('<3f', data, offset, *getattr(bounds_source.bones[index], 'bounds_' + key))
    refresh_resized_bone_bounds(data, target, overrides)
    file = EcfFile.parse(bytes(data))
    cached_chunk = file.chunks[0x700]
    cached = bytearray(cached_chunk.data)
    _, header = bone_headers(cached, target, cached_chunk.offset)
    accessory_count, _, accessories = struct.unpack_from('<IIQ', cached, header + 16)
    for index in edited_sections:
        section = target.sections[index]
        skins = [vertex_influences(target, section, vertex) for vertex in section.vertices]
        used = {bone for skin in skins for bone in skin}
        if not used:
            continue
        influences = max(map(len, skins))
        if not section.rigid_only and influences > section.max_bones:
            struct.pack_into('<i', cached, section.max_bones_offset - cached_chunk.offset, influences)
        if 0 <= section.accessory_index < accessory_count:
            offset = accessories + section.accessory_index * 24
            first, count = struct.unpack_from('<ii', cached, offset)
            if count:
                used.update((first, first + count - 1))
            struct.pack_into('<ii', cached, offset, min(used), max(used) - min(used) + 1)
    data = replace_ecf_chunk(data, 0x700, bytes(cached))

    if target.granny_info.models:
        names = target.granny_info.models[0].mesh_names
        groups = [[index for index, mesh in enumerate(section_mesh_indices) if mesh == slot]
                  for slot in range(len(names))]
        authored = authored_granny_meshes(target, names, groups)
        required = []
        for index, mesh in enumerate(authored):
            bones = mesh_bones.get(index)
            binding_names = {target.bones[bone].name for bone in bones} if bones else set()
            required.append([binding for binding in mesh.bone_bindings if binding.bone_name in binding_names] if bones else None)
        native = file.chunks[0x703].data
        native = patch_granny_skeleton(native, target.bones, target.bones, required, update_bindings=True)
        data = replace_ecf_chunk(data, 0x703, native)
    return data, True
