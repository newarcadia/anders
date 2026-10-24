"""Keep native skeleton order and its indexed geometry consistent."""
import heapq
import math
import struct

from ..errors import UgxError
from ..formats.ecf import EcfFile
from ..formats.granny import grx_array_header, grx_cstring, grx_ref_array, grx_ref_array_from_header, grx_u64
from ..formats.granny_layout import granny_pointer_fields, validate_granny_layout
from ..formats.matrix import mat4_mul
from .skeleton_patch import inverse_bind_rows, transform_bone_binding_bounds
from .skin import vertex_group_from_source_bone
from .vertices import layout_for

__all__ = ('native_skeleton_order', 'canonicalize_ugx_skeleton')


def native_skeleton_order(parents, lod_errors, *, use_lod=True):
    count = len(parents)
    if len(lod_errors) != count or any(not math.isfinite(value) for value in lod_errors):
        raise UgxError('Native skeleton LOD errors must be finite and match its bone count.')
    children = [[] for _ in parents]
    ready = []
    for index, parent in enumerate(parents):
        if parent < -1 or parent >= count or parent == index:
            raise UgxError(f'Bone {index}: its native parent index is invalid.')
        if parent < 0:
            heapq.heappush(ready, index)
        else:
            children[parent].append(index)
    traversal = []
    while ready:
        index = heapq.heappop(ready)
        traversal.append(index)
        for child in children[index]:
            heapq.heappush(ready, child)
    if len(traversal) != count:
        raise UgxError('Native skeleton parent relationships contain a cycle.')
    if not use_lod:
        return traversal, [lod_errors[index] for index in traversal]
    errors = list(lod_errors)
    for index in reversed(traversal):
        parent = parents[index]
        if parent >= 0:
            errors[parent] = max(errors[parent], errors[index])
    order = sorted(traversal, key=lambda index: -errors[index])
    return order, [errors[index] for index in order]


def bone_headers(cached, model, base):
    bones = model.bones[0].matrix_offset - base - 8
    for header in range(48, min(len(cached)-80, 160), 4):
        if struct.unpack_from('<I', cached, header)[0] != len(model.bones):
            continue
        if struct.unpack_from('<Q', cached, header+8)[0] == bones:
            return bones, header
    raise UgxError('Cannot locate the cached skeleton array header for reordering.')


def reorder_named_binding_bounds(native, model, order):
    """Keep named boxes in the frame selected by Granny's first bone match."""
    before, after = {}, {}
    for index, bone in enumerate(model.bones):
        before.setdefault(bone.name, index)
    for index in order:
        after.setdefault(model.bones[index].name, index)
    changes = {name: mat4_mul(inverse_bind_rows(model.bones[index].matrix), model.bones[after[name]].matrix)
               for name, index in before.items()
               if model.bones[index].matrix != model.bones[after[name]].matrix}
    if not changes:
        return
    meshes = set(grx_ref_array_from_header(native, 84))
    for record in grx_ref_array_from_header(native, 96):
        meshes.update(grx_ref_array(native, *grx_array_header(native, record + 84)))
    visited = set()
    for mesh in meshes:
        count, bindings = grx_array_header(native, mesh + 48)
        for index in range(count):
            offset = bindings + index * 44
            if offset in visited:
                continue
            visited.add(offset)
            delta = changes.get(grx_cstring(native, grx_u64(native, offset)))
            if delta is None:
                continue
            bounds = struct.unpack_from('<6f', native, offset + 8)
            struct.pack_into('<6f', native, offset + 8, *transform_bone_binding_bounds(bounds, delta))


def canonicalize_ugx_skeleton(data, model):
    """Permute complete bone records and remap every known numeric binding."""
    native_bones = model.granny_info.bones
    if not model.bones or len(native_bones) != len(model.bones):
        raise UgxError('Cached and native skeleton bone counts differ during reordering.')
    order, errors = native_skeleton_order([bone.parent for bone in model.bones],
                                         [bone.lod_error for bone in native_bones],
                                         use_lod=model.granny_info.skeleton_lod_type != 0)
    if order == list(range(len(order))) and errors == [bone.lod_error for bone in native_bones]:
        return data, order
    if model.bones[0].big_endian or model.bones[0].name_offset_field_size != 8:
        raise UgxError('Skeleton reordering requires a PC v4 or v6 UGX.')
    old_to_new = {old: new for new, old in enumerate(order)}
    file = EcfFile.parse(bytes(data))
    cached_chunk, native_chunk = file.chunks[0x700], file.chunks[0x703]
    cached = bytearray(cached_chunk.data)
    native = bytearray(native_chunk.data)
    cached_bones, header = bone_headers(cached, model, cached_chunk.offset)
    model_offset = grx_ref_array_from_header(native, 96)[0]
    skeleton = grx_u64(native, model_offset+8)
    count, bones = grx_array_header(native, skeleton+8)
    if count != len(order):
        raise UgxError('Native skeleton array size changed during reordering.')
    original_native = bytes(native)
    for new, old in enumerate(order):
        cached[cached_bones+new*80:cached_bones+(new+1)*80] = cached_chunk.data[cached_bones+old*80:cached_bones+(old+1)*80]
        native[bones+new*164:bones+(new+1)*164] = original_native[bones+old*164:bones+(old+1)*164]
        parent = model.bones[old].parent
        parent = old_to_new[parent] if parent >= 0 else -1
        struct.pack_into('<i', cached, cached_bones+new*80+72, parent)
        struct.pack_into('<i', native, bones+new*164+8, parent)
        struct.pack_into('<f', native, bones+new*164+144, errors[new])

    def relocated(offset):
        if bones <= offset < bones+count*164:
            index, inner = divmod(offset-bones, 164)
            return bones+old_to_new[index]*164+inner
        return offset

    for field in granny_pointer_fields(original_native):
        value = grx_u64(original_native, field)
        struct.pack_into('<Q', native, relocated(field), relocated(value))
    # This pointer owns the complete array, rather than the old first bone.
    struct.pack_into('<Q', native, skeleton+12, bones)
    for bound_header in (header+48, header+64):
        bound_count, _, offset = struct.unpack_from('<IIQ', cached_chunk.data, bound_header)
        if not bound_count:
            continue
        if bound_count != count:
            raise UgxError('Cached bone bounds do not match the reordered skeleton.')
        for new, old in enumerate(order):
            cached[offset+new*12:offset+(new+1)*12] = cached_chunk.data[offset+old*12:offset+(old+1)*12]

    result = bytearray(data)
    for section in model.sections:
        if section.local_to_global_bones:
            local_header = section.section_record_file_offset-cached_chunk.offset+(40 if model.cache_version == 4 else 56)
            _, _, palette = struct.unpack_from('<IIQ', cached, local_header)
            for index, bone in enumerate(section.local_to_global_bones):
                struct.pack_into('<i', cached, palette+index*4, old_to_new.get(bone, bone))
        elif 0 <= section.rigid_bone_index < count:
            struct.pack_into('<i', cached, section.rigid_bone_index_offset-cached_chunk.offset,
                             old_to_new[section.rigid_bone_index])
        if section.local_to_global_bones or section.rigid_only:
            continue
        indices = layout_for(section.layout, 'I')
        if indices is None:
            continue
        if indices.type_id != 6:
            raise UgxError(f'Skeleton reordering cannot write skin index element type {indices.type_id}.')
        for vertex_index, vertex in enumerate(section.vertices):
            offset = section.vertex_file_offset+vertex_index*section.vertex_stride+indices.offset
            for influence, (encoded, weight) in enumerate(zip(vertex.bone_indices, vertex.bone_weights)):
                if weight <= 0:
                    continue
                old = vertex_group_from_source_bone(section, encoded, count)
                if old is None or old_to_new[old] == old:
                    continue
                result[offset+influence] = old_to_new[old]+(model.cache_version == 4)
    accessories_header = header+16
    accessory_count, _, accessories = struct.unpack_from('<IIQ', cached_chunk.data, accessories_header)
    for index in range(accessory_count):
        offset = accessories+index*24
        first, length = struct.unpack_from('<ii', cached_chunk.data, offset)
        indices = [old_to_new[old] for old in range(max(0, first), min(count, first+length))]
        if indices:
            struct.pack_into('<ii', cached, offset, min(indices), max(indices)-min(indices)+1)
    reorder_named_binding_bounds(native, model, order)
    validate_granny_layout(bytes(native))
    result[cached_chunk.offset:cached_chunk.offset+cached_chunk.size] = cached
    result[native_chunk.offset:native_chunk.offset+native_chunk.size] = native
    return result, order
