"""Patch the native skeleton while retaining its complete source graph."""
from collections import defaultdict
import math
import struct
from types import SimpleNamespace

from ..errors import UgxError
from ..formats.granny import (
    GRX_BONE_SIZE, GRX_BONE_BINDING_SIZE, GrxWriter, GrannyTransform,
    grx_array_header, grx_cstring, grx_ref_array, grx_ref_array_from_header,
    grx_u64, read_grx_transform, write_grx_array_header, write_grx_transform,
)
from ..formats.granny_layout import granny_pointer_fields, validate_granny_layout
from ..formats.granny_transform import granny_transform_from_rows, inverse_scale_shear
from ..formats.matrix import mat4_mul

__all__ = ('patch_granny_skeleton', 'patch_granny_skeleton_lod')


def patch_granny_skeleton_lod(data, lod_type, lod_errors):
    if not -0x80000000 <= lod_type <= 0x7fffffff:
        raise UgxError('Native skeleton LOD type must fit a signed 32-bit integer.')
    if any(not math.isfinite(value) or abs(value) > 3.4028234663852886e38 for value in lod_errors):
        raise UgxError('Native bone LOD errors must be finite and fit a 32-bit float.')
    validate_granny_layout(data)
    models = grx_ref_array_from_header(data, 96)
    if len(models) != 1:
        raise UgxError('Skeleton LOD editing requires exactly one native Granny model.')
    skeleton = grx_u64(data, models[0] + 8)
    count, bones = grx_array_header(data, skeleton + 8)
    if count != len(lod_errors):
        raise UgxError('Native bone LOD errors must match the skeleton bone count.')
    fields = [(skeleton + 20, struct.pack('<i', lod_type))]
    fields.extend((bones + index * GRX_BONE_SIZE + 144, struct.pack('<f', value))
                  for index, value in enumerate(lod_errors))
    if all(data[offset:offset + 4] == value for offset, value in fields):
        return data
    if lod_type != 0 and any(value < 0.0 for value in lod_errors):
        raise UgxError('Active skeleton LOD errors must be nonnegative.')
    result = bytearray(data)
    for offset, value in fields:
        result[offset:offset + 4] = value
    return bytes(result)


def inverse_bind_rows(rows):
    linear = inverse_scale_shear([row[:3] for row in rows[:3]])
    return [row + [0.] for row in linear] + [[-sum(rows[3][i]*linear[i][j] for i in range(3)) for j in range(3)] + [1.]]


def local_bind_rows(bones, index):
    bone = bones[index]
    try:
        world = inverse_bind_rows(bone.matrix)
    except UgxError as error:
        raise UgxError(f'Bone {index} {bone.name!r}: its inverse bind matrix is singular.') from error
    return mat4_mul(world, bones[bone.parent].matrix) if bone.parent >= 0 else world


def transform_bone_binding_bounds(bounds, delta):
    corners = [tuple(bounds[axis + (3 if mask & (1 << axis) else 0)] for axis in range(3)) + (1.,)
               for mask in range(8)]
    transformed = [tuple(sum(point[row] * delta[row][axis] for row in range(4)) for axis in range(3))
                   for point in corners]
    values = [operation(point[axis] for point in transformed) for operation in (min, max) for axis in range(3)]
    if any(not math.isfinite(value) or abs(value) > 3.4028234663852886e38 for value in values):
        raise UgxError('Native mesh binding bounds exceed their float range.')
    return values


def patch_granny_skeleton(data, source_bones, target_bones, required_bindings=(), *, source_indices=None, lod_errors=None, update_bindings=False):
    if lod_errors is not None and (len(lod_errors) != len(target_bones) or any(not math.isfinite(value) or value < 0.0 or value > 3.4028234663852886e38 for value in lod_errors)):
        raise UgxError('Native bone LOD errors must be finite, nonnegative and match its bone count.')
    source_indices = list(range(len(target_bones))) if source_indices is None else list(source_indices)
    if len(source_indices) != len(target_bones) or not target_bones:
        raise UgxError('Native skeleton edits require an identity for each target bone and at least one bone.')
    if any(index is not None and not 0 <= index < len(source_bones) for index in source_indices):
        raise UgxError('Native skeleton edit references a missing source bone.')
    for index, bone in enumerate(target_bones):
        if bone.parent < -1 or bone.parent >= len(target_bones) or bone.parent == index:
            raise UgxError(f'Bone {index} {bone.name!r}: its native parent index is invalid.')
    validate_granny_layout(data)
    models = grx_ref_array_from_header(data, 96)
    if len(models) != 1:
        raise UgxError('Skeleton editing requires exactly one native Granny model.')
    skeleton = grx_u64(data, models[0] + 8)
    count, bones_offset = grx_array_header(data, skeleton + 8)
    if count != len(source_bones):
        raise UgxError('Cached and native Granny skeleton bone counts differ.')
    native_bones = []
    for index in range(count):
        offset = bones_offset + index*GRX_BONE_SIZE
        values = struct.unpack_from('<16f', data, offset+80)
        native_bones.append(SimpleNamespace(name=grx_cstring(data, grx_u64(data, offset)),
            parent=struct.unpack_from('<i', data, offset+8)[0], matrix=[list(values[i:i+4]) for i in range(0,16,4)]))
    remapped = source_indices != list(range(count))
    source_to_target = {}
    for index, source in enumerate(source_indices):
        if source is not None:
            source_to_target.setdefault(source, index)
    unchanged = not remapped and all(a.name == b.name and a.parent == b.parent and a.matrix == b.matrix
                                     for a,b in zip(native_bones,target_bones))
    if unchanged and lod_errors is None and not update_bindings:
        return data
    source_bones = native_bones
    writer = GrxWriter()
    writer.data = bytearray(data)
    destination_bones = writer.reserve(len(target_bones)*GRX_BONE_SIZE) if remapped else bones_offset
    if remapped:
        for index, source in enumerate(source_indices):
            if source is not None:
                writer.data[destination_bones+index*GRX_BONE_SIZE:destination_bones+(index+1)*GRX_BONE_SIZE] = data[bones_offset+source*GRX_BONE_SIZE:bones_offset+(source+1)*GRX_BONE_SIZE]
        write_grx_array_header(writer.data, skeleton+8, len(target_bones), destination_bones)
    renamed = defaultdict(list)
    original_indices = defaultdict(list)
    for index, before in enumerate(source_bones):
        renamed[before.name]
        original_indices[before.name].append(index)
    first_target = {}
    for index, bone in enumerate(target_bones):
        first_target.setdefault(bone.name, index)
    for index, after in enumerate(target_bones):
        source = source_indices[index]
        before = source_bones[source] if source is not None else None
        offset = destination_bones + index*GRX_BONE_SIZE
        if before is not None:
            renamed[before.name].append(after.name)
        if before is None or before.name != after.name:
            writer.string_ref(offset, after.name)
        struct.pack_into('<i', writer.data, offset+8, after.parent)
        if before is None or before.matrix != after.matrix:
            struct.pack_into('<16f', writer.data, offset+80, *(v for row in after.matrix for v in row))
        if lod_errors is not None:
            struct.pack_into('<f', writer.data, offset+144, lod_errors[index])
        parent_source = source_indices[after.parent] if after.parent >= 0 else None
        previous_parent = before.parent if before is not None else -1
        local_changed = before is None or before.matrix != after.matrix
        local_changed |= previous_parent != (parent_source if parent_source is not None else -1)
        local_changed |= after.parent >= 0 and (parent_source is None or source_bones[parent_source].matrix != target_bones[after.parent].matrix)
        if not local_changed:
            continue
        source_local = local_bind_rows(source_bones, source) if before is not None else None
        target_local = local_bind_rows(target_bones, index)
        previous = read_grx_transform(data, bones_offset+source*GRX_BONE_SIZE+12) if before is not None else GrannyTransform()
        if source_local is not None and all(abs(source_local[r][c]-target_local[r][c]) < 1e-12 for r in range(3) for c in range(3)):
            position = tuple(target_local[3][:3])
            transform = GrannyTransform((previous.flags & ~1) | int(any(v != 0 for v in position)),
                                        position, previous.orientation, previous.scale_shear)
        else:
            reference = previous.orientation if previous.flags & 6 else (0., 0., 0., 1.)
            transform = granny_transform_from_rows(target_local, reference)
            transform.flags |= previous.flags & ~7
        write_grx_transform(writer.data, offset+12, transform)
    if remapped:
        def target_pointer(value):
            if bones_offset <= value < bones_offset+count*GRX_BONE_SIZE:
                old, inner = divmod(value-bones_offset, GRX_BONE_SIZE)
                if old in source_to_target:
                    return destination_bones+source_to_target[old]*GRX_BONE_SIZE+inner
            return value
        for field in granny_pointer_fields(data):
            locations = [field]
            if bones_offset <= field < bones_offset+count*GRX_BONE_SIZE:
                old, inner = divmod(field-bones_offset, GRX_BONE_SIZE)
                locations = [destination_bones+index*GRX_BONE_SIZE+inner for index, source in enumerate(source_indices) if source == old] or [field]
            for location in locations:
                struct.pack_into('<Q', writer.data, location, target_pointer(grx_u64(writer.data, location)))
        write_grx_array_header(writer.data, skeleton+8, len(target_bones), destination_bones)
    model_meshes = grx_ref_array(data, *grx_array_header(data, models[0] + 84))
    requested = {offset: bindings for offset, bindings in zip(model_meshes, required_bindings) if bindings is not None}
    meshes = set(grx_ref_array_from_header(data, 84)) | set(model_meshes)
    for mesh in sorted(meshes):
        count, offset = grx_array_header(data, mesh + 48)
        records = [bytearray(data[offset+i*GRX_BONE_BINDING_SIZE:offset+(i+1)*GRX_BONE_BINDING_SIZE]) for i in range(count)]
        names = []
        retained = []
        original_offsets = []
        bounds = {}
        for binding in requested.get(mesh, ()):
            previous_bounds = bounds.get(binding.bone_name)
            if previous_bounds is None:
                bounds[binding.bone_name] = SimpleNamespace(
                    bounds_min=binding.bounds_min, bounds_max=binding.bounds_max)
            else:
                previous_bounds.bounds_min = tuple(min(a, b) for a, b in zip(previous_bounds.bounds_min, binding.bounds_min))
                previous_bounds.bounds_max = tuple(max(a, b) for a, b in zip(previous_bounds.bounds_max, binding.bounds_max))
        for record_index, record in enumerate(records):
            name = grx_cstring(data, struct.unpack_from('<Q', record)[0])
            choices = renamed.get(name)
            if choices == []:
                continue
            target_name = (name if name in choices else choices[0]) if choices else name
            names.append(target_name)
            retained.append(record)
            original_offsets.append(offset+record_index*GRX_BONE_BINDING_SIZE)
            indices = original_indices.get(name, ())
            target = bounds.get(target_name)
            target_index = first_target.get(target_name)
            frame_changed = bool(indices and target_index is not None and source_bones[indices[0]].matrix != target_bones[target_index].matrix)
            if target is not None and (update_bindings or frame_changed or len(set(choices or ())) > 1 or any(source_bones[i].matrix != target_bones[source_to_target[i]].matrix for i in indices if i in source_to_target)):
                struct.pack_into('<3f3f', record, 8, *target.bounds_min, *target.bounds_max)
            elif frame_changed:
                delta = mat4_mul(inverse_bind_rows(source_bones[indices[0]].matrix), target_bones[target_index].matrix)
                struct.pack_into('<6f', record, 8, *transform_bone_binding_bounds(struct.unpack_from('<6f', record, 8), delta))
        records = retained
        for binding in requested.get(mesh, ()):
            if binding.bone_name not in names:
                record = bytearray(GRX_BONE_BINDING_SIZE)
                struct.pack_into('<3f3f', record, 8, *binding.bounds_min, *binding.bounds_max)
                records.append(record)
                names.append(binding.bone_name)
                original_offsets.append(None)
        records_changed = len(records) != count or any(old is None or record != data[old:old+GRX_BONE_BINDING_SIZE] for old, record in zip(original_offsets, records))
        names_changed = any(old is None or name != grx_cstring(data, grx_u64(data, old)) for old, name in zip(original_offsets, names))
        if not records_changed and not names_changed:
            continue
        destination = writer.reserve(len(records)*GRX_BONE_BINDING_SIZE)
        for index, (record, name) in enumerate(zip(records, names)):
            location = destination + index*GRX_BONE_BINDING_SIZE
            writer.data[location:location+GRX_BONE_BINDING_SIZE] = record
            if original_offsets[index] is None or name != grx_cstring(data, grx_u64(data, original_offsets[index])):
                writer.string_ref(location, name)
        write_grx_array_header(writer.data, mesh + 48, len(records), destination)
    writer.finish_strings()
    result = bytes(writer.data)
    validate_granny_layout(result)
    return result
