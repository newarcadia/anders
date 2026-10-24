"""Edit native part names without changing shared model or bone strings."""
import struct

from ..errors import UgxError
from ..formats.granny import (
    GRX_MESH_SIZE, GrxWriter, grx_array_header, grx_cstring, grx_ref_array,
    grx_ref_array_from_header, grx_u64, safe_grx_name, write_grx_array_header,
)
from ..formats.granny_layout import validate_granny_layout

__all__ = ('patch_granny_model_name', 'patch_granny_mesh_names')


def patch_granny_model_name(data, name):
    validate_granny_layout(data)
    models = grx_ref_array_from_header(data, 96)
    if len(models) != 1:
        raise UgxError('Native model naming requires exactly one model.')
    name = safe_grx_name(name, '')
    if name == grx_cstring(data, grx_u64(data, models[0])):
        return bytes(data)
    writer = GrxWriter()
    writer.data = bytearray(data)
    writer.string_ref(models[0], name)
    writer.finish_strings()
    validate_granny_layout(writer.data)
    return bytes(writer.data)


def patch_granny_mesh_names(data, replacements):
    if not replacements:
        return bytes(data)
    validate_granny_layout(data)
    models = grx_ref_array_from_header(data, 96)
    if len(models) != 1:
        raise UgxError('Native part naming requires exactly one model.')
    model = models[0]
    meshes = grx_ref_array(data, *grx_array_header(data, model + 84))
    if any(not 0 <= index < len(meshes) for index in replacements):
        raise UgxError('Native part rename references a missing mesh.')
    writer = GrxWriter()
    writer.data = bytearray(data)
    groups = {}
    additions = []
    for index, mesh in enumerate(meshes):
        name = replacements.get(index, grx_cstring(data, grx_u64(data, mesh)))
        names = groups.setdefault(mesh, {})
        if name not in names:
            target = mesh
            if names:
                target = writer.reserve(GRX_MESH_SIZE)
                writer.data[target:target + GRX_MESH_SIZE] = data[mesh:mesh + GRX_MESH_SIZE]
                # Clones share immutable payloads while each array retains its
                # own pointer storage for the native relocation pass.
                for header, stride in ((16, 20), (36, 8), (48, 44)):
                    count, source = grx_array_header(data, mesh + header)
                    if count:
                        destination = writer.reserve(count * stride)
                        writer.data[destination:destination + count * stride] = data[source:source + count * stride]
                        write_grx_array_header(writer.data, target + header, count, destination)
                additions.append(target)
            names[name] = target
            if target != mesh or name != grx_cstring(data, grx_u64(data, mesh)):
                writer.string_ref(target, name)
        meshes[index] = names[name]
    if additions:
        model_refs = writer.reserve(len(meshes) * 8)
        for index, mesh in enumerate(meshes):
            struct.pack_into('<Q', writer.data, model_refs + index * 8, mesh)
        write_grx_array_header(writer.data, model + 84, len(meshes), model_refs)
        file_meshes = grx_ref_array_from_header(data, 84) + additions
        file_refs = writer.reserve(len(file_meshes) * 8)
        for index, mesh in enumerate(file_meshes):
            struct.pack_into('<Q', writer.data, file_refs + index * 8, mesh)
        write_grx_array_header(writer.data, 84, len(file_meshes), file_refs)
    writer.finish_strings()
    validate_granny_layout(writer.data)
    return bytes(writer.data)
