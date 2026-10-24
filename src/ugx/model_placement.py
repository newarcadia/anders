"""Retain the model's native initial placement independently of its skeleton."""
from ..errors import UgxError
from ..formats.ecf import EcfFile, replace_ecf_chunk
from ..formats.granny import GRX_TRANSFORM_SIZE, grx_ref_array_from_header, write_grx_transform
from .metadata import ugx_source_bytes_for_export

__all__ = ('restore_ugx_model_placement', 'patch_armature_model_placement')

_PLACEMENT_PROPERTY = 'hwek_ugx_granny_model_placement'


def model_placement_record(native):
    models = grx_ref_array_from_header(native, 96)
    if not models:
        return None
    if len(models) != 1:
        raise UgxError('Model placement requires exactly one native Granny model.')
    offset = models[0] + 16
    if offset < 16 or offset + GRX_TRANSFORM_SIZE > len(native):
        raise UgxError('Native model placement is outside the Granny chunk.')
    return offset


def restore_ugx_model_placement(armature, source_data=None):
    if armature is None or _PLACEMENT_PROPERTY in armature:
        return
    if source_data is None:
        if int(armature.get('hwek_ugx_cache_version', 0)) not in (4, 6):
            return
        # Authored rigs can declare their game without having an imported
        # placement to restore. Imported rigs still require their source.
        if not any(armature.get('hwek_ugx_' + key) for key in ('source_path', 'source_cache_path', 'source_sha256')):
            return
        source_data, mode = ugx_source_bytes_for_export(armature)
        if mode == 'rebased_source':
            raise UgxError('The model source changed before its original placement could be recovered. Reimport the UGX before rebuilding it.')
    chunk = EcfFile.parse(bytes(source_data)).chunks.get(0x703)
    if chunk is not None:
        offset = model_placement_record(chunk.data)
        if offset is not None:
            armature[_PLACEMENT_PROPERTY] = chunk.data[offset:offset + GRX_TRANSFORM_SIZE].hex()


def patch_armature_model_placement(data, model, armature):
    if armature is None or _PLACEMENT_PROPERTY not in armature:
        return data
    try:
        placement = bytes.fromhex(str(armature[_PLACEMENT_PROPERTY]))
    except ValueError as exc:
        raise UgxError('Model placement metadata is not a native transform. Reimport the UGX to recover it.') from exc
    if len(placement) != GRX_TRANSFORM_SIZE:
        raise UgxError('Model placement metadata must contain a 68-byte native transform.')
    if model.granny_info.models:
        current = bytearray(GRX_TRANSFORM_SIZE)
        write_grx_transform(current, 0, model.granny_info.models[0].initial_placement)
        if current == placement:
            return data
    native = EcfFile.parse(bytes(data)).chunks[0x703].data
    offset = model_placement_record(native)
    if offset is None:
        raise UgxError('Model placement requires a native Granny model.')
    if native[offset:offset + GRX_TRANSFORM_SIZE] == placement:
        return data
    patched = bytearray(native)
    patched[offset:offset + GRX_TRANSFORM_SIZE] = placement
    return replace_ecf_chunk(bytearray(data), 0x703, bytes(patched))
