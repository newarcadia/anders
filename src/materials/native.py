"""Retain native material records independently of their source model files."""
import json
import os
import zlib

from ..errors import UgxError
from ..formats.ecf import EcfFile
from ..ugx.read import read_materials
from .tree import rebuild_material_tree

__all__ = ('remember_native_material', 'load_native_material', 'retain_native_materials')


def remember_native_material(material, source):
    if source.bdt_data and source.bdt_index >= 0:
        material['hwek_ugx_native_source'] = json.dumps((source.bdt_index, zlib.compress(source.bdt_data).hex()))


def load_native_material(material, cache_version):
    if material is None or int(material.get('hwek_ugx_cache_version', 0)) != cache_version:
        return None
    snapshot = material.get('hwek_ugx_native_source')
    try:
        if snapshot:
            index, payload = json.loads(snapshot)
            payload = zlib.decompress(bytes.fromhex(payload))
        else:
            path = str(material.get('hwek_ugx_source_path', ''))
            if not path or not os.path.isfile(path):
                return None
            with open(path, 'rb') as handle:
                chunk = EcfFile.parse(handle.read()).chunks.get(0x704)
            if chunk is None:
                return None
            payload = chunk.data
            index = int(material.get('hwek_ugx_material_index', -1))
        sources = read_materials(payload, 0)
        if not isinstance(index, int) or not 0 <= index < len(sources):
            raise UgxError('the source material index is outside its native table')
        return sources[index]
    except (OSError, ValueError, TypeError, zlib.error, UgxError) as error:
        raise UgxError(f'Material {material.name}: could not read its retained native data: {error}.') from error


def retain_native_materials(target, sources):
    source = next((material.bdt_data for material in sources if material is not None and material.bdt_data), b'')
    return rebuild_material_tree(source, target, sources) if source else target
