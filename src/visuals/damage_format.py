"""Keep damage documents bound to their game and owning model."""
from pathlib import Path
import zlib

from ..errors import UgxError
from ..formats.native_xml import read_native_xml_document

__all__ = ('model_damage_version', 'source_damage_version', 'damage_document_version', 'require_damage_version')


def model_damage_version(owner):
    return int(owner.get('hwek_ugx_cache_version', owner.get('hwek_ugx_port_target', 4)))


def source_damage_version(document, path, fallback):
    if document.binary:
        return 6
    tags = {node.tag.casefold() for node in document.root.iter() if isinstance(node.tag, str)}
    if tags & {'intacttofracturedthreshold', 'fracturedtodestructiblethreshold', 'offsetlocaltoattachment'}:
        return 6
    if any(node.get('type', '').casefold() in {'explosivedeath', 'dissolvedeath', 'camerashake', 'swaplightstate',
                                             'puredelay', 'toggleparticle', 'environmentforce'} for node in document.root.iter('action')):
        return 6
    if tags & {'cryopercentagebased', 'shatterdeathevent'}:
        return 4
    parts = [part.casefold() for part in Path(path).parts]
    if 'art' in parts:
        return 4
    if 'data' in parts:
        return 6
    # Plain XML using only common actions is valid in either game. Its owner
    # supplies the format until a native source or game-specific field exists.
    return fallback


def damage_document_version(owner):
    text = owner.get('hwek_damage_text')
    if text is not None and 'hwek_damage_game' not in text:
        source = text.get('hwek_damage_source')
        document = read_native_xml_document(zlib.decompress(bytes.fromhex(source)) if source else text.as_string().encode())
        text['hwek_damage_game'] = source_damage_version(document, text.get('hwek_damage_source_path', ''), model_damage_version(owner))
    return int(text['hwek_damage_game']) if text is not None else model_damage_version(owner)


def require_damage_version(owner, version, source_version=None):
    if source_version is None:
        source_version = damage_document_version(owner)
    if source_version != version:
        source_game = 'Halo Wars 2' if source_version == 6 else 'Halo Wars 1'
        target_game = 'Halo Wars 2' if version == 6 else 'Halo Wars 1'
        raise UgxError(f'This damage template belongs to {source_game}. Load or create a {target_game} damage template for this model.')
