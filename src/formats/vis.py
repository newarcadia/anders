"""Validate VIS roots using the native XML document codec."""
from .native_xml import (
    NativeXmlDocument as VisualDocument,
    encode_native_xml_document as encode_visual_document,
    native_xml_text as visual_xml,
    read_native_xml_document,
    xml_elements as visual_elements,
    xml_signature as visual_signature,
)
from ..errors import UgxError

__all__ = ('VisualDocument', 'read_visual_document', 'visual_xml', 'visual_elements', 'visual_signature', 'encode_visual_document')


def read_visual_document(source):
    document = read_native_xml_document(source)
    if document.root.tag.lower() != 'visual':
        raise UgxError('VIS document must have a visual root element.')
    return document
