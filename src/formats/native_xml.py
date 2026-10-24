"""Read and write native XML documents while retaining unknown native fields."""
from dataclasses import dataclass
import io
import math
import struct
import xml.etree.ElementTree as ET
from .ecf import build_ecf_file
from .xmb import (
    XmbElement, XmbVariant, build_compressed_xml_stream, build_hw2_xmb, decompress_hw2_binary_xml,
    parse_hw2_xmb, wrap_hw2_damage_xmb, xmb_string_variant,
)
from ..errors import UgxError

__all__ = ('NativeXmlDocument', 'read_native_xml_document', 'native_xml_text', 'xml_elements', 'xml_signature', 'encode_native_xml_document')


@dataclass
class NativeXmlDocument:
    root: ET.Element
    source: bytes
    binary: bool = False
    elements: list = None


def xml_elements(root):
    result = {}
    def visit(element, path):
        result[path] = element
        for index, child in enumerate(element):
            visit(child, path + (index,))
    visit(root, ())
    return result


def xml_signature(root):
    return (root.tag, tuple(sorted(root.attrib.items())), (root.text or '').strip(),
            tuple(xml_signature(child) for child in root))


def read_native_xml_document(source):
    binary = not source.lstrip().startswith(b'<') and not source.startswith((b'\xff\xfe', b'\xfe\xff', b'\xef\xbb\xbf'))
    elements = None
    try:
        if binary:
            payload = decompress_hw2_binary_xml(source)[0] if source[:4] == b'\xda\xba\x77\x37' else source
            elements = parse_hw2_xmb(payload)
            nodes = [ET.Element(item.name.value, {key.value: value.value for key, value in item.attributes}) for item in elements]
            for index, item in enumerate(elements):
                nodes[index].text = item.text.value or None
                for child in item.children:
                    if not 0 <= child < len(nodes) or elements[child].parent != index or child <= index:
                        raise UgxError('XML XMB has an invalid child reference.')
                    nodes[index].append(nodes[child])
            root = nodes[0]
        else:
            parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
            root = ET.fromstring(source, parser=parser)
            for _, (prefix, uri) in ET.iterparse(io.BytesIO(source), events=('start-ns',)):
                root.set('xmlns' + (':' + prefix if prefix else ''), uri)
        return NativeXmlDocument(root, source, binary, elements)
    except (ET.ParseError, ValueError, IndexError, struct.error) as exc:
        raise UgxError(f'Could not read XML document: {exc}') from exc


def native_xml_text(document):
    if not document.binary:
        return document.source.decode('utf-16' if document.source.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig')
    ET.indent(document.root, space='  ')
    return ET.tostring(document.root, encoding='unicode')


def encode_native_xml_document(xml, source=b'', binary=False):
    edited = read_native_xml_document(xml.encode('utf-8'))
    original = read_native_xml_document(source) if source else None
    if original is not None and original.binary == binary and xml_signature(original.root) == xml_signature(edited.root):
        return source
    if not binary:
        return ET.tostring(edited.root, encoding='utf-8', xml_declaration=True)
    templates = {}
    if original is not None and original.elements:
        def visit(index, path):
            templates[path] = original.elements[index]
            for offset, child in enumerate(original.elements[index].children):
                visit(child, path + (offset,))
        visit(0, ())
    elements = []
    def variant(value, template=None):
        if template is not None and value == template.value:
            return template
        if value.lower() in ('true', 'false'):
            boolean = int(value.lower() == 'true')
            return XmbVariant((7 << 24) | boolean, 7, 0, 1, value)
        if template is not None and template.type_id in (1, 2, 5, 6):
            try:
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError('non-finite number')
                type_id = 6 if template.type_id == 6 else 2
                payload = struct.pack('<d' if type_id == 6 else '<f', number)
                return XmbVariant((2 << 30) | (type_id << 24), type_id, 2, 1, value, payload)
            except (ValueError, OverflowError) as exc:
                raise UgxError(f'Invalid numeric XML value: {value}') from exc
        if template is not None and template.type_id in (3, 4):
            try:
                number = int(value, 0) if value.lower().startswith(('0x', '-0x')) else int(value)
            except ValueError:
                # XMB can store an integral float attribute as an integer.
                # Editing its fractional part must promote the representation.
                try:
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError('non-finite number')
                    payload = struct.pack('<f', number)
                except (ValueError, OverflowError, struct.error) as exc:
                    raise UgxError(f'Invalid numeric XML value: {value}') from exc
                return XmbVariant((2 << 30) | (2 << 24), 2, 2, 1, value, payload)
            try:
                unsigned = bool(template.flags & 1)
                payload = struct.pack('<I' if unsigned else '<i', number)
                flags = 2 | int(unsigned)
                return XmbVariant((flags << 30) | (4 << 24), 4, flags, 1, value, payload)
            except struct.error as exc:
                raise UgxError(f'Invalid integer XML value: {value}') from exc
        return xmb_string_variant(value, template)
    def visit(node, path, parent=-1):
        if not isinstance(node.tag, str):
            return None
        old = templates.get(path)
        if old is not None and old.name.value != node.tag:
            old = None
        old_attrs = {key.value: (key, value) for key, value in old.attributes} if old is not None else {}
        attrs = []
        for key, value in node.attrib.items():
            pair = old_attrs.get(key, (None, None))
            attrs.append((variant(key, pair[0]), variant(value, pair[1])))
        entry = XmbElement(parent, variant(node.tag, old.name if old else None),
                           variant((node.text or '').strip(), old.text if old else None), attrs)
        index = len(elements)
        elements.append(entry)
        for offset, child in enumerate(node):
            child_index = visit(child, path + (offset,), index)
            if child_index is not None:
                entry.children.append(child_index)
        return index
    visit(edited.root, ())
    payload = build_hw2_xmb(elements)
    if original is not None and original.binary and source[:4] == b'\xda\xba\x77\x37':
        _, header, record = decompress_hw2_binary_xml(source)
        return wrap_hw2_damage_xmb(source, payload, header, record)
    # Retail HW2 BinaryXML ECF file and payload identifiers.
    return build_ecf_file([(0xA9C96500, build_compressed_xml_stream(payload))], file_id=0xE43ABC00)
