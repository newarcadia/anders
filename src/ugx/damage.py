"""Update paired HW2 damage bindings after model edits."""

from typing import Dict, List, Tuple
import bpy
import copy
import os
import xml.etree.ElementTree as ET
from ..formats.xmb import (
    XmbElement,
    build_hw2_xmb,
    decompress_hw2_binary_xml,
    parse_hw2_xmb,
    wrap_hw2_damage_xmb,
    xmb_string_variant,
)
from .data import UgxModel
from .naming import current_ugx_mesh_name

__all__ = ('hw2_mesh_binding_changes', 'update_hw2_damage_elements', 'export_paired_hw2_damage_xmb')

def hw2_mesh_binding_changes(model: UgxModel, objects: List[bpy.types.Object]) -> Tuple[Dict[str, str], Dict[str, List[str]], set]:
    source_names = model.granny_info.models[0].mesh_names if model.granny_info.models else []
    by_index: Dict[int, List[bpy.types.Object]] = {}
    for obj in objects:
        by_index.setdefault(int(obj.get("hwek_ugx_mesh_index", -1)), []).append(obj)
    renamed: Dict[str, str] = {}
    additions: Dict[str, List[str]] = {}
    removed = set()
    for index, source_name in enumerate(source_names):
        candidates = by_index.get(index, [])
        if not candidates:
            removed.add(source_name)
            continue
        primary = next((item for item in candidates if item.name == str(item.get("hwek_ugx_mesh_import_name", source_name))), None)
        if primary is None:
            primary = sorted(candidates, key=lambda item: item.name)[0]
        target = current_ugx_mesh_name(primary)
        if target != source_name:
            renamed[source_name] = target
        clones = [current_ugx_mesh_name(item) for item in candidates if item != primary]
        if clones:
            additions[source_name] = list(dict.fromkeys(clones))
    for index, candidates in by_index.items():
        if 0 <= index < len(source_names):
            continue
        for item in candidates:
            anchor = str(item.get("hwek_ugx_mesh_addition_anchor", ""))
            if anchor not in source_names:
                continue
            target = current_ugx_mesh_name(item)
            if target and target != anchor:
                additions.setdefault(anchor, []).append(target)
    for source_name, names in list(additions.items()):
        additions[source_name] = list(dict.fromkeys(names))
    return renamed, additions, removed


def update_hw2_damage_elements(
    elements: List[XmbElement],
    renamed: Dict[str, str],
    additions: Dict[str, List[str]],
    removed: set,
) -> List[XmbElement]:
    original_count = len(elements)
    unlink = set()
    clones_after: Dict[Tuple[int, int], List[int]] = {}

    def update_text(value: str) -> Tuple[str, List[str], bool]:
        tokens = [item.strip() for item in value.split(",")]
        if len(tokens) <= 1:
            token = tokens[0] if tokens else ""
            if token in removed:
                return "", additions.get(token, []), True
            return renamed.get(token, token), additions.get(token, []), False
        updated = []
        for token in tokens:
            if token in removed:
                continue
            updated.append(renamed.get(token, token))
            updated.extend(additions.get(token, []))
        return ",".join(dict.fromkeys(updated)), [], not updated

    for index in range(original_count):
        element = elements[index]
        old_text = element.text.value
        new_text, extra, empty = update_text(old_text)
        element_name = element.name.value.lower()
        if empty and old_text:
            unlink.add(index)
        elif new_text != old_text:
            element.text = xmb_string_variant(new_text, element.text)
        if extra and element_name == "mesh":
            parent = element.parent
            for name in extra:
                clone = copy.deepcopy(element)
                clone.parent = parent
                clone.text = xmb_string_variant(name, element.text)
                elements.append(clone)
                clones_after.setdefault((parent, index), []).append(len(elements) - 1)
        for attr_index, (key, value) in enumerate(element.attributes):
            updated, attr_extra, attr_empty = update_text(value.value)
            if attr_empty and key.value.lower() == "mesh":
                unlink.add(index)
                continue
            if updated != value.value and not attr_empty:
                element.attributes[attr_index] = (key, xmb_string_variant(updated, value))
            if attr_extra and key.value.lower() == "mesh":
                parent = element.parent
                for name in attr_extra:
                    clone = copy.deepcopy(element)
                    clone.parent = parent
                    clone.attributes[attr_index] = (copy.deepcopy(key), xmb_string_variant(name, value))
                    elements.append(clone)
                    clones_after.setdefault((parent, index), []).append(len(elements) - 1)

    for parent_index, parent in enumerate(elements[:original_count]):
        children = []
        for child in parent.children:
            if child not in unlink:
                children.append(child)
            children.extend(clones_after.get((parent_index, child), []))
        parent.children = children
    reachable = []
    seen = set()
    def visit(index: int):
        if index in seen or index < 0 or index >= len(elements):
            return
        seen.add(index)
        reachable.append(index)
        for child in elements[index].children:
            visit(child)
    visit(0)
    remap = {old: new for new, old in enumerate(reachable)}
    compact = [copy.deepcopy(elements[index]) for index in reachable]
    for element in compact:
        element.parent = remap.get(element.parent, -1)
        element.children = [remap[index] for index in element.children if index in remap]
    return compact


def export_paired_hw2_damage_xmb(
    source_path: str,
    output_ugx_path: str,
    model: UgxModel,
    objects: List[bpy.types.Object],
) -> str:
    if not source_path or not os.path.isfile(source_path):
        return ""
    renamed, additions, removed = hw2_mesh_binding_changes(model, objects)
    if not renamed and not additions and not removed:
        return ""
    with open(source_path, "rb") as handle:
        source = handle.read()
    payload, stream_header, record = decompress_hw2_binary_xml(source)
    source_elements = parse_hw2_xmb(payload)
    affected_names = set(renamed) | set(additions) | set(removed)
    referenced_names = {
        token.strip()
        for element in source_elements
        for value in [element.text.value] + [item.value for _, item in element.attributes]
        for token in value.split(",")
        if token.strip()
    }
    if not (affected_names & referenced_names):
        return ""
    def element_signature(elements: List[XmbElement]):
        return tuple(
            (
                element.parent,
                element.name.type_id,
                element.name.value,
                element.text.type_id,
                element.text.value,
                tuple((key.type_id, key.value, value.type_id, value.value) for key, value in element.attributes),
                tuple(element.children),
            )
            for element in elements
        )
    original_signature = element_signature(source_elements)
    elements = update_hw2_damage_elements(source_elements, renamed, additions, removed)
    if element_signature(elements) == original_signature:
        return ""
    rewritten = wrap_hw2_damage_xmb(source, build_hw2_xmb(elements), stream_header, record)
    output_path = os.path.join(os.path.dirname(os.path.abspath(output_ugx_path)), os.path.basename(source_path))
    with open(output_path, "wb") as handle:
        handle.write(rewritten)
    source_xml_path = source_path[:-4] if source_path.lower().endswith(".xmb") else ""
    if source_xml_path and os.path.isfile(source_xml_path):
        def xml_element(index: int):
            source_element = elements[index]
            node = ET.Element(source_element.name.value or "Node", {
                key.value: value.value
                for key, value in source_element.attributes
                if key.value and key.value != "xmlns"
            })
            if source_element.text.value:
                node.text = source_element.text.value
            for child in source_element.children:
                node.append(xml_element(child))
            return node
        root = xml_element(0)
        try:
            ET.indent(root, space="  ")
        except AttributeError:
            pass
        xml_output_path = os.path.join(os.path.dirname(os.path.abspath(output_ugx_path)), os.path.basename(source_xml_path))
        ET.ElementTree(root).write(xml_output_path, encoding="utf-8", xml_declaration=True)
    return output_path
