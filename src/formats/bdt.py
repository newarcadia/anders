"""Read, rebuild and patch binary data tree records and checksums."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math
import struct
import zlib
from ..errors import UgxError
from .binary import Reader, cstring_capacity_at, read_cstring_at, u32le, write_u16_le, write_u32_le

__all__ = ('BDT_LE_SIGNATURE', 'BDT_BE_SIGNATURE', 'BdtValue', 'BdtNode', 'read_bdt_value_text', 'bdt_stored_value_bytes', 'read_bdt', 'child_named', 'bdt_int', 'bdt_float_pair', 'crc16_phoenix_update', 'bdt_header_crc8', 'refresh_bdt_crc_in_place', 'bdt_rewrite_cstrings', 'bdt_null_flags', 'bdt_stored_string_flags', 'bdt_direct_int_flags', 'bdt_direct_bool_flags', 'bdt_direct_float_flags', 'bdt_direct_byte_flags', 'bdt_direct_short_flags', 'bdt_stored_float_flags', 'bdt_stored_byte_flags', 'bdt_node_topology', 'validate_bdt_node_topology', 'build_bdt_node_chunk', 'clean_bdt_string')

BDT_LE_SIGNATURE = 0x3E


BDT_BE_SIGNATURE = 0xE3


@dataclass
class BdtValue:
    name: str
    text: str
    strings: List[str] = field(default_factory=list)
    string_slots: List[Dict[str, int]] = field(default_factory=list)
    file_offset: int = 0
    string_capacity: int = 0
    entry_offset: int = 0
    raw_value: int = 0
    flags: int = 0
    value_type: int = 0
    value_size: int = 0
    total_size: int = 0
    direct: bool = False
    value_file_offset: int = 0
    value_byte_size: int = 0
    raw_bytes: bytes = b""


@dataclass
class BdtNode:
    name: str = ""
    value: str = ""
    strings: List[str] = field(default_factory=list)
    string_slots: List[Dict[str, int]] = field(default_factory=list)
    value_info: Optional[BdtValue] = None
    attrs: Dict[str, BdtValue] = field(default_factory=dict)
    children: List["BdtNode"] = field(default_factory=list)


def read_bdt_value_text(reader: Reader, data: bytes, value_type: int, value_size: int, total_size: int, raw_value: int, values_start: int, file_base_offset: int, direct: bool = False) -> Tuple[str, List[str], List[Dict[str, int]], int, int]:
    if value_type == 0:
        return "", [], [], 0, 0
    if value_type == 1:
        return ("true" if raw_value else "false"), [], [], 0, 0
    if value_type == 2:
        return str(raw_value), [], [], 0, 0
    if value_type == 3 and (direct or total_size <= 1):
        return str(struct.unpack("<f", struct.pack("<I", raw_value))[0]), [], [], 0, 0
    if value_type == 3 and values_start + raw_value < len(data):
        back = reader.pos
        reader.seek(values_start + raw_value)
        count = max(1, total_size // value_size) if value_size > 0 else total_size
        values = []
        for _ in range(count):
            if reader.can_read(value_size):
                values.append(str(reader.f32()))
        reader.seek(back)
        return ",".join(values), [], [], 0, 0
    if value_type == 4 and values_start + raw_value < len(data):
        value_offset = values_start + raw_value
        text = read_cstring_at(data, value_offset)
        strings = []
        string_slots = []
        cursor = value_offset
        limit = min(len(data), cursor + total_size) if total_size > 0 else len(data)
        while cursor < limit and data[cursor] != 0:
            text_end = data.find(b"\0", cursor, limit)
            if text_end < 0 or text_end <= cursor:
                break
            chunk = data[cursor:text_end]
            if any(c < 0x20 or c > 0x7E for c in chunk):
                break
            strings.append(chunk.decode("utf-8", "replace"))
            string_slots.append({"offset": file_base_offset + cursor, "capacity": cstring_capacity_at(data, cursor)})
            cursor = text_end + 1
        return text, strings, string_slots, file_base_offset + value_offset, cstring_capacity_at(data, value_offset)
    return "", [], [], 0, 0


def bdt_stored_value_bytes(data: bytes, value_type: int, value_size: int, total_size: int, raw_value: int, values_start: int, file_base_offset: int, direct: bool) -> Tuple[int, int, bytes]:
    if direct or value_type == 0 or value_type == 1:
        return 0, 0, b""
    if value_type not in (2, 3, 4):
        return 0, 0, b""
    if raw_value < 0 or values_start + raw_value >= len(data):
        return 0, 0, b""
    size = max(0, int(total_size))
    if value_type == 4 and total_size <= 0:
        size = cstring_capacity_at(data, values_start + raw_value) + 1
    if size <= 0:
        return 0, 0, b""
    value_offset = values_start + raw_value
    size = min(size, len(data) - value_offset)
    return file_base_offset + value_offset, size, bytes(data[value_offset:value_offset + size])


def read_bdt(data: bytes, file_base_offset: int = 0) -> List[BdtNode]:
    if len(data) < 28:
        return []
    probe = Reader(data, False)
    signature = probe.u8()
    if signature == BDT_LE_SIGNATURE:
        big = False
    elif signature == BDT_BE_SIGNATURE:
        big = True
    else:
        return []
    reader = Reader(data, big)
    reader.seek(1)
    header_dwords = reader.u8()
    reader.skip(2)
    reader.u32()
    reader.u32()
    node_bytes = reader.u32()
    name_value_bytes = reader.u32()
    name_bytes = reader.u32()
    value_bytes = reader.u32()
    header_bytes = header_dwords * 4
    reader.seek(header_bytes+data[3]*12)
    nodes_offset = reader.pos
    name_values_offset = nodes_offset + node_bytes
    name_data_offset = name_values_offset + name_value_bytes
    value_padding_offset = name_data_offset + name_bytes
    value_data_offset = ((value_padding_offset + 15) // 16) * 16 if value_bytes else value_padding_offset
    if value_data_offset + value_bytes > len(data):
        return []

    node_count = node_bytes // 8
    name_value_count = name_value_bytes // 8
    nodes: List[BdtNode] = [BdtNode() for _ in range(node_count)]
    parents: List[int] = [0xFFFF] * node_count
    children: List[List[int]] = [[] for _ in range(node_count)]

    for index in range(node_count):
        reader.seek(nodes_offset + index * 8)
        parent = reader.u16()
        reader.u16()
        name_value_offset = reader.u16()
        name_value_count_for_node = reader.u8()
        reader.u8()
        parents[index] = parent
        if parent != 0xFFFF and parent < node_count:
            children[parent].append(index)
        if name_value_offset + name_value_count_for_node > name_value_count:
            continue
        for n in range(name_value_count_for_node):
            entry_offset = name_values_offset + (name_value_offset + n) * 8
            reader.seek(entry_offset)
            raw_value = reader.u32()
            name_offset = reader.u16()
            flags = reader.u16()
            value_type = (flags >> 2) & 0b111
            value_size = 1 << ((flags >> 5) & 0b111)
            total_size = (flags >> 9) & 0x7F
            direct = bool(flags & 2)
            if total_size == 127:
                if direct or raw_value < 4 or raw_value > value_bytes:
                    raise UgxError("Material BDT has an invalid indirect size offset.")
                total_size = struct.unpack_from(
                    ">I" if big else "<I", data, value_data_offset + raw_value - 4,
                )[0]
                if total_size < 127 or raw_value + total_size > value_bytes:
                    raise UgxError("Material BDT indirect value exceeds its value pool.")
            name = read_cstring_at(data, name_data_offset + name_offset)
            text, strings, string_slots, file_offset, capacity = read_bdt_value_text(reader, data, value_type, value_size, total_size, raw_value, value_data_offset, file_base_offset, direct)
            value_file_offset, value_byte_size, raw_bytes = bdt_stored_value_bytes(data, value_type, value_size, total_size, raw_value, value_data_offset, file_base_offset, direct)
            value = BdtValue(
                name,
                text,
                strings,
                string_slots,
                file_offset,
                capacity,
                file_base_offset + entry_offset,
                raw_value,
                flags,
                value_type,
                value_size,
                total_size,
                direct,
                value_file_offset,
                value_byte_size,
                raw_bytes,
            )
            if n == 0:
                nodes[index].name = name
                nodes[index].value = text
                nodes[index].strings = strings
                nodes[index].string_slots = string_slots
                nodes[index].value_info = value
            else:
                nodes[index].attrs[name] = value

    def clone(index: int) -> BdtNode:
        node = BdtNode(
            name=nodes[index].name,
            value=nodes[index].value,
            strings=list(nodes[index].strings),
            string_slots=list(nodes[index].string_slots),
            value_info=nodes[index].value_info,
            attrs=dict(nodes[index].attrs),
            children=[],
        )
        node.children = [clone(child) for child in children[index]]
        return node

    return [clone(index) for index, parent in enumerate(parents) if parent == 0xFFFF]


def child_named(node: BdtNode, name: str) -> Optional[BdtNode]:
    return next((child for child in node.children if child.name == name), None)


def bdt_int(value: Optional[BdtValue], default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value.text, 0)
    except (TypeError, ValueError):
        return default


def bdt_float_pair(value: Optional[BdtValue]) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    text = value.text
    for char in ",;()[]\t":
        text = text.replace(char, " ")
    parts = text.split()
    if not parts:
        return None
    try:
        u = float(parts[0])
        v = float(parts[1]) if len(parts) > 1 else 0.0
    except (TypeError, ValueError):
        return None
    if not math.isfinite(u) or not math.isfinite(v):
        return None
    return u, v


def crc16_phoenix_update(crc: int, value: int) -> int:
    crc ^= (value & 0xFF) << 8
    for _ in range(8):
        crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def bdt_header_crc8(data_crc: int, data_size: int, node_bytes: int, name_value_bytes: int, name_bytes: int, value_bytes: int, user_sections: int = 0) -> int:
    crc = 0xFFFF
    for value in (BDT_LE_SIGNATURE, 7, 0, user_sections & 0xFF):
        crc = crc16_phoenix_update(crc, value)
    for value in (data_crc, data_size, node_bytes, name_value_bytes, name_bytes, value_bytes):
        for shift in (0, 8, 16, 24):
            crc = crc16_phoenix_update(crc, (value >> shift) & 0xFF)
    return (crc ^ 0xFFFF) & 0xFF


def refresh_bdt_crc_in_place(data: bytearray, offset: int, size: int):
    if size < 28 or offset < 0 or offset + size > len(data) or data[offset] != BDT_LE_SIGNATURE:
        return
    header_dwords = data[offset + 1]
    if header_dwords != 7:
        return
    data_size = u32le(data, offset + 8)
    if data_size + 28 > size:
        return
    node_bytes = u32le(data, offset + 12)
    name_value_bytes = u32le(data, offset + 16)
    name_bytes = u32le(data, offset + 20)
    value_bytes = u32le(data, offset + 24)
    data_crc = zlib.crc32(bytes(data[offset + 28:offset + 28 + data_size])) & 0xFFFFFFFF
    write_u32_le(data, offset + 4, data_crc)
    data[offset + 2] = bdt_header_crc8(data_crc, data_size, node_bytes, name_value_bytes, name_bytes, value_bytes, data[offset + 3])


def bdt_rewrite_cstrings(chunk: bytes, replacements: Dict[int, str], entry_replacements: Optional[Dict[int, str]] = None) -> bytes:
    entry_replacements = entry_replacements or {}
    if not replacements and not entry_replacements:
        return chunk
    if len(chunk) < 28 or chunk[0] != BDT_LE_SIGNATURE:
        raise UgxError("Growing material strings currently requires a little-endian BinaryDataTree material chunk.")
    header_dwords = chunk[1]
    if header_dwords != 7:
        raise UgxError("Unsupported BinaryDataTree header layout.")
    data = bytearray(chunk)
    node_bytes = u32le(data, 12)
    name_value_bytes = u32le(data, 16)
    name_bytes = u32le(data, 20)
    value_bytes = u32le(data, 24)
    header_bytes = header_dwords * 4
    name_values_offset = header_bytes + data[3]*12 + node_bytes
    value_padding_offset = name_values_offset + name_value_bytes + name_bytes
    value_data_offset = ((value_padding_offset + 15) // 16) * 16 if value_bytes else value_padding_offset
    value_data_end = value_data_offset + value_bytes
    if value_data_end > len(data):
        raise UgxError("BinaryDataTree value data points outside the material chunk.")
    for original_local in replacements:
        if 0 <= original_local < len(data):
            end = data.find(b"\0", original_local)
            if end >= original_local:
                value_data_end = max(value_data_end, end + 1)
                value_bytes = max(value_bytes, value_data_end - value_data_offset)
    entries = []
    for entry_offset in range(name_values_offset, name_values_offset + name_value_bytes, 8):
        raw = u32le(data, entry_offset)
        flags = struct.unpack_from("<H", data, entry_offset + 6)[0]
        total = (flags >> 9) & 0x7F
        actual_size = u32le(data, value_data_offset + raw - 4) if total == 0x7F and not flags & 2 else total
        entries.append({
            "entry": entry_offset,
            "raw": raw,
            "original_raw": raw,
            "flags": flags,
            "type": (flags >> 2) & 0b111,
            "value_size": 1 << ((flags >> 5) & 0b111),
            "direct": bool(flags & 2),
            "total": total,
            "original_size": actual_size,
            "size": actual_size,
        })

    original_data_size = u32le(data, 8)
    original_value_bytes = u32le(data, 24)

    def relocate_user_sections(position, delta):
        for index in range(data[3]):
            field = header_bytes+index*12+8
            offset = u32le(data, field)
            if offset >= position:
                write_u32_le(data, field, offset+delta)

    shift = 0
    for original_local, value in sorted(replacements.items()):
        current = original_local + shift
        if current < value_data_offset or current >= value_data_end:
            raise UgxError("Material texture string is not inside BinaryDataTree value data.")
        raw = value.encode("utf-8")
        if b"\0" in raw:
            raise UgxError("Material texture path contains a NUL byte.")
        end = data.find(b"\0", current)
        if end < 0:
            raise UgxError("Material texture string is not NUL-terminated.")
        old_size = end - current + 1
        new_size = len(raw) + 1
        delta = new_size - old_size
        data[current:end + 1] = raw + b"\0"
        if delta:
            relocate_user_sections(current+old_size, delta)
            local_rel = original_local - value_data_offset
            for entry in entries:
                if entry["direct"] or entry["type"] not in (2, 3, 4):
                    continue
                entry_raw = entry["original_raw"]
                if entry_raw > local_rel:
                    entry["raw"] += delta
                    write_u32_le(data, entry["entry"], entry["raw"])
                if entry["type"] == 4 and entry_raw <= local_rel < entry_raw + entry["original_size"]:
                    entry["size"] += delta
                    if entry["total"] == 0x7F:
                        write_u32_le(data, value_data_offset + entry["raw"] - 4, entry["size"])
                        continue
                    if entry["size"] >= 0x7F:
                        raise UgxError("Material texture path grew past the compact BinaryDataTree string limit.")
                    flags = (entry["flags"] & ~(0x7F << 9)) | (entry["size"] << 9)
                    entry["flags"] = flags
                    entry["total"] = entry["size"]
                    write_u16_le(data, entry["entry"] + 6, flags)
            value_bytes += delta
            value_data_end += delta
            shift += delta

    entries_by_offset = {entry["entry"]: entry for entry in entries}
    for entry_offset, value in sorted(entry_replacements.items()):
        entry = entries_by_offset.get(entry_offset)
        if not entry or entry["direct"] or entry["type"] != 4 or entry["value_size"] != 1:
            raise UgxError("Material name entry is not a patchable BinaryDataTree string.")
        raw = value.encode("utf-8")
        if b"\0" in raw:
            raise UgxError("Material name contains a NUL byte.")
        new_size = len(raw) + 1
        if new_size >= 0x7F:
            raise UgxError("Material name grew past the compact BinaryDataTree string limit.")
        insert_at = value_data_offset + value_bytes
        data[insert_at:insert_at] = raw + b"\0"
        relocate_user_sections(insert_at, new_size)
        entry["raw"] = value_bytes
        write_u32_le(data, entry["entry"], value_bytes)
        flags = (entry["flags"] & ~(0x7F << 9)) | (new_size << 9)
        entry["flags"] = flags
        entry["total"] = new_size
        write_u16_le(data, entry["entry"] + 6, flags)
        value_bytes += new_size
        value_data_end += new_size

    write_u32_le(data, 24, value_bytes)
    write_u32_le(data, 8, original_data_size+value_bytes-original_value_bytes)
    refresh_bdt_crc_in_place(data, 0, len(data))
    return bytes(data)


def bdt_null_flags(last: bool = True) -> int:
    return (1 if last else 0) << 8


def bdt_stored_string_flags(size: int, last: bool = True) -> int:
    return 1 | (4 << 2) | ((1 if last else 0) << 8) | (max(0, min(127, int(size))) << 9)


def bdt_direct_int_flags(last: bool = True) -> int:
    return 1 | 2 | (2 << 2) | (2 << 5) | ((1 if last else 0) << 8) | (4 << 9)


def bdt_direct_bool_flags(last: bool = True) -> int:
    return 1 | 2 | (1 << 2) | ((1 if last else 0) << 8) | (1 << 9)


def bdt_direct_float_flags(last: bool = True) -> int:
    return 2 | (3 << 2) | (2 << 5) | ((1 if last else 0) << 8) | (4 << 9)


def bdt_direct_byte_flags(last: bool = True, unsigned: bool = True) -> int:
    return (1 if unsigned else 0) | 2 | (2 << 2) | ((1 if last else 0) << 8) | (1 << 9)


def bdt_direct_short_flags(last: bool = True, unsigned: bool = False) -> int:
    return (1 if unsigned else 0) | 2 | (2 << 2) | (1 << 5) | ((1 if last else 0) << 8) | (2 << 9)


def bdt_stored_float_flags(size: int, last: bool = True) -> int:
    return (3 << 2) | (2 << 5) | ((1 if last else 0) << 8) | (max(0, min(127, int(size))) << 9)


def bdt_stored_byte_flags(size: int, last: bool = True) -> int:
    return 1 | (2 << 2) | ((1 if last else 0) << 8) | (max(0, min(127, int(size))) << 9)


def bdt_node_topology(data: bytes) -> Tuple[int, List[Tuple[int, int, int, int, int]], Dict[int, List[int]], List[int]]:
    if len(data) < 28 or data[0] != BDT_LE_SIGNATURE:
        raise UgxError("Material BDT has an invalid little-endian header.")
    header_bytes = int(data[1]) * 4
    node_bytes = u32le(data, 12)
    nodes_offset=header_bytes+data[3]*12
    if header_bytes < 28 or node_bytes % 8 or nodes_offset + node_bytes > len(data):
        raise UgxError("Material BDT has an invalid node table.")
    nodes = [
        struct.unpack_from("<HHHBB", data, nodes_offset + index * 8)
        for index in range(node_bytes // 8)
    ]
    children: Dict[int, List[int]] = {index: [] for index in range(len(nodes))}
    roots: List[int] = []
    for index, (parent, _, _, _, _) in enumerate(nodes):
        if parent == 0xFFFF:
            roots.append(index)
        elif parent < len(nodes):
            children[parent].append(index)
        else:
            raise UgxError(f"Material BDT node {index} has invalid parent index {parent}.")
    return header_bytes, nodes, children, roots


def validate_bdt_node_topology(data: bytes):
    _, nodes, children, _ = bdt_node_topology(data)
    for index, (_, first_child, _, _, child_count) in enumerate(nodes):
        expected = list(range(first_child, first_child + child_count)) if child_count else []
        if children[index] != expected:
            raise UgxError(
                f"Material BDT node {index} advertises children {expected[:4]}, "
                f"but its parent links resolve to {children[index][:4]}."
            )


def build_bdt_node_chunk(nodes: List[Dict[str, Any]], values: bytes) -> bytes:
    names = bytearray()
    name_offsets: Dict[str, int] = {}

    def name_offset(value: str) -> int:
        if value not in name_offsets:
            name_offsets[value] = len(names)
            names.extend(value.encode("utf-8") + b"\0")
        return name_offsets[value]

    children: Dict[int, List[int]] = {index: [] for index in range(len(nodes))}
    roots: List[int] = []
    for index, node in enumerate(nodes):
        parent = int(node.get("parent", 0xFFFF))
        if 0 <= parent < len(nodes):
            children[parent].append(index)
        elif parent == 0xFFFF:
            roots.append(index)
        else:
            raise UgxError(f"BDT node {index} has invalid parent index {parent}.")

    # BDT stores only a first-child index plus an 8-bit child count.  The
    # runtime therefore requires every node's direct children to occupy one
    # contiguous range in the node table.  A breadth-first ordering provides
    # that layout; writing each complete subtree in depth-first order makes
    # sibling ranges point into grandchildren and crashes the retail loader.
    ordered_indices = list(roots)
    seen = set(ordered_indices)
    cursor = 0
    while cursor < len(ordered_indices):
        parent_index = ordered_indices[cursor]
        for child_index in children[parent_index]:
            if child_index in seen:
                raise UgxError(f"BDT node {child_index} is linked more than once.")
            seen.add(child_index)
            ordered_indices.append(child_index)
        cursor += 1
    if len(ordered_indices) != len(nodes):
        missing = sorted(set(range(len(nodes))) - seen)
        raise UgxError(f"BDT node graph is cyclic or disconnected at node {missing[0]}.")

    old_to_new = {old: new for new, old in enumerate(ordered_indices)}

    name_values: List[Tuple[int, str, int]] = []
    node_records = []
    for old_index in ordered_indices:
        node = nodes[old_index]
        node_name_value_offset = len(name_values)
        node_values = list(node.get("name_values", []))
        name_values.extend((int(raw), str(name), int(flags)) for raw, name, flags in node_values)
        child_indices = [old_to_new[index] for index in children[old_index]]
        parent = int(node.get("parent", 0xFFFF))
        if parent != 0xFFFF:
            parent = old_to_new[parent]
        if len(child_indices) > 0xFF:
            raise UgxError(f"BDT node {old_index} has too many children ({len(child_indices)} > 255).")
        if len(node_values) > 0xFF:
            raise UgxError(f"BDT node {old_index} has too many name/value entries ({len(node_values)} > 255).")
        if node_name_value_offset > 0xFFFF:
            raise UgxError("BDT name/value table exceeds the 16-bit node offset limit.")
        node_records.append((
            parent,
            child_indices[0] if child_indices else 0xFFFF,
            node_name_value_offset,
            len(node_values),
            len(child_indices),
        ))

    node_bytes = bytearray()
    for parent, first_child, name_value, name_value_count, child_count in node_records:
        node_bytes.extend(struct.pack("<HHHBB", parent, first_child, name_value, name_value_count, child_count))

    name_value_bytes = bytearray()
    for raw_value, name, flags in name_values:
        name_value_bytes.extend(struct.pack("<IHH", raw_value & 0xFFFFFFFF, name_offset(name), flags))

    value_bytes = bytes(values)
    body = bytearray(node_bytes + name_value_bytes + names)
    if value_bytes:
        while (28 + len(body)) % 16:
            body.append(0)
        body.extend(value_bytes)

    out = bytearray(struct.pack(
        "<BBBBII4I",
        BDT_LE_SIGNATURE,
        7,
        0,
        0,
        0,
        len(body),
        len(node_bytes),
        len(name_value_bytes),
        len(names),
        len(value_bytes),
    ))
    out.extend(body)
    refresh_bdt_crc_in_place(out, 0, len(out))
    validate_bdt_node_topology(bytes(out))
    return bytes(out)


def clean_bdt_string(value: str, fallback: str) -> str:
    text = str(value or fallback).replace("\0", "").strip()
    return text or fallback
