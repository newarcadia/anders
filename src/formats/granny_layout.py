"""Validate packed Granny references before the games rebase them in place."""

from dataclasses import dataclass
from bisect import bisect_right
import struct

from ..errors import UgxError
from .granny import GRX_SCHEMA_DEFS, GRX_TYPE_RECORD_SIZE, grx_primitive_size

__all__ = ('granny_pointer_fields', 'validate_granny_layout', 'compact_granny_data')


@dataclass(frozen=True)
class _Member:
    kind: int
    name: str
    reference: object
    width: int
    offset: int
    size: int


class _PointerGraph:
    def __init__(self, data):
        self.data = data
        self.layouts = {}
        self.building = set()
        self.types = set()
        self.types_by_offset = {}
        self.strings = {}
        self.objects = set()
        self.pointers = set()
        self.plain_ranges = []
        self.pending = [("FileInfo", 0)]

    def fail(self, context, message):
        raise UgxError(f"Granny {context}: {message}.")

    def range(self, offset, size, context):
        if offset < 0 or size < 0 or offset > len(self.data) - size:
            self.fail(context, f"range {offset:#x} + {size:#x} is outside the data chunk")

    def plain(self, offset, size):
        if size:
            self.plain_ranges.append((offset, offset + size))

    def pointer(self, field, context):
        self.range(field, 8, context)
        target = struct.unpack_from('<Q', self.data, field)[0]
        if target and field in self.pointers:
            self.fail(context, 'shared pointer storage would be rebased more than once')
        self.pointers.add(field)
        if target:
            self.range(target, 0, context)
        return target

    def string(self, target, context):
        if not target:
            return ""
        if target in self.strings:
            return self.strings[target]
        self.range(target, 1, context)
        end = self.data.find(b'\0', target)
        if end < 0:
            self.fail(context, f"string at {target:#x} is not terminated")
        self.plain(target, end + 1 - target)
        value = self.data[target:end].decode('utf-8', 'replace')
        self.strings[target] = value
        return value

    def members(self, schema):
        if isinstance(schema, str):
            return GRX_SCHEMA_DEFS[schema]
        # The native type walker visits referenced definitions even when the
        # corresponding object pointer is null. Definitions can be recursive.
        pending = [schema]
        while pending:
            offset = pending.pop()
            if not offset or offset in self.types:
                continue
            self.types.add(offset)
            members = []
            cursor = offset
            while True:
                self.range(cursor, GRX_TYPE_RECORD_SIZE, f"type at {offset:#x}")
                kind = struct.unpack_from('<I', self.data, cursor)[0]
                self.plain(cursor, 4)
                if not kind:
                    self.plain(cursor + 4, GRX_TYPE_RECORD_SIZE - 4)
                    break
                name = self.string(self.pointer(cursor + 4, 'member name'), 'member name')
                reference = self.pointer(cursor + 12, 'member type')
                width = struct.unpack_from('<i', self.data, cursor + 20)[0]
                self.plain(cursor + 20, GRX_TYPE_RECORD_SIZE - 20)
                if kind not in range(1, 23) or kind == 6:
                    self.fail(f"type at {cursor:#x}", f"unsupported member kind {kind}")
                if width < 0 or kind in (2, 3, 4, 5, 7, 8, 22) and width:
                    self.fail(f"type at {cursor:#x}", f"invalid array width {width} for member kind {kind}")
                if reference:
                    pending.append(reference)
                members.append((kind, name, reference, width))
                cursor += GRX_TYPE_RECORD_SIZE
            self.types_by_offset[offset] = members
        return self.types_by_offset[schema]

    def layout(self, schema):
        pending = [(schema, False)]
        while pending:
            current, finish = pending.pop()
            if current in self.layouts:
                continue
            if not current or not finish and current in self.building:
                self.fail('type layout', 'missing type or recursive inline member')
            members = self.members(current)
            if not finish:
                self.building.add(current)
                pending.append((current, True))
                pending.extend((reference, False) for kind, _, reference, _ in members if kind == 1)
                continue
            cursor, fields = 0, []
            for kind, name, reference, width in members:
                width = max(1, width)
                if kind == 1:
                    size = self.layouts[reference][0]
                elif kind in (2, 8, 22):
                    size = 8
                elif kind in (3, 4):
                    size = 12
                elif kind == 5:
                    size = 16
                elif kind == 7:
                    size = 20
                elif kind == 9:
                    size = 68
                else:
                    size = grx_primitive_size(kind)
                if not size:
                    self.fail('type layout', f"member {name} has no storage")
                total = size * width
                self.range(0, cursor + total, 'type layout')
                fields.append(_Member(kind, name, reference, width, cursor, size))
                cursor += total
            self.building.remove(current)
            self.layouts[current] = cursor, fields
        return self.layouts[schema]

    def enqueue(self, schema, target, context, inline=False):
        if target or inline:
            size, _ = self.layout(schema)
            self.range(target, size, context)
            if (schema, target) not in self.objects:
                self.pending.append((schema, target))

    def array(self, schema, field, references, context):
        self.plain(field, 4)
        count = struct.unpack_from('<i', self.data, field)[0]
        target = self.pointer(field + 4, context)
        if count < 0:
            self.fail(context, 'array count is negative')
        if not count:
            return
        if not target:
            self.fail(context, 'nonempty array has a null pointer')
        stride, fields = self.layout(schema)
        if references:
            stride = 8
        if not stride:
            self.fail(context, 'nonempty array has an empty element type')
        self.range(target, count * stride, context)
        if not references and all(member.kind >= 9 for member in fields):
            self.plain(target, count * stride)
            return
        for index in range(count):
            element = target + index * stride
            self.enqueue(schema, self.pointer(element, context) if references else element, context)

    def walk(self):
        while self.pending:
            schema, target = self.pending.pop()
            key = schema, target
            if key in self.objects:
                continue
            self.objects.add(key)
            size, fields = self.layout(schema)
            self.range(target, size, f"object at {target:#x}")
            for member in fields:
                context = f"{member.name} at {target + member.offset:#x}"
                for index in range(member.width):
                    field = target + member.offset + index * member.size
                    if member.kind == 1:
                        self.enqueue(member.reference, field, context, inline=True)
                    elif member.kind == 2:
                        self.enqueue(member.reference, self.pointer(field, context), context)
                    elif member.kind == 3 or member.kind == 4:
                        self.array(member.reference, field, member.kind == 4, context)
                    elif member.kind == 5 or member.kind == 7:
                        variant_type = self.pointer(field, context)
                        if variant_type:
                            self.members(variant_type)
                        if member.kind == 5:
                            value = self.pointer(field + 8, context)
                            if value:
                                self.enqueue(variant_type, value, context)
                        else:
                            self.array(variant_type, field + 8, False, context)
                    elif member.kind == 8:
                        self.string(self.pointer(field, context), context)
                    else:
                        # EmptyReference is an opaque value. Both games leave
                        # it untouched when no reference callback is supplied.
                        self.plain(field, member.size)
        pointers = sorted(self.pointers)
        for left, right in zip(pointers, pointers[1:]):
            if left + 8 > right:
                self.fail(f"pointer at {right:#x}", 'overlaps another pointer')
        index = 0
        for start, end in sorted(self.plain_ranges):
            while index < len(pointers) and pointers[index] + 8 <= start:
                index += 1
            if index < len(pointers) and pointers[index] < end:
                self.fail(f"pointer at {pointers[index]:#x}", 'overlaps scalar, string or type storage')
        return frozenset(pointers)


def granny_pointer_fields(data):
    """Return every rebased pointer field after validating the complete graph."""
    return _PointerGraph(data).walk()


def validate_granny_layout(data):
    granny_pointer_fields(data)


def compact_granny_data(data):
    """Copy reachable native storage and relocate its packed references."""
    graph = _PointerGraph(data)
    pointers = graph.walk()
    ranges = list(graph.plain_ranges) + [(field, field + 8) for field in pointers]
    for field in pointers:
        target = struct.unpack_from('<Q', data, field)[0]
        if target:
            ranges.append((target, min(target + 1, len(data))))
    merged = []
    for first, last in sorted(ranges):
        if merged and first <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(last, merged[-1][1]))
        else:
            merged.append((first, last))
    result, offsets = bytearray(), []
    for first, last in merged:
        # Keep each native address's alignment when removing unreachable gaps.
        result.extend(b'\0' * ((first - len(result)) % 16))
        offsets.append(len(result))
        result.extend(data[first:last])
    starts = [first for first, _last in merged]

    def relocate(offset):
        index = bisect_right(starts, offset) - 1
        return offsets[index] + offset - starts[index]

    for field in pointers:
        target = struct.unpack_from('<Q', data, field)[0]
        struct.pack_into('<Q', result, relocate(field), relocate(target) if target else 0)
    validate_granny_layout(result)
    return bytes(result)
