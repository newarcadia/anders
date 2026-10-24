"""Refit native static-geometry culling trees after vertex edits."""

import struct

from ..errors import UgxError
from ..formats.ecf import EcfFile, replace_ecf_chunk

__all__ = ('patch_tree_bounds',)


def patch_tree_bounds(data, source, target):
    file = EcfFile.parse(bytes(data))
    if 0x705 not in file.chunks:
        return data
    changed = set()
    for before, after in zip(source.sections, target.sections):
        if (before.accessory_index != after.accessory_index or
                len(before.vertices) != len(after.vertices) or
                any(a.pos != b.pos for a, b in zip(before.vertices, after.vertices))):
            changed.update((before.accessory_index, after.accessory_index))
    if len(source.sections) != len(target.sections):
        changed.update(section.accessory_index for section in source.sections + target.sections)
    if not changed:
        return data

    tree = bytearray(file.chunks[0x705].data)
    endian = '>' if tree[:4] == b'\x33\x44\x00\x02' else '<'
    if len(tree) < 12 or struct.unpack_from(endian + 'I', tree)[0] != 0x33440002:
        raise UgxError('Cannot refit the UGX culling tree: invalid stream version.')
    count = struct.unpack_from(endian + 'I', tree, 4)[0]
    if not count or count > (len(tree) - 12) // 48:
        raise UgxError('Cannot refit the UGX culling tree: invalid node count.')
    # Serialized references retain each game's native in-memory node stride.
    stride = 80 if target.cache_version == 6 else 72
    nodes = []
    offset = 8
    for _ in range(count):
        if offset + 48 > len(tree) - 4:
            raise UgxError('Cannot refit the UGX culling tree: truncated node.')
        values = struct.unpack_from(endian + '6f5I', tree, offset)
        children = []
        for reference in values[7:9]:
            if reference == 0xFFFFFFFF:
                continue
            if reference % stride or reference // stride >= count:
                raise UgxError('Cannot refit the UGX culling tree: invalid child reference.')
            children.append(reference // stride)
        if len(children) == 1:
            raise UgxError('Cannot refit the UGX culling tree: incomplete branch.')
        nodes.append((offset, values[:3], values[3:6], children))
        offset += 48 + values[10] * 4
    if offset + 4 != len(tree) or struct.unpack_from(endian + 'I', tree, offset)[0] != 0x33440002:
        raise UgxError('Cannot refit the UGX culling tree: invalid stream trailer.')
    if any(index < 0 or index >= count or nodes[index][3] for index in changed):
        raise UgxError('Edited geometry must retain its culling-tree leaf accessories.')

    bounds = [(low, high) for _, low, high, _ in nodes]
    for index in changed:
        points = [vertex.pos for section in target.sections if section.accessory_index == index
                  for vertex in section.vertices]
        if not points:
            continue
        low = [min(point[axis] for point in points) for axis in range(3)]
        high = [max(point[axis] for point in points) for axis in range(3)]
        for axis in range(3):
            if low[axis] == high[axis]:
                padding = max(0.0000125, abs(low[axis]) * 2 ** -23)
                low[axis] -= padding
                high[axis] += padding
        bounds[index] = (low, high)

    state = [0] * count
    stack = [(0, False)]
    while stack:
        index, visited = stack.pop()
        if visited:
            children = nodes[index][3]
            if any(child in changed for child in children):
                bounds[index] = ([min(bounds[child][0][axis] for child in children) for axis in range(3)],
                                 [max(bounds[child][1][axis] for child in children) for axis in range(3)])
                changed.add(index)
            state[index] = 2
            continue
        if state[index]:
            raise UgxError('Cannot refit the UGX culling tree: repeated or cyclic node.')
        state[index] = 1
        stack.append((index, True))
        stack.extend((child, False) for child in nodes[index][3])
    if any(value != 2 for value in state):
        raise UgxError('Cannot refit the UGX culling tree: disconnected node.')
    for index in changed:
        struct.pack_into(endian + '6f', tree, nodes[index][0], *bounds[index][0], *bounds[index][1])
    return replace_ecf_chunk(data, 0x705, bytes(tree))
