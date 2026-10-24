"""Native model-group and transform-track name lookup."""

__all__ = ('granny_model_name_key', 'granny_track_name_key', 'granny_model_group_index',
           'granny_track_index', 'granny_track_group_sort_flags')


def granny_model_name_key(name):
    # Granny folds ASCII capitals only. Spaces and punctuation remain significant.
    return str(name).encode('utf-8').lower()


def granny_track_name_key(name):
    # The retail binary-search comparator subtracts signed char values. Include
    # the terminator so a non-ASCII suffix also compares correctly with a prefix.
    return tuple(value if value < 128 else value - 256 for value in str(name).encode('utf-8')) + (0,)


def granny_model_group_index(groups, model_name):
    wanted = granny_model_name_key(model_name)
    return next((index for index, group in enumerate(groups)
                 if not group.flags & 8 and granny_model_name_key(group.name) == wanted), -1)


def granny_track_index(names, bone_name, flags=0):
    if not flags & 2:
        return next((index for index, name in enumerate(names) if name == bone_name), -1)
    wanted = granny_track_name_key(bone_name)
    start, count = 0, len(names)
    while count:
        middle = start + count // 2
        key = granny_track_name_key(names[middle])
        if wanted == key:
            return middle
        if wanted > key:
            count -= middle - start + 1
            start = middle + 1
        else:
            count = middle - start
    return -1


def granny_track_group_sort_flags(flags, names):
    if flags & 2:
        keys = [granny_track_name_key(name) for name in names]
        if any(left > right for left, right in zip(keys, keys[1:])):
            return flags & ~2
    return flags
