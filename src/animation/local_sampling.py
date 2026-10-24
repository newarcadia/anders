"""Compile independent local channels where the rig basis stays rigid."""
from mathutils import Matrix, Quaternion, Vector

from .channels import sample_uax_keys
from .curves import sample_granny_curve_batch

__all__ = ('local_channel_sampler', 'local_matrix_channels_representable', 'local_export_channel_sampler')


def matrix_error(left, right):
    return max(abs(left[r][c]-right[r][c]) for r in range(4) for c in range(4))


def local_matrix_channels_representable(left, right, track, rest_local):
    identity = Matrix.Identity(4)
    mirror = Matrix.Diagonal((-1, 1, 1, 1))
    for matrix in (left, right):
        linear = matrix.to_3x3().to_4x4()
        if matrix_error(linear.transposed() @ linear, identity) > 1e-5:
            return False
    position, rotation, scale = rest_local.decompose()
    if matrix_error(Matrix.LocRotScale(position, rotation, scale), rest_local) > 1e-5:
        return False
    matrices = [value for _time, value in track.scale_shear_keys.values] if track is not None and track.scale_shear_keys else []
    offset = getattr(track, 'retarget_offset', None)
    if offset is not None:
        matrices.append(offset.scale_shear)
    for matrix in matrices:
        if any(abs(matrix[r][c]) > 1e-5 for r in range(3) for c in range(3) if r != c):
            return False
    if matrix_error(mirror @ right, identity) > 1e-5:
        if any(max(matrix[i][i] for i in range(3))-min(matrix[i][i] for i in range(3)) > 1e-5 for matrix in matrices):
            return False
        if max(scale)-min(scale) > 1e-5:
            return False
    return True


def local_channel_sampler(left, right, track, rest_local, include_matrix=False, project_basis=False):
    identity = Matrix.Identity(4)
    mirror = Matrix.Diagonal((-1, 1, 1, 1))
    basis_tolerance = .001 if project_basis else 1e-5
    if max(abs(v) for v in right.translation) > 1e-6:
        return None
    for matrix in (left, right):
        linear = matrix.to_3x3().to_4x4()
        if matrix_error(linear.transposed() @ linear, identity) > basis_tolerance:
            return None
        if abs(linear.determinant()+1) > basis_tolerance:
            return None
    if track is not None and getattr(track, 'retarget_offset', None) is not None:
        return None
    position, orientation, scale = rest_local.decompose()
    if matrix_error(Matrix.LocRotScale(position, orientation, scale), rest_local) > (.001 if project_basis else 1e-6):
        return None
    right_rotation = (mirror @ right).to_quaternion()
    left_rotation = (left @ mirror).to_quaternion()
    direct_scale = matrix_error(mirror @ right, identity) <= 1e-6
    scales = track.scale_shear_keys if track is not None else ()
    if scales:
        for _time, value in scales.values:
            if any(abs(value[r][c]) > (.001 if project_basis else 1e-8) for r in range(3) for c in range(3) if r != c):
                return None
            if not direct_scale and max(value[i][i] for i in range(3))-min(value[i][i] for i in range(3)) > (.001 if project_basis else 1e-7):
                return None
    elif not direct_scale and max(scale)-min(scale) > 1e-7:
        return None
    position_keys = track.position_keys if track is not None else ()
    rotation_keys = track.rotation_keys if track is not None else ()
    linear, translation = left.to_3x3(), left.translation
    def sample_batch(times, left_limit=False):
        import numpy as np

        native_position = (sample_granny_curve_batch(position_keys, times, left_limit)
            if position_keys else np.broadcast_to(position, (len(times), 3)))
        native_rotation = (sample_granny_curve_batch(rotation_keys, times, left_limit)
            if rotation_keys else np.broadcast_to(orientation, (len(times), 4)))
        if scales:
            data = sample_granny_curve_batch(scales, times, left_limit)
            native_scale = data[:, (0, 4, 8)]
            if not rotation_keys:
                native_scale = native_scale*np.asarray(scale)
        else:
            native_scale = np.broadcast_to((1., 1., 1.) if rotation_keys else scale, (len(times), 3))
        return np.concatenate((native_position @ position_transform.T+position_offset,
            native_rotation @ rotation_transform.T, native_scale), axis=1)
    def sample(time, left_limit=False):
        native_position = sample_uax_keys(position_keys, time, left_limit=left_limit) if position_keys else position
        native_rotation = Quaternion(sample_uax_keys(rotation_keys, time, left_limit=left_limit)) if rotation_keys else orientation
        if scales:
            data = sample_uax_keys(scales, time, matrix=True, left_limit=left_limit)
            native_scale = (data[0], data[4], data[8])
            if not rotation_keys:
                native_scale = tuple(a*b for a, b in zip(native_scale, scale))
        else:
            native_scale = (1., 1., 1.) if rotation_keys else tuple(scale)
        reflected = Quaternion((native_rotation.w, native_rotation.x, -native_rotation.y, -native_rotation.z))
        rotation = left_rotation @ reflected @ right_rotation
        location = linear @ Vector(native_position) + translation
        values = tuple(location) + tuple(rotation) + tuple(native_scale)
        if include_matrix:
            normalized = rotation.normalized()
            matrix = Matrix.LocRotScale(location, normalized, Vector(native_scale))
            values += tuple(v for row in matrix for v in row)
        return values
    if not include_matrix:
        import numpy as np

        position_transform, position_offset = np.array(linear), np.array(translation)
        rotation_transform = np.array([tuple(left_rotation @ Quaternion(tuple(
            (1 if index < 2 else -1) if component == index else 0 for component in range(4))) @ right_rotation)
            for index in range(4)]).T
        sample.batch = sample_batch
        sample_batch.polynomial = True
    if track is None:
        constant = sample(0.)
        def sample_constant(_time, left_limit=False):
            return constant
        if not include_matrix:
            sample_constant.batch = lambda times, left_limit=False: np.broadcast_to(constant, (len(times), len(constant)))
        return sample_constant
    return sample


def local_export_channel_sampler(left, right, pose, curves):
    identity = Matrix.Identity(4)
    mirror = Matrix.Diagonal((-1, 1, 1, 1))
    if pose.rotation_mode != 'QUATERNION' or max(abs(v) for v in right.translation) > 1e-6:
        return None
    for matrix in (left, right):
        linear = matrix.to_3x3().to_4x4()
        if (matrix_error(linear.transposed() @ linear, identity) > 1e-5
                or abs(linear.determinant()+1) > 1e-5):
            return None
    defaults = tuple(pose.location) + tuple(pose.rotation_quaternion) + tuple(pose.scale)
    channels = [None]*10
    paths = {pose.path_from_id(field): (offset, size) for field, offset, size in
             (('location', 0, 3), ('rotation_quaternion', 3, 4), ('scale', 7, 3))}
    for curve in curves:
        field = paths.get(curve.data_path)
        if field is None or not 0 <= curve.array_index < field[1]:
            return None
        index = field[0]+curve.array_index
        if channels[index] is not None:
            return None
        channels[index] = curve
    left_rotation = (left @ mirror).to_quaternion().normalized()
    right_rotation = (mirror @ right).to_quaternion().normalized()
    linear, translation = left.to_3x3(), left.translation
    scale_basis = right_rotation.to_matrix()

    def sample(frame):
        values = [curve.evaluate(frame) if curve is not None else value
                  for curve, value in zip(channels, defaults)]
        location = linear @ Vector(values[:3]) + translation
        rotation = Quaternion(values[3:7]).normalized()
        reflected = Quaternion((rotation.w, rotation.x, -rotation.y, -rotation.z))
        rotation = left_rotation @ reflected @ right_rotation
        scale = values[7:10]
        if max(scale) == min(scale):
            shear = tuple(scale[row] if row == column else 0. for row in range(3) for column in range(3))
        else:
            # Conjugating the diagonal scale retains its native shear instead
            # of decomposing a world matrix and introducing roundoff per parent.
            shear = tuple(sum(scale_basis[k][row]*scale[k]*scale_basis[k][column] for k in range(3))
                          for row in range(3) for column in range(3))
        return (*location, rotation.x, rotation.y, rotation.z, rotation.w, *shear)
    return sample
