"""Native Granny local transforms, composition and skeleton retargeting."""

from .granny import GrannyTransform
import math

from ..errors import UgxError
from .matrix import mat3_mul, uax_matrix3_identity

__all__ = ('granny_rotation_matrix', 'granny_transform_rows', 'granny_inverse_transform',
           'granny_multiply_transforms', 'granny_retarget_transform', 'granny_unretarget_transform',
           'granny_transform_from_rows')


def granny_rotation_matrix(orientation):
    x, y, z, w = orientation
    return [[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]]


def transpose3(matrix):
    return [list(row) for row in zip(*matrix)]


def transform_vector(matrix, vector):
    return tuple(sum(row[index] * vector[index] for index in range(3)) for row in matrix)


def multiply_quaternions(left, right):
    x, y, z, w = left
    a, b, c, d = right
    return (w*a + x*d + y*c - z*b, w*b - x*c + y*d + z*a,
            w*c + x*b - y*a + z*d, w*d - x*a - y*b - z*c)


def granny_transform_rows(transform):
    # Native world-pose dispatch uses the most complex active component.
    # Rotation includes position, and scale/shear includes all three fields.
    flags = transform.flags
    rotation = granny_rotation_matrix(transform.orientation) if flags & 6 else uax_matrix3_identity()
    linear = mat3_mul(rotation, transform.scale_shear) if flags & 4 else rotation
    rows = [row + [0.0] for row in transpose3(linear)]
    rows.append(list(transform.position if flags & 7 else (0.0, 0.0, 0.0)) + [1.0])
    return rows


def inverse_scale_shear(matrix):
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    cofactors = [[e*i-f*h, c*h-b*i, b*f-c*e],
                 [f*g-d*i, a*i-c*g, c*d-a*f],
                 [d*h-e*g, b*g-a*h, a*e-b*d]]
    determinant = a*cofactors[0][0] + b*cofactors[1][0] + c*cofactors[2][0]
    if determinant == 0 or not math.isfinite(determinant):
        raise UgxError('Animation source skeleton has singular scale/shear; its basis cannot be retargeted.')
    return [[v / determinant for v in row] for row in cofactors]


def quaternion_from_rotation(matrix):
    trace = sum(matrix[i][i] for i in range(3))
    if trace > 0:
        scale = math.sqrt(trace + 1) * 2
        result = ((matrix[2][1]-matrix[1][2])/scale, (matrix[0][2]-matrix[2][0])/scale,
                  (matrix[1][0]-matrix[0][1])/scale, scale/4)
    else:
        axis = max(range(3), key=lambda i: matrix[i][i])
        next_axis, last_axis = (axis+1)%3, (axis+2)%3
        scale = math.sqrt(max(0., 1+matrix[axis][axis]-matrix[next_axis][next_axis]-matrix[last_axis][last_axis])) * 2
        result = [0., 0., 0., (matrix[last_axis][next_axis]-matrix[next_axis][last_axis])/scale]
        result[axis] = scale/4
        result[next_axis] = (matrix[next_axis][axis]+matrix[axis][next_axis])/scale
        result[last_axis] = (matrix[last_axis][axis]+matrix[axis][last_axis])/scale
    length = math.sqrt(sum(value*value for value in result))
    return tuple(value/length for value in result)


def granny_transform_from_rows(rows, reference_orientation=(0., 0., 0., 1.)):
    if len(rows) != 4 or any(len(row) != 4 for row in rows) or not all(math.isfinite(value) for row in rows for value in row):
        raise UgxError('Bone local transform must be a finite 4x4 matrix.')
    if any(abs(rows[i][3]-(i == 3)) > 1e-6 for i in range(4)):
        raise UgxError('Bone local transform must be affine.')
    linear = [[rows[column][row] for column in range(3)] for row in range(3)]
    rotation = [row[:] for row in linear]
    # Polar decomposition keeps an orthogonal rotation and retains every
    # scale/shear component, including reflections, in the residual matrix.
    for _ in range(64):
        inverse = transpose3(inverse_scale_shear(rotation))
        updated = [[(a+b)*.5 for a, b in zip(left, right)] for left, right in zip(rotation, inverse)]
        error = max(abs(a-b) for left, right in zip(rotation, updated) for a, b in zip(left, right))
        rotation = updated
        if error < 1e-12:
            break
    else:
        raise UgxError('Bone local transform rotation did not converge.')
    determinant = (rotation[0][0]*(rotation[1][1]*rotation[2][2]-rotation[1][2]*rotation[2][1])
                   - rotation[0][1]*(rotation[1][0]*rotation[2][2]-rotation[1][2]*rotation[2][0])
                   + rotation[0][2]*(rotation[1][0]*rotation[2][1]-rotation[1][1]*rotation[2][0]))
    if determinant < 0:
        reference = granny_rotation_matrix(reference_orientation)
        candidates = [[[value*(-1 if column == axis else 1) for column, value in enumerate(row)] for row in rotation]
                      for axis in range(3)]
        rotation = min(candidates, key=lambda candidate: sum((a-b)**2 for left, right in zip(candidate, reference) for a, b in zip(left, right)))
    orientation = quaternion_from_rotation(rotation)
    if sum(a*b for a, b in zip(orientation, reference_orientation)) < 0:
        orientation = tuple(-value for value in orientation)
    scale = mat3_mul(transpose3(rotation), linear)
    position = tuple(rows[3][:3])
    flags = int(any(value != 0 for value in position))
    if any(value != 0 for value in orientation[:3]) or abs(orientation[3]) != 1:
        flags |= 2
    if any(value != (row == column) for row, values in enumerate(scale) for column, value in enumerate(values)):
        flags |= 4
    return GrannyTransform(flags, position, orientation, tuple(tuple(row) for row in scale))


def granny_inverse_transform(transform):
    inverse_q = tuple(-value for value in transform.orientation[:3]) + (transform.orientation[3],)
    inverse_rotation = granny_rotation_matrix(inverse_q)
    scale = mat3_mul(mat3_mul(transpose3(inverse_rotation), inverse_scale_shear(transform.scale_shear)), inverse_rotation)
    position = transform_vector(mat3_mul(inverse_rotation, scale), tuple(-v for v in transform.position))
    flags = int(sum(abs(v) for v in position) >= .001)
    flags |= 2 if sum(abs(a-b) for a, b in zip(inverse_q, (0, 0, 0, 1))) >= .0001 else 0
    flags |= 4 if sum(abs(scale[r][c] - (r == c)) for r in range(3) for c in range(3)) >= .001 else 0
    return GrannyTransform(flags, position, inverse_q, tuple(tuple(row) for row in scale))


def granny_multiply_transforms(left, right):
    left_rotation = granny_rotation_matrix(left.orientation)
    right_rotation = granny_rotation_matrix(right.orientation)
    scaled = bool((left.flags | right.flags) & 4)
    linear = mat3_mul(left_rotation, left.scale_shear) if scaled else left_rotation
    position = tuple(a+b for a, b in zip(left.position, transform_vector(linear, right.position)))
    if scaled:
        left_scale = left.scale_shear if left.flags & 4 else uax_matrix3_identity()
        right_scale = right.scale_shear if right.flags & 4 else uax_matrix3_identity()
        scale = mat3_mul(mat3_mul(mat3_mul(transpose3(right_rotation), left_scale), right_rotation), right_scale)
    else:
        scale = uax_matrix3_identity()
    return GrannyTransform(left.flags | right.flags, position, multiply_quaternions(left.orientation, right.orientation),
                           tuple(tuple(row) for row in scale))


def granny_retarget_transform(sample, offset):
    rotation = granny_rotation_matrix(offset.orientation)
    position = transform_vector(rotation, sample.position) if offset.flags & 2 else sample.position
    position = tuple(a+b for a, b in zip(position, offset.position))
    scaled = bool((sample.flags | offset.flags) & 4)
    if scaled:
        offset_scale = offset.scale_shear if offset.flags & 4 else uax_matrix3_identity()
        scale = mat3_mul(mat3_mul(mat3_mul(rotation, offset_scale), transpose3(rotation)), sample.scale_shear)
    else:
        scale = uax_matrix3_identity()
    # The retargeter rotates translation without scaling it and composes the
    # quaternion in the opposite order to a matrix applied after the sample.
    return GrannyTransform(7 if scaled else 3, position, multiply_quaternions(sample.orientation, offset.orientation),
                           tuple(tuple(row) for row in scale))


def granny_unretarget_transform(sample, offset):
    rotation = granny_rotation_matrix(offset.orientation)
    position = tuple(a-b for a, b in zip(sample.position, offset.position))
    if offset.flags & 2:
        position = transform_vector(transpose3(rotation), position)
    inverse_q = tuple(-value for value in offset.orientation[:3]) + (offset.orientation[3],)
    offset_scale = offset.scale_shear if offset.flags & 4 else uax_matrix3_identity()
    correction = mat3_mul(mat3_mul(rotation, offset_scale), transpose3(rotation))
    scale = mat3_mul(inverse_scale_shear(correction), sample.scale_shear)
    return GrannyTransform(7, position, multiply_quaternions(sample.orientation, inverse_q),
                           tuple(tuple(row) for row in scale))
