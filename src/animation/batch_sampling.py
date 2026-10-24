"""Batch native pose samples for rigs with standard parent inheritance."""
import numpy as np

from ..errors import UgxError
from .curves import sample_granny_curve_batch


def channel_matrices(values):
    rotation = values[:, 3:7]
    lengths = np.linalg.norm(rotation, axis=1)
    rotation = np.divide(rotation, lengths[:, None], out=np.zeros_like(rotation), where=lengths[:, None] > 1e-12)
    rotation[lengths <= 1e-12, 0] = 1
    w, x, y, z = rotation.T
    result = np.zeros((len(values), 4, 4))
    result[:, 0, 0] = 1-2*(y*y+z*z)
    result[:, 0, 1] = 2*(x*y-z*w)
    result[:, 0, 2] = 2*(x*z+y*w)
    result[:, 1, 0] = 2*(x*y+z*w)
    result[:, 1, 1] = 1-2*(x*x+z*z)
    result[:, 1, 2] = 2*(y*z-x*w)
    result[:, 2, 0] = 2*(x*z-y*w)
    result[:, 2, 1] = 2*(y*z+x*w)
    result[:, 2, 2] = 1-2*(x*x+y*y)
    result[:, :3, :3] *= values[:, None, 7:10]
    result[:, :3, 3] = values[:, :3]
    result[:, 3, 3] = 1
    return result


def channel_fit_errors(expected, actual):
    left = channel_matrices(expected).astype(np.float32).astype(float)
    right = channel_matrices(actual).astype(np.float32).astype(float)
    magnitude = np.maximum(np.abs(left), np.abs(right))
    spacing = np.ldexp(np.ones_like(magnitude), np.maximum(-149, np.frexp(magnitude)[1]-24))
    errors = np.max(np.abs(left-right)-spacing, axis=(1, 2))
    return np.maximum(0, errors)


def matrix_channels(matrices):
    matrices = matrices.astype(np.float32).astype(float)
    linear = matrices[:, :3, :3]
    scale = np.linalg.norm(linear, axis=1)
    scale[np.linalg.det(linear) < 0] *= -1
    rotation = np.divide(linear, scale[:, None, :], out=np.zeros_like(linear), where=scale[:, None, :] != 0)
    diagonal = np.diagonal(rotation, axis1=1, axis2=2)
    trace = np.sum(diagonal, axis=1)
    quaternion = np.zeros((len(matrices), 4))
    positive = trace > 0
    root = np.sqrt(np.maximum(0, trace[positive]+1))
    quaternion[positive, 0] = root*.5
    factor = .5/root
    for component, (row, column) in enumerate(((2, 1), (0, 2), (1, 0)), start=1):
        quaternion[positive, component] = (rotation[positive, row, column]-rotation[positive, column, row])*factor
    largest = np.argmax(diagonal, axis=1)
    for index in range(3):
        selected = ~positive & (largest == index)
        first, second = (index+1)%3, (index+2)%3
        root = np.sqrt(np.maximum(0, 1+diagonal[selected, index]-diagonal[selected, first]-diagonal[selected, second]))
        factor = .5/root
        quaternion[selected, index+1] = root*.5
        quaternion[selected, 0] = (rotation[selected, second, first]-rotation[selected, first, second])*factor
        quaternion[selected, first+1] = (rotation[selected, index, first]+rotation[selected, first, index])*factor
        quaternion[selected, second+1] = (rotation[selected, index, second]+rotation[selected, second, index])*factor
    quaternion /= np.linalg.norm(quaternion, axis=1)[:, None]
    return np.concatenate((matrices[:, :3, 3], quaternion, scale), axis=1).astype(np.float32).astype(float)


class NativePoseBatchSampler:
    def __init__(self, sample_bones, corrections, axis):
        self.records = {}
        self.cache = {}
        self.max_decompose_error = 0.0
        for binding, bone, correction, track, rest in sample_bones:
            parent = bone.parent
            parent_name = parent.name if parent is not None else None
            left = bone.matrix_local.inverted_safe() @ (parent.matrix_local if parent is not None else axis)
            self.records[bone.name] = (parent_name, track, np.array(rest), np.array(left),
                np.array(corrections[bone.name]),
                np.array(corrections[parent_name].inverted_safe()) if parent_name is not None else np.eye(4),
                correction is not None)

    @classmethod
    def for_rig(cls, sample_bones, corrections, axis):
        excluded = set()
        for binding, bone, _correction, track, _rest in sample_bones:
            source_parent = binding.parent.blender_name if binding.parent is not None else binding.parent_source_name
            if ((bone.parent.name if bone.parent is not None else '') != source_parent
                    or not bone.use_inherit_rotation or bone.inherit_scale != 'FULL' or not bone.use_local_location
                    or getattr(track, 'retarget_offset', None) is not None):
                return None
            if abs(_rest.determinant()) < 1e-8 or (bone.parent is not None and bone.parent.name in excluded):
                excluded.add(bone.name)
            if track is not None and track.scale_shear_keys:
                scales = np.asarray(track.scale_shear_keys.controls).reshape(-1, 3, 3)
                # Positive symmetric parts keep every spline blend invertible.
                if np.any(np.linalg.eigvalsh((scales+scales.transpose(0, 2, 1))*.5) <= 1e-8):
                    excluded.add(bone.name)
        sampler = cls(sample_bones, corrections, axis)
        sampler.excluded = excluded
        return sampler

    def sample(self, name, times, left_limit=False):
        times = np.asarray(times, dtype=float)
        time_key = times.tobytes(), left_limit

        def evaluate(bone_name):
            cache_key = bone_name, time_key
            if cache_key in self.cache:
                return self.cache[cache_key]
            parent, track, rest, left, right, parent_right_inverse, matched = self.records[bone_name]
            local = np.broadcast_to(rest, (len(times), 4, 4)).copy()
            if track is not None:
                if track.rotation_keys:
                    rotation = sample_granny_curve_batch(track.rotation_keys, times, left_limit)
                    if np.any(np.linalg.norm(rotation, axis=1) <= 1e-12):
                        raise UgxError('Granny rotation curve samples to an invalid quaternion.')
                    channels = np.zeros((len(times), 10))
                    channels[:, 3:7] = rotation
                    channels[:, 7:10] = 1
                    local = channel_matrices(channels)
                if track.scale_shear_keys:
                    scale = sample_granny_curve_batch(track.scale_shear_keys, times, left_limit).reshape(-1, 3, 3)
                    local[:, :3, :3] = local[:, :3, :3] @ scale
                local[:, :3, 3] = (sample_granny_curve_batch(track.position_keys, times, left_limit)
                    if track.position_keys else rest[:3, 3])
            if parent is not None:
                _parent_values, residual = evaluate(parent)
                basis = left @ residual @ parent_right_inverse @ local @ right
            else:
                basis = left @ local @ right
            if matched:
                values = matrix_channels(basis)
                projected = channel_matrices(values)
                self.max_decompose_error = max(self.max_decompose_error, float(np.max(np.abs(basis-projected))))
            else:
                values = np.zeros((len(times), 10))
                values[:, 3] = 1
                values[:, 7:10] = 1
                projected = np.broadcast_to(np.eye(4), basis.shape)
            # Carry only the projection residual down the hierarchy. Forming
            # and cancelling world matrices amplifies float32 translation noise.
            if matched:
                inverse_scale = 1/values[:, 7:10]
                inverse_linear = projected[:, :3, :3].transpose(0, 2, 1)*inverse_scale[:, :, None]**2
                residual = np.broadcast_to(np.eye(4), basis.shape).copy()
                residual[:, :3, :3] = inverse_linear @ basis[:, :3, :3]
                residual[:, :3, 3] = (inverse_linear @ (basis[:, :3, 3]-values[:, :3])[:, :, None])[:, :, 0]
            else:
                residual = basis
            self.cache[cache_key] = values, residual
            return values, residual

        return evaluate(name)[0]
