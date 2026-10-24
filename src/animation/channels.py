"""Prepared native animation channels and spline sampling."""
from dataclasses import dataclass
from functools import cached_property
import copy
from .curves import sample_granny_curve

__all__ = ('UaxSampleKeys', 'prepare_uax_track', 'sample_uax_keys', 'uax_curve_break_times')

@dataclass(frozen=True)
class UaxSampleKeys:
    values: tuple
    times: tuple
    degree: int = 1
    controls: tuple = ()
    duration: float = 0.0
    looping: bool = False
    quaternion: bool = False

    def __len__(self):
        return len(self.values)

    @cached_property
    def spline_arrays(self):
        import numpy as np
        return np.asarray(self.times), np.asarray(self.controls)


def prepare_uax_track(track, duration=0.0, looping=False):
    prepared = copy.copy(track)
    for name, curve in (('position_keys', track.position_curve_info),
                        ('rotation_keys', track.orientation_curve_info),
                        ('scale_shear_keys', track.scale_shear_curve_info)):
        values = tuple(sorted(getattr(track, name), key=lambda item: item[0]))
        degree = curve.header_degree if curve is not None else 1
        if curve is not None and 'DaKeyframes32f' in curve.type_name:
            degree = 1
        controls = tuple(tuple(component for row in item[1] for component in row)
                         if name == 'scale_shear_keys' else item[1] for item in values)
        setattr(prepared, name, UaxSampleKeys(values, tuple(item[0] for item in values), degree, controls, duration, looping, name == 'rotation_keys'))
    return prepared



def sample_uax_keys(keys, time, normalize=False, matrix=False, left_limit=False):
    if isinstance(keys, UaxSampleKeys):
        if len(keys) == 1 and not normalize and keys.degree in (0, 1, 2, 3):
            return keys.controls[0]
        times, degree, controls = keys.times, keys.degree, keys.controls
    else:
        ordered = sorted(keys, key=lambda item: item[0])
        times, degree = tuple(item[0] for item in ordered), 1
        controls = tuple(tuple(component for row in item[1] for component in row) if matrix else item[1]
                         for item in ordered)
    duration = keys.duration if isinstance(keys, UaxSampleKeys) else 0.0
    looping = keys.looping if isinstance(keys, UaxSampleKeys) else False
    quaternion = keys.quaternion if isinstance(keys, UaxSampleKeys) else False
    return sample_granny_curve(times, controls, degree, time, normalize, duration, looping, looping, quaternion, left_limit)


def uax_curve_break_times(keys):
    if len(keys) < 2:
        return set()
    candidates = set()
    for index, time in enumerate(keys.times):
        if time <= 0 or (keys.duration > 0 and time > keys.duration):
            continue
        if keys.degree == 0 or (index >= keys.degree and keys.times[index-keys.degree] == time):
            candidates.add(time)
    return {time for time in candidates
            if max(abs(a-b) for a, b in zip(sample_uax_keys(keys, time, normalize=keys.quaternion),
                sample_uax_keys(keys, time, normalize=keys.quaternion, left_limit=True))) > 1e-6}
