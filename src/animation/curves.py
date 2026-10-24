"""Evaluate decoded Granny spline controls without Blender scene state."""

from bisect import bisect_left, bisect_right
import math

from ..errors import UgxError

__all__ = ('sample_granny_curve', 'sample_granny_curve_batch')


def sample_granny_curve_batch(keys, times, left_limit=False):
    import numpy as np

    knots, controls = keys.spline_arrays
    degree = keys.degree
    count = len(knots)
    if not count:
        raise UgxError('Cannot sample a Granny curve without controls.')
    if degree not in (0, 1, 2, 3):
        raise UgxError(f'Unsupported Granny spline degree {degree}.')
    if keys.looping and (not math.isfinite(keys.duration) or keys.duration <= 0):
        raise UgxError('Looped Granny curve sampling requires a positive clip duration.')
    times = np.clip(times, 0 if keys.duration > 0 else knots[0], keys.duration if keys.duration > 0 else knots[-1])
    ahead = np.minimum(np.searchsorted(knots, times, side='left' if left_limit else 'right'), count-1)
    if count == 1 or degree == 0:
        return controls[np.where((ahead == count-1) & keys.looping & (count > 1), 0, ahead)]
    local_count = count-1 if keys.looping else count
    indices = ahead[:, None] + np.arange(2*degree) - degree
    if keys.looping:
        cycles, indices = np.divmod(indices, count-1)
        window_knots = knots[indices] + cycles*keys.duration
    else:
        indices = np.clip(indices, 0, count-1)
        window_knots = knots[indices]
    values = controls[indices[:, :degree+1]].copy()
    if keys.quaternion:
        boundary = (ahead < degree) | (ahead+degree > local_count)
        for index in range(1, degree+1):
            flip = boundary & (np.sum(values[:, index-1]*values[:, index], axis=1) < 0)
            values[flip, index] *= -1
    for level in range(1, degree+1):
        for index in range(degree, level-1, -1):
            left = window_knots[:, index-1]
            width = window_knots[:, index+degree-level]-left
            ratio = np.divide(times-left, width, out=np.zeros_like(times), where=width != 0)
            values[:, index] = ((1-ratio[:, None])*values[:, index-1] + ratio[:, None]*values[:, index])
    return values[:, degree]


def sample_granny_curve(times, controls, degree, time, normalize=False, duration=0.0,
                       underflow_loop=False, overflow_loop=False, quaternion=False, left_limit=False):
    if not times:
        raise UgxError("Cannot sample a Granny curve without controls.")
    if degree not in (0, 1, 2, 3):
        raise UgxError(f"Unsupported Granny spline degree {degree}.")
    count = len(times)
    if (underflow_loop or overflow_loop) and (not math.isfinite(duration) or duration <= 0):
        raise UgxError("Looped Granny curve sampling requires a positive clip duration.")
    time = max(0.0, min(duration, time)) if duration > 0 else max(times[0], min(times[-1], time))
    knot_index = min((bisect_left if left_limit else bisect_right)(times, time), count - 1)
    if count == 1 or degree == 0:
        index = 0 if count > 1 and overflow_loop and knot_index == count-1 else knot_index
        value = tuple(controls[index])
    else:
        # Granny centers its window on the knot ahead of the sample. Endpoint
        # windows repeat the first or last knot and control, without lowering
        # the authored degree when a curve has few controls.
        # A forward loop treats the final knot as a duration marker. Wrapped
        # controls come from the adjacent cycle, excluding that final marker.
        local_count = count - 1 if overflow_loop else count
        wrapped_count = count - 1
        knots, window = [], []
        for i in range(2 * degree):
            index, offset = knot_index + i - degree, 0.0
            if index < 0:
                if underflow_loop:
                    cycle, index = divmod(index, wrapped_count)
                    offset = cycle * duration
                else:
                    index = 0
            elif index >= local_count:
                if overflow_loop:
                    cycle, index = divmod(index - local_count, wrapped_count)
                    offset = (cycle + 1) * duration
                else:
                    index = count - 1
            knots.append(times[index] + offset)
            if i <= degree:
                window.append(tuple(controls[index]))
        if (normalize or quaternion) and (knot_index < degree or knot_index + degree > local_count):
            for i in range(1, len(window)):
                if sum(a * b for a, b in zip(window[i - 1], window[i])) < 0:
                    window[i] = tuple(-component for component in window[i])
        coefficients = [0.0] * (degree + 1)

        def accumulate(level, at, weight, control):
            if level == 0:
                coefficients[control] += weight
                return
            left, right = knots[at - 1], knots[at + degree - level]
            ratio = (time - left) / (right - left) if right != left else 0.0
            accumulate(level - 1, at - 1, weight * (1.0 - ratio), control - 1)
            accumulate(level - 1, at, weight * ratio, control)

        accumulate(degree, degree, 1.0, degree)
        value = tuple(sum(weight * control[i] for weight, control in zip(coefficients, window))
                      for i in range(len(window[0])))
    if normalize:
        length = math.sqrt(sum(component * component for component in value))
        if not math.isfinite(length) or length <= 1e-12:
            raise UgxError("Granny rotation curve samples to an invalid quaternion.")
        value = tuple(component / length for component in value)
    return value
