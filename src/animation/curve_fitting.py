"""Fit editable cubic spans to sampled animation channels."""
from dataclasses import dataclass
from bisect import bisect_left, bisect_right
import math
import struct

from ..errors import UgxError

__all__ = ('BezierSpan', 'bezier_value', 'float32_curve_error', 'fit_bezier_steps', 'bezier_granny_controls')


@dataclass(frozen=True)
class BezierSpan:
    start: float
    end: float
    controls: tuple
    end_value: tuple = ()


def bezier_value(controls, ratio):
    inverse = 1 - ratio
    weights = (inverse**3, 3*inverse*inverse*ratio, 3*inverse*ratio*ratio, ratio**3)
    a, b, c, d = weights
    return tuple(0.+a*w+b*x+c*y+d*z for w, x, y, z in zip(*controls))


def float32_curve_error(left, right):
    maximum = 0.0
    for a, b in zip(left, right):
        difference = abs(a-b)
        if difference <= maximum:
            continue
        if not math.isfinite(a) or not math.isfinite(b):
            return math.inf
        # Native transforms and Blender channels round to float32. Refining
        # below one representable step cannot remove that quantization noise.
        magnitude = max(abs(a), abs(b))
        exponent = math.frexp(magnitude)[1] - 24
        spacing = math.ldexp(1.0, max(-149, exponent))
        maximum = max(maximum, difference-spacing)
    return maximum


def align_quaternions(value, reference, offsets):
    if not offsets:
        return tuple(value)
    result = list(value)
    for offset in offsets:
        if sum(a*b for a, b in zip(result[offset:offset+4], reference[offset:offset+4])) < 0:
            result[offset:offset+4] = [-v for v in result[offset:offset+4]]
    return tuple(result)


def fit_bezier_steps(times, sample, error=None, quaternion_offsets=(), tolerance=1e-5, max_depth=20,
                     check_times=(), sample_left=None, native_clock=False):
    if error is None:
        error = lambda left, right: max(abs(a-b) for a, b in zip(left, right))
    times = sorted({struct.unpack('<f', struct.pack('<f', time))[0] for time in times}) if native_clock else sorted(set(times))
    # Small clock values can need more subdivisions to reach adjacent float32
    # samples than the continuous fitter's ordinary refinement limit allows.
    if native_clock:
        max_depth = max(max_depth, 32)
    check_times = sorted(set(check_times))
    if len(times) < 2:
        return
    samples = {}
    def sampled(time):
        if time in samples:
            return samples[time]
        values = tuple(sample(time))
        if not all(math.isfinite(v) for v in values):
            raise UgxError(f'Animation contains a non-finite transform at {time:.9g} seconds.')
        samples[time] = values
        return values
    left = sampled(times[0])
    previous = left
    for start, end in zip(times, times[1:]):
        after = align_quaternions(sampled(end), left, quaternion_offsets)
        before = sample_left(end) if sample_left is not None else None
        right = align_quaternions(tuple(before), left, quaternion_offsets) if before is not None else after
        if not all(math.isfinite(v) for v in right):
            raise UgxError(f'Animation contains a non-finite left limit at {end:.9g} seconds.')
        pending = [(start, end, left, right, 0)]
        while pending:
            a, b, p0, p3, depth = pending.pop()
            # Refinement can cross the quaternion hemisphere chosen for the
            # original span. Keep both child endpoints on the continuous branch.
            p0 = align_quaternions(p0, previous, quaternion_offsets)
            p3 = align_quaternions(p3, p0, quaternion_offsets)
            width = b-a
            def at(ratio, time=None):
                reference = tuple((1-ratio)*x+ratio*y for x, y in zip(p0, p3))
                return align_quaternions(sampled(a+width*ratio if time is None else time), reference, quaternion_offsets)
            atomic = native_clock and (
                struct.unpack('<I', struct.pack('<f', b))[0] -
                struct.unpack('<I', struct.pack('<f', a))[0] <= 1)
            if atomic:
                p1 = tuple((2*x+y)/3 for x, y in zip(p0, p3))
                p2 = tuple((x+2*y)/3 for x, y in zip(p0, p3))
            elif native_clock:
                interior = sorted({struct.unpack('<f', struct.pack('<f', a+width*ratio))[0]
                                   for ratio in (1/3, 2/3)}-{a, b})
                rows = []
                for time in interior:
                    ratio = (time-a)/width
                    inverse = 1-ratio
                    rows.append((3*inverse*inverse*ratio, 3*inverse*ratio*ratio,
                                 tuple(value-inverse**3*x-ratio**3*y
                                       for value, x, y in zip(at(ratio, time), p0, p3))))
                if len(rows) == 2:
                    a1, b1, v1 = rows[0]
                    a2, b2, v2 = rows[1]
                    determinant = a1*b2-a2*b1
                    p1 = tuple((x*b2-y*b1)/determinant for x, y in zip(v1, v2))
                    p2 = tuple((y*a1-x*a2)/determinant for x, y in zip(v1, v2))
                else:
                    # Two-tick intervals have only one interior native sample.
                    # Fit that sample without evaluating invented clock values.
                    a1, b1, values = rows[0]
                    linear1 = tuple((2*x+y)/3 for x, y in zip(p0, p3))
                    linear2 = tuple((x+2*y)/3 for x, y in zip(p0, p3))
                    correction = tuple((value-a1*x-b1*y)/(a1+b1)
                                       for value, x, y in zip(values, linear1, linear2))
                    p1 = tuple(value+delta for value, delta in zip(linear1, correction))
                    p2 = tuple(value+delta for value, delta in zip(linear2, correction))
            else:
                first, second = at(1/3), at(2/3)
                p1 = tuple(3*x-1.5*y-5*z/6+w/3 for x, y, z, w in zip(first, second, p0, p3))
                p2 = tuple(3*y-1.5*x+z/3-5*w/6 for x, y, z, w in zip(first, second, p0, p3))
            controls = (p0, p1, p2, p3)
            ratios = [.1464466094067262, .5, .8535533905932737]
            ratios.extend((time-a)/width for time in check_times[bisect_right(check_times, a):bisect_left(check_times, b)])
            probes = []
            for ratio in dict.fromkeys(ratios):
                if native_clock:
                    time = struct.unpack('<f', struct.pack('<f', a+width*ratio))[0]
                    # Adjacent native clocks have no intermediate sample. A
                    # Blender jump between them does not need a smooth fit.
                    if not a < time < b:
                        continue
                    ratio = (time-a)/width
                probes.append((ratio, at(ratio, time) if native_clock else at(ratio)))
            deviation = max((error(value, bezier_value(controls, ratio)) for ratio, value in probes), default=0.)
            if deviation <= tolerance:
                endpoint = align_quaternions(after, p3, quaternion_offsets) if b == end and error(after, p3) > 1e-12 else ()
                previous = endpoint or p3
                yield BezierSpan(a, b, controls, endpoint)
            else:
                middle = a+width*.5
                if native_clock:
                    # Bisect representable clocks, rather than their durations.
                    # Near zero, arithmetic halving can exhaust the depth limit
                    # while many native samples still straddle a discontinuity.
                    first = struct.unpack('<I', struct.pack('<f', a))[0]
                    last = struct.unpack('<I', struct.pack('<f', b))[0]
                    middle = struct.unpack('<f', struct.pack('<I', (first+last)//2))[0]
                if depth >= max_depth or middle in (a, b):
                    raise UgxError(f'Animation curve cannot meet {tolerance:g} transform precision between {a:.9g} and {b:.9g} seconds (error {deviation:g}).')
                pm = at((middle-a)/width, middle) if native_clock else at(.5)
                pending.append((middle, b, pm, p3, depth+1))
                pending.append((a, middle, p0, pm, depth+1))
                yield None
        left = align_quaternions(after, previous, quaternion_offsets)


def native_bezier_steps(times, sample, error):
    import numpy as np

    times = np.asarray(times)
    if len(times) < 2:
        return
    starts, ends = times[:-1], times[1:]
    widths = ends-starts
    sample_times, inverse = np.unique(np.concatenate((starts, starts+widths/3, starts+2*widths/3, ends)), return_inverse=True)
    values = sample(sample_times)[inverse]
    before = sample(ends, True)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(before)):
        raise UgxError('Animation contains a non-finite transform.')
    left, first, second, after = np.split(values, 4)
    # Rigid basis conversion is affine in the native channels. Each native
    # interval has degree at most three, so these controls reproduce it exactly.
    p1 = 3*first-1.5*second-5*left/6+before/3
    p2 = 3*second-1.5*first+left/3-5*before/6
    controls = np.stack((left, p1, p2, before), axis=1)
    jumps = error(after, before) > 1e-12
    previous = left[0, 3:7]
    for index, (start, end) in enumerate(zip(starts, ends)):
        points = controls[index].copy()
        if np.dot(points[0, 3:7], previous) < 0:
            points[:, 3:7] *= -1
        post = after[index].copy()
        if np.dot(post[3:7], points[-1, 3:7]) < 0:
            post[3:7] *= -1
        previous = post[3:7] if jumps[index] else points[-1, 3:7]
        yield BezierSpan(float(start), float(end), tuple(map(tuple, points.tolist())),
            tuple(post.tolist()) if jumps[index] else ())


def fit_native_candidate_steps(times, candidate, sample, error, check_times=(), breaks=(), tolerance=1e-5):
    import numpy as np

    spans = list(native_bezier_steps(times, candidate, error))
    if not spans:
        return
    owners, probes, ratios = [], [], []
    check_times = sorted(set(check_times))
    for index, span in enumerate(spans):
        values = [span.start, span.end]
        values.extend(span.start+(span.end-span.start)*ratio for ratio in (.1464466094067262, .5, .8535533905932737))
        values.extend(check_times[bisect_right(check_times, span.start):bisect_left(check_times, span.end)])
        values = sorted(set(values))
        owners.extend([index]*len(values))
        probes.extend(values)
        ratios.extend((value-span.start)/(span.end-span.start) for value in values)
    owners, ratios = np.asarray(owners), np.asarray(ratios)
    unique, inverse = np.unique(probes, return_inverse=True)
    expected = sample(unique)[inverse]
    controls = np.asarray([span.controls for span in spans])
    weights = np.stack(((1-ratios)**3, 3*(1-ratios)**2*ratios, 3*(1-ratios)*ratios**2, ratios**3), axis=1)
    actual = np.sum(weights[:, :, None]*controls[owners], axis=1)
    for index, span in enumerate(spans):
        if span.end_value:
            actual[(owners == index) & (ratios == 1)] = span.end_value
    deviations = error(sample(np.asarray(times[1:]), True), controls[:, -1])
    np.maximum.at(deviations, owners, error(expected, actual))
    accepted = []
    index = 0
    while index < len(spans):
        if deviations[index] <= tolerance:
            accepted.append(spans[index])
            index += 1
        else:
            first = index
            while index < len(spans) and deviations[index] > tolerance:
                index += 1
            for span in fit_bezier_batch_steps(times[first:index+1], sample, error, check_times, breaks, tolerance):
                if span is not None:
                    accepted.append(span)
                yield None
    previous = accepted[0].controls[0]
    for span in accepted:
        points = span.controls
        if sum(a*b for a, b in zip(points[0][3:7], previous[3:7])) < 0:
            points = tuple(point[:3]+tuple(-value for value in point[3:7])+point[7:] for point in points)
        post = align_quaternions(span.end_value, points[-1], (3,)) if span.end_value else ()
        previous = post or points[-1]
        yield BezierSpan(span.start, span.end, points, post)


def fit_bezier_batch_steps(times, sample, error, check_times=(), breaks=(), tolerance=1e-5, max_depth=20):
    import numpy as np

    times = np.asarray(sorted(set(times)))
    if len(times) < 2:
        return

    def sampled(values, left_limit=False):
        unique, inverse = np.unique(values, return_inverse=True)
        result = sample(unique, left_limit)[inverse]
        if not np.all(np.isfinite(result)):
            raise UgxError('Animation contains a non-finite transform.')
        return result

    def aligned(values, reference):
        values = values.copy()
        flip = np.sum(values[:, 3:7]*reference[:, 3:7], axis=1) < 0
        values[flip, 3:7] *= -1
        return values

    endpoints = sampled(times)
    for index in range(1, len(endpoints)):
        if np.dot(endpoints[index, 3:7], endpoints[index-1, 3:7]) < 0:
            endpoints[index, 3:7] *= -1
    p0, after = endpoints[:-1], endpoints[1:]
    p3 = after.copy()
    discontinuities = np.isin(times[1:], list(breaks))
    if np.any(discontinuities):
        p3[discontinuities] = aligned(sampled(times[1:][discontinuities], True), p0[discontinuities])
    jumps = error(after, p3) > 1e-12
    pending = [(a, b, left, right, post if jump else (), 0)
        for a, b, left, right, post, jump in zip(times[:-1], times[1:], p0, p3, after, jumps)]
    accepted = []
    check_times = np.asarray(sorted(set(check_times)))
    while pending:
        starts = np.array([row[0] for row in pending])
        ends = np.array([row[1] for row in pending])
        left = np.array([row[2] for row in pending])
        right = aligned(np.array([row[3] for row in pending]), left)
        widths = ends-starts
        owners, ratios = [], []
        for index, (a, b) in enumerate(zip(starts, ends)):
            probes = [.1464466094067262, .5, .8535533905932737]
            probes.extend((check_times[np.searchsorted(check_times, a, side='right'):
                np.searchsorted(check_times, b)]-a)/(b-a))
            ratios.extend(probes)
            owners.extend([index]*len(probes))
        owners, ratios = np.asarray(owners), np.asarray(ratios)
        probes = starts[owners]+widths[owners]*ratios
        values = sampled(np.concatenate((starts+widths/3, starts+2*widths/3, probes)))
        first = aligned(values[:len(pending)], (2*left+right)/3)
        second = aligned(values[len(pending):2*len(pending)], (left+2*right)/3)
        p1 = 3*first-1.5*second-5*left/6+right/3
        p2 = 3*second-1.5*first+left/3-5*right/6
        controls = np.stack((left, p1, p2, right), axis=1)
        reference = (1-ratios[:, None])*left[owners]+ratios[:, None]*right[owners]
        actual = aligned(values[2*len(pending):], reference)
        weights = np.stack(((1-ratios)**3, 3*(1-ratios)**2*ratios,
            3*(1-ratios)*ratios**2, ratios**3), axis=1)
        predicted = np.sum(weights[:, :, None]*controls[owners], axis=1)
        deviations = np.zeros(len(pending))
        np.maximum.at(deviations, owners, error(actual, predicted))
        next_pending = []
        for index, row in enumerate(pending):
            a, b, left_value, right_value, post, depth = row
            right_value = right[index]
            if deviations[index] <= tolerance:
                accepted.append((a, b, controls[index], post))
                continue
            middle = (a+b)*.5
            if depth >= max_depth or middle in (a, b):
                raise UgxError(f'Animation curve cannot meet {tolerance:g} transform precision between {a:.9g} and {b:.9g} seconds (error {deviations[index]:g}).')
            middle_value = actual[(owners == index) & (ratios == .5)][0]
            next_pending.append((a, middle, left_value, middle_value, (), depth+1))
            next_pending.append((middle, b, middle_value, right_value, post, depth+1))
        pending = next_pending
        yield None
    previous = endpoints[0]
    for start, end, values, endpoint in sorted(accepted, key=lambda item: item[0]):
        values = values.copy()
        if np.dot(values[0, 3:7], previous[3:7]) < 0:
            values[:, 3:7] *= -1
        post = aligned(np.array([endpoint]), values[-1:])[0] if len(endpoint) else ()
        previous = post if len(post) else values[-1]
        yield BezierSpan(float(start), float(end), tuple(map(tuple, values.tolist())), tuple(post.tolist()) if len(post) else ())


def bezier_granny_controls(spans):
    if not spans:
        return [], []
    duration = struct.unpack('<f', struct.pack('<f', spans[-1].end))[0]
    bits = struct.unpack('<I', struct.pack('<f', duration))[0]
    before_end = struct.unpack('<f', struct.pack('<I', bits-1))[0]
    terminal_value = spans[-1].end_value or spans[-1].controls[-1]
    closing_start = None
    if not before_end > spans[-1].start:
        if len(spans) < 2 or spans[-1].start != before_end:
            raise UgxError('Animation timing is finer than native float32 precision at the final curve interval.')
        # The final fitted span already occupies the reserved marker interval.
        # Keep its endpoint states without trying to reserve a second interval.
        closing_start = spans[-1].controls[0]
        spans = spans[:-1]
    # Retain the authored endpoint as a real control before Granny's final
    # timing marker. A loop skips the marker and wraps to its first control.
    # The closing interval occupies one float32 time step, so it does not move
    # the authored curve at any intermediate native sample time.
    knots = [spans[0].start]*3
    controls = [spans[0].controls[0]]
    for index, span in enumerate(spans):
        if index and controls[-1] != span.controls[0]:
            controls.append(span.controls[0])
            knots.append(span.start)
        endpoint = before_end if index == len(spans)-1 else span.end
        controls.extend(span.controls[1:])
        knots.extend([endpoint]*3)
    if closing_start is not None and controls[-1] != closing_start:
        controls.append(closing_start)
        knots.append(before_end)
    controls.extend((terminal_value, terminal_value, terminal_value))
    knots.append(duration)
    return knots, controls
