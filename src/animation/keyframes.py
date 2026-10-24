"""Convert fitted native channels to editable Blender keys and free handles."""
from dataclasses import dataclass
from bisect import bisect_left, bisect_right
import math
import struct
from typing import List, Tuple

from ..errors import UgxError
from .curve_fitting import BezierSpan

__all__ = ('uax_frame', 'uax_reduce_scalar_keys', 'uax_bezier_channel_keys')


def float32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def uax_scene_evaluation_frame(frame):
    # Scene.frame_set rounds the stored subframe before adding the integer frame.
    whole = math.floor(frame)
    return float32(whole+float32(frame-whole))


def uax_scalar_span_value(spans, ends, time):
    span = spans[min(bisect_right(ends, time), len(spans)-1)]
    if time >= span.end:
        return span.end_value[0] if span.end_value else span.controls[-1][0]
    ratio = max(0., (time-span.start)/(span.end-span.start))
    inverse = 1-ratio
    (a,), (b,), (c,), (d,) = span.controls
    return inverse**3*a+3*inverse*inverse*ratio*b+3*inverse*ratio*ratio*c+ratio**3*d


def uax_pulse_channel(spans, component, fps, tolerance=1e-5):
    if not spans:
        return None
    raw_changes = []
    rapid = False
    for span in spans:
        controls = tuple(point[component] for point in span.controls)
        atomic = (struct.unpack('<I', struct.pack('<f', span.end))[0] -
                  struct.unpack('<I', struct.pack('<f', span.start))[0] <= 1)
        post = span.end_value[component] if span.end_value else controls[-1]
        if abs(post-controls[-1]) > tolerance:
            raw_changes.append(span.end)
            rapid |= (span.end-span.start)*fps < .011
        if ((atomic and abs(post-controls[0]) > tolerance)
                or ((span.end-span.start)*fps < .011 and max(controls)-min(controls) > tolerance)):
            raw_changes.extend((span.start, span.end))
            rapid = True
    raw_changes = sorted(set(raw_changes))
    rapid |= any((b-a)*fps < .011 for a, b in zip(raw_changes, raw_changes[1:]))
    if not rapid:
        return None
    native = [BezierSpan(span.start, span.end, tuple((point[component],) for point in span.controls),
                         (span.end_value[component],) if span.end_value else ()) for span in spans]
    ends = [span.end for span in native]
    windows = []
    for time in raw_changes:
        first = float32(max(0., time-.02/fps))
        last = float32(min(spans[-1].end, time+.02/fps))
        if windows and first <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(last, windows[-1][1]))
        else:
            windows.append((first, last))
    base = []
    window_index = 0
    for span in native:
        while window_index < len(windows) and windows[window_index][1] <= span.start:
            window_index += 1
        cursor = span.start
        pieces = []
        for low, high in windows[window_index:]:
            if low >= span.end:
                break
            if cursor < low:
                pieces.append((cursor, low))
            cursor = max(cursor, high)
            if cursor >= span.end:
                break
        if cursor < span.end:
            pieces.append((cursor, span.end))
        for first, last in pieces:
            controls = tuple(point[0] for point in span.controls)
            if first > span.start:
                controls = split_controls(controls, (first-span.start)/(span.end-span.start))[1]
            if last < span.end:
                controls = split_controls(controls, (last-first)/(span.end-first))[0]
            base.append(BezierSpan(first, last, tuple((value,) for value in controls),
                                   span.end_value if last == span.end else ()))
    for first, last in windows:
        if first >= last:
            continue
        a, b = uax_scalar_span_value(native, ends, first), uax_scalar_span_value(native, ends, last)
        base.append(BezierSpan(first, last, ((a,), ((2*a+b)/3,), ((a+2*b)/3,), (b,))))
    return sorted(base, key=lambda span: span.start), native, windows


def add_uax_pulse_envelope(curve, spans, fps, windows):
    import numpy as np
    rotation_channel = curve.data_path.endswith('rotation_quaternion')
    ends = [span.end for span in spans]
    end = float32(uax_frame(spans[-1].end, fps))
    curve.update()
    frame_windows = [(uax_scene_evaluation_frame(uax_frame(first, fps)),
                      uax_scene_evaluation_frame(uax_frame(last, fps))) for first, last in windows]
    window_ends = [last for _first, last in frame_windows]

    def inside(frame):
        index = bisect_left(window_ends, frame)
        return index < len(frame_windows) and frame_windows[index][0] <= frame

    def bits(value):
        return struct.unpack('<I', struct.pack('<f', value))[0]

    def from_bits(value):
        return struct.unpack('<f', struct.pack('<I', value))[0]

    seeds = {1., end}
    for first, last in frame_windows:
        seeds.update((first, last))
        for frame, delta in ((first, -1), (last, 1)):
            value = from_bits(bits(frame)+delta)
            if 1. <= value <= end:
                seeds.add(value)
    centers = set()
    for span in spans:
        values = [point[0] for point in span.controls]
        if (max(values)-min(values) > 1e-5
                or (span.end_value and abs(span.end_value[0]-values[-1]) > 1e-5)):
            for time in (span.start, span.end):
                frame = uax_scene_evaluation_frame(uax_frame(time, fps))
                if inside(frame):
                    centers.add(frame)
    centers.update(float(key.co.x) for key in curve.keyframe_points)
    for center in centers:
        for frame in (center-.0005, center-.0001, center, center+.0001, center+.0005):
            base = bits(float32(max(1., min(end, frame))))
            for delta in range(-4, 5):
                value = from_bits(base+delta)
                if 1. <= value <= end:
                    seeds.add(value)
    starts_array = np.array([span.start for span in spans], dtype=np.float64)
    ends_array = np.array(ends, dtype=np.float64)
    controls = np.array([[point[0] for point in span.controls] for span in spans], dtype=np.float64)
    posts = np.array([span.end_value[0] if span.end_value else span.controls[-1][0]
                      for span in spans], dtype=np.float64)
    window_first = np.array([first for first, _last in frame_windows], dtype=np.float64)
    window_last = np.array(window_ends, dtype=np.float64)

    # Choose the same native clock for each representable Blender frame.
    def residual(frames):
        clocks = ((frames-1)/fps).astype(np.float32).astype(np.float64)
        clock_bits = clocks.astype(np.float32).view(np.uint32).astype(np.int64)
        pending = np.arange(len(frames))
        for delta in (0, -1, 1, -2, 2, -3, 3, -4, 4):
            candidate_bits = clock_bits[pending]+delta
            candidate = np.maximum(candidate_bits, 0).astype(np.uint32).view(np.float32).astype(np.float64)
            native_frame = 1.+candidate*fps
            whole = np.floor(native_frame)
            mapped = (whole+(native_frame-whole).astype(np.float32).astype(np.float64))
            mapped = mapped.astype(np.float32).astype(np.float64)
            matched = (candidate_bits >= 0) & (mapped == frames[pending])
            clocks[pending[matched]] = candidate[matched]
            pending = pending[~matched]
            if not len(pending):
                break
        indices = np.minimum(np.searchsorted(ends_array, clocks, side='right'), len(spans)-1)
        ratio = np.clip((clocks-starts_array[indices])/(ends_array[indices]-starts_array[indices]), 0., 1.)
        inverse = 1-ratio
        a, b, c, d = controls[indices].T
        # Preserve scalar pow rounding before storing float32 residuals.
        inverse_cubed = np.fromiter((float(value)**3 for value in inverse),
                                   dtype=np.float64, count=len(inverse))
        ratio_cubed = np.fromiter((float(value)**3 for value in ratio), dtype=np.float64, count=len(ratio))
        values = inverse_cubed*a+3*inverse*inverse*ratio*b+3*inverse*ratio*ratio*c+ratio_cubed*d
        past = clocks >= ends_array[indices]
        values[past] = posts[indices[past]]
        base_values = np.fromiter((curve.evaluate(float(frame)) for frame in frames),
                                  dtype=np.float64, count=len(frames))
        return (values-base_values).astype(np.float32).astype(np.float64), values

    ordered = np.array(sorted(seeds), dtype=np.float64)
    if len(ordered) < 2:
        return
    initial, _values = residual(ordered)
    first, last = ordered[:-1], ordered[1:]
    low, high = initial[:-1], initial[1:]
    accepted_frames, accepted_values = [], []
    # Independent intervals refine together; restore their frame order below.
    while len(first):
        first_bits = first.astype(np.float32).view(np.uint32).astype(np.uint64)
        last_bits = last.astype(np.float32).view(np.uint32).astype(np.uint64)
        active = last_bits-first_bits > 1
        split = np.zeros(len(first), dtype=bool)
        middle_frames = np.full(len(first), np.nan)
        middle_values = np.empty(len(first))
        for probe_index, ratio in enumerate((.1464466094067262, .5, .8535533905932737)):
            pending = np.flatnonzero(active & ~split)
            frames = (first[pending]+(last[pending]-first[pending])*ratio)
            frames = frames.astype(np.float32).astype(np.float64)
            valid = (frames > first[pending]) & (frames < last[pending])
            pending, frames = pending[valid], frames[valid]
            if not len(frames):
                continue
            actual, values = residual(frames)
            if probe_index == 1:
                middle_frames[pending] = frames
                middle_values[pending] = actual
            expected = (low[pending]+(high[pending]-low[pending])*(frames-first[pending])
                        /(last[pending]-first[pending]))
            predicted = values-actual+expected
            tolerance = np.full(len(frames), 1e-6 if rotation_channel else 1e-5)
            if not rotation_channel and len(window_last):
                indices = np.searchsorted(window_last, frames, side='left')
                in_window = ((indices < len(window_last))
                             & (window_first[np.minimum(indices, len(window_last)-1)] <= frames))
                tolerance[in_window] = 1e-6
            difference = np.abs(values-predicted)
            exponent = np.frexp(np.maximum(np.abs(values), np.abs(predicted)))[1]-24
            spacing = np.ldexp(np.ones(len(frames)), np.maximum(-149, exponent))
            split[pending] = (~(difference <= tolerance)
                              & (~np.isfinite(values) | ~np.isfinite(predicted)
                                 | (difference-spacing > tolerance)))
        accepted_frames.append(last[~split])
        accepted_values.append(high[~split])
        if not np.any(split):
            break
        first, last, low, high = first[split], last[split], low[split], high[split]
        middle_bits = ((first_bits[split]+last_bits[split])//2).astype(np.uint32)
        middle = middle_bits.view(np.float32).astype(np.float64)
        cached = middle_frames[split] == middle
        value = middle_values[split]
        if not np.all(cached):
            value[~cached], _values = residual(middle[~cached])
        first, last = np.concatenate((first, middle)), np.concatenate((middle, last))
        low, high = np.concatenate((low, value)), np.concatenate((value, high))
    frames = np.concatenate(accepted_frames)
    values = np.concatenate(accepted_values)
    order = np.argsort(frames)
    frames = np.concatenate((ordered[:1], frames[order]))
    values = np.concatenate((initial[:1], values[order]))
    # Keep both endpoints of each constant run.
    keep = np.ones(len(frames), dtype=bool)
    keep[1:-1] = (values[:-2] != values[1:-1]) | (values[1:-1] != values[2:])
    frames, values = frames[keep], values[keep]
    peak = float(np.max(np.abs(values)))
    if peak < 1e-6:
        return
    # Equal-width ranges add detail without changing ordinary key edits.
    modifier = curve.modifiers.new('ENVELOPE')
    modifier.name = 'Subframe detail'
    modifier.reference_value = 0.
    modifier.default_min = 0.
    radius = 2.**math.ceil(math.log2(max(1., peak)))
    modifier.default_max = radius
    # Allocate separate frames before assigning nearby native timestamps.
    for index in range(len(frames)):
        modifier.control_points.add(float(index))
    # Bulk writes avoid the invalid control-point callbacks in older Blender.
    modifier.control_points.foreach_set('frame', frames.astype(np.float32))
    modifier.control_points.foreach_set('min', values.astype(np.float32))
    modifier.control_points.foreach_set('max', (values+radius).astype(np.float32))
    curve.id_data.update_tag()


def step_key_frame(frame):
    # Counter Blender's 0.0001-frame exact-key evaluation window. Keep the
    # jump at its native frame instead of advancing it on every import.
    epsilon = float32(.0001)
    key = float32(frame+epsilon)
    if key-frame > epsilon:
        bits = struct.unpack('<I', struct.pack('<f', key))[0]
        key = struct.unpack('<f', struct.pack('<I', bits-1))[0]
    return key


@dataclass
class ChannelKey:
    frame: float
    value: float
    left: tuple
    right: tuple
    interpolation: str = 'BEZIER'


def split_controls(controls, ratio):
    a, b, c, d = controls
    ab, bc, cd = (a+(b-a)*ratio, b+(c-b)*ratio, c+(d-c)*ratio)
    abc, bcd = ab+(bc-ab)*ratio, bc+(cd-bc)*ratio
    value = abc+(bcd-abc)*ratio
    return (a, ab, abc, value), (value, bcd, cd, d)


def discontinuous_channel_keys(spans, component, fps, jump_tolerance=1e-6):
    points = []
    resume = None
    def append(frame, value):
        point = ChannelKey(frame, value, (frame, value), (frame, value))
        points.append(point)
        return point
    for span in spans:
        controls = tuple(value[component] for value in span.controls)
        native_start = float32(uax_frame(span.start, fps))
        end = float32(uax_frame(span.end, fps))
        post = span.end_value[component] if span.end_value else controls[-1]
        jump = abs(post-controls[-1]) > jump_tolerance
        constant = max(controls)-min(controls) < 1e-6
        start_bits = struct.unpack('<I', struct.pack('<f', span.start))[0]
        end_bits = struct.unpack('<I', struct.pack('<f', span.end))[0]
        atomic = end_bits-start_bits <= 1 and not constant
        if not points:
            append(native_start, controls[0])
        first = points[-1]
        if atomic:
            # No native float32 sample exists between these endpoints.
            frame = float32(end-.0002)
            if len(points) > 1:
                previous = points[-2]
                if frame <= previous.frame:
                    raise UgxError('Native animation resets are too close for Blender to represent separately.')
                ratio = (frame-previous.frame)/(first.frame-previous.frame)
                previous.right = (previous.frame+(previous.right[0]-previous.frame)*ratio, previous.right[1])
                first.left = (previous.frame+(first.left[0]-previous.frame)*ratio, first.left[1])
            first.frame = frame
            first.interpolation = 'CONSTANT'
            append(step_key_frame(end), post)
            resume = float32(end+.0003)
            continue
        if resume is not None:
            if constant and abs(first.value-controls[0]) < 1e-6:
                resume = None
            else:
                if end <= resume:
                    if jump:
                        raise UgxError('Discontinuous animation keys are closer than Blender can represent at this frame.')
                    continue
                ratio = max(0., min(1., (resume-native_start)/(end-native_start)))
                controls = split_controls(controls, ratio)[1]
                first.interpolation = 'LINEAR'
                first = append(resume, controls[0])
                native_start = resume
                resume = None
        start = first.frame
        finish = step_key_frame(end) if jump else end
        if finish <= start and constant and abs(post-first.value) < 1e-6:
            continue
        if finish <= start:
            raise UgxError('Discontinuous animation keys are closer than Blender can represent at this frame.')
        if constant:
            first.interpolation = 'CONSTANT'
            append(finish, post)
        else:
            curve_end = float32(end-.0002) if jump else end
            if curve_end <= start:
                raise UgxError('A curved animation segment is too short for Blender to retain its endpoint jump.')
            if jump:
                ratio = (curve_end-native_start)/(end-native_start)
                controls = split_controls(controls, ratio)[0]
            p0, p1, p2, p3 = controls
            first.right = (start+(curve_end-start)/3, p1)
            last = append(curve_end, p3)
            last.left = (start+2*(curve_end-start)/3, p2)
            if jump:
                last.interpolation = 'CONSTANT'
                append(finish, post)
        if jump:
            resume = float32(end+.0003)
    reduced = []
    for point in points:
        # Redundant hold keys can win Blender's exact-key lookup before the
        # following reset, delaying that reset by its entire snapping window.
        if (reduced and reduced[-1].interpolation == point.interpolation == 'CONSTANT'
                and reduced[-1].value == point.value):
            continue
        reduced.append(point)
    points = reduced
    return ([(point.frame, point.value) for point in points],
            [(point.left, point.right) for point in points],
            [point.interpolation for point in points])


def uax_frame(time_seconds: float, fps: float) -> float:
    if not math.isfinite(float(time_seconds)):
        return 1.0
    if not math.isfinite(fps) or fps <= 0:
        raise UgxError('Animation frame rate must be finite and positive.')
    return 1.0 + max(0.0, float(time_seconds)) * fps


def uax_reduce_scalar_keys(keys: List[Tuple[float, float]], tolerance: float = 0.000001) -> List[Tuple[float, float]]:
    if len(keys) <= 2:
        return keys
    reduced = [keys[0]]
    last_value = float(keys[0][1])
    for index in range(1, len(keys) - 1):
        frame, value = keys[index]
        current = float(value)
        next_value = float(keys[index + 1][1])
        # Keep the end of a hold before movement starts. Removing it makes a
        # linear F-curve start moving at the beginning of the hold instead.
        if abs(current - last_value) > tolerance or abs(next_value - current) > tolerance:
            reduced.append((frame, current))
            last_value = current
    if reduced[-1][0] != keys[-1][0] or abs(float(keys[-1][1]) - float(reduced[-1][1])) > tolerance:
        reduced.append(keys[-1])
    elif len(reduced) == 1:
        reduced.append(keys[-1])
    return reduced


def uax_bezier_channel_keys(spans, component, fps, jump_tolerance=1e-6):
    if any(span.end_value and abs(span.end_value[component]-span.controls[-1][component]) > jump_tolerance for span in spans):
        return discontinuous_channel_keys(spans, component, fps, jump_tolerance)
    parts = []
    for span in spans:
        start = struct.unpack('<f', struct.pack('<f', uax_frame(span.start, fps)))[0]
        end = struct.unpack('<f', struct.pack('<f', uax_frame(span.end, fps)))[0]
        controls = tuple(point[component] for point in span.controls)
        if end <= start:
            if max(controls)-min(controls) <= 1e-6:
                continue
            bits = struct.unpack('<I', struct.pack('<f', end))[0]
            start = struct.unpack('<f', struct.pack('<I', bits-1))[0]
            # Native seconds and Blender frames have different float32 spacing.
            # Keep a discontinuous terminal value on the preceding frame tick;
            # a collapsed constant marker must not replace its incoming handle.
            while parts and parts[-1][0] >= start:
                previous = parts.pop()
                controls = (previous[2][0], controls[1], controls[2], controls[3])
            if parts:
                a, _b, values = parts[-1]
                parts[-1] = (a, start, values)
        parts.append((start, end, controls))
    if not parts:
        first, last = spans[0], spans[-1]
        return [(uax_frame(first.start, fps), first.controls[0][component]),
                (uax_frame(last.end, fps), last.controls[-1][component])], None, None
    keys = [(parts[0][0], parts[0][2][0])]
    left, right = [keys[0]], []
    linear = True
    for start, end, controls in parts:
        p0, p1, p2, p3 = controls
        right.append((start+(end-start)/3, p1))
        left.append((start+2*(end-start)/3, p2))
        keys.append((end, p3))
        linear &= abs(p1-(2*p0+p3)/3) <= 1e-7 and abs(p2-(p0+2*p3)/3) <= 1e-7
    right.append(keys[-1])
    if linear:
        return uax_reduce_scalar_keys(keys), None, None
    return keys, list(zip(left, right)), None


def uax_bezier_transform_keys(spans, fps, jump_tolerance=1e-6):
    import numpy as np

    if not math.isfinite(fps) or fps <= 0:
        raise UgxError('Animation frame rate must be finite and positive.')
    frames = np.asarray([(span.start, span.end) for span in spans])
    frames = (1+np.maximum(0, frames)*fps).astype(np.float32).astype(float)
    controls = np.asarray([span.controls for span in spans])
    collapsed = np.any(frames[:, 1] <= frames[:, 0])
    jumps = np.zeros(10, dtype=bool)
    for span in spans:
        if span.end_value:
            jumps |= np.abs(np.asarray(span.end_value[:10])-np.asarray(span.controls[-1][:10])) > jump_tolerance
    linear = np.all((np.abs(controls[:, 1, :10]-(2*controls[:, 0, :10]+controls[:, 3, :10])/3) <= 1e-7)
        & (np.abs(controls[:, 2, :10]-(controls[:, 0, :10]+2*controls[:, 3, :10])/3) <= 1e-7), axis=0)
    constant = linear & (np.ptp(controls[:, :, :10], axis=(0, 1)) <= 1e-6) & ~jumps
    times = np.concatenate((frames[:1, 0], frames[:, 1])).tolist()
    widths = frames[:, 1]-frames[:, 0]
    left_times = np.concatenate((frames[:1, 0], frames[:, 0]+2*widths/3)).tolist()
    right_times = np.concatenate((frames[:, 0]+widths/3, frames[-1:, 1])).tolist()
    result = []
    for index in range(10):
        if constant[index]:
            result.append(([(times[0], float(controls[0, 0, index])),
                (times[-1], float(controls[-1, 3, index]))], None, None))
            continue
        if jumps[index] or collapsed:
            result.append(uax_bezier_channel_keys(spans, index, fps, jump_tolerance=jump_tolerance))
            continue
        values = np.concatenate((controls[:1, 0, index], controls[:, 3, index])).tolist()
        keys = list(zip(times, values))
        if linear[index]:
            result.append((uax_reduce_scalar_keys(keys), None, None))
            continue
        left = list(zip(left_times, [values[0]]+controls[:, 2, index].tolist()))
        right = list(zip(right_times, controls[:, 1, index].tolist()+[values[-1]]))
        result.append((keys, list(zip(left, right)), None))
    return result
