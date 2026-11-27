from fractions import Fraction

import numpy as np

from .jobs import SmootherError

MOVES = {"pan", "tilt", "slide", "push-in", "pullback"}
MIN_SPEED, MAX_SPEED = .55, 1.80


def safe_auto_pair(pair):
    return pair["reliable"] and pair["kind"] in MOVES and pair["confidence"] >= .75


def sections(pairs):
    start = 0
    while start < len(pairs):
        first = pairs[start]
        kind = first["kind"]
        reliable = first["reliable"]
        axis = np.array(first["vector"])
        axis /= max(np.linalg.norm(axis), 1e-9)
        end = start + 1
        while end < len(pairs):
            following = pairs[end]
            if kind != following["kind"] or reliable != following["reliable"] or kind == "cut":
                break
            direction = np.array(following["vector"])
            if kind in MOVES and direction @ axis < .75 * np.linalg.norm(direction):
                break
            end += 1
        yield start, end, kind, reliable, axis
        start = end


def window_rates(progress, dt):
    speed = progress / dt
    padded = np.pad(speed, 2, mode="edge")
    median = np.median(np.lib.stride_tricks.sliding_window_view(padded, 5), axis=1)
    padded = np.pad(median, 2, mode="edge")
    smoothed = np.convolve(padded, [1/9, 2/9, 3/9, 2/9, 1/9], mode="valid")
    return np.maximum(smoothed, .02 * np.mean(smoothed))


def stretch(progress, dt):
    speed = window_rates(progress, dt)
    ease_frames = max(2, min(8, round(.24 / np.median(dt))))
    distance = np.minimum(np.arange(len(dt)), np.arange(len(dt))[::-1])
    edge = np.clip(distance / ease_frames, 0, 1)
    ease = edge * edge * (3 - 2 * edge)
    relative = speed / np.average(speed, weights=dt)
    lo, hi = 0., 2 / max(1e-8, relative.min())
    for _ in range(36):
        center = (lo + hi) / 2
        factors = 1 + ease * (np.clip(center * relative, 1 / MAX_SPEED, 1 / MIN_SPEED) - 1)
        if np.dot(factors, dt) < dt.sum():
            lo = center
        else:
            hi = center
    return factors * dt


def output_frame_count(media, fps, target_duration=None):
    duration = media["duration"] if target_duration is None else target_duration
    if duration <= 0:
        raise SmootherError("Output duration must be positive.")
    frames = duration * float(fps)
    if frames > 2_000_000.5:
        raise SmootherError("Requested output exceeds the prototype limit of two million frames.")
    count = max(2 if len(media["timestamps"]) > 1 else 1, round(frames))
    if abs(count / float(fps) - duration) > 1 / float(fps) + 1e-8:
        raise SmootherError("Choose a higher frame rate to preserve both endpoint views within one output frame.")
    return count


def original_indices(media, fps, count, duration_scale=Fraction(1)):
    """Select the source picture actually on screen at each CFR clock tick."""
    last = len(media["timestamps"]) - 1
    ticks = media.get("timestamp_ticks")
    clock = fps * duration_scale
    if ticks is None:
        # In-memory timelines from callers can omit the FFprobe time base.
        source = media["timestamps"]
        indices = np.searchsorted(source, np.arange(count) * clock.denominator / clock.numerator,
                                  side="right") - 1
        indices[-1] = last
        yield from map(int, indices)
        return
    time_base = Fraction(media["time_base"])
    source_factor = time_base.numerator * clock.numerator
    output_factor = clock.denominator * time_base.denominator
    current = 0
    for index in range(count - 1):
        while current < last and ticks[current + 1] * source_factor <= index * output_factor:
            current += 1
        yield current
    yield last


def plan(media, pairs, fps, target_duration=None):
    source = np.array(media["timestamps"])
    if len(source) != len(pairs) + 1:
        raise SmootherError("Analysis and source frame counts disagree.")
    duration = media["duration"] if target_duration is None else target_duration
    duration_scale = Fraction(str(duration)) / Fraction(str(media["duration"]))
    count = output_frame_count(media, fps, duration)
    output = np.arange(count) * fps.denominator / fps.numerator
    output_anchors = source.copy()
    intervals = []
    adjusted = np.zeros(len(pairs), bool)
    movement = np.zeros(len(pairs), bool)
    input_speed = np.zeros(len(pairs))
    for start, end, kind, reliable, axis in sections(pairs):
        rows = pairs[start:end]
        dt = np.diff(source[start:end+1])
        progress = np.maximum(0, np.array([p["vector"] for p in rows]) @ axis)
        input_speed[start:end] = progress / dt
        applied = False
        reasons = sorted({p["reason"] for p in rows if p["reason"]})
        if kind in MOVES and reliable:
            movement[start:end] = True
            if end - start >= 8 and dt.sum() >= .30:
                widths = stretch(progress, dt)
                if np.max(abs(widths / dt - 1)) > .025:
                    output_anchors[start+1:end] = source[start] + np.cumsum(widths)[:-1]
                    adjusted[start:end] = True
                    applied = True
            else:
                reasons.append("Reliable move is too brief to ease safely.")
        intervals.append({"start_pair": start, "end_pair": end,
                          "start": float(source[start]), "end": float(source[end]), "kind": kind,
                          "confidence": float(np.mean([p["confidence"] for p in rows])),
                          "reliable": reliable, "retimed": applied, "reasons": reasons})

    cut_starts = [i + 1 for i, item in enumerate(pairs) if item["kind"] == "cut"]
    output_anchors *= float(duration_scale)
    speed_output = input_speed * np.divide(np.diff(source), np.diff(output_anchors))
    brackets = np.searchsorted(output_anchors, output, side="right") - 1
    mapping = []
    candidates = auto_candidates = 0
    for index, active in enumerate(original_indices(media, fps, count, duration_scale)):
        time = float(output[index])
        left = int(brackets[index])
        if (0 < index < count - 1 and 0 <= left < len(adjusted)
                and (adjusted[left] or (duration_scale > 1 and movement[left]))):
            right = left + 1
            span = output_anchors[right] - output_anchors[left]
            step = source[right] - source[left]
            fraction = float((time - output_anchors[left]) / span)
            sampled = float(source[left] + fraction * step)
            nearest = right if fraction >= .5 else left
            allow_flow = .10 < fraction < .90 and step / span < .97
            candidates += int(allow_flow)
            auto_candidates += int(allow_flow and safe_auto_pair(pairs[left]))
        else:
            left = right = nearest = active
            sampled = float(source[-1]) if index == count - 1 else min(time / float(duration_scale), float(source[-1]))
            fraction = 0.
            allow_flow = False
        mapping.append({"output_time": time, "source_time": sampled, "left_frame": left,
                        "right_frame": right, "fraction": fraction, "original_frame": nearest,
                        "interpolation_candidate": bool(allow_flow)})
    motion_types = sorted({item["kind"] for item in intervals if item["kind"] in MOVES and item["reliable"]})
    warnings = sorted({reason for item in intervals for reason in item["reasons"]})
    if len(motion_types) > 1:
        warnings.append("Multiple apparent motion types were detected. See the segment boundaries.")
    if cut_starts:
        warnings.append("Cut boundaries are excluded from correction and interpolation.")
        selected = np.array([item["original_frame"] for item in mapping])
        boundaries = [0, *cut_starts, len(source)]
        first_samples = np.searchsorted(selected, boundaries)
        skipped = [f"{source[start]:.6f}s" for start, size in zip(boundaries, np.diff(first_samples)) if size == 0]
        if skipped:
            warnings.append(f"Output frame grid skips {len(skipped)} detected shot(s) starting at "
                            f"{', '.join(skipped[:8])}. Increase the output frame rate to preserve them.")
    uncertain = [segment for segment in intervals if not segment["reliable"] and segment["kind"] != "cut"]
    retimed = sum(segment["retimed"] for segment in intervals)
    recommendation = {
        "retime": "recommended" if retimed else ("review-first" if uncertain else "little-change"),
        "retimed_segments": retimed,
        "uncertain_section_count": len(uncertain),
        "uncertain_sections": [{key: section[key] for key in ("start", "end", "kind", "reasons")}
                               for section in uncertain[:12]],
        "frame_generation": "preview-auto" if auto_candidates else "captured-frames",
        "interpolation_candidates": candidates,
        "auto_candidates": auto_candidates,
    }
    return {"fps": str(fps), "duration": count / float(fps), "target_duration": duration,
            "duration_scale": str(duration_scale),
            "motion_types": motion_types, "segments": intervals, "warnings": media["warnings"] + warnings,
            "time_map": mapping, "output_anchors": output_anchors.tolist(), "recommendation": recommendation,
            "source_speed": input_speed.tolist(), "retimed_speed": speed_output.tolist(),
            "speed_unit": "Per-model apparent progress per second. Compare within a segment.",
            "speed_bounds": [MIN_SPEED / float(duration_scale), MAX_SPEED / float(duration_scale)]}
