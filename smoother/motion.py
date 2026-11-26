import math

import cv2
import numpy as np

from .jobs import SmootherError
from .video import decode_args, pipe, read_frame

ANALYSIS_VERSION = 4
THUMBNAIL_SIZE = (48, 32)

GRID_X, GRID_Y = 5, 4
STATIC_PX = .19
MIN_CONFIDENCE = .58


def seeds(frame, carried, cells):
    points = [carried] if len(carried) else []
    for x0, y0, x1, y1 in cells:
        relevant = carried[(carried[:, 0] >= x0) & (carried[:, 0] < x1)
                           & (carried[:, 1] >= y0) & (carried[:, 1] < y1)]
        needed = 24 - len(relevant)
        if needed < 8:
            continue
        region = frame[y0:y1, x0:x1]
        mask = np.full(region.shape, 255, np.uint8)
        for x, y in relevant:
            cv2.circle(mask, (round(float(x)) - x0, round(float(y)) - y0), 5, 0, -1)
        new = cv2.goodFeaturesToTrack(region, needed, .012, 6, mask=mask, blockSize=5)
        if new is not None:
            points.append(new.reshape(-1, 2) + np.array([x0, y0], dtype=np.float32))
    return np.vstack(points) if points else np.empty((0, 2), np.float32)


def track(first, second, points):
    if len(points) < 4:
        return points[:0], points[:0]
    options = {"winSize": (23, 23), "maxLevel": 3,
               "criteria": (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 24, .01)}
    a = points.reshape(-1, 1, 2)
    b, forward, error = cv2.calcOpticalFlowPyrLK(first, second, a, None, **options)
    back, backward, _ = cv2.calcOpticalFlowPyrLK(second, first, b, None, **options)
    b = b.reshape(-1, 2)
    h, w = first.shape
    valid = ((forward.ravel() == 1) & (backward.ravel() == 1) & (error.ravel() < 29)
             & (np.linalg.norm(points - back.reshape(-1, 2), axis=1) < 1.0)
             & (b[:, 0] >= 3) & (b[:, 0] < w - 3) & (b[:, 1] >= 3) & (b[:, 1] < h - 3))
    return points[valid], b[valid]


def fit(design, flow, weights):
    """Cell-balanced, iteratively reweighted least squares for 2D flow models."""
    for _ in range(3):
        w = np.sqrt(weights)
        result = np.linalg.lstsq((design * w[:, None, None]).reshape(-1, design.shape[2]),
                                 (flow * w[:, None]).reshape(-1), rcond=None)[0]
        errors = np.linalg.norm(flow - design @ result, axis=1)
        weights = weights / np.maximum(1, errors / max(.22, np.median(errors) * 1.8))
    return result, errors


def pair(kind, confidence, vector=(0., 0.), reason=None, regions=0):
    return {"kind": kind, "confidence": float(confidence), "vector": list(map(float, vector)),
            "reason": reason, "regions": regions, "reliable": confidence >= MIN_CONFIDENCE
            and kind not in {"cut", "mixed", "unknown"}}


def measure(a, b, shape, bounds, fov, hint):
    height, width = shape
    x0, y0, x1, y1 = bounds
    keep = (a[:, 0] >= x0) & (a[:, 0] < x1) & (a[:, 1] >= y0) & (a[:, 1] < y1)
    a, flow = a[keep], b[keep] - a[keep]
    row = np.clip(((a[:, 1] - y0) * GRID_Y / (y1-y0)).astype(int), 0, GRID_Y-1)
    col = np.clip(((a[:, 0] - x0) * GRID_X / (x1-x0)).astype(int), 0, GRID_X-1)
    cell_id = row * GRID_X + col
    groups = [np.flatnonzero(cell_id == cell) for cell in np.unique(cell_id)]
    groups = [group for group in groups if len(group) >= 3]
    n = len(groups)
    if n < 4 or len(a) < 20:
        return pair("unknown", .20, reason="Too few distributed, consistent feature tracks.", regions=n)
    region_flow = np.array([np.median(flow[group], axis=0) for group in groups])
    magnitude = np.linalg.norm(region_flow, axis=1)
    signal = float(np.median(magnitude))
    if signal < STATIC_PX and np.percentile(magnitude, 65) < STATIC_PX * 1.5:
        return pair("hold", min(.99, .65 + .025 * n), regions=n)

    selected = np.concatenate(groups)
    weights = np.repeat([1 / len(g) for g in groups], [len(g) for g in groups])
    locations, flow = a[selected], flow[selected]
    xy = (locations - [width / 2, height / 2]) / (width / 2)
    x, y = xy.T
    trans = np.broadcast_to(np.eye(2), (len(x), 2, 2))
    sim = np.stack([trans[:, :, 0], trans[:, :, 1], xy, np.column_stack([-y, x])], axis=2)
    similarity, sim_error = fit(sim, flow, weights)
    tx, ty, radial, roll = similarity
    # Use within-cell divergence to distinguish shared expansion from layer displacement.
    local_scale = []
    for group in np.split(np.arange(len(x)), np.cumsum([len(g) for g in groups])[:-1]):
        pos = xy[group] - np.mean(xy[group], axis=0)
        denom = np.sum(pos * pos)
        if denom > .004 and len(group) >= 5:
            local_scale.append(float(np.sum(pos * (flow[group] - np.mean(flow[group], axis=0))) / denom))
    local_scale = np.array(local_scale)
    scale_support = (np.mean(local_scale * np.sign(radial) > .24 * abs(radial))
                     if len(local_scale) >= 4 else 0.)
    central = np.hypot(tx, ty)
    translation = np.median(region_flow, axis=0)
    focal = width / (2 * math.tan(math.radians(fov) / 2))
    u, v = (locations - [width / 2, height / 2]).T / focal
    rot = np.stack([np.column_stack([1 + u * u, u * v]),
                    np.column_stack([u * v, 1 + v * v])], axis=2)
    angle_flow, rot_error = fit(rot, flow, weights)
    trans_error = np.linalg.norm(flow - translation, axis=1)
    rot_err, sim_err, trans_err = map(np.median, (rot_error, sim_error, trans_error))
    cover = min(1., n / 10)
    expansion = (abs(radial) > .40 and scale_support >= .60
                 and sim_err < .24 + .72 * min(rot_err, trans_err))
    if abs(roll) > max(.50, 1.1 * abs(radial), .52 * central) and sim_err < trans_err * .8:
        return pair("mixed", .45, reason="Roll or mixed camera motion is unsupported.", regions=n)
    if hint in {"push", "pull"} or (hint == "auto" and expansion):
        if expansion and central <= max(.60, .80 * abs(radial)):
            confidence = .51 + .25 * float(scale_support) + .17 * cover
            return pair("push-in" if radial > 0 else "pullback", confidence, (radial / (width / 2), 0),
                        "Expansion/contraction alone cannot distinguish a dolly move from optical zoom.", n)
        if hint != "auto" or central < 3 * abs(radial):
            return pair("mixed", .45, reason="Expansion is weak or combined with substantial directional motion.",
                        regions=n)

    direction_len = np.linalg.norm(translation)
    dot = region_flow @ translation / np.maximum(magnitude * direction_len, 1e-9)
    agreement = float(np.mean((dot > .85) & (magnitude > STATIC_PX * .65)))
    if agreement < .60 or direction_len < STATIC_PX:
        return pair("unknown", .42, reason="Regions do not agree on the apparent motion.", regions=n)

    rotation = hint in {"pan", "tilt"} or (hint == "auto" and rot_err < .75 * trans_err
               and trans_err - rot_err > .042 and rot_err < .23 + .13 * direction_len)
    directional_error = rot_err if rotation else float(np.median(
        abs(region_flow[:, 0] * translation[1] - region_flow[:, 1] * translation[0])
        / direction_len))
    confidence = min(.99, .31 + .25 * agreement + .22 * cover + .18 * max(0, 1 - directional_error /
                                                                         (.45 + .22 * direction_len)))
    if rotation:
        horizontal, vertical = angle_flow / focal
        axis = hint if hint != "auto" else ("pan" if abs(horizontal) > abs(vertical) else "tilt")
        main, other = (horizontal, vertical) if axis == "pan" else (vertical, horizontal)
        if abs(other) > abs(main) * .52:
            return pair("mixed", .48, reason="Substantial simultaneous pan and tilt.", regions=n)
        return pair(axis, confidence, (main, 0), regions=n)
    return pair("slide", confidence, translation / (width / 2),
                "Translation-like image motion. A narrow-angle pan can look similar." if hint == "auto" else None, n)


def cut_signature(frame, chroma):
    thumb = cv2.resize(frame, THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA)
    hist = np.bincount((frame // 16).ravel(), minlength=16).astype(float)
    hist /= hist.sum()
    color = [cv2.resize(plane, THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA) for plane in chroma]
    return thumb.astype(np.int16), hist, np.array(color, np.int16)


def chroma_cut_after_tracking(previous, current, points, tracked):
    """Test the suspected edit on image regions that the motion model aligns."""
    if len(points) < 12:
        return True
    transform, inliers = cv2.findHomography(points, tracked, cv2.RANSAC, 1.5, maxIters=250)
    if transform is None or not np.all(np.isfinite(transform)) or np.count_nonzero(inliers) < max(12, .6 * len(points)):
        return True
    height, width = previous[0].shape
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    valid = cv2.warpPerspective(np.ones((height, width), np.float32), transform, (width, height), flags=flags) > .99
    if valid.mean() < .60:
        return True
    # Compare the same area-averaged cells as the initial cut test. Pixels
    # outside the overlap do not contribute to the cell mean.
    coverage = cv2.resize(valid.astype(np.float32), THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA)
    eligible = coverage >= .85
    if eligible.mean() < .60:
        return True
    signed_deltas, pixel_deltas = [], []
    for before, after in zip(previous, current):
        aligned = cv2.warpPerspective(after, transform, (width, height), flags=flags)
        residual = (before.astype(np.float32) - aligned) * valid
        cells = cv2.resize(residual, THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA)
        signed_deltas.append(abs(cells[eligible] / coverage[eligible]))
        pixel_deltas.append(abs(residual))
    if np.median(np.maximum(*signed_deltas)) > 24:
        return True
    # Count changed pixels before pooling so opposite signs and changes in
    # different color planes contribute even when they share a cell.
    changed = np.maximum(*pixel_deltas) > 12
    support = cv2.resize(changed.astype(np.float32), THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA)
    return np.median(support[eligible] / coverage[eligible]) > .60


def analyze(source, media, reference, fov, hint, scratch):
    cv2.setNumThreads(2)
    factor = min(1., 640 / max(media["width"], media["height"]))
    width, height = round(media["width"] * factor), round(media["height"] * factor)
    if min(width, height) < 40:
        raise SmootherError("Image is too small or narrow for distributed feature tracking.")
    rx, ry, rw, rh = reference or (0, 0, 1, 1)
    xx = np.linspace(rx * width, (rx + rw) * width, GRID_X + 1).round().astype(int)
    yy = np.linspace(ry * height, (ry + rh) * height, GRID_Y + 1).round().astype(int)
    if min(np.diff(xx)) < 8 or min(np.diff(yy)) < 8:
        raise SmootherError("Reference region is too small at analysis resolution.")
    cells = [(x0, y0, x1, y1) for y0, y1 in zip(yy[:-1], yy[1:])
             for x0, x1 in zip(xx[:-1], xx[1:])]
    bounds = int(xx[0]), int(yy[0]), int(xx[-1]), int(yy[-1])
    empty = np.empty((0, 2), np.float32)
    results = []
    size = width * height
    with pipe(decode_args(source, media, "yuv444p", f"scale={width}:{height}:flags=area:out_range=pc"),
              "stdout", scratch) as stream:
        previous, *old_chroma = np.frombuffer(read_frame(stream, size * 3), np.uint8).reshape(3, height, width)
        old_thumb, old_hist, old_color = cut_signature(previous, old_chroma)
        carried = empty
        for index in range(1, media["frame_count"]):
            current, *chroma = np.frombuffer(read_frame(stream, size * 3), np.uint8).reshape(3, height, width)
            thumb, hist, color = cut_signature(current, chroma)
            change = float(np.mean(abs(thumb - old_thumb)))
            histogram_change = float(np.abs(hist - old_hist).sum() / 2)
            suspect_color = np.median(np.max(abs(color - old_color), axis=0)) > 24
            identical_luma = np.array_equal(previous, current)
            if identical_luma and not suspect_color:
                result = pair("hold", .99)
            else:
                points = seeds(previous, carried, cells)
                a, carried = track(previous, current, points)
                survival = len(a) / max(1, len(points))
                if suspect_color and chroma_cut_after_tracking(old_chroma, chroma, a, carried):
                    result = pair("cut", 0., reason="Abrupt chroma change at this frame boundary.")
                    carried = empty
                elif ((change > 11 and histogram_change > .42) or (change > 15 and survival < .26)
                        or change > 50):
                    result = pair("cut", 0., reason="Abrupt image change at this frame boundary.")
                    carried = empty
                elif change < .7 and np.percentile(abs(thumb - old_thumb), 90) <= 1 and len(a) < 20:
                    result = pair("hold", .78, reason="No substantial image change on a low-texture pair.")
                else:
                    result = measure(a, carried, previous.shape, bounds, fov, hint)
            results.append(result)
            previous, old_chroma = current, chroma
            old_thumb, old_hist, old_color = thumb, hist, color
        if stream.read(1):
            raise SmootherError("Decoder emitted extra frames.")
    return results
