import cv2
import numpy as np

from .flow import planes
from .motion import GRID_X, GRID_Y, seeds, track

BACKEND = "opencv-camera-transform-cpu"


def prepare(first, second, shapes, dtype):
    """Fit one image transform and verify it on the overlapping picture."""
    height, width = shapes[0]
    scale = min(1., 960 / max(height, width))
    width, height = round(width * scale), round(height * scale)
    if min(width, height) < 40:
        return None
    size = (width, height)

    def thumbnail(plane):
        small = cv2.resize(plane, size, interpolation=cv2.INTER_AREA)
        return (small >> 2).astype(np.uint8) if dtype.itemsize == 2 else small

    first_planes, second_planes = planes(first, shapes, dtype), planes(second, shapes, dtype)
    a, b = thumbnail(next(first_planes)), thumbnail(next(second_planes))
    xx = np.linspace(0, width, GRID_X + 1).round().astype(int)
    yy = np.linspace(0, height, GRID_Y + 1).round().astype(int)
    cells = [(x0, y0, x1, y1) for y0, y1 in zip(yy, yy[1:]) for x0, x1 in zip(xx, xx[1:])]
    points, tracked = track(a, b, seeds(a, np.empty((0, 2), np.float32), cells))
    if len(points) < 32:
        return None
    transform, mask = cv2.findHomography(points, tracked, cv2.RANSAC, 1.15, maxIters=500)
    if transform is None or mask is None or not np.isfinite(transform).all():
        return None
    inliers = mask.ravel() != 0
    extent = np.ptp(points[inliers], axis=0) if np.any(inliers) else [0, 0]
    if inliers.sum() < 28 or inliers.mean() < .64 or extent[0] < .45 * width or extent[1] < .40 * height:
        return None
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    projected = cv2.perspectiveTransform(corners[None], transform)[0]
    if (not np.isfinite(projected).all() or not cv2.isContourConvex(projected)
            or np.linalg.norm(projected - corners, axis=1).max() > .045 * max(size)):
        return None

    valid = cv2.warpPerspective(np.ones((height, width), np.uint8), transform, size,
                                flags=cv2.INTER_NEAREST).astype(bool)
    if valid.mean() < .88:
        return None
    luma = cv2.absdiff(cv2.warpPerspective(a, transform, size), b)
    bad = (luma > 28) & valid
    if (np.median(luma[valid]) >= 9 or bad.sum() / valid.sum() >= .13
            or any(bad[y0:y1, x0:x1].mean() > .30 for x0, y0, x1, y1 in cells)):
        return None
    color = np.zeros((height, width), np.uint8)
    for before, after in zip(first_planes, second_planes):
        aligned = cv2.warpPerspective(thumbnail(before), transform, size)
        color = np.maximum(color, cv2.absdiff(aligned, thumbnail(after)))
    if np.mean(color[valid] > 14) > .45:
        return None
    try:
        inverse = np.linalg.inv(transform)
    except np.linalg.LinAlgError:
        return None
    return corners, projected, inverse, size


def synthesize(first, second, alpha, prepared, shapes, dtype):
    corners, projected, inverse, (width, height) = prepared
    between = corners + alpha * (projected - corners)
    from_first = cv2.getPerspectiveTransform(corners, between)
    from_second = from_first @ inverse
    if alpha > .5:
        first, second = second, first
        from_first, from_second = from_second, from_first

    output, coverage = [], {}
    for a, b in zip(planes(first, shapes, dtype), planes(second, shapes, dtype)):
        h, w = a.shape
        if a.shape not in coverage:
            scale = np.diag([w / width, h / height, 1.])
            unscale = np.diag([width / w, height / h, 1.])
            main = scale @ from_first @ unscale
            alternate = scale @ from_second @ unscale
            mask = np.full(a.shape, 255, np.uint8)
            valid = cv2.warpPerspective(mask, main, (w, h), flags=cv2.INTER_NEAREST)
            other_valid = cv2.warpPerspective(mask, alternate, (w, h), flags=cv2.INTER_NEAREST)
            uncovered = cv2.bitwise_and(cv2.bitwise_not(valid), other_valid)
            coverage[a.shape] = main, alternate, uncovered
        main, alternate, uncovered = coverage[a.shape]
        # Use one aligned picture in the interior to avoid ghosting fine edges.
        picture = cv2.warpPerspective(a, main, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if cv2.countNonZero(uncovered):
            other = cv2.warpPerspective(b, alternate, (w, h), borderMode=cv2.BORDER_REPLICATE)
            cv2.copyTo(other, uncovered, picture)
        output.append(picture.tobytes())
    return b"".join(output)
