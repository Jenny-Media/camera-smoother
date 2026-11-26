import cv2
import numpy as np

BACKEND = "opencv-farneback-cpu"


def planes(data, shapes, dtype):
    values = np.frombuffer(data, dtype)
    offset = 0
    for height, width in shapes:
        size = height * width
        yield values[offset:offset+size].reshape(height, width)
        offset += size


def grid(height, width):
    return np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))


def consistent(flow, opposite, x, y):
    x2, y2 = x + flow[..., 0], y + flow[..., 1]
    other = cv2.remap(opposite, x2, y2, cv2.INTER_LINEAR)
    error = np.linalg.norm(flow + other, axis=2)
    return ((x2 >= 0) & (x2 <= x.shape[1] - 1) & (y2 >= 0) & (y2 <= y.shape[0] - 1)
            & (error < 1.2 + .12 * np.linalg.norm(flow, axis=2)))


def prepare(first, second, shapes, dtype):
    first_planes = planes(first, shapes, dtype)
    second_planes = planes(second, shapes, dtype)
    first_y = next(first_planes)
    second_y = next(second_planes)
    height, width = first_y.shape
    scale = min(1, 720 / max(height, width))
    size = (max(1, round(scale * width)), max(1, round(scale * height)))
    if min(size) < 24:
        return None
    def resized(plane):
        scaled = cv2.resize(plane, size, interpolation=cv2.INTER_AREA)
        return (scaled >> 2).astype(np.uint8) if dtype.itemsize == 2 else scaled

    a, b = resized(first_y), resized(second_y)
    parameters = (.5, 5, 23, 4, 7, 1.5, 0)
    forward = cv2.calcOpticalFlowFarneback(a, b, None, *parameters)
    backward = cv2.calcOpticalFlowFarneback(b, a, None, *parameters)
    x, y = grid(*a.shape)
    good_f = consistent(forward, backward, x, y)
    good_b = consistent(backward, forward, x, y)
    if min(good_f.mean(), good_b.mean()) < .78:
        return None
    map_x, map_y = x + forward[..., 0], y + forward[..., 1]
    for first_plane, second_plane in [(a, b), *((resized(p), resized(q)) for p, q in zip(first_planes, second_planes))]:
        predicted = cv2.remap(second_plane, map_x, map_y, cv2.INTER_LINEAR)
        mismatch = np.abs(predicted.astype(np.int16) - first_plane)[good_f]
        if np.mean(mismatch > 28) > .17 or np.mean(mismatch) > 11:
            return None
    return (forward, backward, good_f, good_b)


def sample_maps(flow, mask, amount, height, width):
    h, w = mask.shape
    x, y = grid(height, width)
    dx, dy = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
    dx *= width / w * amount
    dy *= height / h * amount
    map_x, map_y = x - dx, y - dy
    for _ in range(2):
        map_x, map_y = (x - cv2.remap(dx, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
                        y - cv2.remap(dy, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    valid = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    valid = cv2.remap(valid, map_x, map_y, cv2.INTER_NEAREST).astype(bool)
    return map_x, map_y, valid


def synthesize(first, second, alpha, prepared, shapes, dtype):
    forward, backward, good_f, good_b = prepared
    output = []
    cached = {}
    for a, b in zip(planes(first, shapes, dtype), planes(second, shapes, dtype)):
        if a.shape not in cached:
            cached[a.shape] = (sample_maps(forward, good_f, alpha, *a.shape),
                               sample_maps(backward, good_b, 1-alpha, *a.shape))
        (ax, ay, av), (bx, by, bv) = cached[a.shape]
        wa = cv2.remap(a, ax, ay, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        wb = cv2.remap(b, bx, by, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mix = cv2.addWeighted(wa, 1-alpha, wb, alpha, 0)
        mix[~av & bv] = wb[~av & bv]
        mix[av & ~bv] = wa[av & ~bv]
        neither = ~av & ~bv
        mix[neither] = (a if alpha < .5 else b)[neither]
        output.append(mix.tobytes())
    return b"".join(output)
