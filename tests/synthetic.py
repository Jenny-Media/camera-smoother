import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT, FPS = 640, 360, 24


def texture(seed, base):
    rng = np.random.default_rng(seed)
    image = np.full((HEIGHT * 2, WIDTH * 2, 3), base, np.uint8)
    for _ in range(1300):
        x, y = rng.integers([0, 0], [2 * WIDTH, 2 * HEIGHT])
        color = tuple(int(v) for v in np.clip(np.array(base) + rng.integers(-75, 79, 3), 0, 255))
        r = int(rng.integers(2, 22))
        if r % 3:
            cv2.circle(image, (int(x), int(y)), r, color, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(image, (int(x-r), int(y-r)), (int(x+r), int(y+r)), color, -1)
    return image


def eased_progress(n):
    x = np.arange(n) / (n - 1)
    return .46 * x + .54 * (x - np.sin(2 * math.pi * x) / (2 * math.pi))


def make(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"fps": FPS, "scenes": [], "frames": []}
    focal = WIDTH / (2 * math.tan(math.radians(60) / 2))
    k = np.array([[focal, 0, WIDTH/2], [0, focal, HEIGHT/2], [0, 0, 1]])
    inv = np.linalg.inv(k)
    crop_offset = np.array([[1, 0, -WIDTH/2], [0, 1, -HEIGHT/2], [0, 0, 1]])
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
               "-i", "-", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-map", "0:v", "-map", "1:a",
               "-vf", "scale=out_color_matrix=bt709,format=yuv420p,"
                      "setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "12",
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
               "-c:a", "aac", "-shortest", str(path)]
    shots = [
        ("pan", 68, 0., .48, 0, (74, 146, 188)),
        ("hold", 18, .48, .48, 0, (74, 146, 188)),
        ("pan-reverse", 66, .48, -.04, 0, (74, 146, 188)),
        ("tilt", 70, 0., .39, 1, (105, 65, 33)),
        ("slide", 74, 0., 175., 2, (175, 193, 219)),
        ("push-in", 72, 0., .43, 3, (73, 75, 62)),
        ("pullback", 74, .43, 0., 3, (73, 75, 62)),
        ("static", 40, 0., 0., 5, (198, 174, 99)),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as ffmpeg:
        old_seed = None
        for label, count, start, end, seed, color in shots:
            if seed != old_seed:
                background = texture(seed, color)
                old_seed = seed
            begin = len(data["frames"])
            data["scenes"].append({"name": label, "first": begin, "end": begin + count})
            for position in start + (end - start) * eased_progress(count):
                matrix = crop_offset.copy()
                if label in {"pan", "hold", "pan-reverse", "tilt"}:
                    c, s = math.cos(position), math.sin(position)
                    rotation = (np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) if label == "tilt"
                                else np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]))
                    matrix = k @ rotation @ inv @ crop_offset
                elif label == "slide":
                    matrix[0, 2] += position
                    matrix[1, 2] += .31 * position
                elif label in {"push-in", "pullback"}:
                    mag = math.exp(position)
                    matrix = np.array([[mag, 0, WIDTH/2*(1-mag)],
                                       [0, mag, HEIGHT/2*(1-mag)], [0, 0, 1]]) @ crop_offset
                frame = cv2.warpPerspective(background, matrix, (WIDTH, HEIGHT), borderMode=cv2.BORDER_REFLECT)
                if label == "slide":
                    # A near layer crosses part of the far layer at a different speed.
                    x, y = round(-260 + 1.65 * position), round(98 + .52 * position)
                    for shift in [0, 60, 120]:
                        cv2.line(frame, (x+shift, y), (x+shift, y+240), (28, 45, 132), 16, cv2.LINE_AA)
                    cv2.line(frame, (x, y+44), (x+120, y+44), (40, 57, 153), 12, cv2.LINE_AA)
                ffmpeg.stdin.write(frame.tobytes())
                data["frames"].append(float(position))
        ffmpeg.stdin.close()
        if ffmpeg.wait():
            raise RuntimeError("Synthetic video encoding failed.")
    path.with_suffix(".truth.json").write_text(json.dumps(data, indent=2))
    print(path)
    return data


if __name__ == "__main__":
    make(Path(sys.argv[1]))
