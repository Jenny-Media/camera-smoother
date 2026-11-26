import json
import subprocess
import tempfile
from bisect import bisect_left
from contextlib import contextmanager
from fractions import Fraction

import numpy as np

from .jobs import SmootherError

FRAME_COLOR_FIELDS = {"color_range": "range", "color_space": "colorspace",
                      "color_primaries": "color_primaries", "color_transfer": "color_trc"}
COLOR_FIELDS = {"color_range": "-color_range", "color_space": "-colorspace",
                "color_primaries": "-color_primaries", "color_transfer": "-color_trc",
                "chroma_location": "-chroma_sample_location"}
SCALE_PROXY = "scale=w='min(1280,iw)':h='min(720,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2:reset_sar=1"
FFMPEG = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-filter_threads", "2"]


def capture(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise SmootherError(f"{command[0]} failed: {result.stderr.strip()[-3000:]}")
    return json.loads(result.stdout)


def rate(value):
    try:
        return Fraction(value).limit_denominator(100_000)
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def exported_media(path):
    return capture(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                    "-of", "json", str(path)])


def probe(path):
    info = capture(["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "V:0", "-show_streams",
                    "-show_frames", "-show_entries",
                    "stream:frame=best_effort_timestamp,duration,pkt_duration:frame_side_data=side_data_type",
                    "-of", "json", str(path)])
    streams = info.get("streams", [])
    frames = info.get("frames", [])
    if not streams or not frames:
        raise SmootherError("No decodable video frames found.")
    stream = streams[0]
    try:
        time_base = Fraction(stream["time_base"])
        pts = [int(frame["best_effort_timestamp"]) for frame in frames]
        if time_base <= 0 or any(a >= b for a, b in zip(pts, pts[1:])):
            raise ValueError("Non-increasing presentation timestamps.")
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise SmootherError("Missing or invalid source frame presentation timestamps.") from exc
    ticks = [value - pts[0] for value in pts]
    timestamps = np.array([float(value * time_base) for value in ticks])
    steps = np.diff(timestamps)
    if not np.all(np.isfinite(timestamps)) or np.any(steps <= 0):
        raise SmootherError("Expected strictly increasing source presentation timestamps.")
    nominal = rate(stream.get("r_frame_rate"))
    average = rate(stream.get("avg_frame_rate"))
    if not nominal or not 1 <= nominal <= 240:
        nominal = average if average and 1 <= average <= 240 else Fraction(30)
    vfr = bool(len(steps) and np.any(abs(steps - 1 / float(nominal)) > max(.001, .03 / float(nominal))))
    chosen = nominal
    if vfr:
        chosen = average if average and 1 <= average <= 240 else Fraction(float(1 / np.median(steps))).limit_denominator(1000)
        if not 1 <= chosen <= 240:
            chosen = Fraction(30)
    last = frames[-1]
    try:
        last_duration = float(int(last.get("duration", last.get("pkt_duration", 0))) * time_base)
    except (TypeError, ValueError):
        last_duration = 0.
    if not np.isfinite(last_duration) or last_duration <= 0:
        last_duration = float(np.median(steps)) if len(steps) else 1 / float(chosen)
    rotation = 0
    side_data = [item.get("side_data_type", "") for f in frames for item in f.get("side_data_list", [])]
    for item in stream.get("side_data_list", []):
        side_data.append(item.get("side_data_type", ""))
        if "rotation" in item:
            rotation = float(item["rotation"])
    if abs(rotation % 90) > .01 and abs(rotation % 90 - 90) > .01:
        raise SmootherError("Non-right-angle display rotation is unsupported.")
    width, height = stream["width"], stream["height"]
    sar = rate(stream.get("sample_aspect_ratio", "1:1").replace(":", "/")) or Fraction(1)
    if round(rotation / 90) % 2:
        width, height, sar = height, width, 1 / sar
    colors = {key: stream[key] for key in COLOR_FIELDS
              if stream.get(key) not in (None, "unknown", "unspecified", "reserved", "N/A")}
    pix_fmt = stream.get("pix_fmt", "")
    high_depth = bool(stream.get("bits_per_raw_sample", "").isdigit()
                      and int(stream["bits_per_raw_sample"]) > 8) or any(
        label in pix_fmt for label in ("10", "12", "14", "16", "p010", "p012", "float", "xyz"))
    known_sdr = colors.get("color_transfer") in {
        "bt709", "smpte170m", "smpte240m", "bt470m", "bt470bg", "iec61966-2-1", "gamma22", "gamma28"
    }
    unsupported_color = (
        colors.get("color_transfer") in {"smpte2084", "arib-std-b67", "log", "log_sqrt", "log100", "log316", "vlog", "bt2020-10", "bt2020-12"}
        or colors.get("color_primaries") == "bt2020"
        or any(any(label in side.lower() for label in ("dolby", "dovi", "hdr", "mastering display", "content light"))
               for side in side_data)
        or (high_depth and not known_sdr)
    )
    warnings = []
    if vfr:
        warnings.append(f"Variable or irregular source timestamps. Default CFR output is {chosen} fps.")
    if rotation:
        warnings.append("Display rotation is applied to frames. Export dimensions follow the displayed orientation.")
    if unsupported_color:
        warnings.append("HDR, log or unclassified high-bit-depth color handling is unverified. Rendering is disabled for this input.")
    return {
        "width": width, "height": height, "sar": str(sar), "frame_count": len(frames),
        "timestamps": timestamps.tolist(), "timestamp_ticks": ticks, "time_base": str(time_base),
        "timestamp_origin": float(pts[0] * time_base),
        "stream_index": stream["index"], "duration": float(timestamps[-1] + last_duration),
        "fps": str(chosen), "variable_frame_rate": vfr, "colors": colors,
        "source_pixel_format": pix_fmt, "high_bit_depth": high_depth,
        "renderable_color": not unsupported_color, "warnings": warnings,
    }


def trim_media(media, start=None, end=None):
    if start is None and end is None:
        return media
    start = 0. if start is None else start
    end = media["duration"] if end is None else end
    if not 0 <= start < end <= media["duration"]:
        raise SmootherError(f"Trim must be a nonempty range within the {media['duration']:.6f}s source.")
    ticks = media["timestamp_ticks"]
    time_base = Fraction(media["time_base"])
    first = bisect_left(ticks, Fraction(str(start)) / time_base)
    last = bisect_left(ticks, Fraction(str(end)) / time_base)
    if first == last:
        raise SmootherError("The trim does not contain a source frame.")
    first_time = float(ticks[first] * time_base)
    if last == len(ticks):
        last_time = media["duration"]
        duration = last_time - first_time
    else:
        last_time = float(ticks[last] * time_base)
        duration = float((ticks[last] - ticks[first]) * time_base)
    selected = [tick - ticks[first] for tick in ticks[first:last]]
    return {**media, "frame_count": len(selected), "timestamp_ticks": selected,
            "timestamps": [float(tick * time_base) for tick in selected],
            "timestamp_origin": media["timestamp_origin"] + first_time, "duration": duration,
            "trim": {"first_frame": first, "end_frame": last,
                     "source_start": first_time, "source_end": last_time}}


@contextmanager
def pipe(command, direction, scratch, pass_fds=()):
    with tempfile.TemporaryFile(dir=scratch) as log:
        process = subprocess.Popen(command, stderr=log, pass_fds=pass_fds, **{direction: subprocess.PIPE})
        channel = getattr(process, direction)
        try:
            yield channel
            channel.close()
            code = process.wait()
            if code:
                log.seek(0)
                raise SmootherError(f"FFmpeg failed: {log.read().decode(errors='replace')[-3000:]}")
        except BaseException as exc:
            channel.close()
            if process.poll() is None:
                process.terminate()
            process.wait()
            if isinstance(exc, BrokenPipeError):
                log.seek(0)
                raise SmootherError(f"FFmpeg pipe closed: {log.read().decode(errors='replace')[-3000:]}") from exc
            raise


def decode_args(source, media, pixel, vf=None):
    command = FFMPEG + ["-threads", "2", "-protocol_whitelist", "file", "-i", str(source),
                        "-map", f"0:{media['stream_index']}",
                        "-an", "-sn", "-dn", "-fps_mode", "passthrough"]
    color_range = media["colors"].get("color_range")
    if not vf and color_range in ("pc", "tv"):
        vf = f"scale=in_range={color_range}:out_range={color_range},format={pixel}"
    trim = media.get("trim")
    if trim:
        filters = [f"trim=start_frame={trim['first_frame']}:end_frame={trim['end_frame']}", "setpts=PTS-STARTPTS"]
        if vf:
            filters.append(vf)
        vf = ",".join(filters)
        command += ["-frames:v", str(media["frame_count"])]
    if vf:
        command += ["-vf", vf]
    return command + ["-threads", "2", "-pix_fmt", pixel, "-f", "rawvideo", "pipe:1"]


def output_args(pixel, width, height, fps, count, fd, colors, sar, codec):
    command = FFMPEG + ["-f", "rawvideo", "-pix_fmt", pixel, "-video_size", f"{width}x{height}",
                        "-framerate", str(fps)]
    color_range = colors.get("color_range")
    if color_range in ("pc", "tv"):
        command += ["-color_range", color_range]
    command += ["-i", "pipe:0", "-an", "-map_metadata", "-1"]
    filters = []
    if sar != "1":
        filters.append(f"setsar={sar}")
    if codec == "prores" and color_range == "pc":
        filters.append("scale=in_range=pc:out_range=tv")
        colors = {**colors, "color_range": "tv"}
    if codec == "proxy":
        proxy_scale = SCALE_PROXY
        if color_range in ("pc", "tv"):
            proxy_scale += f":in_range={color_range}:out_range={color_range}"
        filters += [proxy_scale, "format=yuv420p"]
    frame_tags = [f"{field}={colors[key]}" for key, field in FRAME_COLOR_FIELDS.items() if key in colors]
    if frame_tags:
        filters.append("setparams=" + ":".join(frame_tags))
    if filters:
        command += ["-vf", ",".join(filters)]
    if codec in {"h264", "proxy"}:
        if codec == "proxy":
            limit = 8_000_000
        else:
            limit = h264_limit(width, height, fps)
        # Reserve space for the MP4 index before media. Faststart reopens the
        # destination and cannot safely shift data on an inherited descriptor.
        command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19" if codec == "proxy" else "17",
                    "-maxrate", str(limit), "-bufsize", str(2 * limit),
                    "-moov_size", str(8192 + 64 * count)]
    else:
        command += ["-c:v", "prores_ks", "-profile:v", "2", "-vendor", "apl0"]
    container = "mov" if codec == "prores" else "mp4"
    return command + color_args(colors, pixel, codec) + ["-threads", "2", "-f", container, "-fd", str(fd), "fd:"]


def color_args(colors, pixel, codec):
    # Chroma sample positioning only applies when the output has subsampled chroma.
    return [arg for key, value in colors.items() if key != "chroma_location" or
            (pixel != "yuv444p" or codec == "proxy") for arg in (COLOR_FIELDS[key], value)]


def h264_limit(width, height, fps):
    return max(3_000_000, round(width * height * float(fps) * .5))


def layout(width, height, pixel):
    if pixel == "yuv422p10le":
        return [(height, width), (height, (width + 1) // 2), (height, (width + 1) // 2)], np.dtype("<u2")
    if pixel == "yuv420p":
        return [(height, width), (height // 2, width // 2), (height // 2, width // 2)], np.dtype("u1")
    return [(height, width)] * 3, np.dtype("u1")


def read_frame(stream, count):
    data = stream.read(count)
    if len(data) != count:
        raise SmootherError("Decoded frame count does not match the source timestamps.")
    return data


def validate_export(path, media, fps, count, codec):
    info = exported_media(path)
    streams = info.get("streams", [])
    expected_codec = "h264" if codec == "h264" else "prores"
    if len(streams) != 1 or streams[0].get("codec_name") != expected_codec:
        raise SmootherError("Unexpected video codec or additional streams in export.")
    stream = streams[0]
    try:
        actual = {"codec": expected_codec, "width": stream["width"], "height": stream["height"],
                  "fps": stream["avg_frame_rate"], "duration": float(stream["duration"]),
                  "frame_count": int(stream["nb_frames"]), "colors": {
                      key: stream[key] for key in COLOR_FIELDS if key in stream}}
    except (KeyError, TypeError, ValueError) as exc:
        raise SmootherError(f"Missing or invalid export stream metadata: {path}") from exc
    if ((actual["width"], actual["height"]) != (media["width"], media["height"])
            or rate(actual["fps"]) != fps or actual["frame_count"] != count or not np.isfinite(actual["duration"])
            or abs(actual["duration"] - media["duration"]) > 1 / float(fps) + .00001):
        raise SmootherError(f"Export validation failed: {actual}")
    result = subprocess.run(FFMPEG + ["-xerror", "-i", str(path), "-map", "0:v:0",
                                      "-f", "null", "-"], capture_output=True, text=True)
    if result.returncode:
        raise SmootherError(f"Export failed playback decode: {result.stderr.strip()[-3000:]}")
    actual["decoded_all_frames"] = True
    return actual
