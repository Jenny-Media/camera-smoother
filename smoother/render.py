import json
import os
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path

import cv2

from . import camera, flow
from .jobs import PROXIES, SmootherError, budget, checksum, fingerprint, open_reserved, owner_identity
from .presets import CODEC_PRESETS, PRESET_CODECS, export_dimensions
from .retime import original_indices, output_frame_count, plan, safe_auto_pair
from .video import decode_args, export_bitrate_limit, layout, output_args, pipe, read_frame, validate_export


def original_map(media, fps, count, duration_scale):
    for index in original_indices(media, fps, count, duration_scale):
        yield {"right_frame": index, "original_frame": index, "interpolation_candidate": False}


def encode(source, media, pixel, fps, mapping, destinations, scratch, *, count, owners,
           backend=None, auto_pairs=None, share_size=None):
    width, height = media["width"], media["height"]
    shapes, dtype = layout(width, height, pixel)
    bytes_per_frame = sum(h * w for h, w in shapes) * dtype.itemsize
    counts = {"original_frames": 0, "generated_frames": 0, "interpolation_rejected": 0}
    generated, rejected = [], []
    index = -1
    current = previous = None
    with ExitStack() as stack:
        input_stream = stack.enter_context(pipe(decode_args(source, media, pixel), "stdout", scratch))
        encoders = []
        for path, codec in destinations:
            fd = open_reserved(path, owners[path])
            stack.callback(os.close, fd)
            stack.callback(os.fsync, fd)
            command = output_args(pixel, width, height, fps, count, fd, media["colors"], media["sar"], codec,
                                  share_size=share_size if codec == "h264" else None)
            encoders.append(stack.enter_context(pipe(command, "stdin", scratch, pass_fds=(fd,))))
        prepared_index = -1
        prepared = None
        for frame_num, item in enumerate(mapping):
            right = item["right_frame"]
            while index < right:
                previous = current
                current = read_frame(input_stream, bytes_per_frame)
                index += 1
            data = current if item["original_frame"] == index else previous
            if (backend and item["interpolation_candidate"]
                    and (auto_pairs is None or safe_auto_pair(auto_pairs[right - 1]))):
                if prepared_index != index:
                    prepared = backend.prepare(previous, current, shapes, dtype)
                    prepared_index = index
                if prepared is None:
                    rejected.append(frame_num)
                else:
                    data = backend.synthesize(previous, current, item["fraction"], prepared, shapes, dtype)
                    generated.append(frame_num)
            for encoder in encoders:
                encoder.write(data)
        if index != media["frame_count"] - 1 or input_stream.read(1):
            raise SmootherError("Decoder and timestamp analysis disagree on the final frame.")
    counts["generated_frames"] = len(generated)
    counts["interpolation_rejected"] = len(rejected)
    counts["original_frames"] = count - len(generated)
    return counts, generated, rejected


def describe_export(job, report, fps, preset, interpolation, output_name, *, create=False):
    media = report["source_media"]
    codec = PRESET_CODECS[preset]
    if not media["renderable_color"]:
        raise SmootherError("HDR, log or unclassified high-bit-depth color handling is unverified. Export stopped.")
    width, height = export_dimensions(media, preset)
    if codec == "prores" and width % 2:
        raise SmootherError("ProRes 422 requires even source width.")
    if codec == "hevc" and (width % 2 or height % 2):
        raise SmootherError("Apple Playback requires even source dimensions. Use Playback for this clip.")
    target_duration = report["analysis_options"].get("duration")
    count = output_frame_count(media, fps, target_duration)
    suffix = ".mov" if codec == "prores" else ".mp4"
    export = job.export_file(output_name or (job.id + suffix), suffix, create=create)
    return {"preset": preset, "codec": codec, "output": export.name, "export_path": str(export),
            "width": width, "height": height, "fps": f"{fps.numerator}/{fps.denominator}", "frame_count": count,
            "duration": count / float(fps), "interpolate": interpolation,
            "bitrate_cap_bps": None if codec == "prores" else export_bitrate_limit(
                codec, width, height, fps, share=preset == "share")}


def render(job, manifest, report, fps, codec, interpolation, output_name, *, preset=None):
    if manifest["status"] not in {"analyzed", "render_failed"}:
        raise SmootherError("Render requires an analyzed job with no completed export.")
    preset = preset or CODEC_PRESETS[codec]
    settings = describe_export(job, report, fps, preset, interpolation, output_name, create=True)
    export = Path(settings["export_path"])
    media, pairs = report["source_media"], report["pairs"]
    backend = {"none": None, "flow": flow, "camera": camera, "auto": camera}[interpolation]
    width, height = media["width"], media["height"]
    share_size = (settings["width"], settings["height"]) if preset == "share" else None
    auto_pairs = pairs if interpolation == "auto" else None
    pixel = "yuv422p10le" if codec == "prores" else (
        "yuv444p" if codec != "hevc" and (width % 2 or height % 2
                                         or media["source_pixel_format"].startswith("yuv444")) else "yuv420p")
    target_duration = report["analysis_options"].get("duration")
    count, duration = settings["frame_count"], settings["duration"]
    proxy_count = 3 if backend else 2
    # Compressed exports use a capped rate. ProRes uses a conservative four raw bytes per pixel budget.
    final_budget = width * height * 4 * count if codec == "prores" else round(
        settings["bitrate_cap_bps"] * (duration + 3) / 8)
    disk = budget(job.workspace, final_budget + round(proxy_count * 1_000_000 * (duration + 3))
                  + count * 1800)
    timing = plan(media, pairs, fps, target_duration)
    mapping = timing["time_map"]
    job.rollback_unpublished(manifest)
    for rel in PROXIES.values():
        if job.managed_file(rel, "previews").exists():
            raise SmootherError(f"Preview path already exists: {rel}")
    stage_final = job.scratch_file(manifest, "export" + export.suffix)
    active = ["original", "retime"] + (["interpolated"] if backend else [])
    staged = {key: job.scratch_file(manifest, f"{key}.mp4") for key in active}
    stage_report = job.scratch_file(manifest, "report.json")
    source = manifest["source"]
    cv2.setNumThreads(2)
    cv2.ocl.setUseOpenCL(False)
    proxy_pixel = "yuv420p" if width % 2 == 0 and height % 2 == 0 else "yuv444p"
    scratch = job.path / "scratch"
    owners = {job.path / record["path"]: record["owner"] for record in manifest["scratch"]
              if isinstance(record, dict)}
    reference_map = original_map(media, fps, count, Fraction(timing["duration_scale"]))
    original_counts, _, _ = encode(source, media, proxy_pixel, fps, reference_map,
                                   [(staged["original"], "proxy")], scratch, count=count, owners=owners)
    baseline_outputs = [(staged["retime"], "proxy")]
    if not backend:
        baseline_outputs.append((stage_final, codec))
    baseline_counts, _, _ = encode(source, media, pixel, fps, mapping, baseline_outputs, scratch,
                                   count=count, owners=owners, share_size=share_size)
    all_counts = {"original": original_counts, "retime": baseline_counts}
    generated, rejected = [], []
    if backend:
        info, generated, rejected = encode(source, media, pixel, fps, mapping,
                                           [(staged["interpolated"], "proxy"), (stage_final, codec)], scratch,
                                           count=count, owners=owners, backend=backend,
                                           auto_pairs=auto_pairs, share_size=share_size)
        all_counts["interpolated"] = info

    validation = validate_export(stage_final, {**media, "width": settings["width"], "height": settings["height"],
                                                "duration": timing["target_duration"]}, fps, count, codec)
    expected_colors = media["colors"].copy()
    if codec in {"prores", "hevc"} and expected_colors.get("color_range") == "pc":
        expected_colors["color_range"] = "tv"
        timing["warnings"].append("Full-range SDR samples were converted to limited range for playback.")
    for key, expected in expected_colors.items():
        if validation["colors"].get(key) != expected:
            timing["warnings"].append(f"Export color field {key} differs from expected tag {expected}.")
    if codec == "h264" and not share_size and (width % 2 or height % 2):
        timing["warnings"].append("Odd source dimensions use H.264 4:4:4, which has limited hardware player support.")
    if media.get("high_bit_depth") and codec in {"h264", "hevc"}:
        timing["warnings"].append("This export encodes tagged SDR at 8-bit. ProRes 422 uses a 10-bit encoder.")
    if rejected:
        timing["warnings"].append("Interpolation rejected some candidate frames. Captured frames were reused.")

    export_hash, export_identity = checksum(stage_final)
    job.verify(manifest, "after_render_before_publish")
    settings["bitrate_bps"] = validation["bitrate_bps"]
    report.update(timing, preset=preset, render_settings=settings, frame_counts=all_counts, generated_frames=generated,
                  interpolation=interpolation, interpolation_backend=backend.BACKEND if backend else None,
                  interpolation_fallback_frames=rejected,
                  export={"path": str(export), **validation}, disk=disk)
    with os.fdopen(open_reserved(stage_report, owners[stage_report]), "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())

    outputs = {**{path: PROXIES[key] for key, path in staged.items()},
               stage_report: "render-report.json", stage_final: f"exports/{export.name}"}
    entries = []
    infos = {}
    for path, destination in outputs.items():
        info = infos[path] = path.lstat()
        if owner_identity(info) != owners[path]:
            raise SmootherError(f"Staged file changed: {path}")
        entry = {"destination": destination, "identity": fingerprint(info), "owner": owners[path]}
        if path == stage_final:
            if entry["identity"] != export_identity:
                raise SmootherError(f"Verified export changed: {path}")
            entry["sha256"] = export_hash
        entries.append(entry)
    manifest["publication"] = entries
    job.save(manifest)
    job.finish_publication(manifest)

    # A committed job remains usable if interrupted while releasing scratch links.
    for path, info in infos.items():
        job.remove_owned(path, info)
    manifest["scratch"] = [record for record in manifest["scratch"] if isinstance(record, str)
                           or job.path / record["path"] not in infos]
    job.save(manifest)
    return settings
