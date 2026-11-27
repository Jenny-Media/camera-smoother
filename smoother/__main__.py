import argparse
import json
import math
import os
import shutil
import sys
from fractions import Fraction
from pathlib import Path

from . import __version__
from .jobs import DEFAULT_WORKSPACE, Job, SmootherError, budget, owns_publication, read_json, sync_directory, write_json
from .presets import CODEC_PRESETS, PRESET_CODECS


def fps_value(text):
    from .video import rate
    fps = rate(text)
    if fps is None or not 1 <= fps <= 240:
        raise argparse.ArgumentTypeError("Use a frame rate from 1 to 240, e.g. 24 or 30000/1001.")
    return fps


def seconds_value(text):
    try:
        value = float(text)
        if math.isfinite(value) and value >= 0:
            return value
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Use a nonnegative time in seconds, such as 27.01.")


def roi(text):
    try:
        x, y, w, h = (float(value) for value in text.split(","))
        if 0 <= x < x+w <= 1 and 0 <= y < y+h <= 1:
            return x, y, w, h
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Use normalized x,y,width,height coordinates within [0,1].")


def fov_value(text):
    try:
        value = float(text)
        if 20 <= value <= 130:
            return value
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Horizontal field of view must be 20–130 degrees.")


def parser():
    root = argparse.ArgumentParser(prog="camera-smoother", description="Local camera-move analysis and retiming.")
    root.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    root.add_argument("--workspace", type=Path,
                      default=os.getenv("CAMERA_SMOOTHER_WORKSPACE", DEFAULT_WORKSPACE),
                      help="Managed directory containing separate jobs/ and exports/ directories.")
    commands = root.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="Track source motion and build the baseline time map.")
    analyze.add_argument("source", type=Path)
    analyze.add_argument("--reference", type=roi, metavar="X,Y,W,H")
    analyze.add_argument("--fov", type=fov_value, default=60., help="Approximate horizontal field of view in degrees.")
    analyze.add_argument("--motion", choices=["auto", "pan", "tilt", "slide", "push", "pull"], default="auto")
    analyze.add_argument("--fps", type=fps_value, help="Plan an explicit CFR output rate.")
    analyze.add_argument("--start", type=seconds_value, help="Trim start in source seconds.")
    analyze.add_argument("--end", type=seconds_value, help="Exclusive trim end in source seconds.")
    analyze.add_argument("--duration", type=seconds_value, help="Target duration in seconds after trim.")
    render = commands.add_parser("render", help="Render a previously analyzed job.")
    render.add_argument("--job", required=True)
    render.add_argument("--fps", type=fps_value)
    mode = render.add_mutually_exclusive_group()
    mode.add_argument("--preset", choices=PRESET_CODECS, help="Export preset. Defaults to playback.")
    mode.add_argument("--codec", choices=CODEC_PRESETS, help="Choose the full-size export codec directly.")
    render.add_argument("--interpolate", choices=["none", "auto", "flow", "camera"],
                        help="Generate frames selectively. Auto uses stricter confidence and alignment checks.")
    render.add_argument("--output", metavar="FILENAME", help="New filename inside the managed exports directory.")
    render.add_argument("--plan", action="store_true", help="Show export settings and path without encoding.")
    serve = commands.add_parser("serve", help="Serve a comparison or an edited-video gallery on 127.0.0.1.")
    target = serve.add_mutually_exclusive_group(required=True)
    target.add_argument("--job", help="Serve one job.")
    target.add_argument("--gallery", action="store_true", help="Show edited previews from this workspace.")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--remote-host")
    serve.add_argument("--access-team")
    serve.add_argument("--access-audience")
    serve.add_argument("--access-email")
    serve.add_argument("--approve-proxy", choices=["original", "retime", "interpolated"], action="append")
    cleanup = commands.add_parser("cleanup", help="Dry-run managed cleanup by default.")
    cleanup.add_argument("--job", required=True)
    cleanup.add_argument("--apply", action="store_true", help="Apply the displayed cleanup scope.")
    cleanup.add_argument("--previews", action="store_true", help="Include this job's previews.")
    commands.add_parser("check-update", help="Check GitHub for a newer stable release.")
    return root


def require_video_tools():
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        raise SmootherError("Install FFmpeg and make ffmpeg and ffprobe available on PATH.")


def analyze_job(args, job, manifest):
    from .motion import ANALYSIS_VERSION, analyze
    from .retime import output_frame_count, plan
    from .video import probe, trim_media
    media = trim_media(probe(manifest["source"]), args.start, args.end)
    fps = args.fps or fps_value(media["fps"])
    count = output_frame_count(media, fps, args.duration)
    available = budget(job.workspace, media["frame_count"] * 2600 + count * 1024)
    pairs = analyze(manifest["source"], media, args.reference, args.fov, args.motion, job.path / "scratch")
    timing = plan(media, pairs, fps, args.duration)
    job.verify(manifest, "after_analysis")
    report = {"version": 1, "analysis_version": ANALYSIS_VERSION, "job": job.id, "source_sha256": manifest["sha256"],
              "source_media": media, "pairs": pairs, "analysis_options": {
                  "reference": args.reference, "fov": args.fov, "motion_hint": args.motion,
                  "start": args.start, "end": args.end, "duration": args.duration},
              "frame_counts": None, "generated_frames": [], "export": None, "disk": available,
              **timing}
    write_json(job.report_path, report)
    manifest["status"] = "analyzed"
    job.save(manifest)
    return timing["recommendation"]


def export_options(args, report):
    fps = args.fps or fps_value(report["fps"])
    preset = args.preset or CODEC_PRESETS[args.codec or "h264"]
    return fps, preset, args.interpolate or "none"


def plan_export(args, job, manifest):
    from .render import describe_export
    if manifest["status"] not in {"analyzed", "render_failed"}:
        raise SmootherError("Planning requires an analyzed job with no completed export.")
    report = read_json(job.report_path)
    fps, preset, interpolation = export_options(args, report)
    return describe_export(job, report, fps, preset, interpolation, args.output)


def render_job(args, job, manifest):
    from .motion import ANALYSIS_VERSION, analyze
    from .render import render
    report = read_json(job.report_path)
    fps, preset, interpolation = export_options(args, report)
    if report.get("analysis_version") != ANALYSIS_VERSION:
        from .retime import output_frame_count
        from .video import probe, trim_media
        options = report["analysis_options"]
        media = trim_media(probe(manifest["source"]), options.get("start"), options.get("end"))
        count = output_frame_count(media, fps, options.get("duration"))
        budget(job.workspace, media["frame_count"] * 2600 + count * 1024)
        report["source_media"] = media
        report["pairs"] = analyze(manifest["source"], media, options["reference"], options["fov"],
                                  options["motion_hint"], job.path / "scratch")
        report["analysis_version"] = ANALYSIS_VERSION
    return render(job, manifest, report, fps, PRESET_CODECS[preset], interpolation, args.output, preset=preset)


def resume_render(args, job, manifest, extra):
    committed = manifest["status"] == "rendered"
    pending = None if committed else job.published_export(manifest)
    if not committed and pending is None:
        return False
    if pending is not None:
        path, info, entry = pending
        if not owns_publication(info, entry):
            raise SmootherError(f"Export ownership could not be verified. Recovery files were retained: {path}")
    report = job.render_report(manifest, pending_path=pending[0] if pending else None)
    export = report["export"]
    interpolation = report.get("interpolation", "flow" if report.get("interpolation_backend") else "none")
    settings = report.get("render_settings") or {"output": Path(export["path"]).name, "fps": export["fps"],
                                                 "codec": export["codec"], "interpolate": interpolation}
    extra["render_settings"] = settings
    requested = {"output": args.output, "fps": args.fps, "codec": args.codec,
                 "interpolate": args.interpolate, "preset": args.preset}
    actual = {**settings, "fps": Fraction(settings["fps"]),
              "preset": report.get("preset", CODEC_PRESETS[export["codec"]])}
    conflicts = [f"--{key}" for key, value in requested.items() if value is not None and value != actual[key]]
    if conflicts:
        raise SmootherError(f"{', '.join(conflicts)} conflict with the existing export. Analyze a new job for different settings.")
    if committed:
        verified = job.validate_committed_export(manifest, report)
        sync_directory(job.path)
        extra["already_rendered"] = True
    else:
        verified = job.recover_publication(manifest, pending, report)
        extra["recovered_publication"] = True
    extra["export_identity_verified"] = verified
    return True


def run_render(args, job, manifest, extra):
    prior_publication = manifest.get("publication")
    starting_render = False
    try:
        if resume_render(args, job, manifest, extra):
            return
        if manifest["status"] not in {"analyzed", "render_failed"}:
            raise SmootherError("Render requires an analyzed job.")
        require_video_tools()
        starting_render = True
        job.verify(manifest, "before_render")
        extra["render_settings"] = render_job(args, job, manifest)
    except BaseException as exc:
        if not starting_render and not isinstance(exc, OSError):
            raise
        # A write can fail on either side of the atomic manifest replacement.
        persisted = read_json(job.manifest_path)
        manifest.clear()
        manifest.update(persisted)
        if isinstance(exc, OSError) and resume_render(args, job, manifest, extra):
            extra.pop("already_rendered", None)
            extra["recovered_publication"] = True
            return
        journal = manifest.get("publication")
        # A failed preflight can leave a previous attempt's journal available for recovery.
        if (starting_render and manifest["status"] != "rendered" and (not journal or journal != prior_publication)
                and job.published_export(manifest) is None):
            try:
                job.rollback_unpublished(manifest)
            finally:
                manifest["status"] = "render_failed"
                job.verify(manifest, "after_failed_render", require=False)
        raise


def main(argv=None):
    args = parser().parse_args(argv)
    job = manifest = None
    extra = {}
    try:
        if args.command == "check-update":
            from .updates import check_update
            print(json.dumps(check_update(), indent=2))
            return 0
        if args.command == "serve":
            from .server import serve, serve_gallery
            if args.gallery:
                serve_gallery(args.workspace, args)
            else:
                serve(Job(args.workspace, args.job), args)
            return 0
        if args.command == "analyze":
            require_video_tools()
            job = Job.create(args.workspace, args.source)
        else:
            job = Job(args.workspace, args.job)
        planning = args.command == "render" and args.plan
        with job.lock(shared=planning) as manifest:
            if args.command == "cleanup":
                extra["cleanup"] = job.cleanup(manifest, args.apply, args.previews)
            elif args.command == "analyze":
                try:
                    extra["recommendation"] = analyze_job(args, job, manifest)
                except BaseException:
                    manifest["status"] = "analysis_failed"
                    job.verify(manifest, "after_failed_analysis", require=False)
                    raise
            elif planning:
                extra["render_settings"] = plan_export(args, job, manifest)
            else:
                run_render(args, job, manifest, extra)
            print(json.dumps({**job.summary(manifest), **extra}, indent=2))
        return 0
    except (SmootherError, OSError, ValueError, ImportError, KeyboardInterrupt) as exc:
        result = {"error": str(exc) or "Interrupted.", **extra}
        if job is not None and manifest is not None:
            result.update(job.summary(manifest))
        print(json.dumps(result, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
