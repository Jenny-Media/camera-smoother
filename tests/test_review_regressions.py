import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from smoother.__main__ import main
from smoother.flow import prepare
from smoother.jobs import Job, MARGIN, checksum, fingerprint, read_json, write_json


PROJECT = Path(__file__).resolve().parents[1]
FFMPEG = ["ffmpeg", "-nostdin", "-v", "error"]
WIDTH, HEIGHT = 160, 120


def yuv_frame(y, cb=128, cr=128, width=WIDTH, height=HEIGHT):
    area = width * height
    return bytes([y]) * area + bytes([cb]) * (area // 4) + bytes([cr]) * (area // 4)


def make_clip(path, raw, fps=24, filters=None, size=(WIDTH, HEIGHT), pixel="yuv420p"):
    width, height = size
    command = FFMPEG + ["-f", "rawvideo", "-pix_fmt", pixel, "-s", f"{width}x{height}",
                        "-r", str(fps), "-i", "-"]
    if filters:
        command += ["-vf", filters, "-fps_mode", "vfr"]
    command += ["-c:v", "libx264", "-crf", "0", str(path)]
    subprocess.run(command, input=raw, check=True)


def pixels(path, pixel="yuv420p", one=False):
    command = FFMPEG + ["-i", str(path)]
    if one:
        command += ["-frames:v", "1"]
    return subprocess.check_output(command + ["-f", "rawvideo", "-pix_fmt", pixel, "-"])


def levels(path):
    return list(pixels(path)[::WIDTH * HEIGHT * 3 // 2])


class ReviewRegressionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="smoother-regression-", dir="/private/tmp")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.work = self.root / "managed"

    def cli(self, *args, expected=0, workspace=None):
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            status = main(["--workspace", str(workspace or self.work), *args])
        result = json.loads(output.getvalue() or error.getvalue())
        self.assertEqual(status, expected, result)
        return result

    def analyze(self, path, *args):
        return self.cli("analyze", str(path), *args)["job"]

    def interrupted_export(self, source, *, phase="visible", output=None):
        child = """
import os
import sys
from pathlib import Path
from smoother.__main__ import main
from smoother.jobs import Job

phase = sys.argv.pop(1)
if phase == "committed":
    real_save = Job.save
    def stop_after_commit(self, manifest):
        real_save(self, manifest)
        if manifest["status"] == "rendered":
            os._exit(74)
    Job.save = stop_after_commit
else:
    real_link = os.link
    def stop_after_link(stage, destination, *args, **kwargs):
        real_link(stage, destination, *args, **kwargs)
        if phase == "private" or Path(destination).parent.name == "exports":
            os._exit(74)
    os.link = stop_after_link
main(sys.argv[1:])
"""
        job = Job(self.work, self.analyze(source))
        name = output or f"{job.id}.mp4"
        status = subprocess.run([sys.executable, "-B", "-c", child, phase, "--workspace", str(self.work),
                                 "render", "--job", job.id, "--output", name], cwd=PROJECT).returncode
        self.assertEqual(status, 74)
        export = self.work / "exports" / name
        self.assertEqual(export.is_file(), phase != "private")
        return job, export

    def test_managed_source_guard_and_scratch_ownership(self):
        source = self.root / "input.mp4"
        make_clip(source, yuv_frame(125) * 10)
        job_id = self.analyze(source)
        self.cli("render", "--job", job_id)
        job = Job(self.work, job_id)
        proxy = job.path / "previews" / "original.mp4"
        alternate = self.work / "JOBS" / job_id / "previews" / "original.mp4"
        for candidate in (proxy, alternate):
            if candidate.exists():
                rejected = self.cli("analyze", str(candidate), expected=1)
                self.assertIn("outside managed", rejected["error"])
        rejected = self.cli("analyze", str(proxy), expected=1, workspace=self.root / "another")
        self.assertIn("outside managed", rejected["error"])

        # A legacy manifest can already reference a managed preview.
        peer = Job(self.work, self.analyze(source))
        with peer.lock() as manifest:
            manifest["source"] = str(alternate if alternate.exists() else proxy)
            manifest["sha256"], manifest["source_identity"] = checksum(proxy)
            peer.save(manifest)
        applied = self.cli("cleanup", "--job", job_id, "--previews", "--apply")["cleanup"]
        self.assertIn({"path": str(proxy), "reason": "Used as a job source."}, applied["skipped"])
        self.assertFalse((proxy.parent / "retime.mp4").exists())
        self.assertTrue(proxy.is_file())

        with job.lock() as manifest:
            scratch = job.scratch_file(manifest, "export.mp4")
        scratch.unlink()
        scratch.write_bytes(b"A new file with the same name.")
        with job.lock() as manifest:
            outcome = job.cleanup(manifest, apply=True)
        self.assertTrue(scratch.is_file())
        self.assertEqual(outcome["paths"], [])
        self.assertEqual(outcome["skipped"][0]["path"], str(scratch))

    def test_native_cut_and_vfr_active_frames(self):
        source = self.root / "cut.mp4"
        original = [35, 225, 116, 116, 116, 116, 116]
        make_clip(source, b"".join(map(yuv_frame, original)))
        job_id = self.analyze(source)
        rendered = self.cli("render", "--job", job_id)
        report = read_json(rendered["report"])
        from fractions import Fraction
        media = report["source_media"]
        self.assertEqual(media["timestamp_ticks"][1] * Fraction(media["time_base"]), Fraction(1, 24))
        self.assertEqual([pair["kind"] for pair in report["pairs"][:2]], ["cut", "cut"])
        self.assertEqual(levels(rendered["exports"][0]), original)
        self.assertEqual(levels(Path(rendered["report"]).parent / "previews/original.mp4"), original)

        low_rate = self.cli("render", "--job", self.analyze(source), "--fps", "8")
        self.assertTrue(any("skips 1 detected shot" in warning for warning in read_json(low_rate["report"])["warnings"]))

        source = self.root / "varying.mp4"
        make_clip(source, b"".join(map(yuv_frame, [80, 88, 96, 104])), fps=25,
                  filters="setpts='if(eq(N,0),0,if(eq(N,1),3,if(eq(N,2),15,24)))'")
        vfr = self.cli("render", "--job", self.analyze(source), "--fps", "25")
        report = read_json(vfr["report"])
        self.assertEqual(report["source_media"]["timestamps"], [0., .12, .60, .96])
        self.assertTrue(all(not pair["reliable"] for pair in report["pairs"]))
        for path in (vfr["exports"][0], Path(vfr["report"]).parent / "previews/original.mp4"):
            self.assertEqual(levels(path).index(96), 15)

        trimmed = self.analyze(source, "--start", ".12", "--end", ".96", "--duration", "1.68", "--fps", "25")
        output = self.cli("render", "--job", trimmed, "--interpolate", "camera")
        report = read_json(output["report"])
        self.assertEqual(report["source_media"]["timestamps"], [0., .48])
        self.assertEqual(report["export"]["frame_count"], 42)
        self.assertFalse(report["generated_frames"])
        self.assertEqual(levels(output["exports"][0]), [88] * 24 + [96] * 18)

    def test_chroma_cut_stays_sharp_and_blocks_flow(self):
        width, height = 320, 180
        rng = np.random.default_rng(198)
        texture = cv2.GaussianBlur(rng.integers(0, 256, (height, width + 240), dtype=np.uint8), (5, 5), .9)
        x = np.arange(96) / 95
        positions = np.rint(155 * (x - .55 * np.sin(2 * np.pi * x) / (2 * np.pi))).astype(int)
        shapes = [(height, width), (height // 2, width // 2), (height // 2, width // 2)]
        length = width * height // 4
        warm, cool = bytes([92]) * length + bytes([155]) * length, bytes([165]) * length + bytes([109]) * length
        frames = [texture[:, offset:offset+width].tobytes() + (warm if i < 32 else cool)
                  for i, offset in enumerate(positions)]
        self.assertIsNone(prepare(frames[31], frames[32], shapes, np.dtype("u1")))
        source = self.root / "chroma-cut.mp4"
        make_clip(source, b"".join(frames), size=(width, height))
        result = self.cli("render", "--job", self.analyze(source), "--interpolate", "flow")
        report = read_json(result["report"])
        self.assertEqual(report["pairs"][31]["kind"], "cut")
        self.assertGreater(len(report["generated_frames"]), 5)
        for index in report["generated_frames"]:
            self.assertNotEqual(report["time_map"][index]["left_frame"], 31)
        render = pixels(result["exports"][0])
        samples = [render[i * length * 6 + length * 4] for i in (31, 32)]
        np.testing.assert_allclose(samples, [92, 165], atol=2)

    def test_continuous_slide_past_color_stripes_is_retimed(self):
        width, height = 320, 192
        steps = 2 * np.rint(1.5 + 1.25 * (1 - np.cos(np.linspace(0, 2 * np.pi, 48)))).astype(int)
        positions = np.r_[0, np.cumsum(steps)]
        scene_width = width + positions[-1] + 20
        rng = np.random.default_rng(1237)
        texture = cv2.GaussianBlur(rng.integers(0, 256, (height, scene_width), dtype=np.uint8), (7, 7), 1.1)
        stripe = np.where((np.arange(scene_width // 2) // 6) % 2 == 0, 90, 170).astype(np.uint8)
        chroma = np.broadcast_to(stripe, (height // 2, len(stripe)))
        flat = bytes([128]) * (height * width // 4)
        frames = [texture[:, offset:offset+width].tobytes() +
                  chroma[:, offset//2:(offset+width)//2].tobytes() + flat for offset in positions]
        source = self.root / "continuous-color-slide.mp4"
        make_clip(source, b"".join(frames), size=(width, height))
        job = self.cli("analyze", str(source), "--motion", "slide")
        report = read_json(job["report"])
        self.assertEqual([pair["kind"] for pair in report["pairs"]], ["slide"] * len(steps))
        self.assertTrue(any(segment["retimed"] for segment in report["segments"]))

    def test_distributed_chroma_edit_stays_a_cut_after_alignment(self):
        width, height = 320, 192
        rng = np.random.default_rng(198)
        texture = cv2.GaussianBlur(rng.integers(0, 256, (height, width + 240), dtype=np.uint8), (5, 5), .9)
        x = np.arange(96) / 95
        positions = np.rint(155 * (x - .55 * np.sin(2 * np.pi * x) / (2 * np.pi))).astype(int)
        neutral = np.full((height, width), 128, np.uint8)
        u, v = neutral.copy(), neutral.copy()
        for row in (1, 2):
            u[row::6, :114] = 217
            v[row::6, 114:228] = 217
        frames = [texture[:, offset:offset+width].tobytes() +
                  (neutral.tobytes() * 2 if i < 28 else u.tobytes() + v.tobytes())
                  for i, offset in enumerate(positions)]
        source = self.root / "distributed-chroma-cut.mp4"
        make_clip(source, b"".join(frames), size=(width, height), pixel="yuv444p")
        job = self.analyze(source, "--motion", "slide", "--fps", "48")
        result = self.cli("render", "--job", job, "--interpolate", "flow")
        report = read_json(result["report"])
        self.assertEqual(report["pairs"][27]["kind"], "cut")
        self.assertGreater(len(report["generated_frames"]), 5)
        self.assertTrue(all(report["time_map"][i]["left_frame"] != 27 for i in report["generated_frames"]))
        decoded = np.frombuffer(pixels(result["exports"][0], "yuv444p"), np.uint8).reshape(-1, 3, height, width)
        np.testing.assert_allclose(decoded[[55, 56], 1, 61, 40], [128, 217], atol=3)

    def test_chroma_edits_with_cancelling_or_disjoint_color_bands(self):
        width, cut = 320, 24
        steps = 2 * np.rint(1.5 + 1.25 * (1 - np.cos(np.linspace(0, 2 * np.pi, 64)))).astype(int)
        positions = np.r_[0, np.cumsum(steps)]
        scene_width = width + positions[-1] + 20
        for pixel, height, subsampling in (("yuv420p", 640, 2), ("yuv444p", 192, 1)):
            with self.subTest(pixel=pixel):
                rng = np.random.default_rng(1237)
                texture = cv2.GaussianBlur(rng.integers(0, 256, (height, scene_width), dtype=np.uint8),
                                           (5, 5), .9)
                columns = np.arange(scene_width // subsampling)
                if subsampling == 2:
                    stripe = (128 + 55 * np.cos(2 * np.pi * columns / 20)).astype(np.int16)
                else:
                    stripe = np.where((columns // 10) % 2 == 0, 64, 192).astype(np.int16)
                frames = []
                for i, offset in enumerate(positions):
                    shape = height // subsampling, width // subsampling
                    u = np.broadcast_to(stripe[offset // subsampling:(offset + width) // subsampling], shape).copy()
                    v = np.full(shape, 128, np.int16)
                    sign = 1 if i < cut else -1
                    if subsampling == 2:
                        for row in range(1, 5):
                            u[row::10] += sign * 13
                            v[row + 5::10] -= sign * 13
                    else:
                        u[0::6] += sign * 11
                        u[1::6] -= sign * 11
                        v[2::6] += sign * 11
                        v[3::6] -= sign * 11
                    frames.append(texture[:, offset:offset + width].tobytes()
                                  + u.astype(np.uint8).tobytes() + v.astype(np.uint8).tobytes())
                source = self.root / f"banded-chroma-cut-{pixel}.mp4"
                make_clip(source, b"".join(frames), size=(width, height), pixel=pixel)
                job = self.analyze(source, "--motion", "slide", "--fps", "96")
                analysis_path = Job(self.work, job).report_path
                analysis = read_json(analysis_path)
                self.assertEqual(analysis["pairs"][cut - 1]["kind"], "cut")
                if subsampling == 2:
                    analysis["analysis_version"] = 3
                    analysis["pairs"][cut - 1]["kind"] = "slide"
                    write_json(analysis_path, analysis)
                result = self.cli("render", "--job", job, "--interpolate", "flow")
                report = read_json(result["report"])
                self.assertEqual(report["pairs"][cut - 1]["kind"], "cut")
                self.assertAlmostEqual(report["output_anchors"][cut], report["source_media"]["timestamps"][cut])
                self.assertGreater(len(report["generated_frames"]), 5)
                self.assertTrue(all(report["time_map"][i]["left_frame"] != cut - 1
                                    for i in report["generated_frames"]))

    def test_full_range_prores_preserves_decoded_levels(self):
        width, height = 224, 120
        expected = [0, 8, 16, 128, 235, 247, 255]
        patches = np.repeat(np.array(expected, np.uint8), 32)
        frame = np.broadcast_to(patches[None, :, None], (height, width, 3)).tobytes()
        source = self.root / "full-range.mp4"
        make_clip(source, frame * 8, size=(width, height), pixel="rgb24",
                  filters="scale=out_color_matrix=bt709:out_range=pc,format=yuv420p,"
                          "setparams=range=pc:color_primaries=bt709:color_trc=bt709:colorspace=bt709")
        job = self.cli("render", "--job", self.analyze(source), "--codec", "prores")
        report = read_json(job["report"])
        self.assertEqual(report["source_media"]["colors"]["color_range"], "pc")
        self.assertEqual(report["export"]["colors"]["color_range"], "tv")
        for path in (source, job["exports"][0], Path(job["report"]).parent / "previews/retime.mp4"):
            decoded = np.frombuffer(pixels(path, "rgb24", one=True), np.uint8).reshape(height, width, 3)
            np.testing.assert_allclose(decoded[height // 2, 16::32, 0], expected, atol=3, err_msg=str(path))

    def test_transient_commit_error_finishes_visible_export(self):
        source = self.root / "source.mp4"
        make_clip(source, yuv_frame(125) * 9)
        original_save = Job.save
        for after_replace in (False, True):
            with self.subTest(after_replace=after_replace):
                job_id = self.analyze(source)
                failed_once = False

                def failing_save(job, manifest):
                    nonlocal failed_once
                    if manifest["status"] == "rendered" and not failed_once:
                        failed_once = True
                        if after_replace:
                            original_save(job, manifest)
                        raise OSError("Injected transient commit failure.")
                    original_save(job, manifest)

                with patch.object(Job, "save", failing_save):
                    completed = self.cli("render", "--job", job_id)
                self.assertTrue(failed_once)
                self.assertTrue(completed["recovered_publication"])
                self.assertEqual(completed["status"], "rendered")
                self.assertTrue(Path(completed["exports"][0]).is_file())
                self.assertEqual(Path(completed["report"]).name, "render-report.json")
                self.assertTrue(read_json(completed["report"])["export"]["decoded_all_frames"])

    def test_hard_exit_preserves_recovery_plan_and_allows_retry(self):
        source = self.root / "source.mp4"
        make_clip(source, yuv_frame(125) * 9)
        job_id = self.analyze(source)
        child = """
import os
import sys
from smoother.__main__ import main
from smoother import render

real_link = os.link

def stop_after_link(*args, **kwargs):
    real_link(*args, **kwargs)
    os._exit(73)

render.os.link = stop_after_link
main(sys.argv[1:])
"""
        for use_cleanup, current_id in [(True, job_id), (False, self.analyze(source))]:
            with self.subTest(use_cleanup=use_cleanup):
                status = subprocess.run([sys.executable, "-B", "-c", child, "--workspace", str(self.work),
                                         "render", "--job", current_id], cwd=PROJECT).returncode
                self.assertEqual(status, 73)
                job = Job(self.work, current_id)
                pending = job.path / "previews/original.mp4"
                self.assertTrue(pending.exists())
                self.assertTrue(read_json(job.manifest_path)["publication"])
                dry_run = self.cli("cleanup", "--job", current_id)
                self.assertIn(str(pending), [entry["path"] for entry in dry_run["cleanup"]["paths"]])
                self.assertTrue(pending.exists())
                if use_cleanup:
                    self.cli("cleanup", "--job", current_id, "--apply")
                    self.assertFalse(pending.exists())
                    self.assertNotIn("publication", read_json(job.manifest_path))
                self.assertEqual(self.cli("render", "--job", current_id)["status"], "rendered")

    def test_visible_export_remains_available_to_a_second_workspace(self):
        source = self.root / "source.mp4"
        make_clip(source, yuv_frame(125) * 9)
        job, export = self.interrupted_export(source)
        job_id = job.id
        identity = checksum(export)
        other_work = self.root / "downstream"
        dependent = self.cli("analyze", str(export), workspace=other_work)["job"]
        for flags in ((), ("--apply",)):
            result = self.cli("cleanup", "--job", job_id, *flags)["cleanup"]
            self.assertNotIn(str(export), [entry["path"] for entry in result["paths"]])
            self.assertIn(str(export), [entry["path"] for entry in result["skipped"]])
            self.assertEqual(checksum(export), identity)
        with patch("smoother.render.encode", side_effect=AssertionError("Recovery should reuse the visible export.")):
            recovered = self.cli("render", "--job", job_id)
        self.assertTrue(recovered["recovered_publication"])
        self.assertEqual(recovered["status"], "rendered")
        self.assertEqual(recovered["exports"], [str(export)])
        self.cli("cleanup", "--job", job_id, "--apply")
        self.assertEqual(checksum(export), identity)
        self.assertEqual(self.cli("render", "--job", dependent, workspace=other_work)["status"], "rendered")

    def test_timestamp_change_keeps_recovery_files_and_verified_export(self):
        source = self.root / "source.mp4"
        make_clip(source, yuv_frame(125) * 9)
        job, export = self.interrupted_export(source)
        digest, _ = checksum(export)
        old = export.stat()
        os.utime(export, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000_000))
        pending = read_json(job.manifest_path)["publication"]
        for flags in ((), ("--apply",)):
            result = self.cli("cleanup", "--job", job.id, *flags)["cleanup"]
            self.assertEqual(result["paths"], [])
            self.assertTrue(any(item["path"] == str(export) and "retained" in item["reason"]
                                for item in result["skipped"]))
            self.assertEqual(read_json(job.manifest_path)["publication"], pending)
        with patch("smoother.render.encode", side_effect=AssertionError("Recovery should reuse the export.")):
            self.assertTrue(self.cli("render", "--job", job.id)["recovered_publication"])
        info = read_json(job.manifest_path)["exports"][0]
        self.assertEqual(info["sha256"], digest)
        self.assertEqual(info["identity"], fingerprint(export.stat()))

    def test_changed_export_bytes_are_rejected_and_recovery_is_retained(self):
        source = self.root / "source.mp4"
        make_clip(source, yuv_frame(125) * 9)
        job, export = self.interrupted_export(source)
        before = read_json(job.manifest_path)
        stat_before = export.stat()
        with export.open("r+b") as file:
            first = file.read(1)
            file.seek(0)
            file.write(bytes([first[0] ^ 1]))
        os.utime(export, ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns))
        self.assertEqual(fingerprint(export.stat()), fingerprint(stat_before))
        error = self.cli("render", "--job", job.id, expected=1)
        self.assertIn("failed verification", error["error"])
        self.assertEqual(read_json(job.manifest_path), before)
        self.assertEqual(self.cli("cleanup", "--job", job.id, "--apply")["cleanup"]["paths"], [])
        self.assertTrue((job.path / "render-report.json").is_file())

    def test_recovery_rejects_bytes_changed_after_the_checksum(self):
        source = self.root / "source.mp4"
        make_clip(source, yuv_frame(125) * 9)
        job, export = self.interrupted_export(source)
        journal = job.manifest_path.read_bytes()
        original = export.read_bytes()
        finish = Job.finish_publication

        def change_after_hash(instance, manifest, **kwargs):
            before = export.stat()
            export.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            os.utime(export, ns=(before.st_atime_ns, before.st_mtime_ns))
            self.assertEqual(fingerprint(export.stat()), fingerprint(before))
            return finish(instance, manifest, **kwargs)

        with patch.object(Job, "finish_publication", change_after_hash):
            error = self.cli("render", "--job", job.id, expected=1)
        self.assertIn("Export changed during verification", error["error"])
        self.assertNotIn("export_identity_verified", error)
        self.assertEqual(job.manifest_path.read_bytes(), journal)
        export.write_bytes(original)
        self.assertTrue(self.cli("render", "--job", job.id)["export_identity_verified"])

    def test_ambiguous_export_replacement_preserves_recovery(self):
        source = self.root / "source.mp4"
        make_clip(source, yuv_frame(125) * 9)
        job, export = self.interrupted_export(source)
        manifest = read_json(job.manifest_path)
        entry = next(item for item in manifest["publication"] if item["destination"].startswith("exports/"))
        staged = job.staged_path(manifest, entry)
        original = checksum(staged)
        moved_export = self.root / "original-export.mp4"
        export.rename(moved_export)
        export.write_bytes(b"Unrelated file occupying the name.")
        replacement = checksum(export)

        for flags in ((), ("--apply",)):
            plan = self.cli("cleanup", "--job", job.id, *flags)["cleanup"]
            self.assertEqual(plan["paths"], [])
            self.assertTrue(any(item["path"] == str(export) and "ownership" in item["reason"]
                                for item in plan["skipped"]))
            self.assertEqual(checksum(staged), original)
            self.assertEqual(read_json(job.manifest_path)["publication"], manifest["publication"])
        before = job.manifest_path.read_bytes()
        moved_source = self.root / "moved-source.mp4"
        source.rename(moved_source)
        error = self.cli("render", "--job", job.id, "--output", "retry.mp4", expected=1)
        self.assertIn("ownership could not be verified", error["error"])
        self.assertEqual(job.manifest_path.read_bytes(), before)
        self.assertEqual(checksum(staged), original)
        self.assertEqual(checksum(export), replacement)

        export.rename(self.root / "replacement-export.mp4")
        moved_export.rename(export)
        self.assertTrue(self.cli("render", "--job", job.id)["export_identity_verified"])
        self.assertEqual(checksum(export), original)

    def test_cleanup_rechecks_destination_before_removing_recovery_files(self):
        source = self.root / "source.mp4"
        make_clip(source, yuv_frame(125) * 9)
        job, export = self.interrupted_export(source, phase="private")
        journal = read_json(job.manifest_path)["publication"]
        dry = self.cli("cleanup", "--job", job.id)["cleanup"]
        candidates = [Path(item["path"]) for item in dry["paths"]]
        self.assertTrue(candidates)
        verify = Job.verify

        def occupy_after_verification(instance, manifest, stage, **kwargs):
            result = verify(instance, manifest, stage, **kwargs)
            if stage == "before_cleanup":
                export.write_bytes(b"Unrelated file occupying the name.")
            return result

        with patch.object(Job, "verify", occupy_after_verification):
            error = self.cli("cleanup", "--job", job.id, "--apply", expected=1)
        self.assertIn("changed during cleanup", error["error"])
        self.assertTrue(all(path.exists() for path in candidates))
        self.assertEqual(read_json(job.manifest_path)["publication"], journal)
        self.assertEqual(export.read_bytes(), b"Unrelated file occupying the name.")

    def test_late_filename_collision_allows_retry_without_touching_other_export(self):
        source, other_source = self.root / "source.mp4", self.root / "other.mp4"
        make_clip(source, yuv_frame(125) * 9)
        make_clip(other_source, yuv_frame(80) * 9)
        source_before = checksum(source)
        for apply in (False, True):
            with self.subTest(apply=apply):
                name = f"shared-{apply}.mp4"
                job, export = self.interrupted_export(source, phase="private", output=name)
                other_job = self.analyze(other_source)
                self.cli("render", "--job", other_job, "--output", name)
                winner = checksum(export)
                dry = self.cli("cleanup", "--job", job.id)["cleanup"]
                self.assertIn(str(job.path / "previews/original.mp4"), [entry["path"] for entry in dry["paths"]])
                self.assertNotIn(str(export), [entry["path"] for entry in dry["paths"]])
                if apply:
                    self.cli("cleanup", "--job", job.id, "--apply")
                else:
                    journal = read_json(job.manifest_path)["publication"]
                    error = self.cli("render", "--job", job.id, "--output", name, expected=1)
                    self.assertIn("already exists", error["error"])
                    self.assertEqual(read_json(job.manifest_path)["publication"], journal)
                    moved_source = self.root / "held-source.mp4"
                    source.rename(moved_source)
                    try:
                        self.cli("render", "--job", job.id, "--output", "retry.mp4", expected=1)
                        self.assertEqual(read_json(job.manifest_path)["publication"], journal)
                    finally:
                        moved_source.rename(source)
                result = self.cli("render", "--job", job.id, "--output", f"retry-{apply}.mp4")
                self.assertEqual(result["status"], "rendered")
                self.assertEqual(result["exports"], [str(export.with_name(f"retry-{apply}.mp4"))])
                self.assertEqual(checksum(export), winner)
                self.assertEqual(checksum(source), source_before)
                self.assertTrue(self.cli("render", "--job", other_job)["export_identity_verified"])

    def test_missing_video_tools_preserves_unpublished_recovery(self):
        source = self.root / "source.mp4"
        make_clip(source, yuv_frame(125) * 9)
        job, _ = self.interrupted_export(source, phase="private")
        before = job.manifest_path.read_bytes()
        preview = job.path / "previews/original.mp4"
        identity = checksum(preview)
        with patch("smoother.__main__.shutil.which", return_value=None):
            error = self.cli("render", "--job", job.id, expected=1)
        self.assertIn("Install FFmpeg", error["error"])
        self.assertEqual(job.manifest_path.read_bytes(), before)
        self.assertEqual(checksum(preview), identity)
        self.assertEqual(self.cli("render", "--job", job.id)["status"], "rendered")

    def test_disappearing_export_preserves_recovery_without_relinking(self):
        source = self.root / "source.mp4"
        make_clip(source, yuv_frame(125) * 9)
        job, export = self.interrupted_export(source)
        moved = self.root / "moved.mp4"
        before = job.manifest_path.read_bytes()
        identity = checksum(export)
        lookup, finish = Job.published_export, Job.finish_publication

        def move_after_lookup(instance, manifest):
            result = lookup(instance, manifest)
            export.rename(moved)
            return result

        def move_before_commit(instance, manifest, **kwargs):
            export.rename(moved)
            return finish(instance, manifest, **kwargs)

        for name, replacement in (("published_export", move_after_lookup), ("finish_publication", move_before_commit)):
            with self.subTest(name=name), patch.object(Job, name, replacement):
                error = self.cli("render", "--job", job.id, expected=1)
                self.assertIn("disappeared", error["error"])
                self.assertFalse(export.exists())
                self.assertEqual(checksum(moved), identity)
                self.assertEqual(job.manifest_path.read_bytes(), before)
                moved.rename(export)
        self.assertTrue(self.cli("render", "--job", job.id)["recovered_publication"])

    def test_legacy_export_recovery_requires_decode_and_flags_unverified_identity(self):
        source = self.root / "source.mp4"
        make_clip(source, yuv_frame(125) * 9)
        job, export = self.interrupted_export(source)
        with job.lock() as manifest:
            next(entry for entry in manifest["publication"] if entry["destination"].startswith("exports/")).pop("sha256")
            job.save(manifest)
        pending = job.manifest_path.read_bytes()
        original = export.read_bytes()
        info = export.stat()

        def replace_bytes(data):
            export.write_bytes(data)
            os.utime(export, ns=(info.st_atime_ns, info.st_mtime_ns))
            self.assertEqual(fingerprint(export.stat()), fingerprint(info))

        replace_bytes(b"\0" * len(original))
        error = self.cli("render", "--job", job.id, expected=1)
        self.assertNotIn("recovered_publication", error)
        self.assertEqual(job.manifest_path.read_bytes(), pending)
        replace_bytes(original)
        remuxed = self.root / "no-stream-duration.mkv"
        subprocess.run(FFMPEG + ["-i", str(export), "-c", "copy", str(remuxed)], check=True)
        replacement = remuxed.read_bytes()
        self.assertLessEqual(len(replacement), len(original))
        replace_bytes(replacement.ljust(len(original), b"\0"))
        error = self.cli("render", "--job", job.id, expected=1)
        self.assertIn("Missing or invalid export stream metadata", error["error"])
        self.assertEqual(job.manifest_path.read_bytes(), pending)
        replace_bytes(original)
        source.rename(self.root / "moved-source.mp4")
        receipt = self.cli("render", "--job", job.id)
        self.assertTrue(receipt["recovered_publication"])
        self.assertFalse(receipt["export_identity_verified"])
        self.assertIsNone(read_json(job.manifest_path)["exports"][0]["sha256"])
        receipt = self.cli("render", "--job", job.id)
        self.assertTrue(receipt["already_rendered"])
        self.assertFalse(receipt["export_identity_verified"])
        replace_bytes(b"\0" * len(original))
        error = self.cli("render", "--job", job.id, expected=1)
        self.assertNotIn("already_rendered", error)
        replace_bytes(original)
        with job.lock() as manifest:
            manifest["exports"][0] = {"path": str(export), "bytes": info.st_size}
            job.save(manifest)
        receipt = self.cli("render", "--job", job.id)
        self.assertFalse(receipt["export_identity_verified"])

    def test_retry_uses_published_files_after_original_is_unavailable(self):
        for committed in (False, True):
            with self.subTest(committed=committed):
                source = self.root / f"source-{committed}.mp4"
                make_clip(source, yuv_frame(125) * 9)
                job, export = self.interrupted_export(source, phase="committed" if committed else "visible")
                digest, _ = checksum(export)
                source.rename(source.with_name(f"moved-{committed}.mp4"))
                with patch("smoother.__main__.require_video_tools", side_effect=AssertionError("No new encoding.")):
                    result = self.cli("render", "--job", job.id)
                flag = "already_rendered" if committed else "recovered_publication"
                self.assertTrue(result[flag])
                self.assertTrue(result["export_identity_verified"])
                self.assertEqual(result["status"], "rendered")
                self.assertEqual(result["exports"], [str(export)])
                self.assertEqual(checksum(export)[0], digest)

    def test_recovery_rejects_conflicting_settings(self):
        source = self.root / "source.mp4"
        make_clip(source, yuv_frame(125) * 9)
        job, export = self.interrupted_export(source)
        for committed in (False, True):
            with self.subTest(committed=committed):
                before = job.manifest_path.read_bytes()
                error = self.cli("render", "--job", job.id, "--output", "requested.mov", "--codec", "prores",
                                 "--fps", "12", "--interpolate", "flow", expected=1)
                self.assertIn("conflict", error["error"])
                settings = error["render_settings"]
                self.assertEqual({key: settings[key] for key in ("output", "codec", "fps", "interpolate", "preset")},
                                 {"output": export.name, "codec": "h264", "fps": "24/1", "interpolate": "none",
                                  "preset": "playback"})
                self.assertEqual(job.manifest_path.read_bytes(), before)
                completed = self.cli("render", "--job", job.id, "--output", export.name, "--fps", "24")
                self.assertEqual(completed["status"], "rendered")
        self.assertFalse((export.parent / "requested.mov").exists())

    def test_same_workspace_source_only_excludes_protected_scratch(self):
        source = self.root / "source.mp4"
        make_clip(source, yuv_frame(125) * 9)
        job, export = self.interrupted_export(source)
        dependent = self.analyze(export)
        self.cli("render", "--job", job.id)
        before = self.cli("cleanup", "--job", job.id, "--previews")["cleanup"]
        applied = self.cli("cleanup", "--job", job.id, "--previews", "--apply")["cleanup"]
        self.assertEqual(applied["paths"], before["paths"])
        self.assertEqual(applied["skipped"], before["skipped"])
        self.assertEqual(len(applied["skipped"]), 1)
        protected = Path(applied["skipped"][0]["path"])
        self.assertTrue(protected.samefile(export))
        self.assertTrue(all(not Path(item["path"]).exists() for item in applied["paths"]))
        self.assertIn(str(job.path / "previews/original.mp4"), [item["path"] for item in applied["paths"]])
        self.assertEqual(self.cli("render", "--job", dependent)["status"], "rendered")

    def test_scratch_only_cleanup_ignores_retained_preview(self):
        source = self.root / "source.mp4"
        source.write_bytes(b"Sample source bytes.")
        job = Job.create(self.work, source)
        with job.lock() as manifest:
            scratch = job.scratch_file(manifest, "export.mp4")
            scratch.write_bytes(b"Sample scratch bytes.")
            proxy = job.path / "previews/original.mp4"
            proxy.write_bytes(b"Sample proxy bytes.")
            manifest["previews"]["original"] = {"path": "previews/original.mp4",
                                                "identity": fingerprint(proxy.stat())}
            job.save(manifest)
        proxy.unlink()
        proxy.symlink_to(source)
        self.cli("cleanup", "--job", job.id, "--apply")
        persisted = read_json(job.manifest_path)
        self.assertFalse(scratch.exists())
        self.assertFalse(persisted["scratch"])
        self.assertTrue(proxy.is_symlink())
        self.assertEqual(persisted["integrity"][-1]["stage"], "after_cleanup")
        self.assertTrue(persisted["integrity"][-1]["unchanged"])

    def test_preview_cleanup_skips_replacement_and_removes_owned_sibling(self):
        source = self.root / "source.mp4"
        make_clip(source, yuv_frame(125) * 9)
        before = checksum(source)
        job = Job(self.work, self.analyze(source))
        completed = self.cli("render", "--job", job.id)
        original, retime = (job.path / "previews" / name for name in ("original.mp4", "retime.mp4"))
        replacement = self.root / "replacement.mp4"
        replacement.write_bytes(b"Unrelated replacement preview.")
        replacement.replace(original)
        other = checksum(original)
        dry = self.cli("cleanup", "--job", job.id, "--previews")["cleanup"]
        self.assertEqual([entry["path"] for entry in dry["paths"]], [str(retime)])
        self.assertEqual([entry["path"] for entry in dry["skipped"]], [str(original)])
        self.assertTrue(retime.exists())
        applied = self.cli("cleanup", "--job", job.id, "--previews", "--apply")["cleanup"]
        self.assertEqual(applied["paths"], dry["paths"])
        self.assertEqual(applied["skipped"], dry["skipped"])
        self.assertFalse(retime.exists())
        self.assertEqual(checksum(original), other)
        self.assertEqual(checksum(source), before)
        self.assertTrue(Path(completed["exports"][0]).is_file())

    def test_invalid_report_path_does_not_hide_cleanup_receipt(self):
        source = self.root / "input.mp4"
        make_clip(source, yuv_frame(125) * 10)
        job_id = self.analyze(source)
        completed = self.cli("render", "--job", job_id)
        report = Path(completed["report"])
        retained = self.root / "retained-report.json"
        report.rename(retained)
        report.symlink_to(retained)
        result = self.cli("cleanup", "--job", job_id, "--previews", "--apply")
        self.assertIsNone(result["report"])
        self.assertIn("Symlink", result["report_warning"])
        self.assertTrue(result["cleanup"]["applied"])
        self.assertEqual(len(result["cleanup"]["paths"]), 2)
        self.assertTrue(all(not Path(item["path"]).exists() for item in result["cleanup"]["paths"]))
        self.assertTrue(retained.is_file())
        self.assertTrue(Path(completed["exports"][0]).is_file())

    def test_analysis_budgets_output_rows_before_tracking(self):
        source = self.root / "source.mp4"
        source.write_bytes(b"Sample source bytes.")
        media = {"frame_count": 100, "fps": "1", "duration": 100., "timestamps": list(range(100))}
        with (patch("smoother.video.probe", return_value=media),
              patch("smoother.motion.analyze") as analyze,
              patch("smoother.jobs.shutil.disk_usage", return_value=SimpleNamespace(free=MARGIN + 2 * 1024**2))):
            result = self.cli("analyze", str(source), "--fps", "240", expected=1)
        self.assertIn("Insufficient disk space", result["error"])
        analyze.assert_not_called()

    @unittest.skipUnless(shutil.which("node"), "Node is needed for the shared-playhead test.")
    def test_pause_aligns_all_panels_to_the_same_frame(self):
        script = r"""
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const code = fs.readFileSync("smoother/web/app.js", "utf8");
const context = {
    window: {matchMedia: () => ({matches:false})},
    document: {getElementById: () => ({getBoundingClientRect: () => ({width:0,height:0})})},
    assert
};
vm.runInNewContext(code.replace(/\ninit\(\)\.catch[^\n]*\n?$/, "") + `
    report = {fps:24, duration:2};
    const video = currentTime => ({currentTime, duration:2, seeking:false, paused:false,
                                   pause() { this.paused = true; }});
    views.push({video:video(1.00)}, {video:video(.89)}, {video:video(1.12)});
    playing = true;
    stop();
    assert.equal(time, 1);
    for (const {video} of views) {
        assert.equal(video.paused, true);
        assert.equal(Math.floor(video.currentTime * report.fps), 24);
    }
    report = {fps:24000/1001, duration:1001/240};
    for (const view of views) {
        view.figure = {classList:{toggle(){}}};
        view.button = {setAttribute(){}};
        view.video.duration = report.duration;
    }
    views[0].video.currentTime = 1.06;
    views[1].video.duration = NaN;
    stop();
    views[1].video.duration = report.duration;
    move(views[1].video, sharedSeekTime());
    choose(1);
    assert.equal(outputFrame(time), 25);
    for (const {video} of views) assert.equal(video.currentTime, (25.5 / report.fps));
`, context);
"""
        subprocess.run(["node", "-e", script], cwd=PROJECT, check=True)


if __name__ == "__main__":
    unittest.main()
