import contextlib
import http.client
import io
import json
import os
import subprocess
import tempfile
import threading
import time
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from synthetic import make

from smoother import camera
from smoother.__main__ import main
from smoother.flow import prepare, synthesize
from smoother.jobs import Job, SmootherError, budget, checksum, fingerprint, read_json
from smoother.motion import measure
from smoother.render import encode
from smoother.retime import plan, safe_auto_pair
from smoother.server import Access, ThreadingHTTPServer, comparison_pages, handler_type, preview_data
from smoother.video import FFMPEG, exported_media


def load(path):
    return json.loads(Path(path).read_text())


def gray_frames(path):
    decoded = subprocess.check_output(FFMPEG + ["-i", str(path), "-map", "0:V:0", "-vf", "scale=160:90",
                                               "-pix_fmt", "gray", "-f", "rawvideo", "-"])
    return np.frombuffer(decoded, np.uint8).reshape(-1, 90, 160).astype(np.int16)


def variation(values):
    return np.std(values) / np.mean(values)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="camera-smoother-test-", dir="/private/tmp")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.work = cls.root / "managed"
        cls.source = cls.root / "camera-moves.mp4"
        cls.truth = make(cls.source)
        cls.original_hash, _ = checksum(cls.source)
        analyzed = cls.cli("analyze", str(cls.source))
        cls.job_id = analyzed["job"]
        cls.summary = cls.cli("render", "--job", cls.job_id, "--interpolate", "flow")
        cls.report = load(cls.summary["report"])
        cls.job = Job(cls.work, cls.job_id)
        cls.manifest = read_json(cls.job.manifest_path)

    @classmethod
    def cli(cls, *args, expected=0):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            status = main(["--workspace", str(cls.work), *args])
        raw = output.getvalue() or errors.getvalue()
        if status != expected:
            raise AssertionError(f"CLI status {status}, expected {expected}: {raw}")
        return json.loads(raw)

    def test_motion_speed_and_landmarks(self):
        r = self.report
        ts = np.array(r["source_media"]["timestamps"])
        anchors = np.array(r["output_anchors"])
        pos = np.array(self.truth["frames"])
        speed_bounds = np.diff(ts) / np.diff(anchors)
        self.assertGreaterEqual(speed_bounds.min(), .55 - 1e-8)
        self.assertLessEqual(speed_bounds.max(), 1.8 + 1e-8)
        mapped = np.array([item["source_time"] for item in r["time_map"]])
        self.assertTrue(np.all(np.diff(mapped) >= 0))
        self.assertEqual((mapped[0], mapped[-1]), (ts[0], ts[-1]))
        window = np.ones(8) / 8
        cuts = []
        for scene in self.truth["scenes"]:
            name, first, end = (scene[k] for k in ("name", "first", "end"))
            motion = "pan" if name == "pan-reverse" else name
            rows = r["pairs"][first:end-1]
            s = slice(first, end)
            bounds = slice(first, end-1)
            if name in {"hold", "static"}:
                self.assertTrue(all(row["kind"] == "hold" for row in rows))
                np.testing.assert_allclose(ts[s], anchors[s])
            else:
                self.assertGreater(sum(row["kind"] == motion for row in rows) / len(rows), .75)
                displacement = np.abs(np.diff(pos[s]))
                before, after = variation(displacement / np.diff(ts)[bounds]), variation(
                    displacement / np.diff(anchors)[bounds])
                self.assertLess(after, .88 * before, name)
                out = r["time_map"][first:end]
                realized = np.array([pos[item["original_frame"]] for item in out])
                baseline_cv = variation(np.convolve(abs(np.diff(realized)), window, "valid"))
                original_cv = variation(np.convolve(displacement, window, "valid"))
                self.assertLess(baseline_cv, original_cv, name)
                print(f"{name:12s} map speed CV {before:.3f} -> {after:.3f}, "
                      f"8-frame baseline CV {original_cv:.3f} -> {baseline_cv:.3f}")
            if first and r["pairs"][first-1]["kind"] == "cut":
                cuts.append(first)
                self.assertAlmostEqual(anchors[first], ts[first])
        self.assertEqual(cuts, [152, 222, 296, 442])
        tracks = [p["vector"][0] for p in r["pairs"]]
        self.assertGreater(tracks[25] * -tracks[116], 0)
        self.assertIn("Expansion/contraction", " ".join(r["warnings"]))

    def test_export_playback_frames_color_and_selective_flow(self):
        r = self.report
        movie = self.summary["exports"][0]
        info = exported_media(movie)["streams"]
        self.assertEqual(len(info), 1)
        self.assertEqual((info[0]["codec_name"], info[0]["width"], info[0]["height"]), ("h264", 640, 360))
        self.assertEqual(info[0]["avg_frame_rate"], "24/1")
        self.assertTrue(r["export"]["decoded_all_frames"])
        self.assertEqual(r["export"]["colors"]["color_transfer"], "bt709")
        self.assertAlmostEqual(r["duration"], r["target_duration"], delta=1 / 24)
        actual = gray_frames(movie)
        source = gray_frames(self.source)
        baseline = gray_frames(self.job.path / "previews/retime.mp4")
        self.assertEqual(len(actual), len(r["time_map"]))
        for frames in (actual, baseline):
            self.assertLess(np.mean(abs(frames[0] - source[0])), 2.0)
            self.assertLess(np.mean(abs(frames[-1] - source[-1])), 2.0)
            self.assertLess(np.max(np.mean(abs(frames[449:] - frames[448]), axis=(1, 2))), .8)
        generated = set(r["generated_frames"])
        self.assertGreater(len(generated), 100)
        for index in generated:
            row = r["time_map"][index]
            pair = r["pairs"][row["left_frame"]]
            self.assertTrue(row["interpolation_candidate"] and pair["reliable"])
            self.assertNotEqual(pair["kind"], "cut")
        counts = r["frame_counts"]["interpolated"]
        self.assertEqual(counts["generated_frames"], len(generated))
        self.assertEqual(counts["original_frames"] + counts["generated_frames"], len(actual))
        self.assertEqual(r["frame_counts"]["retime"]["generated_frames"], 0)
        self.assertEqual(checksum(self.source)[0], self.original_hash)
        self.assertEqual(self.manifest["integrity"][-1]["stage"], "after_render_before_publish")
        self.assertTrue(self.manifest["integrity"][-1]["unchanged"])

    def test_vfr_prores_and_cleanup(self):
        source = self.root / "vfr.mp4"
        subprocess.run(FFMPEG + ["-i", str(self.source), "-t", "1.5", "-an", "-vf",
                                  "select='not(eq(mod(n,7),3))'", "-fps_mode", "vfr", "-c:v", "libx264",
                                  "-color_range", "pc", str(source)], check=True)
        digest = checksum(source)[0]
        job_id = self.cli("analyze", str(source), "--motion", "pan", "--reference", ".1,.1,.8,.8")["job"]
        rendered = self.cli("render", "--job", job_id, "--codec", "prores", "--fps", "30000/1001")
        report = load(rendered["report"])
        self.assertTrue(report["source_media"]["variable_frame_rate"])
        self.assertGreater(len(set(np.round(np.diff(report["source_media"]["timestamps"]), 4))), 1)
        self.assertAlmostEqual(report["duration"], report["target_duration"], delta=1001 / 30000)
        self.assertEqual(report["export"]["codec"], "prores")
        self.assertEqual(report["export"]["fps"], "30000/1001")
        self.assertEqual(exported_media(rendered["exports"][0])["streams"][0]["profile"], "Standard")
        self.assertEqual(report["frame_counts"]["retime"]["generated_frames"], 0)
        dry = self.cli("cleanup", "--job", job_id, "--previews")["cleanup"]
        self.assertFalse(dry["applied"])
        self.assertEqual(len(dry["paths"]), 2)
        self.assertGreater(dry["reclaimable_bytes"], 0)
        self.assertTrue(all(Path(item["path"]).exists() for item in dry["paths"]))
        kept = self.cli("cleanup", "--job", job_id, "--apply")["cleanup"]
        self.assertEqual(kept["paths"], [])
        self.cli("cleanup", "--job", job_id, "--apply", "--previews")
        self.assertTrue(Path(rendered["report"]).is_file())
        self.assertTrue(Path(rendered["exports"][0]).is_file())
        self.assertTrue(all(not Path(item["path"]).exists() for item in dry["paths"]))
        self.assertEqual(checksum(source)[0], digest)

    def test_trimmed_camera_render_preserves_source_rate_and_views(self):
        analyzed = self.cli("analyze", str(self.source), "--start", ".5", "--end", "2.5", "--duration", "4",
                            "--motion", "pan")
        job_id = analyzed["job"]
        draft = self.cli("render", "--job", job_id, "--interpolate", "camera", "--output", "camera-pan.mp4")
        data = load(draft["report"])
        self.assertEqual(data["source_media"]["trim"], {"first_frame": 12, "end_frame": 60,
                                                        "source_start": .5, "source_end": 2.5})
        self.assertEqual(data["source_media"]["duration"], 2)
        self.assertEqual(data["export"]["duration"], 4)
        self.assertEqual(data["export"]["fps"], "24/1")
        self.assertEqual(data["export"]["frame_count"], 96)
        self.assertEqual(len(data["time_map"]), 96)
        self.assertEqual(data["time_map"][0]["original_frame"], 0)
        self.assertEqual(data["time_map"][-1]["original_frame"], 47)
        self.assertEqual(data["interpolation"], "camera")
        self.assertGreater(len(data["generated_frames"]), 24)
        self.assertEqual(len(data["interpolation_fallback_frames"]), data["frame_counts"]["interpolated"]["interpolation_rejected"])
        self.assertTrue(all(data["time_map"][i]["interpolation_candidate"] for i in data["generated_frames"]))
        source = gray_frames(self.source)
        rendered = gray_frames(draft["exports"][0])
        for a, b in zip(source[[12, 59]], rendered[[0, -1]]):
            self.assertLess(np.abs(a - b).mean(), 3)
        preview = preview_data(data, ["original", "retime", "interpolated"])
        self.assertEqual(preview["variants"][0]["name"], "Constant-speed reference")
        self.assertAlmostEqual(preview["segments"][-1]["end"], 2 * data["segments"][-1]["end"])
        replay = self.cli("render", "--job", job_id, "--interpolate", "camera")
        self.assertTrue(replay["already_rendered"])
        self.assertEqual(replay["render_settings"]["interpolate"], "camera")
        self.assertEqual(checksum(self.source)[0], self.original_hash)

    def test_share_plan_and_export_preserve_rate_and_limit_size(self):
        source = self.root / "share-source.mp4"
        subprocess.run(FFMPEG + ["-f", "lavfi", "-i", "color=blue:size=2560x1440:rate=60000/1001",
                                  "-frames:v", "12", "-c:v", "libx264", "-preset", "ultrafast", str(source)],
                       check=True)
        analyzed = self.cli("analyze", str(source))
        self.assertEqual(analyzed["recommendation"]["retimed_segments"], 0)
        args = ("render", "--job", analyzed["job"], "--preset", "share", "--output", "smaller.mp4")
        pending = self.cli(*args, "--plan")
        self.assertEqual(pending["exports"], [])
        settings = pending["render_settings"]
        self.assertEqual(settings["export_path"], str(self.work / "exports/smaller.mp4"))
        self.assertFalse(Path(settings["export_path"]).exists())
        self.assertEqual((settings["width"], settings["height"], settings["fps"]), (1920, 1080, "60000/1001"))
        result = self.cli(*args)
        self.assertEqual(result["exports"], [settings["export_path"]])
        report = load(result["report"])
        self.assertEqual(report["export"]["codec"], "h264")
        self.assertEqual(report["export"]["width"], 1920)
        self.assertEqual(report["export"]["frame_count"], 12)
        self.assertEqual(self.cli(*args)["render_settings"]["preset"], "share")
        conflict = self.cli("render", "--job", analyzed["job"], "--preset", "playback", expected=1)
        self.assertIn("--preset", conflict["error"])

    def test_auto_frame_generation_gates_by_pair_confidence(self):
        analyzed = self.cli("analyze", str(self.source), "--start", ".5", "--end", "2.5", "--duration", "4",
                            "--motion", "pan")
        self.assertEqual(analyzed["recommendation"]["frame_generation"], "preview-auto")
        report_path = Path(analyzed["report"])
        analysis = load(report_path)
        for pair in analysis["pairs"][:20]:
            pair["confidence"] = min(.70, pair["confidence"])
        report_path.write_text(json.dumps(analysis))
        rendered = self.cli("render", "--job", analyzed["job"], "--interpolate", "auto")
        report = load(rendered["report"])
        self.assertGreater(len(report["generated_frames"]), 0)
        candidates = report["recommendation"]
        self.assertLess(candidates["auto_candidates"], candidates["interpolation_candidates"])
        for index in report["generated_frames"]:
            item = report["time_map"][index]
            self.assertGreaterEqual(item["left_frame"], 20)
            self.assertTrue(item["interpolation_candidate"])
            self.assertTrue(safe_auto_pair(report["pairs"][item["left_frame"]]))
        self.assertEqual(checksum(self.source)[0], self.original_hash)

    def test_failure_preserves_input_and_cleanup_scope(self):
        source = self.root / "short.mp4"
        subprocess.run(FFMPEG + ["-i", str(self.source), "-t", "0.8", "-map", "0:v:0",
                                  "-c:v", "copy", str(source)], check=True)
        digest = checksum(source)[0]
        job_id = self.cli("analyze", str(source))["job"]
        export = self.work / "exports" / "original-alias.mp4"
        os.link(source, export)
        failed = self.cli("render", "--job", job_id, "--output", export.name, expected=1)
        self.assertIn("already exists", failed["error"])
        self.assertTrue(os.path.samefile(source, export))
        def fail_after_writing(*args, **kwargs):
            if args[5][0][1] == "proxy" and "retime.mp4" in args[5][0][0].name:
                raise SmootherError("Test encoder interruption.")
            return encode(*args, **kwargs)
        with patch("smoother.render.encode", side_effect=fail_after_writing):
            failed = self.cli("render", "--job", job_id, expected=1)
        self.assertIn("Test encoder interruption", failed["error"])
        self.assertEqual(checksum(source)[0], digest)
        job = Job(self.work, job_id)
        manifest = read_json(job.manifest_path)
        self.assertTrue(manifest["integrity"][-1]["unchanged"])
        self.assertEqual(manifest["integrity"][-1]["stage"], "after_failed_render")
        dry = self.cli("cleanup", "--job", job_id)["cleanup"]
        self.assertEqual(sum(item["bytes"] > 0 for item in dry["paths"]), 1)
        self.assertGreater(dry["reclaimable_bytes"], 0)
        path = next(Path(item["path"]) for item in dry["paths"] if item["bytes"] > 0)
        with job.lock():
            self.assertIn("active", self.cli("cleanup", "--job", job_id, "--apply", expected=1)["error"])
            self.assertTrue(path.is_file())
        held = self.root / "held-owned-scratch.mp4"
        path.rename(held)
        path.symlink_to(source)
        self.assertIn("Symlink", self.cli("cleanup", "--job", job_id, "--apply", expected=1)["error"])
        path.unlink()
        os.link(source, path)
        applied = self.cli("cleanup", "--job", job_id, "--apply")["cleanup"]
        self.assertIn({"path": str(path), "reason": "Used as a job source."}, applied["skipped"])
        path.unlink()
        held.rename(path)
        self.cli("cleanup", "--job", job_id, "--apply")
        self.assertFalse(path.exists())
        self.assertEqual(checksum(source)[0], digest)
        self.assertTrue(os.path.samefile(source, export))
        manifest = read_json(job.manifest_path)
        self.assertTrue(manifest["integrity"][-1]["unchanged"])
        self.assertEqual(manifest["integrity"][-1]["stage"], "after_cleanup")

    def test_hdr_refusal_and_disk_guard(self):
        source = self.root / "hdr-pq.mp4"
        subprocess.run(FFMPEG + ["-i", str(self.source), "-t", "0.4", "-an", "-c:v", "libx264",
                                  "-vf", "setparams=colorspace=bt2020nc:color_primaries=bt2020:color_trc=smpte2084", str(source)], check=True)
        digest = checksum(source)[0]
        analysis = self.cli("analyze", str(source))
        self.assertFalse(load(analysis["report"])["source_media"]["renderable_color"])
        self.assertIn("unverified", self.cli("render", "--job", analysis["job"], expected=1)["error"])
        self.assertEqual(checksum(source)[0], digest)
        with patch("smoother.jobs.shutil.disk_usage", return_value=SimpleNamespace(free=1024)):
            with self.assertRaisesRegex(SmootherError, "Insufficient disk space"):
                budget(self.work, 1024**2)

    def test_local_http_ranges_and_allowlist(self):
        media = self.job.path / "previews/original.mp4"
        handler = handler_type("127.0.0.1", {"/media/original.mp4": (media, fingerprint(media.stat()))},
                               comparison_pages({}), None)
        with running_server(handler) as port:
            status, headers, body = request(port, "/media/original.mp4", {"Range": "bytes=32-199"})
            self.assertEqual((status, len(body)), (206, 168))
            self.assertEqual(body, media.read_bytes()[32:200])
            self.assertTrue(headers["Content-Range"].startswith("bytes 32-199/"))
            self.assertEqual(headers["Cache-Control"], "private, no-store")
            self.assertEqual(headers["Accept-Ranges"], "bytes")
            self.assertEqual(request(port, "/media/original.mp4", {"Range": "bytes=-5"})[2], media.read_bytes()[-5:])
            self.assertEqual(request(port, "/media/original.mp4", {"Range": "bytes=999999999-"})[0], 416)
            self.assertEqual(request(port, "/media/original.mp4", method="HEAD")[2], b"")
            for path in ("/manifest.json", "/report.json", "/media/retime.mp4", "/media/../report.json",
                         "/%2e%2e", "/exports/original.mp4"):
                self.assertEqual(request(port, path)[0], 404, path)
            self.assertEqual(request(port, "/", {"Host": "example.com"})[0], 403)
            self.assertEqual(request(port, "/", method="POST")[0], 501)

    def test_gallery_lists_edited_jobs_and_scopes_media_by_job(self):
        source = self.root / "Short & Scenic.mp4"
        subprocess.run(FFMPEG + ["-i", str(self.source), "-t", "0.4", "-map", "0:v:0",
                                  "-c:v", "copy", str(source)], check=True)
        second_id = self.cli("analyze", str(source))["job"]
        self.cli("render", "--job", second_id)
        unrendered = Job.create(self.work, source)
        prefixes = [f"/jobs/{job}" for job in (self.job_id, second_id)]

        def inspect(configured_port, handler, summary):
            host = {"Host": f"127.0.0.1:{configured_port}"}
            with running_server(handler) as port:
                status, _, body = request(port, "/", host)
                self.assertEqual(status, 200)
                self.assertIn(b"Short &amp; Scenic", body)
                self.assertNotIn(unrendered.id.encode(), body)
                for prefix, variant in zip(prefixes, ("interpolated", "retime")):
                    url = f"{prefix}/media/{variant}.mp4"
                    self.assertIn(url.encode(), body)
                    self.assertEqual(request(port, f"{prefix}/", host)[0], 200)
                    metadata = json.loads(request(port, f"{prefix}/api/preview.json", host)[2])
                    self.assertIn(url, [item["url"] for item in metadata["variants"]])
                    self.assertEqual(request(port, url, {**host, "Range": "bytes=0-31"})[0], 206)
                for unknown in ("/media/original.mp4", "/api/preview.json", "/exports/short.mp4",
                                f"/jobs/{unrendered.id}/", f"{prefixes[1]}/media/interpolated.mp4",
                                f"{prefixes[0]}/../manifest.json"):
                    self.assertEqual(request(port, unknown, host)[0], 404, unknown)
                self.assertEqual(request(port, "/", {"Host": "example.com"})[0], 403)
                self.assertEqual(request(port, "/", host, method="POST")[0], 501)
                with self.assertRaisesRegex(SmootherError, "active"):
                    with Job(self.work, second_id).lock():
                        pass
            print(json.dumps(summary))

        with patch("smoother.server.run_server", side_effect=inspect):
            summary = self.cli("serve", "--gallery")
        self.assertTrue({self.job_id, second_id}.issubset(clip["job"] for clip in summary["videos"]))
        error = self.cli("serve", "--gallery", "--remote-host", "example.com", expected=1)
        self.assertIn("local only", error["error"])

    def test_access_requires_valid_token_on_page_and_direct_media(self):
        try:
            import jwt
            from cryptography.hazmat.primitives.asymmetric import rsa
        except ImportError:
            self.skipTest("Install requirements-remote.txt to exercise JWT validation.")
        private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        access = Access("example-test", "aud-test", "approved@example.com")
        access.keys = SimpleNamespace(get_signing_key_from_jwt=lambda token: SimpleNamespace(key=private.public_key()))
        claims = {"iss": access.issuer, "aud": "aud-test", "exp": int(time.time() + 90),
                  "email": "approved@example.com"}
        media = self.job.path / "previews/original.mp4"
        handler = handler_type("preview.example.com", {"/media/original.mp4": (media, fingerprint(media.stat()))},
                               comparison_pages({}), access)
        token = jwt.encode(claims, private, algorithm="RS256")
        with running_server(handler) as port:
            host = {"Host": "preview.example.com"}
            authed = {**host, "Cf-Access-Jwt-Assertion": token}
            for path in ("/", "/media/original.mp4"):
                self.assertEqual(request(port, path, host)[0], 403)
                self.assertEqual(request(port, path, {**host, "Cf-Access-Authenticated-User-Email":
                                                       "approved@example.com"})[0], 403)
                self.assertEqual(request(port, path, authed)[0], 200)
                for change in ({"email": "other@example.com"}, {"exp": 1}, {"aud": "wrong"},
                               {"iss": "https://invalid.example"}):
                    bad = jwt.encode({**claims, **change}, private, algorithm="RS256")
                    self.assertEqual(request(port, path, {**host, "Cf-Access-Jwt-Assertion": bad})[0], 403)
                self.assertEqual(request(port, path, {**host, "Cf-Access-Jwt-Assertion": token + "bad"})[0], 403)
            self.assertEqual(request(port, "/media/retime.mp4", authed)[0], 404)


@contextlib.contextmanager
def running_server(handler):
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield server.server_port
        finally:
            server.shutdown()
            worker.join()


def request(port, path, headers=None, method="GET"):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=4)
    try:
        conn.request(method, path, headers={"Host": "127.0.0.1", **(headers or {})})
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


class EstimationTests(unittest.TestCase):
    def test_camera_interpolates_coherent_motion_and_rejects_parallax(self):
        rng = np.random.default_rng(231)
        a = cv2.GaussianBlur(rng.integers(0, 255, (192, 320), dtype=np.uint8), (5, 5), .9)

        def shifted(x):
            return cv2.warpAffine(a, np.float32([[1, 0, x], [0, 1, 0]]), (320, 192),
                                  borderMode=cv2.BORDER_REFLECT)

        b = shifted(8)
        shapes, dtype = [a.shape], np.dtype("u1")
        state = camera.prepare(a.tobytes(), b.tobytes(), shapes, dtype)
        self.assertIsNotNone(state)
        synthesized = camera.synthesize(a.tobytes(), b.tobytes(), .5, state, shapes, dtype)
        generated = np.frombuffer(synthesized, np.uint8).reshape(a.shape)
        center = np.s_[20:-20, 20:-20]
        self.assertLess(np.abs(generated[center].astype(np.int16) - shifted(4)[center]).mean(), 2)
        b[110:] = shifted(-10)[110:]
        self.assertIsNone(camera.prepare(a.tobytes(), b.tobytes(), shapes, dtype))

    def test_duration_change_keeps_protected_pairs_out_of_interpolation(self):
        from smoother.motion import pair
        pairs = ([pair("pan", .95, (1, 0)) for _ in range(8)] +
                 [pair("cut", 0), pair("hold", .95), pair("unknown", .2)])
        media = {"timestamps": list(np.arange(12)/24), "duration": .5, "warnings": [],
                 "timestamp_ticks": list(range(12)), "time_base": "1/24"}
        result = plan(media, pairs, Fraction(24), 1)
        rows = result["time_map"]
        self.assertTrue(any(row["interpolation_candidate"] for row in rows[:16]))
        self.assertFalse(any(row["interpolation_candidate"] for row in rows[16:]))
        self.assertEqual([row["original_frame"] for row in rows[16:]], [8, 8, 9, 9, 10, 10, 11, 11])
        self.assertTrue(np.all(np.diff([row["source_time"] for row in rows]) >= 0))

    def test_directional_slides_and_mixed_motion(self):
        xx, yy = np.meshgrid(np.linspace(15, 625, 32), np.linspace(12, 348, 20))
        points = np.column_stack([xx.ravel(), yy.ravel()])
        for dx, dy in ((2, 0), (0, -3), (2, 1.7)):
            result = measure(points, points + [dx, dy], (360, 640), (0, 0, 640, 360), 60, "auto")
            self.assertTrue(result["reliable"])
            self.assertEqual(result["kind"], "slide")
            np.testing.assert_allclose(np.array(result["vector"]) * 320, [dx, dy], atol=.01)
        centered = points - [320, 180]
        roll = points + np.column_stack([-centered[:, 1], centered[:, 0]]) * .08
        result = measure(points, roll, (360, 640), (0, 0, 640, 360), 60, "auto")
        self.assertEqual(result["kind"], "mixed")
        self.assertFalse(result["reliable"])

    def test_arbitrary_timestep_matches_known_translation(self):
        rng = np.random.default_rng(30)
        a = cv2.GaussianBlur(rng.integers(0, 255, (144, 224), dtype=np.uint8), (7, 7), 1.4)
        def translated(fraction):
            return cv2.warpAffine(a, np.float32([[1, 0, 9*fraction], [0, 1, -4*fraction]]), (224, 144),
                                  borderMode=cv2.BORDER_REFLECT)
        b = translated(1)
        shapes, dtype = [a.shape], np.dtype("u1")
        state = prepare(a.tobytes(), b.tobytes(), shapes, dtype)
        self.assertIsNotNone(state)
        center = np.s_[18:-18, 18:-18]
        for alpha in (.27, .73):
            result = np.frombuffer(synthesize(a.tobytes(), b.tobytes(), alpha, state, shapes, dtype),
                                   dtype=np.uint8).reshape(a.shape)
            expected = translated(alpha)[center].astype(np.int16)
            reference = a if alpha < .5 else b
            self.assertLess(np.mean(abs(result[center] - expected)),
                            .40 * np.mean(abs(reference[center] - expected)))

    def test_unreliable_ranges_and_reversals_are_anchors(self):
        from fractions import Fraction
        from smoother.motion import pair
        pairs = ([pair("slide", .9, (1 + i / 12, 0)) for i in range(12)] +
                 [pair("slide", .9, (-1 - i / 12, 0)) for i in range(12)] +
                 [pair("unknown", .1), pair("mixed", .4), pair("hold", .95), pair("cut", 0.)])
        source = (np.arange(len(pairs) + 1) / 20).tolist()
        result = plan({"timestamps": source, "duration": source[-1] + .05, "warnings": []}, pairs, Fraction(20))
        expected = [0, 12, 24, 25, 26, 27, 28]
        for i in expected:
            self.assertAlmostEqual(result["output_anchors"][i], source[i])
        self.assertEqual([p["kind"] for p in result["segments"][:2]], ["slide", "slide"])
        self.assertTrue(all(p["retimed"] for p in result["segments"][:2]))
        self.assertFalse(any(p["retimed"] for p in result["segments"][2:]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
