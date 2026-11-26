import json
import mimetypes
import os
import re
import stat
from contextlib import ExitStack
from fractions import Fraction
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .jobs import JOB_ID, PROXIES, Job, SmootherError, fingerprint, read_json, safe_path

WEB = Path(__file__).parent / "web"
VARIANTS = {"original": "Original", "retime": "Retime only", "interpolated": "Retime + interpolation"}
NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
RANGE = re.compile(r"bytes=(\d{0,20})-(\d{0,20})")


def byte_range(header, size):
    match = RANGE.fullmatch(header)
    if match is None or size == 0:
        raise ValueError("Unsupported range.")
    start, end = match.groups()
    if not start:
        amount = int(end)
        if amount <= 0:
            raise ValueError("Empty suffix.")
        return max(0, size - amount), size - 1
    first = int(start)
    last = min(size - 1, int(end)) if end else size - 1
    if first > last:
        raise ValueError("Unsatisfiable range.")
    return first, last


class Access:
    def __init__(self, team, audience, email):
        import jwt
        self.jwt = jwt
        self.issuer = f"https://{team}.cloudflareaccess.com"
        self.keys = jwt.PyJWKClient(f"{self.issuer}/cdn-cgi/access/certs", timeout=5)
        self.audience = audience
        self.email = email.casefold()

    def allows(self, token):
        if not token or len(token) > 8192:
            return False
        try:
            if self.jwt.get_unverified_header(token).get("alg") != "RS256":
                return False
            key = self.keys.get_signing_key_from_jwt(token).key
            claims = self.jwt.decode(token, key, algorithms=["RS256"], issuer=self.issuer,
                                     audience=self.audience, options={"require": ["exp", "iss", "aud", "email"]})
            return isinstance(claims["email"], str) and claims["email"].casefold() == self.email
        except (self.jwt.PyJWTError, ValueError, TypeError, OSError):
            return False


def preview_data(report, variants, prefix=""):
    anchors = report["output_anchors"]
    timestamps = report["source_media"]["timestamps"]
    n = len(report["source_speed"])
    stride = max(1, (n + 1499) // 1500)
    original = "original" in variants
    retimed = any(key != "original" for key in variants)
    scale = float(Fraction(report.get("duration_scale", "1")))
    names = {**VARIANTS, "original": "Constant-speed reference"} if scale != 1 else VARIANTS
    series = [[timestamps[i] * scale, anchors[i], report["source_speed"][i] / scale if original else 0.,
               report["retimed_speed"][i] if retimed else 0.]
              for i in range(0, n, stride)]
    segments = [{**{k: segment[k] for k in ("kind", "reliable", "retimed")},
                 "start": segment["start"] * scale, "end": segment["end"] * scale}
                for segment in report["segments"]]
    generated = report["generated_frames"] if "interpolated" in variants else []
    fps = report["fps"].split("/")
    fps = float(fps[0]) / (float(fps[1]) if len(fps) == 2 else 1)
    return {"duration": report["duration"], "fps": fps, "series": series, "segments": segments,
            "generated": generated, "variants": [
                {"id": key, "name": names[key], "url": f"{prefix}/media/{key}.mp4",
                 "counts": report["frame_counts"][key]} for key in variants]}


def web_page(name):
    path = WEB / name
    return path.read_bytes(), mimetypes.guess_type(path)[0] or "application/octet-stream"


def comparison_pages(data, prefix=""):
    return {f"{prefix}/": web_page("index.html"),
            f"{prefix}/api/preview.json": (json.dumps(data, allow_nan=False).encode(), "application/json; charset=utf-8")}


def handler_type(host, assets, pages, access):
    pages = {"/app.js": web_page("app.js"), "/style.css": web_page("style.css"), **pages}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def end_headers(self):
            self.send_header("Cache-Control", "private, no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Content-Security-Policy",
                             "default-src 'none'; script-src 'self'; style-src 'self'; "
                             "media-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            super().end_headers()

        def log_message(self, fmt, *args):
            pass

        def send_bytes(self, content, kind, body):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            if body:
                self.wfile.write(content)

        def stream_asset(self, path, expected, body):
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as file:
                info = os.fstat(file.fileno())
                if not stat.S_ISREG(info.st_mode) or fingerprint(info) != expected:
                    self.send_error(404)
                    return
                size = info.st_size
                header = self.headers.get("Range")
                try:
                    start, end = byte_range(header, size) if header is not None else (0, size - 1)
                except ValueError:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(206 if header is not None else 200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(end - start + 1))
                if header is not None:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                if body:
                    file.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        chunk = file.read(min(256 * 1024, remaining))
                        if not chunk:
                            self.close_connection = True
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)

        def respond(self, body):
            try:
                if self.headers.get_all("Host") != [host] or (access and not access.allows(
                        self.headers.get("Cf-Access-Jwt-Assertion"))):
                    self.send_error(403)
                    return
                url = urlsplit(self.path)
                if url.netloc or url.scheme or not url.path.startswith("/"):
                    self.send_error(404)
                    return
                route = url.path
                if route in pages:
                    self.send_bytes(*pages[route], body)
                elif route in assets:
                    self.stream_asset(*assets[route], body)
                else:
                    self.send_error(404)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                self.close_connection = True
            except (OSError, ValueError):
                self.send_error(404)

        def do_GET(self):
            self.respond(True)

        def do_HEAD(self):
            self.respond(False)

    return Handler


def local_host(port):
    if not 1 <= port <= 65535:
        raise SmootherError("Port must be between 1 and 65535.")
    return f"127.0.0.1:{port}"


def verified_assets(job, records, variants, prefix=""):
    assets = {}
    for key in variants:
        item = records[key]
        if item["path"] != PROXIES[key]:
            raise SmootherError("Unexpected preview asset in manifest.")
        path = job.managed_file(item["path"], "previews")
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or fingerprint(info) != item["identity"]:
            raise SmootherError(f"Preview asset changed: {key}")
        assets[f"{prefix}/media/{key}.mp4"] = (path, item["identity"])
    return assets


def run_server(port, handler, summary):
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        print(json.dumps(summary, indent=2), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def serve(job, args):
    host = local_host(args.port)
    remote = args.remote_host is not None
    extra = (args.access_team, args.access_audience, args.access_email, args.approve_proxy)
    access = None
    if remote:
        if (not all(extra) or "." not in args.remote_host
                or not all(NAME.fullmatch(item) for item in args.remote_host.split("."))
                or not NAME.fullmatch(args.access_team)
                or not re.fullmatch(r"[^@\s]+@[^@\s]+", args.access_email)):
            raise SmootherError("Remote mode requires a hostname, Access team, audience, email and approved proxies.")
        access = Access(args.access_team, args.access_audience, args.access_email)
        host = args.remote_host
    elif any(extra):
        raise SmootherError("Access and proxy approval arguments require --remote-host.")
    with job.lock(shared=True) as manifest:
        records = manifest["previews"]
        if manifest["status"] != "rendered" or not records:
            raise SmootherError("Render a job with previews before serving it.")
        if remote and set(args.approve_proxy) - set(records):
            raise SmootherError("An approved proxy does not exist for this job.")
        variants = [key for key in VARIANTS if key in records and (not remote or key in args.approve_proxy)]
        assets = verified_assets(job, records, variants)
        data = preview_data(read_json(job.reported_path(manifest)), variants)
        summary = job.summary(manifest, args.port)
        if remote:
            summary["preview_url"] = f"https://{host}/"
            summary["approved_proxies"] = variants
        run_server(args.port, handler_type(host, assets, comparison_pages(data), access), summary)


def gallery_page(clips):
    cards = []
    for clip in clips:
        title = escape(clip["title"])
        cards.append(
            f'<article class="clip-card"><h2>{title}</h2>'
            f'<video aria-label="{title}" src="{clip["video"]}" controls muted playsinline preload="metadata"></video>'
            f'<div class="clip-footer"><span>{clip["duration"]:.1f} s · {clip["variant"]}</span>'
            f'<a href="{clip["comparison"]}" aria-label="Compare {title}">Compare</a></div></article>')
    page = (WEB / "gallery.html").read_text(encoding="utf-8")
    count = f"{len(clips)} {'video' if len(clips) == 1 else 'videos'}"
    page = page.replace("<!-- count -->", count).replace("<!-- clips -->", "\n".join(cards))
    return page.encode(), "text/html; charset=utf-8"


def serve_gallery(workspace, args):
    host = local_host(args.port)
    if any((args.remote_host, args.access_team, args.access_audience, args.access_email, args.approve_proxy)):
        raise SmootherError("The gallery is local only. Remote viewing requires a single --job.")
    jobs_dir = safe_path(workspace / "jobs")
    if not jobs_dir.is_dir():
        raise SmootherError("Render a job with previews before serving the gallery.")
    assets, pages, clips, skipped = {}, {}, [], []
    with ExitStack() as locks:
        for path in sorted(jobs_dir.iterdir()):
            if not JOB_ID.fullmatch(path.name) or path.is_symlink() or not path.is_dir():
                continue
            job = Job(workspace, path.name)
            prefix = f"/jobs/{job.id}"
            with ExitStack() as candidate:
                try:
                    manifest = candidate.enter_context(job.lock(shared=True))
                    records = manifest["previews"]
                    edited = next((key for key in ("interpolated", "retime") if key in records), None)
                    if manifest["status"] != "rendered" or edited is None:
                        continue
                    variants = [key for key in VARIANTS if key in records]
                    job_assets = verified_assets(job, records, variants, prefix)
                    report = read_json(job.reported_path(manifest))
                    stem = Path(report["export"]["path"]).stem
                    if stem == job.id:
                        stem = Path(manifest["source"]).stem
                    title = stem.replace("-", " ").replace("_", " ").title()
                    data = {**preview_data(report, variants, prefix), "title": title}
                    job_pages = comparison_pages(data, prefix)
                except (OSError, SmootherError, ValueError, KeyError, TypeError) as exc:
                    skipped.append({"job": job.id, "reason": str(exc)})
                    continue
                locks.enter_context(candidate.pop_all())
            clips.append({"job": job.id, "title": title, "duration": data["duration"],
                          "video": f"{prefix}/media/{edited}.mp4", "variant": VARIANTS[edited],
                          "comparison": f"{prefix}/"})
            assets.update(job_assets)
            pages.update(job_pages)
        if not clips:
            raise SmootherError("No completed jobs with available edited previews.")
        clips.sort(key=lambda clip: clip["title"].casefold())
        pages["/"] = gallery_page(clips)
        run_server(args.port, handler_type(host, assets, pages, None),
                   {"preview_url": f"http://{host}/", "videos": clips, "skipped": skipped})
