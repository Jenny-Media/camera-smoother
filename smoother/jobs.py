import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_WORKSPACE = PROJECT / "data"
PROXIES = {"original": "previews/original.mp4", "retime": "previews/retime.mp4",
           "interpolated": "previews/interpolated.mp4"}
JOB_ID = re.compile(r"[a-f0-9]{16}")
EXPORT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,195}\.(?:mp4|mov)")
MARGIN = 128 * 1024**2


class SmootherError(Exception):
    pass


def safe_path(path):
    """Reject symlinks in managed paths, including missing leaf paths."""
    path = Path(os.path.abspath(path))
    for part in (*reversed(path.parents), path):
        if part.is_symlink():
            raise SmootherError(f"Symlink in managed path: {part}")
    return path


def fingerprint(info):
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]


def owner_identity(info):
    # Creation time is stable while FFmpeg writes the reserved inode.
    return [info.st_dev, info.st_ino, getattr(info, "st_birthtime_ns", None)]


def owns_publication(info, entry):
    return stat.S_ISREG(info.st_mode) and owner_identity(info) == entry.get("owner")


def matches_publication(info, entry):
    return owns_publication(info, entry) and fingerprint(info) == entry["identity"]


def matches_export(path, info, entry, verify_bytes=False):
    if matches_publication(info, entry) and (not verify_bytes or not entry.get("sha256")):
        return True
    if not owns_publication(info, entry) or info.st_size != entry["identity"][2] or not entry.get("sha256"):
        return False
    digest, identity = checksum(path)
    return digest == entry["sha256"] and identity == fingerprint(info)


def validate_legacy_export(path, report):
    from .presets import export_dimensions
    from .video import rate, validate_export

    media = report["export"]
    fps = rate(media.get("fps"))
    if not fps or fps <= 0:
        raise SmootherError(f"Invalid frame rate in the saved render report: {path}")
    source = report["source_media"]
    width, height = export_dimensions(source, report.get("preset"))
    target = {**source, "width": width, "height": height,
              "duration": report.get("target_duration", source["duration"])}
    validate_export(path, target, fps, media["frame_count"], media["codec"])


def check_export_unchanged(path, expected):
    try:
        current = path.lstat()
    except FileNotFoundError as exc:
        raise SmootherError(f"Export disappeared during verification: {path}") from exc
    if (not stat.S_ISREG(current.st_mode) or owner_identity(current) != owner_identity(expected)
            or fingerprint(current) != fingerprint(expected) or current.st_ctime_ns != expected.st_ctime_ns):
        raise SmootherError(f"Export changed during verification: {path}")
    return current


def open_reserved(path, identity):
    fd = os.open(safe_path(path), os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or owner_identity(info) != identity or info.st_size:
        os.close(fd)
        raise SmootherError(f"Scratch reservation changed: {path}")
    return fd


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def checksum(path):
    path = Path(path)
    with path.open("rb") as file:
        before = os.fstat(file.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise SmootherError("Input must be a regular local file.")
        digest = hashlib.file_digest(file, "sha256").hexdigest()
        if fingerprint(before) != fingerprint(os.fstat(file.fileno())):
            raise SmootherError("Input changed while calculating its SHA-256.")
    if fingerprint(before) != fingerprint(path.stat()):
        raise SmootherError("Input path changed while calculating its SHA-256.")
    return digest, fingerprint(before)


def budget(path, estimate):
    free = shutil.disk_usage(path).free
    required = estimate + MARGIN
    if free < required:
        raise SmootherError(f"Insufficient disk space. Need {required:,} bytes including reserve, have {free:,}.")
    return {"free_bytes": free, "required_bytes": required, "reserve_bytes": MARGIN}


def read_json(path):
    with safe_path(path).open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path, value):
    path = safe_path(path)
    tmp = path.with_name(f".{path.name}-{uuid.uuid4().hex}")
    try:
        with tmp.open("x", encoding="utf-8") as file:
            json.dump(value, file, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)
        sync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


class Job:
    def __init__(self, workspace, job_id):
        if not JOB_ID.fullmatch(job_id):
            raise SmootherError("Invalid job ID.")
        self.workspace = safe_path(workspace)
        self.id = job_id
        self.path = safe_path(self.workspace / "jobs" / job_id)
        self.manifest_path = self.path / "manifest.json"
        self.report_path = self.path / "report.json"

    @classmethod
    def create(cls, workspace, source):
        source = Path(source).resolve(strict=True)
        if not source.is_file():
            raise SmootherError("Input must be a regular local file.")
        job = cls(workspace, uuid.uuid4().hex[:16])
        jobs = job.workspace / "jobs"
        for parent in source.parents:
            same_jobs = jobs.is_dir() and jobs.samefile(parent)
            other_job = (parent.parent.name.casefold() == "jobs" and JOB_ID.fullmatch(parent.name.casefold())
                         and (parent / "manifest.json").is_file())
            if same_jobs or other_job:
                raise SmootherError("Choose a source outside managed job directories.")
        job.workspace.mkdir(parents=True, exist_ok=True)
        budget(job.workspace, 1024**2)
        original_hash, original_identity = checksum(source)
        job.path.mkdir(mode=0o700, parents=True)
        for folder in ("scratch", "previews"):
            (job.path / folder).mkdir(mode=0o700)
        write_json(job.manifest_path, {
            "version": 1, "job": job.id, "source": str(source),
            "sha256": original_hash, "source_identity": original_identity,
            "status": "created", "scratch": [], "previews": {}, "exports": [],
            "integrity": [],
        })
        return job

    @contextmanager
    def lock(self, shared=False):
        if not self.path.is_dir():
            raise SmootherError(f"Job {self.id} does not exist.")
        path = safe_path(self.path / ".lock")
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise SmootherError("Job lock is not a regular file.")
            try:
                flag = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
                fcntl.flock(fd, flag | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise SmootherError(f"Job {self.id} is active. No changes made.") from exc
            manifest = read_json(self.manifest_path)
            if manifest.get("version") != 1 or manifest.get("job") != self.id:
                raise SmootherError("Invalid job manifest.")
            yield manifest
        finally:
            os.close(fd)

    def verify(self, manifest, stage, require=True):
        try:
            digest, identity = checksum(manifest["source"])
            unchanged = digest == manifest["sha256"] and identity == manifest["source_identity"]
        except (OSError, SmootherError) as exc:
            digest, unchanged = None, False
            error = str(exc)
        else:
            error = None if unchanged else "Source bytes or file identity changed."
        manifest["integrity"].append({"stage": stage, "sha256": digest,
                                      "unchanged": unchanged, "error": error})
        self.save(manifest)
        if require and not unchanged:
            raise SmootherError(error)
        return unchanged

    def save(self, manifest):
        write_json(self.manifest_path, manifest)

    def scratch_file(self, manifest, label):
        rel = f"scratch/{uuid.uuid4().hex}-{label}"
        path = self.managed_file(rel, "scratch")
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            record = {"path": rel, "owner": owner_identity(os.fstat(fd))}
            sync_directory(path.parent)
            manifest["scratch"].append(record)
            self.save(manifest)
        finally:
            os.close(fd)
        return path

    def managed_file(self, rel, area, check_leaf=True):
        parts = Path(rel).parts
        if len(parts) != 2 or parts[0] != area or parts[1] in {".", ".."}:
            raise SmootherError(f"Invalid manifest path: {rel}")
        path = safe_path(self.path / area) / parts[1]
        return safe_path(path) if check_leaf else path

    def reported_path(self, manifest):
        path = self.path / "render-report.json" if manifest.get("render_report") else self.report_path
        return safe_path(path)

    def publication_path(self, entry):
        name = entry["destination"]
        if name == "render-report.json":
            return safe_path(self.path) / name
        if name in PROXIES.values():
            return self.managed_file(name, "previews", check_leaf=False)
        if name.startswith("exports/") and EXPORT_NAME.fullmatch(Path(name).name) and name.count("/") == 1:
            return safe_path(self.workspace / "exports") / Path(name).name
        raise SmootherError(f"Invalid publication destination: {name}")

    def pending_private_links(self, manifest):
        for entry in manifest.get("publication", []):
            if entry["destination"].startswith("exports/"):
                continue
            path = self.publication_path(entry)
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            if matches_publication(info, entry):
                yield path, info

    def published_export(self, manifest):
        """Find an occupied export destination that may belong to this job."""
        for entry in manifest.get("publication", []):
            if entry["destination"].startswith("exports/"):
                path = self.publication_path(entry)
                try:
                    info = path.lstat()
                except FileNotFoundError:
                    return None
                if owns_publication(info, entry) or not self.peer_owns_export(path, info, entry["destination"]):
                    return path, info, entry
                return None
        return None

    def peer_owns_export(self, path, info, destination):
        for peer in self.peer_manifests():
            for record in peer.get("exports", []):
                if record.get("path") == str(path) and owns_publication(info, record):
                    return True
            for record in peer.get("publication", []):
                if record.get("destination") == destination and owns_publication(info, record):
                    return True
        return False

    def staged_path(self, manifest, entry):
        for record in manifest["scratch"]:
            if isinstance(record, dict) and record["owner"] == entry["owner"]:
                return self.managed_file(record["path"], "scratch")
        raise SmootherError(f"Missing staging record for {entry['destination']}")

    def render_report(self, manifest, pending_path=None):
        if manifest["status"] == "rendered":
            records = manifest["exports"]
            if len(records) != 1:
                raise SmootherError("The completed job does not identify one export.")
            record = records[0]
            export = safe_path(record["path"])
            if not export.parent.samefile(safe_path(self.workspace / "exports")) or not EXPORT_NAME.fullmatch(export.name):
                raise SmootherError(f"Invalid completed export path: {export}")
            report_path = self.reported_path(manifest)
            if "report_identity" in manifest:
                info = report_path.lstat()
                if not stat.S_ISREG(info.st_mode) or fingerprint(info) != manifest["report_identity"]:
                    raise SmootherError(f"Completed render report changed: {report_path}")
        else:
            export = pending_path
            entry = next((item for item in manifest["publication"] if item["destination"] == "render-report.json"), None)
            if entry is None:
                raise SmootherError("Missing render report in publication journal.")
            report_path = self.publication_path(entry)
            if not report_path.exists():
                report_path = self.staged_path(manifest, entry)
            if not matches_publication(report_path.lstat(), entry):
                raise SmootherError(f"Publication report changed: {report_path}")
        report = read_json(report_path)
        media = report.get("export") or {}
        if (report.get("job") != self.id or media.get("path") != str(export) or not media.get("decoded_all_frames")
                or media.get("codec") not in {"h264", "hevc", "prores"}):
            raise SmootherError(f"Render report does not describe the verified export: {export}")
        return report

    def validate_committed_export(self, manifest, report):
        record = manifest["exports"][0]
        path = safe_path(record["path"])
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != record["bytes"]:
            raise SmootherError(f"Completed export changed: {path}")
        hash_verified = bool(record.get("sha256"))
        if (hash_verified or "identity" in record) and (
                "identity" not in record or not matches_export(path, info, record, verify_bytes=True)):
            raise SmootherError(f"Completed export failed verification: {path}")
        if not hash_verified:
            validate_legacy_export(path, report)
        check_export_unchanged(path, info)
        return hash_verified

    def recover_publication(self, manifest, published, report):
        path, expected, entry = published
        info = check_export_unchanged(path, expected)
        if not matches_export(path, info, entry, verify_bytes=True):
            raise SmootherError(f"Visible export failed verification. Recovery files were retained: {path}")
        hash_verified = bool(entry.get("sha256"))
        if not hash_verified:
            validate_legacy_export(path, report)
        self.finish_publication(manifest, recovered_export=info)
        return hash_verified

    def finish_publication(self, manifest, recovered_export=None):
        entries = sorted(manifest["publication"], key=lambda entry: entry["destination"].startswith("exports/"))
        previews, exports = {}, []
        report_identity = None
        for entry in entries:
            destination = self.publication_path(entry)
            name = entry["destination"]
            is_export = name.startswith("exports/")
            if is_export and recovered_export is not None:
                info = check_export_unchanged(destination, recovered_export)
            else:
                try:
                    info = destination.lstat()
                except FileNotFoundError:
                    stage = self.staged_path(manifest, entry)
                    if not matches_publication(stage.lstat(), entry):
                        raise SmootherError(f"Staged file changed: {stage}")
                    os.link(stage, destination, follow_symlinks=False)
                    info = destination.lstat()
                if is_export and not matches_export(destination, info, entry):
                    raise SmootherError(f"Visible export failed verification. Recovery files were retained: {destination}")
                if not is_export and not matches_publication(info, entry):
                    raise SmootherError(f"Publication destination already exists or changed: {destination}")
            if is_export:
                exports.append({"path": str(destination), "bytes": info.st_size, "owner": entry["owner"],
                                "identity": fingerprint(info), "sha256": entry.get("sha256")})
            elif name in PROXIES.values():
                previews[destination.stem] = {"path": name, "identity": entry["identity"]}
            elif name == "render-report.json":
                report_identity = entry["identity"]
            sync_directory(destination.parent)
        if len(exports) != 1 or report_identity is None:
            raise SmootherError("Publication is missing its verified export or render report.")
        if recovered_export is not None:
            check_export_unchanged(Path(exports[0]["path"]), recovered_export)
        committed = {**manifest, "status": "rendered", "previews": previews,
                     "render_report": True, "report_identity": report_identity, "exports": exports}
        committed.pop("publication")
        self.save(committed)
        manifest.update(committed)
        manifest.pop("publication")

    def peer_manifests(self):
        for path in safe_path(self.workspace / "jobs").iterdir():
            if not JOB_ID.fullmatch(path.name) or path == self.path or path.is_symlink():
                continue
            try:
                peer = read_json(path / "manifest.json")
            except (FileNotFoundError, ValueError, SmootherError):
                continue
            if isinstance(peer, dict) and peer.get("version") == 1 and peer.get("job") == path.name:
                yield peer

    def source_references(self, manifest):
        """Include previously analyzed sources that predate the containment guard."""
        refs = {tuple(manifest["source_identity"][:2])}
        refs.update(tuple(peer["source_identity"][:2]) for peer in self.peer_manifests())
        return refs

    def remove_owned(self, path, info):
        current = safe_path(path).lstat()
        if fingerprint(current) != fingerprint(info) or owner_identity(current) != owner_identity(info):
            raise SmootherError(f"Cleanup candidate changed: {path}")
        path.unlink()
        sync_directory(path.parent)

    def rollback_unpublished(self, manifest):
        if not manifest.get("publication"):
            return
        if self.published_export(manifest) is not None:
            raise SmootherError("The export destination is occupied. Recovery files were retained.")
        refs = self.source_references(manifest)
        for path, info in self.pending_private_links(manifest):
            if tuple(fingerprint(info)[:2]) not in refs:
                self.remove_owned(path, info)
        manifest.pop("publication")
        manifest["status"] = "render_failed"
        self.save(manifest)

    def export_file(self, name, suffix, *, create=True):
        if not EXPORT_NAME.fullmatch(name) or Path(name).suffix != suffix:
            raise SmootherError(f"Use an export filename ending in {suffix}, within the exports directory.")
        folder = safe_path(self.workspace / "exports")
        if create:
            folder.mkdir(exist_ok=True)
        elif folder.exists() and not folder.is_dir():
            raise SmootherError(f"Exports path is not a directory: {folder}")
        path = safe_path(folder / name)
        if path.exists():
            raise SmootherError(f"Export path already exists: {path}")
        return path

    def summary(self, manifest, port=8765):
        job_bytes = sum(p.lstat().st_size for p in self.path.rglob("*")
                        if not p.is_symlink() and p.is_file())
        export_bytes = sum(item["bytes"] for item in manifest["exports"])
        report_path = None
        warning = {}
        try:
            report = self.reported_path(manifest)
            if report.exists():
                report_path = str(report)
        except (OSError, SmootherError) as exc:
            warning["report_warning"] = str(exc)
        return {"job": self.id, "status": manifest["status"],
                "preview_url": f"http://127.0.0.1:{port}/" if manifest["previews"] else None,
                "exports": [item["path"] for item in manifest["exports"]],
                "report": report_path,
                "job_bytes": job_bytes, "export_bytes": export_bytes,
                "total_bytes": job_bytes + export_bytes, **warning}

    def cleanup(self, manifest, apply=False, previews=False):
        candidates = []
        for record in manifest["scratch"]:
            rel = record if isinstance(record, str) else record["path"]
            owner = None if isinstance(record, str) else record["owner"]
            candidates.append((rel, "scratch", owner))
        if previews:
            candidates.extend((item["path"], "previews", item["identity"])
                              for item in manifest["previews"].values())
        refs = self.source_references(manifest)
        planned = []
        skipped = []
        removed = set()
        published = self.published_export(manifest)
        reserved = {tuple(entry["owner"]) for entry in manifest["publication"]} if published else set()
        pending = [] if published else list(self.pending_private_links(manifest))
        if published:
            path, info, entry = published
            if not owns_publication(info, entry):
                reason = "Export ownership could not be verified. Recovery files were retained."
            elif matches_publication(info, entry):
                reason = "Export is visible. Retry render to finish publication."
            else:
                reason = "Visible export differs from the journal. Recovery files were retained."
            skipped.append({"path": str(path), "reason": reason})
        for path, info in pending:
            if tuple(fingerprint(info)[:2]) in refs:
                skipped.append({"path": str(path), "reason": "Used as a job source."})
            else:
                planned.append((path, info))
        for rel, area, identity in candidates:
            if area == "scratch" and identity is not None and tuple(identity) in reserved:
                path = self.managed_file(rel, area, check_leaf=False)
                skipped.append({"path": str(path), "reason": "Reserved for publication recovery."})
                continue
            path = self.managed_file(rel, area)
            if area == "previews" and rel not in PROXIES.values():
                raise SmootherError(f"Unexpected preview path: {rel}")
            try:
                info = path.lstat()
            except FileNotFoundError:
                removed.add(path)
                continue
            if not stat.S_ISREG(info.st_mode):
                raise SmootherError(f"Unsafe cleanup candidate: {path}")
            if tuple(fingerprint(info)[:2]) in refs:
                skipped.append({"path": str(path), "reason": "Used as a job source."})
                continue
            if area == "scratch" and (identity is None or owner_identity(info) != identity):
                skipped.append({"path": str(path), "reason": "Scratch ownership could not be verified."})
                continue
            if area == "previews" and fingerprint(info) != identity:
                skipped.append({"path": str(path), "reason": "Preview identity changed."})
                continue
            planned.append((path, info))
        links = Counter(tuple(fingerprint(info)[:2]) for _, info in planned)
        result = {"applied": apply, "paths": [], "skipped": skipped}
        for path, info in planned:
            key = tuple(fingerprint(info)[:2])
            blocks = info.st_blocks if links[key] == info.st_nlink else 0
            result["paths"].append({"path": str(path), "bytes": info.st_size, "reclaimable_bytes": blocks * 512})
            links[key] -= 1
        result["reclaimable_bytes"] = sum(item["reclaimable_bytes"] for item in result["paths"])
        if apply:
            self.verify(manifest, "before_cleanup")
            if not published and self.published_export(manifest) is not None:
                raise SmootherError("Export destination changed during cleanup. Recovery files were retained.")
            try:
                for path, info in planned:
                    self.remove_owned(path, info)
                    removed.add(path)
            finally:
                manifest["scratch"] = [record for record in manifest["scratch"]
                                       if self.path / (record if isinstance(record, str) else record["path"]) not in removed]
                if previews:
                    manifest["previews"] = {key: item for key, item in manifest["previews"].items()
                                            if self.path / item["path"] not in removed}
                if not published and all(path in removed for path, _ in pending):
                    manifest.pop("publication", None)
                self.verify(manifest, "after_cleanup")
        return result
