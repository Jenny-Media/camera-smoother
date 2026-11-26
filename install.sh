#!/bin/bash
set -euo pipefail

die() {
    printf 'Camera Smoother: %s\n' "$*" >&2
    exit 1
}

release=v0.1.1
target="${1:-both}"
install_home="${CAMERA_SMOOTHER_INSTALL_HOME:-$HOME}"
case "$target" in
    codex|claude|both|cli) ;;
    *) die 'Choose codex, claude, both, or cli.' ;;
esac
[[ $# -le 1 ]] || die 'Choose a single installation target.'
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] ||
    die 'This installer requires an Apple silicon Mac and an arm64 shell.'
[[ "$install_home" == /* ]] || die 'CAMERA_SMOOTHER_INSTALL_HOME must be an absolute path.'
command -v git >/dev/null || die 'Install Apple Command Line Tools with xcode-select --install.'

# Homebrew's bin directory may be missing from a newly configured shell.
if [[ -x /opt/homebrew/bin/brew ]]; then
    export PATH="/opt/homebrew/bin:$PATH"
fi
ffmpeg_version="$(ffmpeg -version 2>/dev/null || true)"
if ! command -v python3.13 >/dev/null || ! command -v ffprobe >/dev/null ||
    [[ ! "$ffmpeg_version" =~ ^ffmpeg\ version\ ([89]|[1-9][0-9]+)\. ]]; then
    brew="$(command -v brew)" || die 'Install Homebrew from https://brew.sh, then rerun this command.'
    "$brew" install ffmpeg python@3.13
fi
command -v ffprobe >/dev/null || die 'FFprobe must be available on PATH.'
ffmpeg_version="$(ffmpeg -version 2>/dev/null || true)"
[[ "$ffmpeg_version" =~ ^ffmpeg\ version\ ([89]|[1-9][0-9]+)\. ]] ||
    die 'FFmpeg 8 or newer must be available on PATH.'

python3.13 -I - "$install_home" "$target" "$release" <<'PYTHON'
import fcntl
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


REPO = "https://github.com/Jenny-Media/camera-smoother.git"
LEGACY_COMMIT = "fc16d3fad282243409d778a30f198f2f0a9b03c2"
SKILL = Path("skills/camera-move-smoothing")
LAYOUT = "camera-smoother-install-v1\n"


def exists(path):
    return path.exists() or path.is_symlink()


def run(*args, cwd=None, capture=False):
    return subprocess.run(args, cwd=cwd, check=True, text=True, capture_output=capture).stdout


def link_matches(path, target):
    return path.is_symlink() and path.resolve() == target.resolve()


def version_tuple(tag):
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag):
        raise ValueError(f"Invalid release tag: {tag}")
    return tuple(map(int, tag[1:].split(".")))


def check_legacy(root):
    marker = root / ".camera-smoother-release"
    if (not marker.is_file() or marker.is_symlink()
            or marker.read_text().strip() != "v0.1.0"
            or run("git", "rev-parse", "HEAD", cwd=root, capture=True).strip() != LEGACY_COMMIT
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=root).returncode != 0):
        raise ValueError(f"An unrecognized or modified installation occupies {root}.")


def install_release(releases, tag):
    # Keep the tested environment at its original absolute path so venv entry
    # points stay valid after the active release changes.
    runtime = Path(tempfile.mkdtemp(prefix=f"{tag}.", dir=releases))
    try:
        run("git", "init", "--quiet", "--initial-branch=main", cwd=runtime)
        run("git", "fetch", "--quiet", "--depth=1", REPO, f"refs/tags/{tag}", cwd=runtime)
        run("git", "checkout", "--quiet", "--detach", "FETCH_HEAD", cwd=runtime)
        run(sys.executable, "-m", "venv", runtime / ".venv")
        python = runtime / ".venv/bin/python"
        run(python, "-m", "pip", "install", "-r", runtime / "requirements.txt")
        run(python, "-I", "-c", "import cv2, numpy")
        actual = run(runtime / "camera-smoother", "--version", capture=True).strip()
        if actual != f"camera-smoother {tag[1:]}":
            raise ValueError(f"Release {tag} reported {actual!r}.")
        if not (runtime / SKILL / "SKILL.md").is_file():
            raise ValueError(f"Release {tag} is missing its skill.")
        (runtime / ".camera-smoother-release").write_text(f"{tag}\n")
        return runtime
    except BaseException:
        shutil.rmtree(runtime)
        raise


def install(root, links, tag):
    marker = root / ".camera-smoother-managed"
    current = root / ".current"
    releases = root / ".releases"
    entry = root / "camera-smoother"
    skill_entry = root / SKILL / "SKILL.md"

    for link in links:
        if exists(link) and not link_matches(link, root / SKILL):
            raise ValueError(f"A skill already exists at {link}. Move it aside before installing.")
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError(f"An existing path occupies {root}.")
    if not root.exists():
        root.mkdir()
        marker.write_text(LAYOUT)
    elif not exists(marker):
        check_legacy(root)
        if exists(current) or exists(releases):
            raise ValueError(f"Migration paths are already occupied inside {root}.")
        marker.write_text(LAYOUT)
    elif marker.is_symlink() or not marker.is_file() or marker.read_text() != LAYOUT:
        raise ValueError(f"An existing installation or folder occupies {root}.")

    for directory in (releases, skill_entry.parent.parent, skill_entry.parent):
        if directory.is_symlink():
            raise ValueError(f"Unexpected symlink inside the install directory: {directory}.")
    if entry.is_symlink() or (entry.exists() and not entry.is_file()):
        raise ValueError(f"Unexpected launcher at {entry}.")
    releases.mkdir(exist_ok=True)

    runtime = None
    if exists(current):
        if not current.is_symlink() or current.resolve().parent != releases.resolve():
            raise ValueError("The active release path is outside the managed release directory.")
        active = current.resolve(strict=True)
        active_tag = (active / ".camera-smoother-release").read_text().strip()
        if version_tuple(active_tag) > version_tuple(tag):
            raise ValueError(f"{active_tag} is already installed. This installer would downgrade it to {tag}.")
        if active_tag == tag:
            runtime = active

    skill_target = current / SKILL / "SKILL.md"
    if exists(skill_entry) and not link_matches(skill_entry, skill_target):
        # The first release used a real skill file in this stable directory.
        if not skill_entry.is_file() or skill_entry.is_symlink() or not (root / ".git").is_dir():
            raise ValueError(f"An existing skill occupies {skill_entry}.")
        original = run("git", "show", f"{LEGACY_COMMIT}:{SKILL}/SKILL.md", cwd=root, capture=True)
        if skill_entry.read_text() != original:
            raise ValueError(f"The previous Camera Smoother skill was modified at {skill_entry}.")

    if runtime is None:
        runtime = install_release(releases, tag)

    # Switch the runtime only after clone, dependencies and version validation
    # succeed. Older environments remain available to already-running jobs.
    with tempfile.TemporaryDirectory(prefix=".switch-", dir=root) as scratch:
        scratch = Path(scratch)
        if not current.is_symlink() or not entry.is_file():
            shutil.copy2(runtime / "camera-smoother", scratch / "launcher")
            os.replace(scratch / "launcher", entry)
        if not link_matches(current, runtime):
            (scratch / "current").symlink_to(runtime, target_is_directory=True)
            os.replace(scratch / "current", current)
        if not link_matches(skill_entry, skill_target):
            skill_entry.parent.mkdir(parents=True, exist_ok=True)
            (scratch / "skill").symlink_to(skill_target)
            os.replace(scratch / "skill", skill_entry)

    for link in links:
        if not link_matches(link, root / SKILL):
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(root / SKILL, target_is_directory=True)
        print(f"Skill installed at {link}")
    print(f"Installed {tag}. CLI: {entry}")
    if links:
        print("Start a new agent session and ask it to use the camera-move-smoothing skill.")


def main():
    home, target, tag = sys.argv[1:]
    home = Path(home)
    shared = home / ".local/share"
    root = shared / "camera-smoother"
    links = [
        home / folder / SKILL for name, folder in (("codex", ".agents"), ("claude", ".claude"))
        if target in (name, "both")
    ]
    shared.mkdir(parents=True, exist_ok=True)
    lock = os.open(shared / ".camera-smoother-install.lock", os.O_CREAT | os.O_NOFOLLOW | os.O_RDWR, 0o600)
    with os.fdopen(lock, "rb"):
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another install or update is running.") from exc
        install(root, links, tag)


try:
    main()
except (OSError, subprocess.CalledProcessError, ValueError) as exc:
    sys.exit(f"Camera Smoother: {exc}")
PYTHON
