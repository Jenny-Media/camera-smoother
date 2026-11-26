#!/bin/bash
set -euo pipefail

die() {
    printf 'Camera Smoother: %s\n' "$*" >&2
    exit 1
}

check_skill() {
    if [[ -e "$1" || -L "$1" ]]; then
        [[ -L "$1" && "$(readlink "$1")" == "$skill_source" ]] ||
            die "A skill already exists at $1. Move it aside before installing."
    fi
}

install_skill() {
    check_skill "$1"
    if [[ ! -L "$1" ]]; then
        mkdir -p "$(dirname "$1")"
        ln -s "$skill_source" "$1"
    fi
    printf 'Skill installed at %s\n' "$1"
}

install_camera_smoother() {
    local target="${1:-both}" release=v0.1.0
    local repo=https://github.com/jenny-Media/camera-smoother.git
    local install_home="${CAMERA_SMOOTHER_INSTALL_HOME:-$HOME}"
    local skill_source install_dir codex_skill claude_skill marker ffmpeg_version brew
    case "$target" in
        codex|claude|both|cli) ;;
        *) die 'Choose codex, claude, both, or cli.' ;;
    esac
    [[ $# -le 1 ]] || die 'Choose a single installation target.'
    [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] ||
        die 'This installer requires an Apple silicon Mac and an arm64 shell.'
    [[ "$install_home" == /* ]] || die 'CAMERA_SMOOTHER_INSTALL_HOME must be an absolute path.'
    command -v git >/dev/null || die 'Install Apple Command Line Tools with xcode-select --install.'

    install_dir="$install_home/.local/share/camera-smoother"
    marker="$install_dir/.camera-smoother-release"
    skill_source="$install_dir/skills/camera-move-smoothing"
    codex_skill="$install_home/.agents/skills/camera-move-smoothing"
    claude_skill="$install_home/.claude/skills/camera-move-smoothing"
    if [[ "$target" == codex || "$target" == both ]]; then check_skill "$codex_skill"; fi
    if [[ "$target" == claude || "$target" == both ]]; then check_skill "$claude_skill"; fi

    if [[ -e "$install_dir" || -L "$install_dir" ]]; then
        [[ ! -L "$install_dir" && -f "$marker" && "$(cat "$marker")" == "$release" ]] ||
            die "An existing installation or folder occupies $install_dir."
    fi

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

    if [[ ! -f "$marker" ]]; then
        mkdir -p "$(dirname "$install_dir")"
        git clone --quiet --depth 1 --branch "$release" "$repo" "$install_dir"
        printf '%s\n' "$release" > "$marker"
    fi
    python3.13 -m venv "$install_dir/.venv"
    "$install_dir/.venv/bin/python" -m pip install -r "$install_dir/requirements.txt"
    "$install_dir/camera-smoother" --help >/dev/null

    if [[ "$target" == codex || "$target" == both ]]; then install_skill "$codex_skill"; fi
    if [[ "$target" == claude || "$target" == both ]]; then install_skill "$claude_skill"; fi
    printf 'Installed %s. CLI: %s/camera-smoother\n' "$release" "$install_dir"
    if [[ "$target" != cli ]]; then
        printf 'Start a new agent session and ask it to use the camera-move-smoothing skill.\n'
    fi
}

install_camera_smoother "$@"
