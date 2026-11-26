import json
import re
import subprocess

from . import __version__
from .jobs import SmootherError


RELEASES = "https://api.github.com/repos/Jenny-Media/camera-smoother/releases/latest"


def check_update():
    try:
        response = subprocess.run(
            ["curl", "--fail", "--silent", "--show-error", "--max-time", "10", RELEASES],
            check=True, capture_output=True, text=True,
        )
        release = json.loads(response.stdout)
        tag = release["tag_name"]
        if not isinstance(tag, str) or not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag):
            raise ValueError("expected a stable vMAJOR.MINOR.PATCH release tag")
    except subprocess.CalledProcessError as exc:
        raise SmootherError(f"Update check failed: {exc.stderr.strip() or 'GitHub request failed'}") from exc
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SmootherError(f"Update check failed: {exc}") from exc

    latest = tag.removeprefix("v")
    installed_version = tuple(map(int, __version__.split(".")))
    latest_version = tuple(map(int, latest.split(".")))
    return {
        "installed_version": __version__,
        "latest_version": latest,
        "update_available": latest_version > installed_version,
        "release_url": f"https://github.com/Jenny-Media/camera-smoother/releases/tag/{tag}",
        "installer_url": "https://smooth.jenny.media/install.sh",
    }
