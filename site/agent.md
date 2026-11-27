# Use Camera Smoother on local footage

Help the user trim and retime recorded camera moves while preserving the original camera path. Run jobs through the CLI on the user's Apple silicon Mac. Use only source clips the user selects. Keep those clips read-only. Keep previews on `127.0.0.1` unless the user explicitly approves wider access.

## Install and update

The CLI needs Git and Homebrew. If the CLI is absent and the user asks you to install it, run this in a local shell. The default installs the CLI and the `camera-move-smoothing` skill for both Codex and Claude Code.

```sh
installer=$(curl -fsSL https://smooth.jenny.media/install.sh) && bash -c "$installer"
```

Append `-- codex`, `-- claude`, or `-- cli` to select one target. If the agent has no local shell, give the user the command to run in Terminal. Start a new agent session to load an installed or refreshed skill.

```sh
CS="$HOME/.local/share/camera-smoother/camera-smoother"
"$CS" --version
"$CS" check-update
```

Run `check-update` when the user asks about updates. It queries GitHub and reports the installed version, latest stable version, and whether an update is available. Report a newer release and rerun the install command when the user asks to update. An existing v0.1.0 install can upgrade with the same install command.

## Analyze, render, and review

Use the installed `camera-move-smoothing` skill if available. Keep jobs and exports in a dedicated local workspace. Set `SOURCE` to the clip the user selected, `WORK` to the workspace, and `JOB` to the job ID returned by `analyze`.

```sh
"$CS" --workspace "$WORK" analyze "$SOURCE"
"$CS" --workspace "$WORK" render --job "$JOB" --interpolate none
"$CS" --workspace "$WORK" serve --job "$JOB"
```

- Use `--start` and `--end` for trim points in source seconds, and `--duration` when the user wants a new length. Keep the source frame rate by default.
- Read `recommendation` from `analyze` before rendering. `little-change` means the plan found little reliable speed correction. `review-first` flags uncertain motion. Check cuts and trim bounds in the report. If `frame_generation` is `preview-auto`, consider `--interpolate auto` on a comparison job. It uses a stricter frame-pair confidence gate and falls back to captured frames when alignment fails. The default uses captured frames. Inspect foreground reveals, railings, and moving water before recommending the interpolated result.
- Use `--preset playback` for full-size H.264, `--preset apple-playback` for full-size HEVC for Apple devices, `--preset editing` for ProRes 422, or `--preset share` for a smaller H.264 MP4. All presets keep source FPS by default. Use `--fps` only when the user requests a different rate. Analyze a new job for another export.
- Set `--output` to the requested filename inside the workspace's `exports/` directory. Use `render --plan` with the same flags to show the exact destination and settings before encoding.
- Report the final export path, local preview URL, source range, output duration, and frame rate. Use `cleanup --job "$JOB"` to preview managed cleanup. Leave scratch and previews in place unless the user requests removal.

See the [CLI documentation](https://github.com/Jenny-Media/camera-smoother#readme) for options and troubleshooting.
