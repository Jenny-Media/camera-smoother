---
name: camera-move-smoothing
description: Trim and retime camera moves in existing local footage with Camera Smoother. Use when choosing clean edit points, slowing or evening out a pan or other move, or comparing captured-frame and interpolated drafts.
---

# Camera move smoothing

Use the installed CLI at `~/.local/share/camera-smoother/camera-smoother`, or use `./camera-smoother` from a configured source checkout. If the skill is linked into an agent directory, its resolved path is also inside the checkout. If the CLI is missing, point the user to the install guide at <https://smooth.jenny.media/>. Keep job files and exports in a dedicated workspace outside the install directory. Treat source clips as read-only.

When the user asks about updates, run `camera-smoother check-update` using the installed CLI path. Report `update_available` and the installed and latest versions. Use the installer in <https://smooth.jenny.media/agent.md> when the user asks to update.

1. Review the selected footage when cut points or pacing are uncertain. Sample the source around the start and end of the intended move, then narrow the range. Look for setup, camera resets, holds, and reversals. Use source times for trim decisions and output times for frame or preview checks.
2. Run `analyze` with `--start` and `--end` in source seconds when trimming. Add `--duration` for the requested output length. Keep the source frame rate by default. Use a motion hint only when the move is known or the estimated classification needs help.
3. Read `recommendation` and check the resolved trim bounds, reliable movement, warnings, and time map. Keep captured frames by default. Use `--interpolate auto` to preview high-confidence frame pairs when the retime needs in-between samples. For manual comparison, `camera` suits coherent moves and `flow` uses dense optical flow. Inspect large parallax, moving subjects, and soft detail.
4. Choose `--preset playback` for general playback, `apple-playback` for HEVC, `editing` for ProRes 422, or `share` for a smaller MP4. Keep source FPS by default. Use `--output` for a new filename and `render --plan` to confirm the destination. Use `serve --job` to compare versions. When duration changes, compare against the constant-speed reference. Inspect cuts, horizons, railings, surf, and foreground reveals at playback speed. Check the report for generated frames, fallbacks, and export validation.
5. Report the effective source range, output duration and frame rate, and which edit you recommend when the user asks. Link the export and the running preview. Keep previews on loopback unless the user approves wider access.

```sh
CS="$HOME/.local/share/camera-smoother/camera-smoother"
"$CS" --workspace "$WORK" analyze "$SOURCE" --start "$IN" --end "$OUT" --duration "$LENGTH"
"$CS" --workspace "$WORK" render --job "$JOB" --interpolate camera --output "$NAME.mp4"
"$CS" --workspace "$WORK" serve --job "$JOB"
```

Use the job ID from the analysis response for `JOB`. Omit trim and duration options the edit does not need.
