---
name: camera-move-smoothing
description: Trim and retime camera moves in existing local footage with Camera Smoother. Use when choosing clean edit points, slowing or evening out a pan or other move, or comparing captured-frame and interpolated drafts.
---

# Camera move smoothing

Use the installed CLI at `~/.local/share/camera-smoother/camera-smoother`, or use `./camera-smoother` from a configured source checkout. If the skill is linked into an agent directory, its resolved path is also inside the checkout. If the CLI is missing, point the user to the install guide at <https://smooth.jenny.media/>. Keep job files and exports in a dedicated workspace outside the install directory. Treat source clips as read-only.

1. Review the selected footage when cut points or pacing are uncertain. Sample the source around the start and end of the intended move, then narrow the range. Look for setup, camera resets, holds, and reversals. Use source times for trim decisions and output times for frame or preview checks.
2. Run `analyze` with `--start` and `--end` in source seconds when trimming. Add `--duration` for the requested output length. Keep the source frame rate by default. Use a motion hint only when the move is known or the estimated classification needs help.
3. Read the report before rendering. Check the resolved trim bounds, reliable movement segments, warnings, and time map. Choose `--interpolate camera` for coherent moves, such as a distant pan with strong features. Use `--interpolate none` when the user wants captured frames, or as a comparison when large parallax, moving subjects, or soft detail make synthesis questionable. Leave the fallback in place for rejected frames.
4. Render to a new filename and use `serve --job` to review the versions. When the duration changes, the comparison includes a constant-speed reference at the new duration. Inspect the result at playback speed and around important cuts, horizons, railings, surf, or foreground reveals. Check the report for generated frames, `interpolation_fallback_frames`, and export validation.
5. Report the effective source range, output duration and frame rate, and which edit you recommend when the user asks. Link the export and the running preview. Keep previews on loopback unless the user approves wider access.

```sh
CS="$HOME/.local/share/camera-smoother/camera-smoother"
"$CS" --workspace "$WORK" analyze "$SOURCE" --start "$IN" --end "$OUT" --duration "$LENGTH"
"$CS" --workspace "$WORK" render --job "$JOB" --interpolate camera --output "$NAME.mp4"
"$CS" --workspace "$WORK" serve --job "$JOB"
```

Use the job ID from the analysis response for `JOB`. Omit trim and duration options the edit does not need.
