# Camera Smoother

A local CLI for evening out recorded camera moves on Apple silicon. FFmpeg reads and encodes the video. OpenCV tracks the move and can generate selected between-frame samples. A local browser shows the comparison.

## Install

The installer supports Apple silicon Macs with [Homebrew](https://brew.sh/) and Git. It installs missing FFmpeg and Python dependencies through Homebrew and uses a dedicated virtual environment. Review [the installer source](https://github.com/jenny-Media/camera-smoother/blob/v0.1.0/install.sh) before running it in your terminal.

**Codex**

```sh
installer=$(curl -fsSL https://raw.githubusercontent.com/jenny-Media/camera-smoother/v0.1.0/install.sh) && bash -c "$installer" -- codex
```

**Claude Code**

```sh
installer=$(curl -fsSL https://raw.githubusercontent.com/jenny-Media/camera-smoother/v0.1.0/install.sh) && bash -c "$installer" -- claude
```

Use `both` for both agents or `cli` for the standalone CLI. Open a new agent session and ask it to use `camera-move-smoothing` on a local clip. The CLI is installed at `~/.local/share/camera-smoother/camera-smoother`. The installer links the skill into `~/.agents/skills` for Codex and `~/.claude/skills` for Claude Code, and refuses to replace an existing skill at either destination.

```sh
"$HOME/.local/share/camera-smoother/camera-smoother" --help
```

### Manual setup from a checkout

Use Homebrew FFmpeg 8 or later and Python 3.13. Run these commands from a checkout, or `cd ~/.local/share/camera-smoother` after the one-line install to follow the CLI examples below.

```sh
brew install ffmpeg python@3.13
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./camera-smoother --help
```

The prototype was exercised on arm64 with FFmpeg 8.1.2, Python 3.13, OpenCV 5.1.0 and NumPy 2.5.5. The optional interpolator uses CPU optical flow. The application runs offline once its dependencies are installed.

## Analyze, render and review

```sh
SOURCE="/absolute/path/to/clip.mp4"
JOB=$(./camera-smoother analyze "$SOURCE" |
  .venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["job"])')
./camera-smoother render --job "$JOB"
./camera-smoother serve --job "$JOB"
```

Open [the local review page](http://127.0.0.1:8765/). The play button, timeline slider and clickable speed graph share one playhead. On phones, switch versions above the player. Cuts and low-confidence ranges are marked in gold. Purple ticks show generated frames. The server lists proxies for only the selected job and supports HTTP byte ranges. It binds to loopback and serves video, the review page and the graph's derived data.

Use `./camera-smoother --workspace /path/to/workspace serve --gallery` to play the edited proxies for completed jobs on one page. Open **Compare** on a clip to see its original and edited versions on a shared playhead. Gallery mode is local only and holds a shared lock on each listed job until the server stops. Restart the gallery after rendering to include new jobs.

Optional flags and variants are shown below. Each example analyzes a separate job so both exports can be retained.

```sh
# Three comparison versions and an H.264 final using selective interpolation
FLOW_JOB=$(./camera-smoother analyze "$SOURCE" |
  .venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["job"])')
./camera-smoother render --job "$FLOW_JOB" --interpolate flow
./camera-smoother serve --job "$FLOW_JOB" --port 8765

# ProRes 422 editing export, with an optional viewing region and frame-rate override
EDIT_JOB=$(./camera-smoother analyze "$SOURCE" --motion slide --reference .1,.15,.8,.65 |
  .venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["job"])')
./camera-smoother render --job "$EDIT_JOB" --codec prores --fps 30000/1001 --output edit.mov
```

- `--reference x,y,width,height` uses normalized coordinates in the displayed orientation. Choose a broad region with stable features. `--motion auto` estimates the model. Use `pan`, `tilt`, `slide`, `push` or `pull` when the intended move is known. Set `--fov` for a known horizontal field of view. The default is 60°.
- The default final is a silent, source-size H.264 MP4. ProRes 422 is silent too. The exporter keeps the source's displayed orientation and uses the nominal frame rate when timestamps support it. VFR input uses its actual presentation times and exports constant frame rate. A chosen frame rate can shift the duration by up to one output frame. The encoder checks geometry, frame rate, duration, and decoder playback.
- Proxies are separate silent H.264 MP4s sized to fit inside 1280×720 without upscaling. A job's final is outside its preview allowlist. The CLI prints the job ID, report, disk usage and export path. The preview URL becomes available when `serve` is running.
- Tagged SDR color fields are carried through where the codec supports them. The report lists differences. ProRes supports 10-bit encoding for classified SDR input and remaps full-range SDR to limited range to preserve playback levels. H.264 outputs 8-bit and reports when that reduces bit depth. HDR, log and unclassified high-bit-depth sources can be analyzed. Export stops for these inputs because their color handling is unverified.

### Trim and change the duration at the source frame rate

```sh
JOB=$(./camera-smoother analyze "$SOURCE" --start 27 --end 32.5 --duration 11.011 --motion pan |
  .venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["job"])')
./camera-smoother render --job "$JOB" --interpolate camera --output smooth-pan-59.94.mp4
./camera-smoother serve --job "$JOB"
```

Trim times are relative to the first displayed source frame. The selection includes frames whose presentation times are at or after `--start` and before `--end`. The report records the frame boundaries used and the browser shows the selected part of the clip. `--duration` sets the desired length after trimming, rounded to the output frame grid. For a changed duration, the preview's **Constant-speed reference** spreads the selected source frames over that duration. The other panels apply local camera-motion timing as well. The overall duration change also scales intentional holds and the distance in time between cuts.

`--interpolate camera` uses one motion transform per verified adjacent pair. It works best on coherent movement, such as a pan across distant scenery. When the alignment fails, the renderer reuses a captured frame and records its output index in `interpolation_fallback_frames`. Inspect foreground reveals and moving details before using the export. This mode runs locally on the CPU.

## Motion model and limits

Analysis uses actual presentation timestamps and forward/backward-checked feature tracks across a 5×4 grid. It balances image regions, estimates yaw or pitch for rotations, uses coherent regional displacement for slides in any direction, and uses shared expansion or contraction for pushes and pulls. It measures chroma on the same grid before and after aligning confidently tracked regions to distinguish a color cut from camera movement. Expansion alone is ambiguous with optical zoom. Narrow-angle rotation can resemble translation. A motion hint or reference region can help with those cases. Low texture, moving foregrounds, mixed motion, rapid changes and cuts can limit correction.

Within a reliable run, the time map reallocates output time to faster or slower parts of the same source path. It pins segment endpoints to the overall playback timeline and uses eased local rates between 0.55× and 1.80× that timeline's playback rate. The first and last picture are pinned at output presentation boundaries. Holds, reversals, cuts and unreliable ranges split the map. Original and unadjusted regions show the source picture active at each output time. The report warns when the output frame grid skips a detected shot. The original-frame baseline can repeat or skip source pictures in adjusted regions.

Optional bidirectional Farnebäck flow synthesizes arbitrary fractions of adjacent pictures. The renderer uses it only in retimed stretches that need additional sample times. It chooses real pictures near an existing sample. A failed flow consistency or appearance check falls back to a real picture, and interpolation excludes cut and unreliable pairs. Thin objects, occlusions, surf and foreground reveals still need visual review. Interpolation does not require model weights. The graph shows apparent model progress per second; compare curves within each segment since angular, displacement and scale units differ.

See [OpenCV tracking and dense flow documentation](https://docs.opencv.org/doc/doxygen/html/d4/dc2/modules_2video_2include_2opencv2_2video_2tracking_8hpp.html) and [FFprobe frame metadata](https://www.ffmpeg.org/ffprobe.html).

## Jobs and cleanup

The default workspace is `data/` inside the checkout. For an installed CLI, keep jobs in a dedicated folder outside the install directory, for example by running `export CAMERA_SMOOTHER_WORKSPACE="$HOME/Movies/Camera Smoother"`. You can also put `--workspace /absolute/path` before the subcommand.

```text
data/
  jobs/<job-id>/
    manifest.json     source SHA-256, asset inventory and integrity checks
    report.json        analysis, segments, and planned time map
    render-report.json rendered time map, frame counts, and validation
    scratch/
    previews/
  exports/
```

Source files are opened read-only. Use an input outside managed job directories. A new export is published only after verification and must have a new filename inside `exports/`. The CLI checks a conservative disk budget before processing, including the projected number of output frames in the report. It records SHA-256 verification after analysis, rendering, failed processing and applied cleanup.

```sh
./camera-smoother cleanup --job "$JOB"                   # Dry run for scratch
./camera-smoother cleanup --job "$JOB" --apply           # Remove listed scratch
./camera-smoother cleanup --job "$JOB" --previews        # Dry run including previews
./camera-smoother cleanup --job "$JOB" --previews --apply
```

Cleanup uses the job manifest and file identities, and refuses an active job. Stop its server with Ctrl+C before applying cleanup. Completed renders normally release their encoding scratch, so a dry run may report zero bytes. A retry or applied cleanup can roll back unfinished preview and report links if the destination is empty or another job in the workspace owns it. If another job claimed the filename, retry with a free name, for example `render --job "$JOB" --output retry.mp4`. If an unrelated file occupies the destination and ownership cannot be verified, the retry and cleanup retain the staged export and journal. Once the job's export is visible, rerun `render --job "$JOB"` to finish publication or retrieve the existing receipt. The source is only needed when encoding. A conflicting option returns the existing export's settings in `render_settings`. Analyze a new job to request different settings.

Cleanup lists the job's visible pending export and its reserved scratch in `skipped`, including cases where the export's metadata differs from the journal. New exports record SHA-256, which lets retry verify their bytes after a timestamp-only change. Retry decodes legacy exports that lack a saved hash and reports `export_identity_verified: false` when playback and media checks pass. That receipt cannot establish the original export's byte identity. If the available file identity or media checks fail, recovery leaves the files and journal in place. Cleanup skips replaced previews and files referenced as sources. Completed reports and exports are retained. `reclaimable_bytes` reflects allocated blocks and accounts for hard links. The original must remain available for the applied cleanup integrity check.

## Reproduce the synthetic acceptance run

```sh
mkdir -p data/fixtures
FIXTURE="data/fixtures/synthetic-$(date +%Y%m%d-%H%M%S).mp4"
.venv/bin/python tests/synthetic.py "$FIXTURE"
JOB=$(./camera-smoother analyze "$FIXTURE" |
  .venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["job"])')
./camera-smoother render --job "$JOB" --interpolate flow
./camera-smoother cleanup --job "$JOB"
./camera-smoother cleanup --job "$JOB" --previews
./camera-smoother serve --job "$JOB"
```

Install the optional JWT test dependency and run the suite. The HTTP tests listen on a temporary loopback port. The suite creates its own synthetic files and temporary workspace.

```sh
.venv/bin/python -m pip install -r requirements-remote.txt
.venv/bin/python -m unittest discover -s tests -v
```

The fixture includes an uneven pan and reversal, a hold, tilt, diagonal slide with a differently moving foreground rail, push-in and pullback, static tail and four cuts. Tests validate improvement against the known motion, endpoint pictures, holds and static frames, duration, selective interpolation, VFR timing and cut boundaries, ProRes playback levels, range requests, publication recovery, and source integrity. The paused-playhead regression runs when Node is available.

For footage review, provide two original clips of roughly 5–15 seconds each. A scenic pan with surf and a textured horizon works well. For the second clip, use a push-in or slide with a railing or nearby subject crossing a distant background, ideally with a foreground reveal. Leave a short lead-in and tail. SDR footage is ready for a full three-way comparison.

## Optional remote access, after explicit approval

Remote access is off by default. The procedure below involves Cloudflare account, DNS and Access changes. Confirm the hostname, approved email, selected job, and exact proxies before executing it.

1. Create a Cloudflare Access self-hosted application covering the entire approved hostname. Add a single Allow policy using the user's exact approved email and approved identity provider. Keep bypass, service-token and broad allow policies out of this application. Record the team's short name and the application's audience (AUD).
2. Add a Cache Rule matching that hostname with **Bypass cache**. Ensure other rules cannot make the hostname cache-eligible. The origin also returns `Cache-Control: private, no-store`.
3. Install the optional JWT dependency. Start a dedicated origin on another loopback port with the approved variants. Use one `--approve-proxy` flag per approved file. In this example only the Original and Retime only proxies are approved.

   ```sh
   .venv/bin/python -m pip install -r requirements-remote.txt
   ./camera-smoother serve --job "$JOB" --port 8766 \
     --remote-host review.example.com --access-team YOUR_TEAM \
     --access-audience YOUR_APP_AUD --access-email you@example.com \
     --approve-proxy original --approve-proxy retime
   ```

4. Create a **named**, locally managed tunnel. Use the returned UUID and credentials path in a private YAML file outside the project.

   ```sh
   cloudflared tunnel login
   cloudflared tunnel create camera-smoother-review
   ```

   ```yaml
   tunnel: YOUR_TUNNEL_UUID
   credentials-file: /absolute/private/path/YOUR_TUNNEL_UUID.json
   ingress:
     - hostname: review.example.com
       service: http://127.0.0.1:8766
       originRequest:
         httpHostHeader: review.example.com
         access:
           required: true
           teamName: YOUR_TEAM
           audTag:
             - YOUR_APP_AUD
     - service: http_status:404
   ```

5. After checking the Access application and cache rules, validate the tunnel configuration, create the approved route and run the named tunnel. Keep the origin port local.

   ```sh
   TUNNEL_CONFIG="/absolute/private/path/camera-smoother.yml"
   cloudflared tunnel --config "$TUNNEL_CONFIG" ingress validate
   cloudflared tunnel route dns camera-smoother-review review.example.com
   cloudflared tunnel --config "$TUNNEL_CONFIG" run camera-smoother-review
   ```

6. In a signed-out session, test both the page and a direct URL for every approved proxy, with a media range request. Confirm Access redirects or denies each request and sends no media bytes. For an approved signed-in session, verify the page and media carry `private, no-store` and show no cache hit or Age. Check that unapproved proxy URLs return 404 after sign-in and that iPhone Safari can play and seek near the beginning, middle and end. Use the actual approved paths when substituting for this example.

   ```sh
   curl -sS -D - -o /dev/null https://review.example.com/
   curl -sS -D - -o /dev/null -H 'Range: bytes=0-2047' \
     https://review.example.com/media/original.mp4
   ```

The origin validates the JWT signature, issuer, audience, expiry and exact email for every page or asset request. It exposes a fixed allowlist of approved proxies and the information needed to draw the comparison. Local tests covered valid, tampered, expired and wrong-identity JWTs and signed-out direct media access. Live Access and physical iPhone Safari require the approved remote setup.

See [Cloudflare named tunnel setup](https://developers.cloudflare.com/tunnel/features/locally-managed-tunnels/create-local-tunnel/), [tunnel ingress settings](https://developers.cloudflare.com/tunnel/reference/origin-parameters/), [Access JWT validation](https://developers.cloudflare.com/cloudflare-one/access-controls/applications/http-apps/authorization-cookie/validating-json/) and [Cloudflare cache behavior](https://developers.cloudflare.com/cache/concepts/cache-responses/).
