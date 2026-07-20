# Sona

![Sona — Find your note](public/og.png)

**A personal singing studio for practicing pitch, shaping vocal guides, and
recording complete takes from your own music.**

Sona turns an MP3 into an instrumental track and a scrolling pitch contour.
Follow the contour with live microphone feedback, correct the target where the
automatic analysis needs help, and export a finished performance without
leaving the app.

Sona ships with an empty library. Add music you are authorized to use and the
local processing companion will prepare it for the studio.

## What you can do

- **Practice:** sing against a live pitch target with scoring, note feedback,
  guide vocals, voice-range guidance, and microphone timing adjustment.
- **Edit:** drag over any part of the contour to remove it, smooth it, or repair
  it from the lead-vocal analysis. Save your changes as the song's default or
  restore the original contour at any time.
- **Record:** capture a take, pause and resume, restart from the beginning, or
  rewind to the playhead and continue from there. When you finish, download raw
  vocals, the transposed instrumental, and a combined mix.
- **Choose the right key:** move the song and target contour together from
  −16 to +16 semitones, or unlock them to change each independently. Song key
  affects playback and recordings; contour key affects only the target and
  scoring.
- **Train away from a song:** use the built-in vocal exercises for focused
  pitch practice.

## Quick start

### 1. Install the requirements

- [Node.js](https://nodejs.org/) 22.13 or newer
- Python 3.13
- [FFmpeg](https://ffmpeg.org/) and FFprobe available on `PATH`
- A current version of Chrome, Edge, Firefox, or Safari
- A microphone for live feedback and recording

### 2. Set up Sona

```sh
npm install
npm run pipeline:setup
```

The first pipeline setup creates a local Python environment and downloads
about 249 MiB of vocal-separation model files. Downloads are verified by size
and SHA-256 hash before use.

### 3. Start both parts of the app

In one terminal, start the web interface:

```sh
npm run dev
```

In a second terminal, start the local song-processing companion:

```sh
npm run pipeline
```

Open the URL shown by the development server. Choose **Add a song**, select an
MP3, and keep the processing companion running while Sona prepares it. The song
will appear in your library when processing finishes.

## Using the studio

### Practicing

Practicing is the default mode. Enable your microphone and press play to see
your detected pitch against the target contour. Sona shows whether you are on
pitch and scores voiced sections as you sing.

Turn on **Guide vocals** when you want to hear the original singer. Use **Mic
timing** if the detected pitch appears slightly ahead of or behind the music.

### Editing

Editing opens with the original vocal mix so it is easier to hear what the
contour should follow. Drag horizontally across the contour and release to
choose an action:

- **Remove** clears the selected target.
- **Smooth** replaces unstable pitch with the median of the surrounding notes.
- **Repair vocals** copies the lead pYIN analysis directly, including silent
  regions.

Choose **Save as new default** to keep the edited contour in this browser.
Choose **Restore original** whenever you want to return to the generated
version.

### Recording

Recording captures your microphone against the instrumental. Guide vocals can
be enabled for monitoring, but they are never included in the exported backing
track.

- **Restart from beginning** deletes the current take and starts again at 0:00.
- **Rewind to playhead** keeps everything before the current playhead and
  discards everything after it.
- **End recording** prepares three WAV downloads: raw vocals, instrumental, and
  the combined mix.

The instrumental and combined downloads use the **Song key** setting. Changing
only **Contour key** never alters the recorded audio.

Keyboard shortcuts: <kbd>Space</kbd> plays or pauses, and the left and right
arrow keys skip five seconds.

## Performance and hardware support

The processing pipeline runs on macOS, Windows, and Linux. It automatically
selects the fastest available packaged inference backend:

- NVIDIA CUDA on supported NVIDIA systems
- Core ML or Metal Performance Shaders on macOS
- DirectML on Windows
- CPU everywhere else

Every accelerated inference stage retains a CPU fallback, so a missing or
unsupported accelerator should not prevent song processing. Custom ONNX
Runtime installations exposing MIGraphX/ROCm, OpenVINO, QNN, TensorRT, or
WebGPU are also detected.

For diagnostics, set `SONA_ACCELERATOR` to `cpu`, `cuda`, `coreml`, or
`directml` before running setup and the pipeline. The default is `auto`.

## Hosting the web interface

Song processing still happens through the companion running on the user's
computer. If a hosted Sona web interface needs to connect to that companion,
allow the site's exact origin when starting it:

```sh
PIPELINE_ALLOWED_ORIGINS=https://studio.example.com npm run pipeline
```

Separate multiple origins with commas. Local development origins are allowed
automatically; other origins are denied by default.

## Troubleshooting

**The Add a song window says the processing companion is unavailable.**

Run `npm run pipeline` in a second terminal and leave it open while adding the
song. If this is your first run, complete `npm run pipeline:setup` first.

**Microphone feedback or recording does not start.**

Allow microphone access for the Sona page in your browser. A hosted copy must
use HTTPS for browsers to expose microphone access; loopback development URLs
are treated as secure contexts.

**npm omitted the Rolldown native binding.**

Sona's `postinstall` script detects the operating system, CPU architecture, and
Linux libc, then installs the exact binding for the installed Rolldown version.
Run `npm install` again rather than placing a binding in `node_modules`
manually.

## Developer reference

For a song id such as `example`, the pipeline creates:

```text
public/audio/example.mp3
public/audio/example-instrumental.mp3
public/data/example-contour.json
public/data/example-pyin.json
```

It also adds the song to `public/library.json`. Intermediate analysis and cache
data stay under `contour_out/` and are not added to the public library.

The main pipeline files are:

- `server/pipeline-server.mjs` — local upload server and job queue
- `work/process_song.py` — cached end-to-end orchestration
- `work/hardware_acceleration.py` — hardware detection and CPU fallback
- `work/separate_kim_stem.py` — vocal and instrumental separation
- `work/separate_lead_vocals.py` — lead and backing vocal separation
- `contour.py` — CREPE and pYIN extraction
- `work/export_contour.py` — contour correction and browser export
- `work/contour_pipeline_config.py` — named contour presets
- `transcribe_notes.py` — note evidence for conservative contour repairs

Model URLs, sizes, and hashes live in
`contour_out/models/audio-separator/model-assets.json`. The downloaded model
files are ignored by Git and should not be replaced without updating and
revalidating that manifest.

To process a song directly after setup:

```sh
.venv/bin/python work/process_song.py example --mp3 /path/to/example.mp3
```

The pipeline creates missing output directories automatically and can resume
from its integrity-checked stage cache. Add `--force` to rebuild every stage.

### Validate a change

```sh
npm run lint
npm test
npm run pipeline:test
```

`npm test` builds the production web app and runs the JavaScript test suite.
`npm run pipeline:test` validates the Python processing pipeline after setup.
The production web build is written to `dist/`.

## Responsible use

Only process, record, and distribute material you are authorized to use. Sona
does not include songs or generated contours.

## License

Sona is available under the [MIT License](LICENSE). Third-party components and
downloaded model assets remain subject to the terms in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
