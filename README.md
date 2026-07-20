# Sona

Sona is a browser singing studio with contour editing, live pitch practice,
recording, guided vocal exercises, and a local MP3-to-contour pipeline. The
distributed library starts empty; no songs, recordings, or generated contours
are included.

## Requirements

- Node.js 22.13 or newer
- Python 3.13
- FFmpeg and FFprobe on `PATH`
- A current Chromium, Firefox, or Safari browser
- Microphone permission for live pitch feedback and recording

The pipeline runs on macOS, Windows, and Linux. Setup checks the host and
installs the best packaged ONNX Runtime automatically: NVIDIA CUDA when an
NVIDIA GPU is detected, Core ML on macOS, DirectML on Windows, and CPU
elsewhere. Each inference stage checks the providers that actually loaded and
keeps a CPU fallback, so unsupported or unavailable acceleration never makes
the pipeline unusable.

## Install and run

```sh
npm install
npm run pipeline:setup
```

Pipeline setup downloads approximately 249 MiB of public separator weights
from their canonical UVR release URLs. The weights are not stored in Git; every
download is checked against the exact byte size and SHA-256 hash recorded in
`contour_out/models/audio-separator/model-assets.json` before it can be used.

To force a backend for diagnostics, set `SONA_ACCELERATOR` to `cpu`, `cuda`,
`coreml`, or `directml` before running setup and the pipeline. The default is
`auto`. Custom ONNX Runtime installations exposing MIGraphX/ROCm, OpenVINO,
QNN, TensorRT, or WebGPU are also detected and preserved.

`npm install` runs the repository's Rolldown binding repair. It detects the
host OS, CPU architecture, and Linux libc, then installs the exact native
binding required by the installed Rolldown version if npm omitted it.

Run the app and local processing companion in separate terminals:

```sh
npm run dev
npm run pipeline
```

Open the local URL printed by the dev server. The library's **Add a song** card
uploads an MP3 to the companion, shows stage progress, and refreshes the library
when processing completes.

When a hosted copy of the web app needs to reach the companion on a user's
machine, allow that exact site origin when starting it:

```sh
PIPELINE_ALLOWED_ORIGINS=https://studio.example.com npm run pipeline
```

Multiple origins may be comma-separated. Loopback development origins are
always allowed; other origins remain denied by default.

## What the pipeline produces

For a song id such as `example`, the production path creates:

```text
public/audio/example.mp3
public/audio/example-instrumental.mp3
public/data/example-contour.json
public/data/example-pyin.json
```

It also adds metadata to `public/library.json`. Decoded notes are retained only
as internal contour-correction evidence under `contour_out/`; the removed
note-bar lane and its public JSON are not regenerated.

The main production files are:

- `server/pipeline-server.mjs`: local upload server and one-at-a-time job queue
- `work/process_song.py`: cached end-to-end pipeline orchestration
- `work/separate_kim_stem.py`: Kim vocal/instrumental separation
- `work/separate_lead_vocals.py`: lead/backing vocal separation
- `work/hardware_acceleration.py`: cross-platform provider detection and fallback
- `contour.py`: CREPE and pYIN extraction
- `work/export_contour.py`: voicing, correction, and browser JSON export
- `work/contour_pipeline_config.py`: named contour presets
- `transcribe_notes.py`: internal note evidence used for conservative repairs

The small local model catalog and hash manifest are stored in
`contour_out/models/audio-separator/`. The three downloaded weights remain
ignored by Git. Do not replace an asset without updating and revalidating its
canonical URL, size, and hash in `model-assets.json`.

Third-party attribution and licensing information is recorded in
`THIRD_PARTY_NOTICES.md`. The application downloads public model-release
assets rather than redistributing those weights in this repository.

## Manual processing

With the Python environment configured, the UI and server ultimately run:

```sh
.venv/bin/python work/process_song.py example --mp3 /path/to/example.mp3
```

The pipeline creates output directories as needed and uses an integrity-checked
stage cache so an interrupted run can resume. Pass `--force` to rebuild every
stage.

## Validate and build

```sh
npm run lint
npm test
npm run pipeline:test
```

`npm test` performs the production web build and JavaScript tests.
`npm run pipeline:test` validates the restored Python pipeline after
`npm run pipeline:setup` has created `.venv`.

The production web build is written to `dist/`.

Only process and distribute material you are authorized to use.

## License

Sona is available under the MIT License. Third-party components and downloaded
model assets remain subject to the terms described in `THIRD_PARTY_NOTICES.md`.
