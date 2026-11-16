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

### 1. Set up Sona

The setup script checks the small system prerequisites, offers to install
anything missing, and installs the web app. The large ML environment is
optional until you add your first song.

On macOS or Linux:

```sh
./scripts/bootstrap.sh
```

On Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
```

Pass `--check` to see what is missing without installing anything, `--yes` to
skip the confirmation prompt, or `--pipeline` to install song processing and
models eagerly. Re-running is safe. If Node.js is already installed,
`npm run setup` runs the same platform-aware setup.

### 2. Start Sona

```sh
npm start
```

That runs the web interface and its loopback-only local companion together.
Open the printed URL and choose **Add a song**. On first use, Sona offers to
install the Python tools and download the separation models: about 1.7 GiB on
Apple silicon Macs and NVIDIA systems, about 250 MiB elsewhere (see
[Performance and hardware support](#performance-and-hardware-support));
downloads are size- and SHA-256-verified. The rest of the app works before that
optional setup finishes.

<details>
<summary>Running the two parts separately</summary>

`npm start` is a convenience wrapper. You can still run each half in its own
terminal:

```sh
npm run dev        # web interface
npm run companion  # private local library and song processing
```

For the production build, run `npm run build`, then either `npm start -- --prod`
or `npm run serve` alongside `npm run companion`.

</details>

### What Sona needs

The setup script handles the first three. Listed here in case you would rather
install them yourself:

- [Node.js](https://nodejs.org/) 22.13 or newer
- Python 3.13 or newer
- [FFmpeg](https://ffmpeg.org/) and FFprobe available on `PATH`
- A current version of Chrome, Edge, Firefox, or Safari
- A microphone for live feedback and recording

## Choosing a pitch model

Each song's target contour is traced by a pitch model, and you can pick which
one per song. Choose it in the **Add a song** panel, or click the model name
on a song's card to re-analyse that song with a different one.

| Model | Notes |
| --- | --- |
| **CREPE** (default) | The original. Its cleanup steps are the most thoroughly tuned. |
| **RMVPE** | Built for singing with other sound still present. Downloads about 345 MiB the first time it is chosen. |
| **SwiftF0** | Small and very fast, with a decisive sense of when a note starts and stops. |

No model is best on every song, so if a contour looks wrong, trying another
model is often quicker than repairing it by hand. Re-analysis replaces the
song's target contour and leaves its audio alone; contour edits you saved
belong to the analysis they were made on and do not carry over. The companion
normally deletes a song's separated stems once it is published, so re-analysis
separates the song again; start the companion with `SONA_KEEP_INTERMEDIATES=1`
to make switching models take seconds instead of minutes.

## Using the studio

### Practicing

Practicing is the default mode. Enable your microphone and press play to see
your detected pitch against the target contour. Sona shows whether you are on
pitch and scores voiced sections as you sing.

Turn on **Guide vocals** when you want to hear the original singer; the song
keeps its place and your score when you switch. Use **Mic timing** if the
detected pitch appears slightly ahead of or behind the music. The setting is
remembered for each microphone and also aligns recorded vocals.

Headphones give the most accurate tracking. If the music plays out loud
instead, turn on **Using speakers** so the microphone does not score the song
itself. **Any-octave scoring** gives credit for singing the melody an octave
above or below the target.

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

The processing pipeline runs on macOS, Windows, and Linux. It uses one of two
sets of separation models, chosen during setup:

| Model set | Chosen automatically on | Download | Models |
| --- | --- | --- | --- |
| `roformer` | Apple silicon Macs and NVIDIA CUDA systems | about 1.7 GiB | Mel-Band RoFormer vocal and karaoke models |
| `compact` | everything else | about 250 MiB | Kim MDX-Net and UVR karaoke models |

The RoFormer models separate vocals more cleanly but run on PyTorch and are
several times slower than real time without a GPU, so machines that would run
them on the CPU keep the compact set. Set `SONA_SEPARATOR` to `roformer` or
`compact` before running setup and the pipeline to choose explicitly; the
default is `auto`. Only the selected set is downloaded.

Within a model set the pipeline selects the fastest available packaged
inference backend:

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

## Host the interface without hosting user media

The production build is already configured for Cloudflare Workers. Deploy the
UI shell with:

```sh
npm run deploy
```

On a machine using the hosted UI, run only the local companion, naming the
origin you deployed to so it will answer that page:

```sh
PIPELINE_ALLOWED_ORIGINS=https://studio.example.com npm run companion
```

The browser asks before the hosted page connects to `127.0.0.1`. Songs,
generated stems, contours, models, and recordings remain on that computer; the
Cloudflare Worker receives none of them. Sona waits for an explicit **Add a
song** click before requesting local-network access.

### Browser support for the hosted shell

| Browser | Hosted UI + local companion |
| --- | --- |
| Chrome / Edge | Works. Chrome 142 and later show a Local Network Access prompt on the first connection; allow it. |
| Firefox | Works. |
| Safari | Not supported. Run Sona locally instead. |

Loopback is a [potentially trustworthy origin](https://developer.mozilla.org/en-US/docs/Web/Security/Mixed_content),
so an HTTPS page reaching `http://127.0.0.1` is not mixed content. WebKit has
not adopted that rule ([bug 171934](https://bugs.webkit.org/show_bug.cgi?id=171934)),
so Safari blocks the request no matter how the companion is configured. Sona
detects this and says so rather than reporting the companion as down. Safari
users should open `http://localhost:3000` with `npm start`, which is
unaffected.

No hosted origin is trusted by default. Name the exact origin you deploy to
when starting the companion:

```sh
# macOS/Linux
PIPELINE_ALLOWED_ORIGINS=https://studio.example.com npm run companion
```

```powershell
# Windows PowerShell
$env:PIPELINE_ALLOWED_ORIGINS="https://studio.example.com"; npm run companion
```

Separate multiple origins with commas. The companion binds only to
`127.0.0.1`; unlisted browser origins are rejected before request bodies are
read.

## Troubleshooting

**The Add a song window says the processing companion is unavailable.**

The companion is not running. Use `npm start` for the fully local app, or keep
`npm run companion` running when using a hosted copy. In Chrome, allow the
site's Local Network Access request. In Safari, a hosted copy cannot reach the
companion at all — use `npm start` and open `http://localhost:3000`.

**The companion is running, but song processing needs setup.**

Choose **Install locally** in the Add a song panel. If Python or FFmpeg is
missing, run `npm run setup`, restart Sona, and try again. For an eager or
headless install, use `npm run pipeline:setup`.

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

For a song id such as `example`, the companion creates:

```text
.sona-data/library/audio/example.mp3
.sona-data/library/audio/example-instrumental.mp3
.sona-data/library/data/example-contour.json
.sona-data/library/data/example-pyin.json
```

It also updates `.sona-data/library/library.json`. Set `SONA_DATA_DIR` to use a
different private local directory. Intermediate analysis and cache data stay
under ignored local directories. `npm run build` refuses to run if song audio,
contours, or model weights appear in `public/`, preventing an accidental media
deployment.

The main pipeline files are:

- `server/pipeline-server.mjs` — local upload server and job queue
- `work/process_song.py` — cached end-to-end orchestration
- `work/hardware_acceleration.py` — hardware detection and CPU fallback
- `work/separate_stems.py` — RoFormer separation: one pass per stem pair
- `work/separate_kim_stem.py` — compact-set vocal and instrumental separation
- `work/separate_lead_vocals.py` — compact-set lead and backing separation
- `contour.py` — pitch extraction with the song's chosen model, plus pYIN
- `work/pitch_models.py` — the selectable pitch models and each one's own
  thresholds and tuning
- `work/calibrate_pitch_model.py` — measures a model's confidence thresholds
  from kept stems
- `work/export_contour.py` — contour correction and browser export
- `work/contour_pipeline_config.py` — named contour presets
- `transcribe_notes.py` — note evidence for conservative contour repairs

Model URLs, sizes, and hashes live in
`contour_out/models/audio-separator/model-assets.json`. The downloaded model
files are ignored by Git and should not be replaced without updating and
revalidating that manifest.

Each pitch model has its own entry in `work/pitch_models.py`: the native
confidence values it should be held to at each level the pipeline tests, its
voicing-gate levels, and the name of its own contour preset in
`work/contour_pipeline_config.py`. Changing one model's entry or preset never
affects another model, and cached work for that model is redone automatically.
The RMVPE and SwiftF0 thresholds were measured on a single song, so treat them
as a starting point: keep a few songs' stems, run
`.venv/bin/python work/calibrate_pitch_model.py --model rmvpe <song ids>`, and
paste the rows it prints.

To process a song directly into the local library after setup:

```sh
.venv/bin/python work/process_song.py example \
  --mp3 /path/to/example.mp3 \
  --output-dir .sona-data/library \
  --pitch-model crepe
```

`--pitch-model` takes `crepe`, `rmvpe`, or `swiftf0`. RMVPE's weights are
fetched by the companion on first use; from the command line run
`node scripts/model-assets.mjs pitch rmvpe` once beforehand.

The pipeline creates missing output directories automatically and can resume
from its integrity-checked stage cache. Add `--force` to rebuild every stage.

Each song produces four full-length WAV stems (about 190 MB) under
`contour_out/`. The companion deletes them once a song is published, keeping
the small pitch evidence; set `SONA_KEEP_INTERMEDIATES=1` to keep them. Running
`process_song.py` directly keeps them unless you pass `--clean-intermediates`.

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
does not include songs or generated contours. Model weights are downloaded
from their canonical upstream release only after an explicit local install;
public availability is not itself a grant of reuse or redistribution rights.

## License

Sona is available under the [MIT License](LICENSE). Third-party components and
downloaded model assets remain subject to the terms in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
