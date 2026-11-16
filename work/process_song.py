#!/usr/bin/env python3
"""End-to-end pipeline for one uploaded song.

Separation uses one of two interchangeable model sets (see
``hardware_acceleration.separator_profile``):

``roformer`` -- the default wherever Torch has an accelerator:

* ``vocals_mel_band_roformer.ckpt`` splits the song once into the broad vocal
  pitch anchor and, by subtraction, the instrumental published to the web app.
* ``mel_band_roformer_karaoke_gabox.ckpt`` splits that vocal stem into
  lead/backing evidence.

``compact`` -- smaller ONNX/VR models that stay fast on DirectML, CoreML and
plain CPU:

* ``Kim_Inst.onnx`` supplies the instrumental.
* ``Kim_Vocal_2.onnx`` supplies the broad vocal pitch anchor.
* ``5_HP-Karaoke-UVR.pth`` splits that vocal stem into lead/backing evidence.

The resulting app assets are written into a caller-selected local library:

    <output>/audio/<id>.mp3
    <output>/audio/<id>-instrumental.mp3
    <output>/data/<id>-contour.json
    <output>/data/<id>-pyin.json

Decoded notes remain an internal source of conservative octave evidence; the
removed note-bar lane and its public JSON are not regenerated.

Progress is reported as ``::step:<name>`` / ``::done duration=<s>`` /
``::error <message>`` markers for ``server/pipeline-server.mjs``.

Usage:  python work/process_song.py <id> [--mp3 PATH]
The mp3 defaults to <root>/<id>.mp3 (where the companion server saves uploads).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:  # Support both ``python work/process_song.py`` and test/module imports.
    from .contour_pipeline_config import PRESETS, PRODUCTION_CONFIG
    from .hardware_acceleration import inference_hardware, separator_profile
    from .pipeline_cache import StageCache
    from .pitch_models import DEFAULT_PITCH_MODEL, PITCH_MODELS, get_pitch_model
except ImportError:  # pragma: no cover - exercised by the script entry point
    from contour_pipeline_config import PRESETS, PRODUCTION_CONFIG
    from hardware_acceleration import inference_hardware, separator_profile
    from pipeline_cache import StageCache
    from pitch_models import DEFAULT_PITCH_MODEL, PITCH_MODELS, get_pitch_model


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
MODEL_DIR = ROOT / "contour_out" / "models" / "audio-separator"
MODEL_ASSET_MANIFEST = MODEL_DIR / "model-assets.json"
SEPARATOR_ROOT = ROOT / "contour_out" / "separators"
KIM_SEPARATOR = ROOT / "work" / "separate_kim_stem.py"
KIM_INST_MODEL = "Kim_Inst.onnx"
KIM_VOCAL_MODEL = "Kim_Vocal_2.onnx"
LEAD_MODEL = "5_HP-Karaoke-UVR.pth"
STEM_SEPARATOR = ROOT / "work" / "separate_stems.py"
ROFORMER_VOCAL_MODEL = "vocals_mel_band_roformer.ckpt"
ROFORMER_VOCAL_CONFIG = "vocals_mel_band_roformer.yaml"
ROFORMER_LEAD_MODEL = "mel_band_roformer_karaoke_gabox.ckpt"
ROFORMER_LEAD_CONFIG = "config_mel_band_roformer_karaoke_gabox.yaml"
# Weights and their configs, per separator profile, pinned by SHA-256.
_MODEL_MANIFEST = json.loads(MODEL_ASSET_MANIFEST.read_text())
PROFILE_MODEL_HASHES = {
    profile: {asset["filename"]: asset["sha256"] for asset in entry["assets"]}
    for profile, entry in _MODEL_MANIFEST["profiles"].items()
}
# Weights for the selectable pitch models that need a download. They are
# fetched the first time a song asks for that model, not during setup.
PITCH_MODEL_HASHES = {
    name: {asset["filename"]: asset["sha256"] for asset in entry["assets"]}
    for name, entry in _MODEL_MANIFEST.get("pitchModels", {}).items()
}
MODEL_METADATA_HASHES = {
    "download_checks.json": "e3d2e06b67e9f1073d668295bcc76821030bb97359d2583bbd4bed3afbe18264",
    "mdx_model_data.json": "14f711c6cdc3309a47e63e20d41120fe678b2c604cf72f0336aae32f3498108e",
    "vr_model_data.json": "852518e3e51715e6f825879769669570351f2a51293c5cb7c3270233d196745b",
}
PIPELINE_CACHE_REVISION = 1


def marker(text: str) -> None:
    print(text, flush=True)


def run(command: list[str]) -> None:
    """Run a child process, forwarding output; raise on non-zero exit."""
    print("+", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}")


def ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    try:
        return round(float(out.stdout.strip()), 3)
    except ValueError:
        return 0.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_models(hash_file=sha256, profile: str | None = None) -> dict[str, str]:
    """Require the exact benchmarked weights before starting a costly run."""
    model_hashes = PROFILE_MODEL_HASHES[profile or separator_profile()]
    pipeline_assets = {**model_hashes, **MODEL_METADATA_HASHES}
    missing = [
        filename for filename in pipeline_assets if not (MODEL_DIR / filename).exists()
    ]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"missing separator asset(s): {joined}; run `npm run pipeline:setup`"
        )
    for filename, expected_hash in pipeline_assets.items():
        model_path = MODEL_DIR / filename
        actual_hash = hash_file(model_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"separator asset hash mismatch for {filename}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
    return dict(model_hashes)


def validate_pitch_model(name: str, hash_file=sha256) -> dict[str, str]:
    """Require the pinned weights of a pitch model that ships as a download."""
    hashes = PITCH_MODEL_HASHES.get(name, {})
    for filename, expected_hash in hashes.items():
        path = MODEL_DIR / filename
        if not path.exists():
            raise RuntimeError(
                f"missing {name} pitch model asset: {filename}; "
                f"run `node scripts/model-assets.mjs pitch {name}`"
            )
        actual_hash = hash_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"pitch model asset hash mismatch for {filename}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
    return dict(hashes)


def installed_versions(*packages: str) -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def command_version(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    text = (result.stdout or result.stderr).splitlines()
    return text[0].strip() if text else "unknown"


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def separator_command(
    python: Path,
    source: Path,
    model: str,
    stem: str,
    output_name: str,
    output_dir: Path,
    batch_size: int = 1,
) -> list[str]:
    """Build the bounded MDX command used in production and in unit tests."""
    return [
        str(python),
        str(KIM_SEPARATOR),
        str(source),
        "--model",
        model,
        "--stem",
        stem,
        "--output-name",
        output_name,
        "--model-dir",
        str(MODEL_DIR),
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(batch_size),
    ]


def separate_kim_product(
    source: Path,
    destination: Path,
    *,
    model: str,
    stem: str,
    output_name: str,
    batch_size: int = 1,
) -> Path:
    """Run one Kim model and atomically replace only its product stem."""
    if not KIM_SEPARATOR.exists():
        raise RuntimeError(f"Kim separator helper not found: {KIM_SEPARATOR}")
    if batch_size < 1:
        raise ValueError("Kim batch size must be at least 1")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{output_name}.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        run(
            separator_command(
                Path(PYTHON),
                source,
                model,
                stem,
                output_name,
                staging,
                batch_size,
            )
        )
        output = staging / f"{output_name}.wav"
        if not output.is_file():
            raise RuntimeError(f"Kim separator did not produce: {output}")
        output.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(staging)
    return destination


def stem_separator_command(
    python: Path,
    source: Path,
    model: str,
    primary: Path,
    secondary: Path,
) -> list[str]:
    """Build the one-pass Roformer command: a predicted stem and its remainder."""
    return [
        str(python),
        str(STEM_SEPARATOR),
        str(source),
        "--model",
        model,
        "--model-dir",
        str(MODEL_DIR),
        "--primary",
        str(primary),
        "--secondary",
        str(secondary),
    ]


def separate_stem_pair(
    source: Path,
    *,
    model: str,
    primary: Path,
    secondary: Path,
) -> None:
    """Run one Roformer model; the helper replaces both stems atomically."""
    if not STEM_SEPARATOR.exists():
        raise RuntimeError(f"stem separator helper not found: {STEM_SEPARATOR}")
    run(stem_separator_command(Path(PYTHON), source, model, primary, secondary))
    for output in (primary, secondary):
        if not output.is_file():
            raise RuntimeError(f"stem separator did not produce: {output}")


def remove_intermediate_audio(directories: list[Path]) -> int:
    """Delete regenerable WAV stems; return the number of bytes released."""
    released = 0
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in directory.glob("*.wav"):
            released += path.stat().st_size
            path.unlink()
        try:
            directory.rmdir()
        except OSError:
            pass  # Something else lives here; leave it.
    return released


def process(
    song: str,
    mp3: Path,
    *,
    force: bool = False,
    pitch_model: str = DEFAULT_PITCH_MODEL,
    clean_intermediates: bool = False,
    branch_workers: int = 2,
    kim_batch_size: int = 1,
    lead_vr_batch_size: int = 1,
    anchor_batch_size: int = 512,
    lead_batch_size: int = 64,
    pyin_workers: int = 0,
    output_dir: Path | None = None,
) -> float:
    if branch_workers not in (1, 2):
        raise ValueError("branch_workers must be 1 or 2")
    for name, value in (
        ("kim_batch_size", kim_batch_size),
        ("lead_vr_batch_size", lead_vr_batch_size),
        ("anchor_batch_size", anchor_batch_size),
        ("lead_batch_size", lead_batch_size),
    ):
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
    if pyin_workers < 0:
        raise ValueError("pyin_workers must be zero or greater")
    total_started = time.perf_counter()
    separator_dir = SEPARATOR_ROOT / song
    lead_vocals_dir = ROOT / "contour_out" / "lead_vocals" / song
    lead_vocals = lead_vocals_dir / "lead.wav"
    lead_contour = ROOT / "contour_out" / "lead_contours" / f"{song}_contour.csv"
    anchor_csv = ROOT / "contour_out" / f"{song}_contour.csv"
    final_notes = ROOT / "experiments" / "transcription_final" / f"{song}_notes_final.json"
    output_root = (output_dir or (ROOT / ".sona-data" / "library")).resolve()
    public_audio = output_root / "audio"
    public_data = output_root / "data"
    public_audio.mkdir(parents=True, exist_ok=True)
    public_data.mkdir(parents=True, exist_ok=True)

    cache = StageCache(ROOT, song, marker)
    profile = separator_profile()
    marker(f"::profile separator={profile}")
    model_hashes = validate_models(cache.file_hash, profile)
    tuning = get_pitch_model(pitch_model)
    contour_config = PRESETS[tuning.contour_preset]
    marker(f"::pitch-model {tuning.name} preset={tuning.contour_preset}")
    # Everything that should invalidate cached pitch work when this model or
    # its tuning changes.
    pitch_identity = {
        "pitchModel": tuning.name,
        "tuning": tuning.fingerprint(),
        "assets": validate_pitch_model(tuning.name, cache.file_hash),
    }
    crepe = tuning.name == "crepe"

    def code_hashes(*relative_paths: str) -> dict[str, str]:
        return cache.code_hashes(
            [Path(__file__), *(ROOT / path for path in relative_paths)]
        )

    pitch_environment = installed_versions(
        "numpy", "scipy", "librosa", "soundfile", "torch", "torchcrepe", "swift-f0"
    )
    separator_environment = installed_versions(
        "audio-separator",
        "onnxruntime",
        "onnxruntime-gpu",
        "onnxruntime-directml",
        "numpy",
        "scipy",
        "librosa",
        "soundfile",
        "torch",
    )
    hardware_environment = inference_hardware()
    pitch_environment["hardware"] = hardware_environment
    separator_environment["hardware"] = hardware_environment

    # Keep the user's source in caller-owned local storage. Downstream stages
    # use explicit paths and never need a deployable copy in the repository.
    root_mp3 = mp3.resolve()
    source_hash = cache.file_hash(root_mp3)

    # 1. Produce the dependency-critical vocal stem first.  The instrumental
    # has no consumers until publication, so it continues in the background
    # while the vocal-dependent pitch DAG runs below.
    marker("::step:separating")
    instrumental = separator_dir / "instrumental.wav"
    vocals = separator_dir / "vocals.wav"
    public_instrumental = public_audio / f"{song}-instrumental.mp3"
    if profile == "roformer":
        # One pass yields both products: the model predicts the vocals and the
        # instrumental is the input minus that prediction.
        cache.run(
            "stem_separation",
            {
                "revision": PIPELINE_CACHE_REVISION,
                "sourceSha256": source_hash,
                "model": {
                    name: model_hashes[name]
                    for name in (ROFORMER_VOCAL_MODEL, ROFORMER_VOCAL_CONFIG)
                },
                "arguments": {
                    "primary": "vocals",
                    "secondary": "instrumental",
                    "sampleRate": 44_100,
                },
                "environment": separator_environment,
                "code": code_hashes("work/separate_stems.py"),
            },
            [vocals, instrumental],
            lambda: separate_stem_pair(
                root_mp3,
                model=ROFORMER_VOCAL_MODEL,
                primary=vocals,
                secondary=instrumental,
            ),
            force=force,
        )
    else:
        cache.run(
            "vocal_separation",
            {
                "revision": PIPELINE_CACHE_REVISION,
                "sourceSha256": source_hash,
                "model": {KIM_VOCAL_MODEL: model_hashes[KIM_VOCAL_MODEL]},
                "arguments": {
                    "stem": "Vocals",
                    "sampleRate": 44_100,
                    "hopLength": 1024,
                    "segmentSize": 256,
                    "overlap": 0.25,
                    "batchSize": kim_batch_size,
                    "denoise": False,
                },
                "environment": separator_environment,
                "code": code_hashes("work/separate_kim_stem.py"),
            },
            [vocals],
            lambda: separate_kim_product(
                root_mp3,
                vocals,
                model=KIM_VOCAL_MODEL,
                stem="Vocals",
                output_name="vocals",
                batch_size=kim_batch_size,
            ),
            force=force,
        )

    def build_instrumental_branch() -> None:
        if profile != "roformer":
            cache.run(
                "instrumental_separation",
                {
                    "revision": PIPELINE_CACHE_REVISION,
                    "sourceSha256": source_hash,
                    "model": {KIM_INST_MODEL: model_hashes[KIM_INST_MODEL]},
                    "arguments": {
                        "stem": "Instrumental",
                        "sampleRate": 44_100,
                        "hopLength": 1024,
                        "segmentSize": 256,
                        "overlap": 0.25,
                        "batchSize": kim_batch_size,
                        "denoise": False,
                    },
                    "environment": separator_environment,
                    "code": code_hashes("work/separate_kim_stem.py"),
                },
                [instrumental],
                lambda: separate_kim_product(
                    root_mp3,
                    instrumental,
                    model=KIM_INST_MODEL,
                    stem="Instrumental",
                    output_name="instrumental",
                    batch_size=kim_batch_size,
                ),
                force=force,
            )
        cache.run(
            "instrumental_encode",
            {
                "revision": PIPELINE_CACHE_REVISION,
                "instrumentalSha256": cache.file_hash(instrumental),
                "arguments": {"codec": "libmp3lame", "bitrate": "192k"},
                "ffmpeg": command_version(["ffmpeg", "-version"]),
            },
            [public_instrumental],
            lambda: run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(instrumental),
                    "-codec:a",
                    "libmp3lame",
                    "-b:a",
                    "192k",
                    str(public_instrumental),
                ]
            ),
            force=force,
        )

    instrumental_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="instrumental"
    )
    instrumental_future = instrumental_executor.submit(build_instrumental_branch)

    # Once vocals exist the pitch graph forks.  The full-model anchor consumes
    # broad vocals directly, while the other branch isolates lead vocals and
    # then extracts tiny-CREPE/pYIN evidence.  They join for note decoding.
    def isolate_lead_vocals() -> None:
        backing_vocals = lead_vocals_dir / "backing.wav"
        try:
            if profile == "roformer":
                cache.run(
                    "lead_separation",
                    {
                        "revision": PIPELINE_CACHE_REVISION,
                        "sourceSha256": cache.file_hash(vocals),
                        "model": {
                            name: model_hashes[name]
                            for name in (ROFORMER_LEAD_MODEL, ROFORMER_LEAD_CONFIG)
                        },
                        "arguments": {
                            "primary": "lead",
                            "secondary": "backing",
                            "sampleRate": 44_100,
                        },
                        "environment": separator_environment,
                        "code": code_hashes("work/separate_stems.py"),
                    },
                    [lead_vocals, backing_vocals],
                    lambda: separate_stem_pair(
                        vocals,
                        model=ROFORMER_LEAD_MODEL,
                        primary=lead_vocals,
                        secondary=backing_vocals,
                    ),
                    force=force,
                )
            else:
                cache.run(
                    "lead_separation",
                    {
                        "revision": PIPELINE_CACHE_REVISION,
                        "sourceSha256": cache.file_hash(vocals),
                        "model": {LEAD_MODEL: model_hashes[LEAD_MODEL]},
                        "arguments": {
                            "vrBatchSize": lead_vr_batch_size,
                            "vrWindowSize": 512,
                            "outputFormat": "WAV",
                            "soundfile": True,
                        },
                        "environment": separator_environment,
                        "code": code_hashes("work/separate_lead_vocals.py"),
                    },
                    [lead_vocals, backing_vocals],
                    lambda: run(
                        [
                            PYTHON,
                            "work/separate_lead_vocals.py",
                            "--song",
                            song,
                            "--source",
                            str(vocals),
                            "--validated-model-sha256",
                            model_hashes[LEAD_MODEL],
                            "--vr-batch-size",
                            str(lead_vr_batch_size),
                        ]
                    ),
                    force=force,
                )
            if not lead_vocals.exists():
                raise RuntimeError(
                    f"lead-vocal separator did not produce: {lead_vocals}"
                )
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            # Never reuse stale lead evidence when a slug is reprocessed.
            shutil.rmtree(lead_vocals_dir, ignore_errors=True)
            lead_contour.unlink(missing_ok=True)
            marker(
                "::warning lead-vocal isolation failed; using broad Kim vocals: "
                f"{error}"
            )

    def extract_anchor() -> None:
        # CREPE keeps its exact historical command line; other models name
        # themselves and take the anchor row of their confidence tuning.
        estimator_arguments = (
            ["--batch-size", str(anchor_batch_size)]
            if crepe
            else ["--estimator", tuning.name, "--estimator-role", "anchor"]
        )
        cache.run(
            "anchor_extraction",
            {
                "revision": PIPELINE_CACHE_REVISION,
                "sourceSha256": cache.file_hash(vocals),
                "pitch": pitch_identity,
                "arguments": {
                    "model": tuning.anchor_variant,
                    "fmax": 1500,
                    "batchSize": anchor_batch_size if crepe else None,
                    "pitchFilter": contour_config.pitch_filter,
                    "dataOnly": True,
                },
                "environment": pitch_environment,
                "code": code_hashes(
                    "contour.py",
                    "voicing.py",
                    "work/contour_filters.py",
                    "work/contour_pipeline_config.py",
                    "work/pitch_models.py",
                ),
            },
            [anchor_csv],
            lambda: run(
                [
                    PYTHON,
                    "contour.py",
                    str(vocals.relative_to(ROOT)),
                    "--no-separate",
                    "--name",
                    song,
                    "--fmax",
                    "1500",
                    *estimator_arguments,
                    "--pitch-filter",
                    contour_config.pitch_filter,
                    "--data-only",
                ]
            ),
            force=force,
        )
        if not anchor_csv.exists():
            raise RuntimeError(f"pitch anchor not written: {anchor_csv}")

    def extract_lead_pitch() -> tuple[Path, Path]:
        marker("::step:transcribing")
        pitch_vocals = lead_vocals if lead_vocals.exists() else vocals
        pitch_contour = (
            lead_contour
            if pitch_vocals == lead_vocals
            else ROOT
            / "experiments"
            / "transcription_final"
            / "raw_f0"
            / f"{song}_contour.csv"
        )
        cache.run(
            "lead_pitch_extraction",
            {
                "revision": PIPELINE_CACHE_REVISION,
                "sourceSha256": cache.file_hash(pitch_vocals),
                "pitch": pitch_identity,
                "arguments": {
                    "model": tuning.lead_variant,
                    "fmax": 1500,
                    "batchSize": lead_batch_size,
                    "torchThreads": 2,
                    "chunkSeconds": 20,
                    "decoder": "weighted_argmax",
                    "pitchFilter": contour_config.lead_pitch_filter,
                    "secondaryF0": "pyin",
                    "pyinChunkSeconds": 20,
                    "dataOnly": True,
                },
                "environment": pitch_environment,
                "code": code_hashes(
                    "contour.py",
                    "voicing.py",
                    "work/build_final_transcriptions.py",
                    "work/contour_filters.py",
                    "work/contour_pipeline_config.py",
                    "work/pitch_models.py",
                ),
            },
            [pitch_contour],
            lambda: run(
                [
                    PYTHON,
                    "work/build_final_transcriptions.py",
                    song,
                    "--model",
                    "tiny",
                    "--batch-size",
                    str(lead_batch_size),
                    "--pyin-workers",
                    str(pyin_workers),
                    "--pitch-model",
                    tuning.name,
                    "--extract-only",
                ]
            ),
            force=force,
        )
        return pitch_vocals, pitch_contour

    def extract_lead_branch() -> tuple[Path, Path]:
        isolate_lead_vocals()
        return extract_lead_pitch()

    if branch_workers == 2:
        # Keep public progress markers monotonic even though these two steps
        # overlap internally.
        marker("::step:isolating")
        marker("::step:tracking")
        marker("::parallel branches=anchor,lead")
        try:
            with ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="song-dag"
            ) as pool:
                anchor_future = pool.submit(extract_anchor)
                lead_future = pool.submit(extract_lead_branch)
                anchor_future.result()
                pitch_vocals, pitch_contour = lead_future.result()
        except Exception:
            # Unified-memory pressure can make two otherwise valid MPS jobs
            # fail only while they overlap.  Completed stages are cache hits;
            # retry the graph serially before surfacing a real stage failure.
            marker("::warning parallel branches failed; retrying sequentially")
            isolate_lead_vocals()
            extract_anchor()
            pitch_vocals, pitch_contour = extract_lead_pitch()
    else:
        marker("::step:isolating")
        isolate_lead_vocals()
        marker("::step:tracking")
        extract_anchor()
        pitch_vocals, pitch_contour = extract_lead_pitch()

    # The two branches join here: note decoding needs both pitch contours.
    cache.run(
        "note_transcription",
        {
            "revision": PIPELINE_CACHE_REVISION,
            "leadContourSha256": cache.file_hash(pitch_contour),
            "anchorSha256": cache.file_hash(anchor_csv),
            "vocalSha256": cache.file_hash(pitch_vocals),
            "arguments": {"beatTracking": False, "pianoPreview": False},
            "environment": installed_versions("numpy", "scipy", "librosa"),
            "code": code_hashes(
                "transcribe_notes.py",
                "work/analyze_pitch_gate.py",
                "work/contour_pipeline_config.py",
                "work/build_final_transcriptions.py",
            ),
        },
        [final_notes],
        lambda: run(
            [
                PYTHON,
                "work/build_final_transcriptions.py",
                song,
                "--model",
                "tiny",
                "--skip-extraction",
                "--skip-piano",
            ]
        ),
        force=force,
    )
    if not final_notes.exists():
        raise RuntimeError(f"transcription not written: {final_notes}")

    # 5. Publish the web-app assets into the private local library.
    marker("::step:finalizing")
    notes_evidence = ROOT / "contour_out" / f"{song}_notes_evidence.json"
    note_outputs = [notes_evidence]

    def publish_notes() -> None:
        atomic_copy(final_notes, notes_evidence)

    cache.run(
        "note_publish",
        {
            "revision": PIPELINE_CACHE_REVISION,
            "notesSha256": cache.file_hash(final_notes),
            "purpose": "internal-contour-correction-evidence",
        },
        note_outputs,
        publish_notes,
        force=force,
    )

    public_contour = public_data / f"{song}-contour.json"
    public_pyin = public_data / f"{song}-pyin.json"
    contour_outputs = [public_contour, public_pyin]
    reviewed_model_path = ROOT / contour_config.reviewed_model_path
    cache.run(
        "contour_export",
        {
            "revision": PIPELINE_CACHE_REVISION,
            "leadContourSha256": cache.file_hash(pitch_contour),
            "anchorSha256": cache.file_hash(anchor_csv),
            "notesSha256": cache.file_hash(notes_evidence),
            "pitchModel": tuning.name,
            "pipelinePreset": tuning.contour_preset,
            "pipelineConfig": contour_config.to_dict(),
            "reviewedModelSha256": (
                cache.file_hash(reviewed_model_path)
                if contour_config.reviewed_model_enabled
                and reviewed_model_path.exists()
                else None
            ),
            "code": code_hashes(
                "work/export_contour.py",
                "work/analyze_pitch_gate.py",
                "work/octave_correct_contour.py",
                "work/contour_pipeline_config.py",
                "work/reviewed_contour_classifier.py",
            ),
        },
        contour_outputs,
        lambda: run(
            [
                PYTHON,
                "work/export_contour.py",
                ".",
                str(public_data),
                song,
                "--preset",
                tuning.contour_preset,
                "--pitch-model",
                tuning.name,
            ]
        ),
        force=force,
    )
    if not public_contour.exists():
        raise RuntimeError(f"contour export not written for {song}")
    if not public_pyin.exists():
        raise RuntimeError(f"lead-pYIN repair source not written for {song}")

    public_source = public_audio / f"{song}.mp3"
    cache.run(
        "source_publish",
        {"revision": PIPELINE_CACHE_REVISION, "sourceSha256": source_hash},
        [public_source],
        lambda: atomic_copy(root_mp3, public_source),
        force=force,
    )

    # Join at the last possible point.  A failure caused only by accelerator
    # contention is retried now, after the MPS-heavy pitch branch has released
    # its models.  Integrity-checked cache hits make completed work free.
    try:
        instrumental_future.result()
    except Exception:
        marker("::warning background instrumental branch failed; retrying")
        build_instrumental_branch()
    finally:
        instrumental_executor.shutdown(wait=True, cancel_futures=False)

    duration = ffprobe_duration(public_source)
    if clean_intermediates:
        # Each song leaves four full-length WAV stems behind. Nothing reads
        # them after publication; the small CSV/JSON evidence is kept so a
        # contour can still be re-exported without separating again.
        released = remove_intermediate_audio([separator_dir, lead_vocals_dir])
        marker(f"::cleanup releasedBytes={released}")
    marker(
        f"::timing stage=total seconds={time.perf_counter() - total_started:.3f}"
    )
    return duration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("song", help="slug id for the new song")
    parser.add_argument(
        "--mp3",
        type=Path,
        help="path to the uploaded mp3 (default <root>/<id>.mp3)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun every stage even when the integrity-checked cache matches",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="local library root for audio/ and data/ (default: .sona-data/library)",
    )
    parser.add_argument(
        "--pitch-model",
        choices=sorted(PITCH_MODELS),
        default=DEFAULT_PITCH_MODEL,
        help="pitch model for this song; each has its own tuning in work/pitch_models.py",
    )
    parser.add_argument(
        "--clean-intermediates",
        action="store_true",
        help=(
            "delete the separated WAV stems once the song is published; "
            "a later rerun of this song id separates again"
        ),
    )
    parser.add_argument(
        "--branch-workers",
        type=int,
        choices=(1, 2),
        default=2,
        help="independent anchor/lead DAG branches to run concurrently (default 2)",
    )
    parser.add_argument(
        "--kim-batch-size",
        type=int,
        default=1,
        help="compact profile: Kim MDX batch size (benchmarked safe default 1)",
    )
    parser.add_argument(
        "--lead-vr-batch-size",
        type=int,
        default=1,
        help="compact profile: lead-separator VR batch size (benchmarked safe default 1)",
    )
    parser.add_argument(
        "--anchor-batch-size",
        type=int,
        default=512,
        help="full-CREPE anchor batch size (default 512)",
    )
    parser.add_argument(
        "--lead-batch-size",
        type=int,
        default=64,
        help="tiny-CREPE lead batch size (default 64)",
    )
    parser.add_argument(
        "--pyin-workers",
        type=int,
        default=0,
        help="pYIN worker processes; 0 chooses a memory-bounded automatic count",
    )
    args = parser.parse_args()
    for option in (
        "kim_batch_size",
        "lead_vr_batch_size",
        "anchor_batch_size",
        "lead_batch_size",
    ):
        if getattr(args, option) < 1:
            parser.error(f"--{option.replace('_', '-')} must be at least 1")
    if args.pyin_workers < 0:
        parser.error("--pyin-workers must be zero or greater")
    mp3 = args.mp3 or (ROOT / f"{args.song}.mp3")
    if not mp3.exists():
        marker(f"::error missing mp3: {mp3}")
        raise SystemExit(1)
    try:
        duration = process(
            args.song,
            mp3,
            force=args.force,
            pitch_model=args.pitch_model,
            clean_intermediates=args.clean_intermediates,
            branch_workers=args.branch_workers,
            kim_batch_size=args.kim_batch_size,
            lead_vr_batch_size=args.lead_vr_batch_size,
            anchor_batch_size=args.anchor_batch_size,
            lead_batch_size=args.lead_batch_size,
            pyin_workers=args.pyin_workers,
            output_dir=args.output_dir,
        )
    except Exception as error:  # noqa: BLE001 - surface failures to the server
        marker(f"::error {error}")
        raise SystemExit(1)
    marker(f"::done duration={duration}")


if __name__ == "__main__":
    main()
