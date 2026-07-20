#!/usr/bin/env python3
"""End-to-end pipeline for one uploaded song using the selected Kim models.

The commercial-separator bake-off selected a two-model route:

* ``Kim_Inst.onnx`` supplies the instrumental published to the web app.
* ``Kim_Vocal_2.onnx`` supplies the broad vocal pitch anchor.
* ``5_HP-Karaoke-UVR.pth`` splits that vocal stem into lead/backing evidence.

The resulting app assets are written straight into ``public/``:

    public/audio/<id>.mp3
    public/audio/<id>-instrumental.mp3
    public/data/<id>-contour.json
    public/data/<id>-pyin.json

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
    from .contour_pipeline_config import PRODUCTION_CONFIG
    from .hardware_acceleration import inference_hardware
    from .pipeline_cache import StageCache
except ImportError:  # pragma: no cover - exercised by the script entry point
    from contour_pipeline_config import PRODUCTION_CONFIG
    from hardware_acceleration import inference_hardware
    from pipeline_cache import StageCache


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
MODEL_DIR = ROOT / "contour_out" / "models" / "audio-separator"
MODEL_ASSET_MANIFEST = MODEL_DIR / "model-assets.json"
SEPARATOR_ROOT = ROOT / "contour_out" / "separators"
KIM_SEPARATOR = ROOT / "work" / "separate_kim_stem.py"
KIM_INST_MODEL = "Kim_Inst.onnx"
KIM_VOCAL_MODEL = "Kim_Vocal_2.onnx"
LEAD_MODEL = "5_HP-Karaoke-UVR.pth"
MODEL_HASHES = {
    asset["filename"]: asset["sha256"]
    for asset in json.loads(MODEL_ASSET_MANIFEST.read_text())["assets"]
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


def validate_models(hash_file=sha256) -> dict[str, str]:
    """Require the exact benchmarked weights before starting a costly run."""
    pipeline_assets = {**MODEL_HASHES, **MODEL_METADATA_HASHES}
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
    return dict(MODEL_HASHES)


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


def process(
    song: str,
    mp3: Path,
    *,
    force: bool = False,
    branch_workers: int = 2,
    kim_batch_size: int = 1,
    lead_vr_batch_size: int = 1,
    anchor_batch_size: int = 512,
    lead_batch_size: int = 64,
    pyin_workers: int = 0,
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
    public_audio = ROOT / "public" / "audio"
    public_data = ROOT / "public" / "data"
    public_audio.mkdir(parents=True, exist_ok=True)
    public_data.mkdir(parents=True, exist_ok=True)

    cache = StageCache(ROOT, song, marker)
    model_hashes = validate_models(cache.file_hash)

    def code_hashes(*relative_paths: str) -> dict[str, str]:
        return cache.code_hashes(
            [Path(__file__), *(ROOT / path for path in relative_paths)]
        )

    pitch_environment = installed_versions(
        "numpy", "scipy", "librosa", "soundfile", "torch", "torchcrepe"
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

    # The mix must live at <root>/<song>.mp3 so every downstream stem/id lines up.
    root_mp3 = ROOT / f"{song}.mp3"
    if mp3.resolve() != root_mp3.resolve():
        uploaded_hash = cache.file_hash(mp3)
        if not root_mp3.exists() or cache.file_hash(root_mp3) != uploaded_hash:
            atomic_copy(mp3, root_mp3)
    source_hash = cache.file_hash(root_mp3)

    # 1. Produce the dependency-critical vocal stem first.  The instrumental
    # has no consumers until publication, so it continues in the background
    # while the vocal-dependent pitch DAG runs below.
    marker("::step:separating")
    instrumental = separator_dir / "instrumental.wav"
    vocals = separator_dir / "vocals.wav"
    public_instrumental = public_audio / f"{song}-instrumental.mp3"
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
        cache.run(
            "anchor_extraction",
            {
                "revision": PIPELINE_CACHE_REVISION,
                "sourceSha256": cache.file_hash(vocals),
                "arguments": {
                    "model": "full",
                    "fmax": 1500,
                    "batchSize": anchor_batch_size,
                    "pitchFilter": PRODUCTION_CONFIG.pitch_filter,
                    "dataOnly": True,
                },
                "environment": pitch_environment,
                "code": code_hashes(
                    "contour.py",
                    "voicing.py",
                    "work/contour_filters.py",
                    "work/contour_pipeline_config.py",
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
                    "--batch-size",
                    str(anchor_batch_size),
                    "--pitch-filter",
                    PRODUCTION_CONFIG.pitch_filter,
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
                "arguments": {
                    "model": "tiny",
                    "fmax": 1500,
                    "batchSize": lead_batch_size,
                    "torchThreads": 2,
                    "chunkSeconds": 20,
                    "decoder": "weighted_argmax",
                    "pitchFilter": PRODUCTION_CONFIG.lead_pitch_filter,
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

    # 5. Publish the web-app assets into public/.
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
    reviewed_model_path = ROOT / PRODUCTION_CONFIG.reviewed_model_path
    cache.run(
        "contour_export",
        {
            "revision": PIPELINE_CACHE_REVISION,
            "leadContourSha256": cache.file_hash(pitch_contour),
            "anchorSha256": cache.file_hash(anchor_csv),
            "notesSha256": cache.file_hash(notes_evidence),
            "pipelineConfig": PRODUCTION_CONFIG.to_dict(),
            "reviewedModelSha256": (
                cache.file_hash(reviewed_model_path)
                if PRODUCTION_CONFIG.reviewed_model_enabled
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
        lambda: run([PYTHON, "work/export_contour.py", ".", "public/data", song]),
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
        help="Kim MDX batch size (benchmarked safe default 1)",
    )
    parser.add_argument(
        "--lead-vr-batch-size",
        type=int,
        default=1,
        help="lead-separator VR batch size (benchmarked safe default 1)",
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
            branch_workers=args.branch_workers,
            kim_batch_size=args.kim_batch_size,
            lead_vr_batch_size=args.lead_vr_batch_size,
            anchor_batch_size=args.anchor_batch_size,
            lead_batch_size=args.lead_batch_size,
            pyin_workers=args.pyin_workers,
        )
    except Exception as error:  # noqa: BLE001 - surface failures to the server
        marker(f"::error {error}")
        raise SystemExit(1)
    marker(f"::done duration={duration}")


if __name__ == "__main__":
    main()
