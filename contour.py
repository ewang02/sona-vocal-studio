#!/usr/bin/env python3
"""
Karaoke pitch-contour feasibility test.

Pipeline: song.mp3 -> Demucs vocal separation -> torchcrepe f0 + confidence
          -> optional independent pYIN estimate -> confidence-gated contour
          -> outputs:

  1. <song>_contour.png   : contour plot colored by confidence (the "look at it" test)
  2. <song>_contour.csv   : time, f0_hz, midi, confidence, RMS, voiced flag
  3. <song>_sine.wav      : sine-wave resynthesis of the contour (the "listen to it" test)
  4. <song>_mix.wav       : separated vocals + sine overlaid, for A/B by ear
  5. console stats        : % voiced, confidence distribution, gap analysis

Usage:
  python contour_feasibility.py song.mp3
  python contour_feasibility.py song.mp3 --conf 0.6 --hop-ms 10
  python contour_feasibility.py vocals.wav --no-separate   # if you already have a vocal stem

Install (CPU is fine, GPU faster):
  pip install torch torchaudio demucs torchcrepe librosa soundfile matplotlib scipy
"""

import argparse
import multiprocessing
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import soundfile as sf
from voicing import coherent_voicing_mask
from work.contour_filters import filter_f0
from work.contour_pipeline_config import PRESETS
from work.hardware_acceleration import torch_device_name

CREPE_SR = 16000  # torchcrepe expects 16 kHz


def deterministic_weighted_argmax(logits):
    """torchcrepe weighted argmax without its random ±20-cent dither."""
    import torch

    bins = logits.argmax(dim=1)
    indexes = torch.arange(logits.shape[1], device=logits.device)[None, :, None]
    mask = (indexes >= bins[:, None, :] - 4) & (indexes < bins[:, None, :] + 5)
    probabilities = torch.sigmoid(logits).masked_fill(~mask, 0.0)
    cents_by_bin = (
        20.0 * torch.arange(logits.shape[1], device=logits.device)
        + 1997.3794084376191
    )[None, :, None]
    cents = (cents_by_bin * probabilities).sum(dim=1) / probabilities.sum(dim=1)
    frequency = 10.0 * 2.0 ** (cents / 1200.0)
    return bins, frequency


# ----------------------------------------------------------------------------
# Step 1: vocal separation (Demucs htdemucs, two-stem mode)
# ----------------------------------------------------------------------------
def separate_vocals(input_path: Path, work_dir: Path) -> Path:
    print(f"[1/4] Separating vocals with Demucs (this is the slow step)...")
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", "htdemucs",
        "-o", str(work_dir),
        str(input_path),
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit("Demucs failed. Is it installed? (pip install demucs)")

    vocals = work_dir / "htdemucs" / input_path.stem / "vocals.wav"
    if not vocals.exists():
        sys.exit(f"Expected separated vocals at {vocals} but not found.")
    return vocals


# ----------------------------------------------------------------------------
# Step 2: f0 + confidence via torchcrepe
# ----------------------------------------------------------------------------
def extract_f0(
    vocal_path: Path,
    hop_ms: float,
    model: str,
    fmax: float = 1000.0,
    batch_size: int = 128,
    torch_threads: int = 4,
    chunk_seconds: float = 30.0,
    decoder_name: str = "weighted_argmax",
    pitch_filter: str = "mean_hz_90",
):
    import gc
    import librosa
    import torch
    import torchcrepe

    print(f"[2/4] Running CREPE ({model} model, {hop_ms} ms hop, fmax {fmax:.0f} Hz)...")
    audio, _ = librosa.load(str(vocal_path), sr=CREPE_SR, mono=True)
    hop_length = int(CREPE_SR * hop_ms / 1000)
    # Select the fastest Torch backend supported by this installation. Small
    # floating-point drift between devices is below transcription tolerance.
    device = torch_device_name(torch)
    print(f"[hardware] CREPE device={device}")
    if device == "cpu":
        torch.set_num_threads(max(1, torch_threads))
    decoder = (
        deterministic_weighted_argmax
        if decoder_name == "weighted_argmax"
        else torchcrepe.decode.viterbi
    )

    # torchcrepe normally accepts the whole recording, but its frame tensor and
    # model activations can become very large. Decode overlapping, hop-aligned
    # chunks and keep only each chunk's core. Peak memory is then independent
    # of song duration, and one process never holds several songs at once.
    target_length = len(audio) // hop_length + 1
    requested_core_frames = max(
        1, int(round(chunk_seconds * CREPE_SR / hop_length))
    )
    requested_overlap_frames = max(
        4, int(round(0.12 * CREPE_SR / hop_length))
    )
    # torchcrepe runs Viterbi inside inference batches. Keeping every core and
    # overlap boundary on the same batch phase makes a frame independent of
    # which extraction chunk contained it.
    core_frames = (
        (requested_core_frames + batch_size - 1) // batch_size
    ) * batch_size
    overlap_frames = (
        (requested_overlap_frames + batch_size - 1) // batch_size
    ) * batch_size
    def predict_chunks(active_device: str):
        f0_values = np.full(target_length, np.nan, dtype=np.float32)
        periodicity_values = np.zeros(target_length, dtype=np.float32)
        for core_start in range(0, target_length, core_frames):
            core_end = min(target_length, core_start + core_frames)
            input_start_frame = max(0, core_start - overlap_frames)
            input_end_frame = min(target_length, core_end + overlap_frames)
            sample_start = input_start_frame * hop_length
            sample_end = min(len(audio), input_end_frame * hop_length)
            audio_t = torch.from_numpy(audio[sample_start:sample_end]).unsqueeze(0)
            with torch.inference_mode():
                f0_chunk, periodicity_chunk = torchcrepe.predict(
                    audio_t,
                    CREPE_SR,
                    hop_length=hop_length,
                    fmin=65.0,     # ~C2, below any sung melody
                    # Leave headroom above the song's top note so confidence
                    # does not collapse near the model search ceiling.
                    fmax=fmax,
                    model=model,
                    batch_size=batch_size,
                    device=active_device,
                    decoder=decoder,
                    return_periodicity=True,
                )
            local_start = core_start - input_start_frame
            local_end = local_start + (core_end - core_start)
            predicted = min(local_end, f0_chunk.shape[1]) - local_start
            if predicted > 0:
                f0_values[core_start : core_start + predicted] = (
                    f0_chunk[0, local_start : local_start + predicted].cpu().numpy()
                )
                periodicity_values[core_start : core_start + predicted] = (
                    periodicity_chunk[0, local_start : local_start + predicted]
                    .cpu()
                    .numpy()
                )
            del audio_t, f0_chunk, periodicity_chunk
            gc.collect()
        return f0_values, periodicity_values

    try:
        f0_raw_np, periodicity_raw_np = predict_chunks(device)
    except Exception as error:
        if device == "cpu":
            raise
        print(f"[hardware] CREPE {device} failed; retrying on CPU: {error}")
        if hasattr(torchcrepe.infer, "model"):
            delattr(torchcrepe.infer, "model")
        accelerator = getattr(torch, device, None)
        empty_cache = getattr(accelerator, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
        gc.collect()
        device = "cpu"
        torch.set_num_threads(max(1, torch_threads))
        f0_raw_np, periodicity_raw_np = predict_chunks(device)

    # Preserve the frame-level model output before the standard display
    # cleanup. A 90 ms mean is useful for looking at a contour but can erase
    # much of a 100–150 ms note; note transcription needs access to both.
    f0_raw = torch.from_numpy(f0_raw_np).unsqueeze(0)
    periodicity_raw = torch.from_numpy(periodicity_raw_np).unsqueeze(0)

    # Confidence remains median-filtered. Pitch filtering is configurable and
    # robust candidates operate in MIDI/log-frequency so octave outliers are
    # symmetric and note edges are not smeared in linear Hz.
    win = max(3, int(round(0.09 / (hop_ms / 1000))))  # ~90 ms window
    if win % 2 == 0:
        win += 1
    periodicity = torchcrepe.filter.median(periodicity_raw, win)
    f0 = filter_f0(
        f0_raw_np,
        periodicity_raw_np,
        hop_ms,
        pitch_filter,
    )

    conf = periodicity.squeeze(0).cpu().numpy()
    f0_raw = f0_raw.squeeze(0).cpu().numpy()
    conf_raw = periodicity_raw.squeeze(0).cpu().numpy()
    times = np.arange(len(f0)) * hop_ms / 1000.0

    # RMS energy gate so silence between phrases can't count as voiced
    frame = hop_length * 2
    rms = np.array([
        np.sqrt(np.mean(audio[i * hop_length: i * hop_length + frame] ** 2))
        if i * hop_length < len(audio) else 0.0
        for i in range(len(f0))
    ])
    rms_floor = np.percentile(rms[rms > 0], 20) * 0.5 if np.any(rms > 0) else 0.0

    # torchcrepe caches the network globally. Release it before optional pYIN
    # so two pitch models are never resident together in this process.
    if hasattr(torchcrepe.infer, "model"):
        delattr(torchcrepe.infer, "model")
    gc.collect()

    return times, f0, conf, rms, rms_floor, audio, f0_raw, conf_raw


def hz_to_midi(f0):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 69.0 + 12.0 * np.log2(f0 / 440.0)


def _pyin_chunk(job):
    """Run one independent pYIN window in a worker process."""
    (
        core_start,
        core_end,
        input_start_frame,
        audio,
        sample_rate,
        hop_length,
        fmax,
    ) = job
    import librosa

    try:
        from threadpoolctl import threadpool_limits

        thread_limit = threadpool_limits(limits=1)
    except ImportError:  # pragma: no cover - optional scipy dependency helper
        thread_limit = nullcontext()
    with thread_limit:
        f0, _, voiced_probability = librosa.pyin(
            audio,
            fmin=65.0,
            fmax=fmax,
            sr=sample_rate,
            frame_length=2048,
            hop_length=hop_length,
            center=True,
            fill_na=np.nan,
        )
    return core_start, core_end, input_start_frame, f0, voiced_probability


def automatic_pyin_workers(job_count: int) -> int:
    """Use useful CPU parallelism without multiplying memory without bound."""
    if job_count <= 1:
        return 1
    cpu_count = os.cpu_count() or 1
    return min(job_count, 4, max(1, cpu_count // 2))


def extract_pyin(
    audio,
    sample_rate,
    hop_length,
    fmax,
    target_length,
    chunk_seconds=20.0,
    active_mask=None,
    workers=1,
):
    """Independent F0 evidence used as confidence, never as a hard override."""
    import gc

    if workers < 0:
        raise ValueError("pYIN workers must be zero (auto) or a positive integer")
    aligned_f0 = np.full(target_length, np.nan, dtype=float)
    aligned_probability = np.zeros(target_length, dtype=float)
    core_frames = max(1, int(round(chunk_seconds * sample_rate / hop_length)))
    overlap_frames = max(4, int(round(0.25 * sample_rate / hop_length)))
    jobs = []
    for core_start in range(0, target_length, core_frames):
        core_end = min(target_length, core_start + core_frames)
        if active_mask is not None and not np.any(active_mask[core_start:core_end]):
            continue
        input_start_frame = max(0, core_start - overlap_frames)
        input_end_frame = min(target_length, core_end + overlap_frames)
        sample_start = input_start_frame * hop_length
        sample_end = min(len(audio), input_end_frame * hop_length)
        jobs.append(
            (
                core_start,
                core_end,
                input_start_frame,
                np.ascontiguousarray(audio[sample_start:sample_end]),
                sample_rate,
                hop_length,
                fmax,
            )
        )

    worker_count = automatic_pyin_workers(len(jobs)) if workers == 0 else workers
    worker_count = min(max(1, worker_count), max(1, len(jobs)))
    print(
        "[2b/4] Running independent pYIN confidence check "
        f"in chunks (workers={worker_count})..."
    )
    executor = None
    if worker_count > 1:
        # Spawn is safe even when PyTorch initialized a GPU/MPS runtime earlier
        # in this process.  Each job receives only its bounded overlapping
        # audio window rather than a full-song copy.
        try:
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("spawn"),
            )
        except (NotImplementedError, OSError, PermissionError) as error:
            print(
                "[2b/4] Multiprocessing unavailable; falling back to serial "
                f"pYIN ({error})"
            )
            worker_count = 1
    results = (
        executor.map(_pyin_chunk, jobs)
        if executor is not None
        else map(_pyin_chunk, jobs)
    )
    try:
        for (
            core_start,
            core_end,
            input_start_frame,
            f0,
            voiced_probability,
        ) in results:
            local_start = core_start - input_start_frame
            wanted = core_end - core_start
            count = min(wanted, max(0, len(f0) - local_start))
            if count > 0:
                aligned_f0[core_start : core_start + count] = f0[
                    local_start : local_start + count
                ]
                aligned_probability[core_start : core_start + count] = np.nan_to_num(
                    voiced_probability[local_start : local_start + count], nan=0.0
                )
            del f0, voiced_probability
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    gc.collect()
    return aligned_f0, aligned_probability


# ----------------------------------------------------------------------------
# Step 3: analysis + plot
# ----------------------------------------------------------------------------
def analyze_and_plot(times, f0, conf, voiced, out_png: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("[3/4] Plotting + stats...")
    midi = hz_to_midi(f0)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(16, 10),
        gridspec_kw={"height_ratios": [3, 3, 1]},
    )

    # Full-song overview
    m = midi.copy()
    m[~voiced] = np.nan
    sc = ax1.scatter(times, m, c=conf, cmap="viridis", s=2, vmin=0, vmax=1)
    ax1.set_title(f"{title} — full contour (color = confidence)")
    ax1.set_ylabel("MIDI pitch")
    fig.colorbar(sc, ax=ax1, label="confidence")

    # Zoom on the densest voiced 20 s (likely a verse/chorus) — this is the
    # panel to squint at: is the melody visually obvious as a smooth ribbon?
    if voiced.any():
        w = 20.0
        dens_best, t0 = -1, times[0]
        for start in np.arange(times[0], max(times[-1] - w, times[0]) + 1e-9, 5.0):
            d = np.sum(voiced & (times >= start) & (times < start + w))
            if d > dens_best:
                dens_best, t0 = d, start
        sel = (times >= t0) & (times < t0 + w)
        ax2.scatter(times[sel], m[sel], c=conf[sel], cmap="viridis", s=6, vmin=0, vmax=1)
        ax2.set_xlim(t0, t0 + w)
        ax2.set_title(f"Zoom: {t0:.0f}–{t0 + w:.0f} s — is the melody a clean ribbon here?")
        ax2.set_ylabel("MIDI pitch")
        ax2.grid(True, axis="y", alpha=0.3)
        # semitone gridlines in the zoom window
        lo, hi = np.nanpercentile(m[sel], [1, 99]) if np.any(sel & voiced) else (48, 72)
        ax2.set_yticks(np.arange(np.floor(lo), np.ceil(hi) + 1))

    ax3.hist(conf[voiced], bins=50, range=(0, 1), color="tab:blue", alpha=0.8)
    ax3.set_title("Confidence distribution (voiced frames)")
    ax3.set_xlabel("confidence")

    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close(fig)

    # ---- console verdict data ----
    total_v = voiced.sum()
    pct_voiced = 100.0 * total_v / len(voiced)
    hop = times[1] - times[0] if len(times) > 1 else 0.01

    # gaps: runs of unvoiced frames inside the sung region
    gaps = []
    if total_v:
        idx = np.where(voiced)[0]
        run = 0
        for i in range(idx[0], idx[-1] + 1):
            if not voiced[i]:
                run += 1
            elif run:
                gaps.append(run * hop)
                run = 0
    gaps = np.array(gaps) if gaps else np.array([0.0])

    print("\n================ FEASIBILITY STATS ================")
    print(f"Voiced frames:            {pct_voiced:5.1f} % of song")
    print(f"Mean confidence (voiced): {conf[voiced].mean():.3f}" if total_v else "no voiced frames!")
    print(f"Frames with conf > 0.8:   {100.0 * np.mean(conf[voiced] > 0.8):5.1f} % of voiced" if total_v else "")
    print(f"Dropout gaps > 0.3 s:     {int(np.sum(gaps > 0.3))}  (median gap {np.median(gaps):.2f} s)")
    print("===================================================")
    print("Rough rubric: >70% of voiced frames above 0.8 confidence and a visually")
    print("clean zoom panel = green light. Lots of speckle/octave jumps = trouble.\n")


# ----------------------------------------------------------------------------
# Step 4: sine resynthesis (the listening test)
# ----------------------------------------------------------------------------
def resynthesize(times, f0, voiced, vocal_path: Path, out_sine: Path, out_mix: Path):
    print("[4/4] Rendering sine resynthesis...")
    import librosa

    sr = 22050
    dur = times[-1] + (times[1] - times[0] if len(times) > 1 else 0.01)
    n = int(dur * sr)
    t_samples = np.arange(n) / sr

    f0_i = np.interp(t_samples, times, np.where(voiced, f0, 0.0))
    amp_i = np.interp(t_samples, times, voiced.astype(float))
    # short fades to avoid clicks at note-region boundaries
    from scipy.ndimage import uniform_filter1d
    amp_i = uniform_filter1d(amp_i, size=int(0.01 * sr))

    phase = 2 * np.pi * np.cumsum(f0_i) / sr
    sine = 0.35 * amp_i * np.sin(phase)
    sf.write(out_sine, sine.astype(np.float32), sr)

    vocals, _ = librosa.load(str(vocal_path), sr=sr, mono=True)
    L = min(len(vocals), len(sine))
    mix = 0.6 * vocals[:L] + 0.6 * sine[:L]
    peak = np.max(np.abs(mix)) or 1.0
    sf.write(out_mix, (mix / peak * 0.9).astype(np.float32), sr)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Pitch-contour karaoke feasibility test")
    ap.add_argument("input", type=Path, help="song mp3 (or vocal stem with --no-separate)")
    ap.add_argument("--no-separate", action="store_true", help="input is already an isolated vocal")
    ap.add_argument("--conf", type=float, default=0.55, help="reliable confidence threshold for voicing (default 0.55)")
    ap.add_argument("--recover-conf", type=float, default=0.30,
                    help="lower confidence floor for locally coherent recovery frames (default 0.30)")
    ap.add_argument("--no-coherent-recover", action="store_true",
                    help="disable low-confidence recovery and use the reliable confidence gate only")
    ap.add_argument("--adaptive-conf", action="store_true",
                    help="derive the reliable confidence threshold from this song's active-frame distribution")
    ap.add_argument("--coherence-st", type=float, default=1.0,
                    help="max semitone deviation for low-confidence recovery (default 1.0)")
    ap.add_argument("--hop-ms", type=float, default=10.0, help="analysis hop in ms (default 10)")
    ap.add_argument("--fmax", type=float, default=1000.0,
                    help="CREPE search ceiling in Hz; leave headroom above the song's top note (default 1000)")
    ap.add_argument("--model", choices=["full", "tiny"], default="full", help="CREPE model size")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="CREPE inference batch size; smaller uses less memory (default 128)",
    )
    ap.add_argument(
        "--torch-threads",
        type=int,
        default=4,
        help="maximum CPU threads used by PyTorch (default 4)",
    )
    ap.add_argument(
        "--chunk-seconds",
        type=float,
        default=30.0,
        help="memory-bounded CREPE chunk duration (default 30)",
    )
    ap.add_argument(
        "--decoder",
        choices=["weighted_argmax", "viterbi"],
        default="weighted_argmax",
        help=(
            "CREPE pitch decoder; weighted_argmax is frame-local and exactly "
            "stable across memory-bounded chunks (default)"
        ),
    )
    ap.add_argument(
        "--pitch-filter",
        choices=sorted({preset.pitch_filter for preset in PRESETS.values()}),
        default="mean_hz_90",
        help="frame-level CREPE pitch cleanup (default preserves legacy 90 ms mean)",
    )
    ap.add_argument(
        "--pyin-chunk-seconds",
        type=float,
        default=20.0,
        help="memory-bounded pYIN chunk duration (default 20)",
    )
    ap.add_argument(
        "--pyin-workers",
        type=int,
        default=1,
        help=(
            "pYIN worker processes; 0 chooses a memory-bounded automatic count "
            "(default 1)"
        ),
    )
    ap.add_argument(
        "--secondary-f0",
        choices=["none", "pyin"],
        default="none",
        help="optional independent F0 evidence written alongside CREPE",
    )
    ap.add_argument(
        "--data-only",
        action="store_true",
        help="write the contour CSV without plot or resynthesis artifacts",
    )
    ap.add_argument("--outdir", type=Path, default=Path("contour_out"), help="output directory")
    ap.add_argument("--name", help="output asset name (defaults to the input filename stem)")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"File not found: {args.input}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = args.name or args.input.stem

    if args.no_separate:
        vocal_path = args.input
    else:
        vocal_path = separate_vocals(args.input, args.outdir / "demucs")

    times, f0, conf, rms, rms_floor, audio, f0_raw, conf_raw = extract_f0(
        vocal_path,
        args.hop_ms,
        args.model,
        args.fmax,
        args.batch_size,
        args.torch_threads,
        args.chunk_seconds,
        args.decoder,
        args.pitch_filter,
    )
    midi = hz_to_midi(f0)
    midi_raw = hz_to_midi(f0_raw)
    voiced, gate = coherent_voicing_mask(
        times, midi, conf, rms, rms_floor,
        conf_hi=args.conf,
        conf_lo=args.recover_conf,
        recover=not args.no_coherent_recover,
        adaptive=args.adaptive_conf,
        coherence_st=args.coherence_st,
    )
    print(
        f"[gate] reliable conf >= {gate['conf_hi']:.3f}, "
        f"recovery conf >= {gate['conf_lo']:.3f}, RMS floor {rms_floor:.6f}"
    )
    print(
        f"[gate] primary={int(gate['primary'].sum())} "
        f"recovered={int(gate['recovered'].sum())} "
        f"rejected_low_conf_pitchless={int(gate['pitchless_low_conf'].sum())}"
    )

    # CSV
    csv_path = args.outdir / f"{stem}_contour.csv"
    columns = [
        times,
        f0,
        midi,
        conf,
        rms,
        voiced.astype(int),
        f0_raw,
        midi_raw,
        conf_raw,
    ]
    names = [
        "time_s",
        "f0_hz",
        "midi",
        "confidence",
        "rms",
        "voiced",
        "f0_hz_raw",
        "midi_raw",
        "confidence_raw",
    ]
    if args.secondary_f0 == "pyin":
        hop_length = int(CREPE_SR * args.hop_ms / 1000)
        f0_pyin, confidence_pyin = extract_pyin(
            audio,
            CREPE_SR,
            hop_length,
            args.fmax,
            len(times),
            args.pyin_chunk_seconds,
            voiced,
            args.pyin_workers,
        )
        columns.extend([f0_pyin, hz_to_midi(f0_pyin), confidence_pyin])
        names.extend(["f0_hz_pyin", "midi_pyin", "confidence_pyin"])
    header = ",".join(names)
    temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    np.savetxt(
        temporary_csv,
        np.column_stack(columns),
        delimiter=",", header=header, comments="", fmt="%.5f",
    )
    temporary_csv.replace(csv_path)

    if not args.data_only:
        analyze_and_plot(
            times, f0, conf, voiced, args.outdir / f"{stem}_contour.png", stem
        )
        resynthesize(
            times,
            f0,
            voiced,
            vocal_path,
            args.outdir / f"{stem}_sine.wav",
            args.outdir / f"{stem}_mix.wav",
        )

    print(f"Outputs in {args.outdir}/ :")
    if not args.data_only:
        print(f"  {stem}_contour.png  <- look: is the melody a clean ribbon?")
        print(f"  {stem}_sine.wav     <- listen: is this recognizably the melody?")
        print(f"  {stem}_mix.wav      <- A/B: sine vs. actual vocal")
    print(f"  {stem}_contour.csv  <- raw data for your own analysis")


if __name__ == "__main__":
    main()
