"""Frame-level acoustic features for breath (inhale) detection.

The same extractor must serve MIR-1K training clips and production lead
stems, so it depends only on the audio: every feature is computable at
inference time on ``lead.wav`` without CREPE or pYIN. Frames use the
pipeline's 10 ms hop at 16 kHz so a predicted breath probability aligns
1:1 with contour CSV rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
HOP_SECONDS = 0.01
N_FFT = 512
_HOP = int(SAMPLE_RATE * HOP_SECONDS)
# Normalized-autocorrelation lag range covering 80-1000 Hz fundamentals.
_MIN_LAG = SAMPLE_RATE // 1000
_MAX_LAG = SAMPLE_RATE // 80
_CONTEXT_FRAMES = 11  # ~±50 ms rolling context


def load_audio_16k(path: Path, channel: int | None = None) -> np.ndarray:
    """Load audio resampled to the extractor rate.

    ``channel`` selects one channel of a multichannel file (MIR-1K keeps the
    voice on channel 1); ``None`` mixes to mono.
    """
    import librosa
    import soundfile as sf

    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if channel is not None and audio.shape[1] > channel:
        mono = audio[:, channel]
    else:
        mono = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=SAMPLE_RATE)
    return np.ascontiguousarray(mono, dtype=np.float32)


def _frame(audio: np.ndarray) -> np.ndarray:
    padded = np.pad(audio, (N_FFT // 2, N_FFT // 2))
    frame_count = 1 + len(audio) // _HOP
    strides = (padded.strides[0] * _HOP, padded.strides[0])
    return np.lib.stride_tricks.as_strided(
        padded, shape=(frame_count, N_FFT), strides=strides, writeable=False
    )


def _rolling(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    return windows.mean(axis=1), windows.std(axis=1)


FEATURE_NAMES = (
    "log_rms",
    "zcr",
    "log_flatness",
    "centroid",
    "rolloff85",
    "band_low",
    "band_mid",
    "band_high",
    "band_top",
    "flux",
    "periodicity",
    "log_rms_mean",
    "log_rms_std",
    "log_flatness_mean",
    "log_flatness_std",
    "periodicity_mean",
    "periodicity_std",
    "zcr_mean",
    "zcr_std",
)


def extract_breath_features(audio: np.ndarray) -> np.ndarray:
    """Return an (n_frames, len(FEATURE_NAMES)) float32 matrix at 10 ms hop."""
    frames = _frame(np.asarray(audio, dtype=np.float32))
    windowed = frames * np.hanning(N_FFT).astype(np.float32)

    spectrum = np.abs(np.fft.rfft(windowed, axis=1)) ** 2
    total = spectrum.sum(axis=1) + 1e-12
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SAMPLE_RATE)

    log_rms = np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-8)
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)

    log_spectrum = np.log(spectrum + 1e-12)
    flatness = np.exp(log_spectrum.mean(axis=1)) / (spectrum.mean(axis=1) + 1e-12)
    log_flatness = np.log10(flatness + 1e-8)

    centroid = (spectrum * freqs).sum(axis=1) / total / (SAMPLE_RATE / 2)
    cumulative = np.cumsum(spectrum, axis=1)
    rolloff_bins = np.argmax(cumulative >= 0.85 * cumulative[:, -1:], axis=1)
    rolloff85 = freqs[rolloff_bins] / (SAMPLE_RATE / 2)

    def band(lo: float, hi: float) -> np.ndarray:
        selection = (freqs >= lo) & (freqs < hi)
        return spectrum[:, selection].sum(axis=1) / total

    flux = np.r_[0.0, np.sqrt(np.mean(np.diff(np.sqrt(spectrum), axis=0) ** 2, axis=1))]
    flux = flux / (np.median(flux[flux > 0]) + 1e-8) if np.any(flux > 0) else flux

    centered = frames - frames.mean(axis=1, keepdims=True)
    transform = np.fft.rfft(centered, n=2 * N_FFT, axis=1)
    autocorr = np.fft.irfft(np.abs(transform) ** 2, axis=1)[:, :N_FFT]
    normalizer = autocorr[:, 0:1] + 1e-12
    periodicity = (autocorr[:, _MIN_LAG:_MAX_LAG] / normalizer).max(axis=1)

    columns = [
        log_rms,
        zcr,
        log_flatness,
        centroid,
        rolloff85,
        band(0, 400),
        band(400, 2000),
        band(2000, 5000),
        band(5000, 8000),
        flux,
        periodicity,
    ]
    for base in (log_rms, log_flatness, periodicity, zcr):
        mean, std = _rolling(base, _CONTEXT_FRAMES)
        columns.extend([mean, std])
    return np.column_stack(columns).astype(np.float32)
