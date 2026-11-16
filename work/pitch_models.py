"""Selectable pitch models and each one's own tuning.

Every song is analysed with one of these models, chosen when the song is
added (or re-analysed). All of them answer the same question -- the sung
pitch and how sure the model is, every 10 ms -- but they disagree about what
"sure" means: CREPE's periodicity, RMVPE's salience and SwiftF0's voicing
score are three different scales. The rest of the pipeline compares
confidence against fixed levels (is this frame reliable enough to seed a
phrase? to outvote the other estimator?), so each model declares which of its
own values corresponds to each of those levels.

To tune a model, edit its entry in ``PITCH_MODELS``:

* ``anchor_confidence`` / ``lead_confidence`` -- the model's own thresholds.
  Entry *i* is the native confidence that should count as
  ``CONFIDENCE_LEVELS[i]``. Raise a value to make the pipeline stricter with
  this model at that level; lower it to make it more permissive. The anchor
  row applies to the broad vocal stem, the lead row to the isolated lead.
* ``voicing_reliable`` / ``voicing_recover`` -- the two levels of the
  extraction voicing gate, on the common scale.
* ``contour_preset`` -- the named ``ContourPipelineConfig`` (in
  ``contour_pipeline_config.py``) that governs pitch smoothing, gating,
  octave repair and edge cleanup for contours from this model. Each model has
  its own preset so changing one never moves another.

``work/calibrate_pitch_model.py`` measures the confidence rows from audio.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# The confidence levels the pipeline tests against, on CREPE's scale (the
# model the downstream thresholds were tuned with).
CONFIDENCE_LEVELS = (0.10, 0.25, 0.30, 0.45, 0.50, 0.55, 0.60, 0.65, 0.75, 0.80, 0.90)

DEFAULT_PITCH_MODEL = "crepe"


@dataclass(frozen=True)
class PitchModelTuning:
    name: str
    label: str
    # torchcrepe network size per role; None for single-size models.
    anchor_variant: str | None
    lead_variant: str | None
    # Native confidence equal to each CONFIDENCE_LEVELS entry, per role.
    # None means the model already reports on the common scale.
    anchor_confidence: tuple[float, ...] | None
    lead_confidence: tuple[float, ...] | None
    voicing_reliable: float
    voicing_recover: float
    contour_preset: str
    # Weight file required beside the separator models, if any.
    asset: str | None = None
    # The model's own published voiced/unvoiced threshold, if it should be
    # honoured: it pins the 0.30 "recoverable" level (and scales the levels
    # below it) instead of letting calibration drag them down to match a
    # more permissive model.
    native_voicing_floor: float | None = None

    def confidence_knots(self, role: str) -> tuple[float, ...] | None:
        if role == "anchor":
            return self.anchor_confidence
        if role == "lead":
            return self.lead_confidence
        raise ValueError(f"Unknown pitch role {role!r}; choose anchor or lead")

    def fingerprint(self) -> str:
        """Stable digest of every tuning value, for stage-cache keys."""
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


PITCH_MODELS: dict[str, PitchModelTuning] = {
    "crepe": PitchModelTuning(
        name="crepe",
        label="CREPE",
        anchor_variant="full",
        lead_variant="tiny",
        anchor_confidence=None,
        lead_confidence=None,
        voicing_reliable=0.55,
        voicing_recover=0.30,
        contour_preset="production-v6r",
    ),
    # The confidence rows below were measured with calibrate_pitch_model.py
    # on one song under both separator profiles, by matching at each level the
    # share of sung frames CREPE accepts. The lead rows moved little between
    # the two; the anchor rows moved more (RMVPE's 0.50 level read 0.11 on one
    # stem set and 0.23 on the other). Treat them as a starting point and
    # re-measure as the library grows.
    #
    # levels:            0.10   0.25   0.30   0.45   0.50   0.55   0.60   0.65   0.75   0.80   0.90
    "rmvpe": PitchModelTuning(
        name="rmvpe",
        label="RMVPE",
        anchor_variant=None,
        lead_variant=None,
        anchor_confidence=(0.010, 0.025, 0.030, 0.067, 0.159, 0.291, 0.422, 0.533, 0.740, 0.814, 0.891),
        lead_confidence=(0.010, 0.025, 0.030, 0.144, 0.334, 0.512, 0.671, 0.784, 0.864, 0.889, 0.962),
        voicing_reliable=0.55,
        voicing_recover=0.30,
        contour_preset="rmvpe-v1",
        asset="rmvpe.onnx",
        # RMVPE's salience is close to all-or-nothing, and its authors treat
        # 0.03 as voiced. Matching CREPE's acceptance rate at the low levels
        # would put "recoverable" near 0.003, trusting pitch the model itself
        # considers absent.
        native_voicing_floor=0.03,
    ),
    "swiftf0": PitchModelTuning(
        name="swiftf0",
        label="SwiftF0",
        anchor_variant=None,
        lead_variant=None,
        anchor_confidence=(0.018, 0.079, 0.121, 0.371, 0.471, 0.573, 0.667, 0.740, 0.850, 0.895, 0.971),
        lead_confidence=(0.008, 0.087, 0.147, 0.458, 0.589, 0.713, 0.823, 0.901, 0.965, 0.979, 0.996),
        voicing_reliable=0.55,
        voicing_recover=0.30,
        contour_preset="swiftf0-v1",
    ),
}


def get_pitch_model(name: str | None) -> PitchModelTuning:
    selected = (name or DEFAULT_PITCH_MODEL).strip().lower()
    try:
        return PITCH_MODELS[selected]
    except KeyError as error:
        raise ValueError(
            f"Unknown pitch model {name!r}; choose from {', '.join(PITCH_MODELS)}"
        ) from error


def calibrate_confidence(
    native: np.ndarray, knots: tuple[float, ...] | None
) -> np.ndarray:
    """Map a model's native confidence onto the pipeline's common scale.

    Piecewise linear through (0, 0), each (knot, level) pair and (1, 1), so it
    is monotonic: ranking frames by confidence is unchanged, only where the
    pipeline's fixed levels fall.
    """
    values = np.clip(np.asarray(native, dtype=float), 0.0, 1.0)
    if knots is None:
        return values
    if len(knots) != len(CONFIDENCE_LEVELS):
        raise ValueError(
            f"expected {len(CONFIDENCE_LEVELS)} confidence knots, got {len(knots)}"
        )
    native_points = np.array([0.0, *knots, 1.0], dtype=float)
    if np.any(np.diff(native_points) <= 0):
        raise ValueError("confidence knots must increase strictly between 0 and 1")
    level_points = np.array([0.0, *CONFIDENCE_LEVELS, 1.0], dtype=float)
    return np.interp(values, native_points, level_points)


# ---------------------------------------------------------------------------
# Estimators. Each returns (f0_hz, native_confidence) with one frame every
# ``hop_length`` samples of 16 kHz audio, frame i centred on sample
# i * hop_length -- the same grid torchcrepe produces.
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000

# RMVPE (Wei et al., 2023). The network reads a log-mel spectrogram and
# returns, per frame, a salience over 360 pitch bins of 20 cents. These
# front-end constants are fixed by the released weights.
RMVPE_HOP = 160
RMVPE_N_FFT = 1024
RMVPE_MELS = 128
RMVPE_MEL_FMIN = 30.0
RMVPE_MEL_FMAX = 8000.0
RMVPE_LOG_FLOOR = 1e-5
RMVPE_CENTS_ORIGIN = 1997.3794084376191
RMVPE_BIN_CENTS = 20.0
RMVPE_FRAME_MULTIPLE = 32
RMVPE_CHUNK_FRAMES = 3200
RMVPE_CONTEXT_FRAMES = 320


def rmvpe_log_mel(audio: np.ndarray) -> np.ndarray:
    import librosa

    magnitude = np.abs(
        librosa.stft(
            np.asarray(audio, dtype=np.float32),
            n_fft=RMVPE_N_FFT,
            hop_length=RMVPE_HOP,
            win_length=RMVPE_N_FFT,
            window="hann",
            center=True,
            pad_mode="reflect",
        )
    )
    basis = librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=RMVPE_N_FFT,
        n_mels=RMVPE_MELS,
        fmin=RMVPE_MEL_FMIN,
        fmax=RMVPE_MEL_FMAX,
        htk=True,
    )
    return np.log(np.maximum(basis @ magnitude, RMVPE_LOG_FLOOR)).astype(np.float32)


def decode_rmvpe_salience(salience: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Salience (frames, 360) -> (f0_hz, confidence).

    The pitch is the salience-weighted mean of the nine bins around the peak,
    which resolves well below the 20-cent bin width; the confidence is the
    peak salience itself.
    """
    salience = np.asarray(salience, dtype=np.float64)
    frames, bins = salience.shape
    peak = np.argmax(salience, axis=1)
    offsets = np.arange(-4, 5)
    indexes = peak[:, None] + offsets[None, :]
    valid = (indexes >= 0) & (indexes < bins)
    local = np.where(
        valid, salience[np.arange(frames)[:, None], np.clip(indexes, 0, bins - 1)], 0.0
    )
    cents = RMVPE_BIN_CENTS * indexes + RMVPE_CENTS_ORIGIN
    weight = local.sum(axis=1)
    weighted = (local * cents).sum(axis=1)
    peak_cents = RMVPE_BIN_CENTS * peak + RMVPE_CENTS_ORIGIN
    mean_cents = np.where(weight > 0, weighted / np.maximum(weight, 1e-12), peak_cents)
    return 10.0 * 2.0 ** (mean_cents / 1200.0), salience.max(axis=1)


def estimate_rmvpe(
    audio: np.ndarray,
    hop_length: int,
    model_path: Path,
    providers: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    import onnxruntime as ort

    if hop_length != RMVPE_HOP:
        raise ValueError(f"RMVPE analyses a fixed {RMVPE_HOP}-sample hop (10 ms)")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"RMVPE weights not found: {model_path}; "
            "run `node scripts/model-assets.mjs pitch rmvpe`"
        )
    mel = rmvpe_log_mel(audio)
    total = mel.shape[1]
    session = ort.InferenceSession(
        str(model_path), providers=providers or ["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    salience = np.zeros((total, 360), dtype=np.float32)
    # Fixed-size, overlapping chunks keep memory independent of song length.
    # Only each chunk's core is kept; the context on either side gives the
    # network's recurrent layers the same view they would have of the whole.
    for core_start in range(0, total, RMVPE_CHUNK_FRAMES):
        core_end = min(total, core_start + RMVPE_CHUNK_FRAMES)
        start = max(0, core_start - RMVPE_CONTEXT_FRAMES)
        end = min(total, core_end + RMVPE_CONTEXT_FRAMES)
        chunk = mel[:, start:end]
        padding = (-chunk.shape[1]) % RMVPE_FRAME_MULTIPLE
        if padding:
            chunk = np.pad(chunk, ((0, 0), (0, padding)), mode="constant")
        output = session.run(None, {input_name: chunk[None, :, :]})[0][0]
        salience[core_start:core_end] = output[
            core_start - start : core_end - start
        ]
    return decode_rmvpe_salience(salience)


def estimate_swiftf0(
    audio: np.ndarray,
    hop_length: int,
    fmin: float,
    fmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    from swift_f0 import SwiftF0

    result = SwiftF0().detect(
        np.asarray(audio, dtype=np.float32), SAMPLE_RATE, fmin=fmin, fmax=fmax
    )
    # SwiftF0 reports every 16 ms; the pipeline's grid is finer. Interpolate
    # pitch in log-frequency so a glide stays a straight line in semitones.
    target_times = np.arange(len(audio) // hop_length + 1) * hop_length / SAMPLE_RATE
    log_pitch = np.log2(np.maximum(result.pitch_hz, 1e-6))
    f0 = 2.0 ** np.interp(target_times, result.timestamps, log_pitch)
    confidence = np.interp(target_times, result.timestamps, result.confidence)
    return f0, confidence
