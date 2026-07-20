#!/usr/bin/env python3
"""Render decoded note JSON as a deterministic piano guide WAV.

The default piano is a bounded-memory physical approximation: struck,
slightly inharmonic strings, pitch-dependent partial decay, a deterministic
hammer transient, and a small soundboard/room response. It needs no external
SoundFont and keeps only one mono song buffer in memory.

Usage:
  .venv/bin/python render_note_transcription.py \
    contour_out/lemon_notes_auto.json --out outputs/lemon-auto-piano.wav
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def _valid_notes(notes: list[dict], maximum_time: float = 1_800.0) -> list[dict]:
    valid: list[dict] = []
    for note in notes:
        try:
            t0 = float(note["t0"])
            t1 = float(note["t1"])
            midi = int(note["midi"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            np.isfinite(t0)
            and np.isfinite(t1)
            and 0 <= t0 < t1 <= maximum_time
            and 0 <= midi <= 127
        ):
            valid.append(note)
    return valid


def _piano_note(
    midi: int,
    key_duration: float,
    confidence: float,
    sample_rate: int,
    seed: int,
) -> np.ndarray:
    release = float(np.clip(0.18 + 0.004 * (72 - midi), 0.10, 0.34))
    duration = key_duration + release
    count = max(1, int(np.ceil(duration * sample_rate)))
    time = np.arange(count, dtype=np.float64) / sample_rate
    frequency = 440.0 * 2 ** ((midi - 69) / 12)

    attack = 1.0 - np.exp(-time / 0.0028)
    release_envelope = np.ones(count, dtype=np.float64)
    released = time > key_duration
    release_envelope[released] = np.exp(
        -(time[released] - key_duration) / max(release * 0.16, 0.018)
    )
    base_decay = float(np.clip(2.8 * 2 ** ((60 - midi) / 30), 0.9, 4.2))
    inharmonicity = float(
        np.clip(1.0e-4 * 2 ** ((midi - 60) / 18), 3.0e-5, 8.0e-4)
    )
    maximum_partial = max(1, min(14, int(0.47 * sample_rate / frequency)))

    tonal = np.zeros(count, dtype=np.float64)
    weight_sum = 0.0
    string_cents = (-0.65, 0.0, 0.82) if midi >= 52 else (0.0,)
    string_weights = (0.24, 0.53, 0.23) if midi >= 52 else (1.0,)
    for partial in range(1, maximum_partial + 1):
        partial_frequency = (
            partial
            * frequency
            * np.sqrt(1.0 + inharmonicity * partial * partial)
        )
        if partial_frequency >= 0.47 * sample_rate:
            break
        partial_weight = (
            partial ** -1.25
            * np.exp(-0.028 * partial * partial)
            * (0.42 + 0.58 * abs(np.sin(np.pi * partial * 0.18)))
        )
        partial_decay = np.exp(
            -time / (base_decay / max(1.0, partial ** 0.35))
        )
        strings = np.zeros(count, dtype=np.float64)
        for cents, string_weight in zip(string_cents, string_weights):
            detuned = partial_frequency * 2 ** (cents / 1200.0)
            phase = ((seed + partial * 131 + int((cents + 1) * 97)) % 6283) / 1000
            strings += string_weight * np.sin(2 * np.pi * detuned * time + phase)
        tonal += partial_weight * partial_decay * strings
        weight_sum += partial_weight
    if weight_sum:
        tonal /= weight_sum

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(count)
    high_passed = np.r_[noise[0], np.diff(noise)]
    hammer_ramp = np.clip(time / 0.0010, 0.0, 1.0)
    hammer = 0.020 * hammer_ramp * high_passed * np.exp(-time / 0.010)
    if not np.isfinite(confidence):
        confidence = 1.0
    velocity = 0.64 + 0.16 * float(np.clip(confidence, 0.0, 1.0))
    return (
        velocity * attack * release_envelope * tonal
        + velocity * hammer
    ).astype(np.float32)


def render_piano(
    notes: list[dict], output: Path, sample_rate: int = 44_100
) -> None:
    notes = _valid_notes(notes)
    end_time = max((float(note["t1"]) for note in notes), default=0.0) + 0.7
    audio = np.zeros(max(1, int(np.ceil(end_time * sample_rate))), dtype=np.float32)
    for note in notes:
        start = max(0, int(round(float(note["t0"]) * sample_rate)))
        key_duration = float(note["t1"]) - float(note["t0"])
        midi = int(note["midi"])
        try:
            confidence = float(note.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        if not np.isfinite(confidence):
            confidence = 1.0
        seed = (
            midi * 1_000_003
            + start * 97
            + int(round(key_duration * sample_rate)) * 193
        ) & 0xFFFFFFFF
        wave = _piano_note(midi, key_duration, confidence, sample_rate, seed)
        end = min(len(audio), start + len(wave))
        if end > start:
            audio[start:end] += wave[: end - start]

    # Fixed early reflections approximate a soundboard/room without a long
    # convolution or a second full-size floating-point buffer.
    dry = audio.copy()
    for delay_s, gain in ((0.029, 0.10), (0.043, 0.075), (0.061, 0.055),
                          (0.079, 0.035), (0.113, 0.020)):
        delay = int(round(delay_s * sample_rate))
        if delay < len(audio):
            audio[delay:] += gain * dry[:-delay]
    del dry

    audio *= 1.15
    np.tanh(audio, out=audio)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0:
        audio *= 0.96 / peak
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.stem + ".tmp" + output.suffix)
    sf.write(temporary_output, audio, sample_rate, subtype="PCM_16")
    temporary_output.replace(output)


def render_sine(
    notes: list[dict], output: Path, sample_rate: int = 22_050
) -> None:
    notes = _valid_notes(notes)
    duration = max((float(note["t1"]) for note in notes), default=0.0) + 0.1
    audio = np.zeros(max(1, int(np.ceil(duration * sample_rate))), dtype=np.float32)
    for note in notes:
        start = max(0, int(round(float(note["t0"]) * sample_rate)))
        end = min(len(audio), int(round(float(note["t1"]) * sample_rate)))
        if end <= start:
            continue
        frequency = 440.0 * 2 ** ((int(note["midi"]) - 69) / 12)
        count = end - start
        phase = 2 * np.pi * frequency * np.arange(count) / sample_rate
        envelope = np.ones(count, dtype=float)
        fade = min(int(0.012 * sample_rate), count // 3)
        if fade > 1:
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade))
            envelope[:fade] *= ramp
            envelope[-fade:] *= ramp[::-1]
        confidence = float(note.get("confidence", 1.0))
        amplitude = 0.16 + 0.12 * np.clip(confidence, 0.0, 1.0)
        audio[start:end] += (amplitude * envelope * np.sin(phase)).astype(np.float32)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0.95:
        audio *= 0.95 / peak
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.stem + ".tmp" + output.suffix)
    sf.write(temporary_output, audio, sample_rate, subtype="PCM_16")
    temporary_output.replace(output)


def render(
    notes: list[dict],
    output: Path,
    sample_rate: int = 44_100,
    instrument: str = "piano",
) -> None:
    if instrument == "sine":
        render_sine(notes, output, sample_rate)
    else:
        render_piano(notes, output, sample_rate)


def default_output(prediction: Path, instrument: str) -> Path:
    stem = prediction.stem.replace("_notes_auto", "-auto")
    return Path("outputs") / f"{stem}-{instrument}.wav"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--instrument", choices=("piano", "sine"), default="piano")
    parser.add_argument("--sample-rate", type=int, default=44_100)
    args = parser.parse_args()

    if not 8_000 <= args.sample_rate <= 192_000:
        parser.error("--sample-rate must be between 8000 and 192000")

    payload = json.loads(args.prediction.read_text())
    output = args.out or default_output(args.prediction, args.instrument)
    render(payload["notes"], output, args.sample_rate, args.instrument)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
