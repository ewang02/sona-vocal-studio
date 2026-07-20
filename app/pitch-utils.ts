export type VoiceLabel =
  | "Bass"
  | "Baritone"
  | "Tenor"
  | "Alto/Contralto"
  | "Mezzo-soprano"
  | "Soprano";

export interface VoiceAlternative {
  label: VoiceLabel;
  range: readonly [number, number];
  overlapSemitones: number;
  score: number;
}

export interface VoiceFit {
  label: VoiceLabel;
  approximateLabel: string;
  range: readonly [number, number];
  observedRange: readonly [number, number];
  overlapSemitones: number;
  score: number;
  confidence: number;
  alternatives: readonly VoiceAlternative[];
}

export interface PitchFrame {
  f0: number | null;
  clarity: number;
  rms: number;
  timeMs: number;
}

export interface FilteredPitchFrame {
  midi: number | null;
  rmsFloor: number;
  clarityFloor: number;
  bridged: boolean;
  calibrating: boolean;
}

export interface AdaptivePitchFilter {
  process(frame: PitchFrame): FilteredPitchFrame;
  consume(frame: PitchFrame): FilteredPitchFrame;
  reset(): void;
}

interface VoiceRangeDefinition {
  label: VoiceLabel;
  low: number;
  high: number;
}

const NOTE_NAMES = [
  "C",
  "C#",
  "D",
  "D#",
  "E",
  "F",
  "F#",
  "G",
  "G#",
  "A",
  "A#",
  "B",
] as const;

const VOICE_RANGES: readonly VoiceRangeDefinition[] = [
  { label: "Bass", low: 40, high: 64 },
  { label: "Baritone", low: 43, high: 67 },
  { label: "Tenor", low: 48, high: 72 },
  { label: "Alto/Contralto", low: 53, high: 77 },
  { label: "Mezzo-soprano", low: 57, high: 81 },
  { label: "Soprano", low: 60, high: 84 },
];

export function clamp(value: number, minimum: number, maximum: number): number {
  const low = Math.min(minimum, maximum);
  const high = Math.max(minimum, maximum);
  return Math.min(high, Math.max(low, value));
}

export function midiToName(midi: number): string {
  if (!Number.isFinite(midi)) return "—";
  const rounded = Math.round(midi);
  const noteIndex = ((rounded % 12) + 12) % 12;
  const octave = Math.floor(rounded / 12) - 1;
  return `${NOTE_NAMES[noteIndex]}${octave}`;
}

export function hzToMidi(hz: number): number {
  return Number.isFinite(hz) && hz > 0 ? 69 + 12 * Math.log2(hz / 440) : Number.NaN;
}

/** Seconds → "m:ss", clamped at zero. Shared by the player and the library. */
export function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const minutes = Math.floor(seconds / 60);
  const remaining = Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${minutes}:${remaining}`;
}

function sortedFinite(values: readonly number[]): number[] {
  return values.filter(Number.isFinite).sort((a, b) => a - b);
}

export function median(values: readonly number[]): number {
  const sorted = sortedFinite(values);
  if (sorted.length === 0) return Number.NaN;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

/** Percentile with p expressed from 0 to 1 and linear interpolation. */
export function percentile(values: readonly number[], p: number): number {
  const sorted = sortedFinite(values);
  if (sorted.length === 0) return Number.NaN;
  const position = clamp(p, 0, 1) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const fraction = position - lower;
  return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
}

export function classifyVoiceRange(lowMidi: number, highMidi: number): VoiceFit {
  if (!Number.isFinite(lowMidi) || !Number.isFinite(highMidi)) {
    throw new RangeError("Voice range endpoints must be finite MIDI values.");
  }

  const low = Math.min(lowMidi, highMidi);
  const high = Math.max(lowMidi, highMidi);
  const observedSpan = Math.max(high - low, 0.5);
  const observedCenter = (low + high) / 2;

  const fits = VOICE_RANGES.map((voice): VoiceAlternative => {
    const overlapLow = Math.max(low, voice.low);
    const overlapHigh = Math.min(high, voice.high);
    const overlaps = overlapHigh >= overlapLow;
    const overlap = overlaps ? Math.max(overlapHigh - overlapLow, 0.5) : 0;
    const voiceSpan = voice.high - voice.low;
    const coverage = clamp(overlap / observedSpan, 0, 1);
    const targetCoverage = clamp(overlap / voiceSpan, 0, 1);
    const centerFit = 1 - clamp(Math.abs(observedCenter - (voice.low + voice.high) / 2) / 24, 0, 1);
    // Low-note proximity is slightly more useful than the sampled high note for
    // resolving common Baritone/Tenor and Alto/Mezzo overlaps.
    const edgeDistance = Math.abs(low - voice.low) * 0.6 + Math.abs(high - voice.high) * 0.4;
    const edgeFit = 1 - clamp(edgeDistance / 24, 0, 1);
    const score = coverage * 0.5 + targetCoverage * 0.2 + centerFit * 0.15 + edgeFit * 0.15;

    return {
      label: voice.label,
      range: [voice.low, voice.high],
      overlapSemitones: overlaps ? Math.max(0, overlapHigh - overlapLow) : 0,
      score,
    };
  }).sort((a, b) => b.score - a.score);

  const primary = fits[0];
  const runnerUp = fits[1];
  const separation = primary.score - runnerUp.score;

  return {
    label: primary.label,
    approximateLabel: `≈ ${primary.label}`,
    range: primary.range,
    observedRange: [low, high],
    overlapSemitones: primary.overlapSemitones,
    score: primary.score,
    confidence: clamp(primary.score * (0.75 + Math.max(0, separation)), 0, 1),
    alternatives: fits.slice(1, 4),
  };
}

export function createAdaptivePitchFilter(): AdaptivePitchFilter {
  // The clarity floor must be able to fall as well as rise. Learning it from
  // frames the gate already accepted is a ratchet: the percentile of a set
  // filtered at >= floor can never sit below the floor, so a clean singer
  // drives it to the ceiling and then loses ordinary vibrato and transition
  // frames. Learn it from every pitched frame instead, and keep the ceiling
  // below the clarity a real voice dips to mid-phrase.
  const MIN_CLARITY = 0.35;
  const MAX_CLARITY = 0.55;
  const CLARITY_PERCENTILE = 0.1;
  const CLARITY_MARGIN = 0.05;
  const ABSOLUTE_RMS_FLOOR = 0.0015;
  const MAX_RMS_FLOOR = 0.03;
  const CLEARLY_UNVOICED_CLARITY = 0.42;
  const NOISE_HISTORY = 140;
  const CLARITY_HISTORY = 240;
  const MEDIAN_WINDOW = 5;
  const JUMP_SEMITONES = 7;
  const JUMP_CONFIRM_FRAMES = 3;
  const JUMP_CLUSTER = 2.5;
  const MAX_BRIDGE_MS = 70;
  // An exact-octave outlier against a held note is a detector glitch, not a
  // leap: fold it back rather than letting it feed the jump confirmer. Real
  // octave leaps push through after OCTAVE_FOLD_LIMIT consecutive folds.
  const OCTAVE_TOLERANCE = 0.75;
  // Downward octave flips are the common YIN subharmonic failure. Never
  // accept one merely because it persisted for a handful of frames: a real
  // downward octave after an articulation is accepted from a reset state.
  // Upward movement must remain possible so an onset initially detected one
  // octave low can recover quickly.
  const UPWARD_OCTAVE_FOLD_LIMIT = 3;
  // Adaptive EMA over the median output: near-stationary wobble is damped,
  // larger moves pass through at full speed (alpha reaches 1), so glides and
  // onsets keep their timing.
  const EMA_BASE_ALPHA = 0.3;
  const EMA_ALPHA_PER_ST = 0.5;

  let noiseRms: number[] = [];
  let pitchedClarity: number[] = [];
  let midiWindow: number[] = [];
  let pendingJump: number[] = [];
  let rmsFloor = 0.003;
  let clarityFloor = 0.45;
  let stableMidi: number | null = null;
  let emaMidi: number | null = null;
  let octaveFoldStreak = 0;
  let lastAcceptedTime: number | null = null;
  let lastFrameTime: number | null = null;
  let frameCount = 0;

  const reset = (): void => {
    noiseRms = [];
    pitchedClarity = [];
    midiWindow = [];
    pendingJump = [];
    rmsFloor = 0.003;
    clarityFloor = 0.45;
    stableMidi = null;
    emaMidi = null;
    octaveFoldStreak = 0;
    lastAcceptedTime = null;
    lastFrameTime = null;
    frameCount = 0;
  };

  const result = (midi: number | null, bridged: boolean): FilteredPitchFrame => ({
    midi,
    rmsFloor,
    clarityFloor,
    bridged,
    calibrating: frameCount < 30 && noiseRms.length < 10,
  });

  const rememberNoise = (rms: number): void => {
    const upperNoise =
      noiseRms.length >= 10 ? Math.max(ABSOLUTE_RMS_FLOOR, percentile(noiseRms, 0.9) * 1.8) : 0.08;
    if (rms > upperNoise) return;
    noiseRms.push(rms);
    if (noiseRms.length > NOISE_HISTORY) noiseRms.shift();
    const learned = percentile(noiseRms, 0.8) * 1.8;
    rmsFloor = clamp(learned, ABSOLUTE_RMS_FLOOR, MAX_RMS_FLOOR);
  };

  const process = (frame: PitchFrame): FilteredPitchFrame => {
    frameCount += 1;
    const clarity = clamp(Number.isFinite(frame.clarity) ? frame.clarity : 0, 0, 1);
    const rms = Math.max(0, Number.isFinite(frame.rms) ? frame.rms : 0);
    const timeMs = Number.isFinite(frame.timeMs)
      ? frame.timeMs
      : (lastFrameTime ?? 0) + 1000 / 60;

    if (lastFrameTime !== null && timeMs < lastFrameTime) {
      midiWindow = [];
      pendingJump = [];
      stableMidi = null;
      lastAcceptedTime = null;
    }
    lastFrameTime = timeMs;

    const rawMidi = frame.f0 === null ? Number.NaN : hzToMidi(frame.f0);
    if (!Number.isFinite(rawMidi) || clarity <= CLEARLY_UNVOICED_CLARITY) {
      rememberNoise(rms);
    }

    // Learn from every frame the detector found a pitch in, loud enough to be
    // the singer -- not only from frames that passed the gate.
    if (Number.isFinite(rawMidi) && rms >= rmsFloor) {
      pitchedClarity.push(clarity);
      if (pitchedClarity.length > CLARITY_HISTORY) pitchedClarity.shift();
      if (pitchedClarity.length >= 20) {
        clarityFloor = clamp(
          percentile(pitchedClarity, CLARITY_PERCENTILE) - CLARITY_MARGIN,
          MIN_CLARITY,
          MAX_CLARITY,
        );
      }
    }

    const voiced = Number.isFinite(rawMidi) && clarity >= clarityFloor && rms >= rmsFloor;
    if (!voiced) {
      pendingJump = [];
      octaveFoldStreak = 0;
      const gap =
        lastAcceptedTime === null ? Number.POSITIVE_INFINITY : timeMs - lastAcceptedTime;
      if (stableMidi !== null && gap >= 0 && gap <= MAX_BRIDGE_MS) {
        return result(emaMidi ?? stableMidi, true);
      }
      midiWindow = [];
      stableMidi = null;
      emaMidi = null;
      return result(null, false);
    }

    // Exact-octave outliers against a held pitch fold back onto it instead of
    // feeding the jump confirmer. Downward subharmonics remain folded during
    // a continuous vowel; upward corrections can confirm after a short hold.
    let effectiveMidi = rawMidi;
    if (stableMidi !== null) {
      const jump = rawMidi - stableMidi;
      const octaves = Math.round(jump / 12);
      if (octaves !== 0 && Math.abs(jump - octaves * 12) <= OCTAVE_TOLERANCE) {
        if (octaves < 0 || octaveFoldStreak < UPWARD_OCTAVE_FOLD_LIMIT) {
          effectiveMidi = rawMidi - octaves * 12;
          octaveFoldStreak = octaves > 0 ? octaveFoldStreak + 1 : 0;
        }
      } else if (Math.abs(jump) > JUMP_SEMITONES) {
        // Leave the streak alone: a still-jumping pitch past the fold limit
        // must stay unfolded so the confirmer can accept the leap.
      } else {
        octaveFoldStreak = 0;
      }
    } else {
      octaveFoldStreak = 0;
    }

    if (stableMidi !== null && Math.abs(effectiveMidi - stableMidi) > JUMP_SEMITONES) {
      const pendingCenter = median(pendingJump);
      if (pendingJump.length === 0 || Math.abs(effectiveMidi - pendingCenter) <= JUMP_CLUSTER) {
        pendingJump.push(effectiveMidi);
      } else {
        pendingJump = [effectiveMidi];
      }
      if (pendingJump.length < JUMP_CONFIRM_FRAMES) return result(emaMidi ?? stableMidi, false);

      midiWindow = pendingJump.slice(-MEDIAN_WINDOW);
      pendingJump = [];
      octaveFoldStreak = 0;
      stableMidi = median(midiWindow);
      // A confirmed jump lands instantly; easing across it would draw a swoop
      // the singer never sang.
      emaMidi = stableMidi;
      lastAcceptedTime = timeMs;
      return result(emaMidi, false);
    }

    pendingJump = [];
    midiWindow.push(effectiveMidi);
    if (midiWindow.length > MEDIAN_WINDOW) midiWindow.shift();
    stableMidi = median(midiWindow);
    if (emaMidi === null) {
      emaMidi = stableMidi;
    } else {
      const delta = stableMidi - emaMidi;
      const alpha = clamp(EMA_BASE_ALPHA + Math.abs(delta) * EMA_ALPHA_PER_ST, EMA_BASE_ALPHA, 1);
      emaMidi += alpha * delta;
    }
    lastAcceptedTime = timeMs;
    return result(emaMidi, false);
  };

  return { process, consume: process, reset };
}
