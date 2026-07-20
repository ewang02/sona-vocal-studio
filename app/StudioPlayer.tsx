"use client";

import type { SoundTouchNode as SoundTouchNodeType } from "@soundtouchjs/audio-worklet";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  clamp,
  classifyVoiceRange,
  createAdaptivePitchFilter,
  formatTime,
  midiToName,
  percentile,
} from "./pitch-utils";
import type { ContourData, Song } from "./songs";
import {
  editContour,
  encodeWav,
  mixPcm,
  normalizeSelection,
  pcmBlockEnd,
  renderPcmTrack,
  trimPcmBlocks,
  type PcmBlock,
  type RepairSourceData,
  type StudioMode,
  type TimeSelection,
} from "./studio-utils";

type InputMode = "demo" | "microphone";

// Manual fine-trim on top of the measured mic-latency model, persisted across
// sessions. The measured chain covers analysis and scheduling exactly but the
// capture path is device-dependent and unqueryable in most browsers.
const MIC_NUDGE_STORAGE_KEY = "sona.micNudgeMs";
const MIC_NUDGE_MIN_MS = -100;
const MIC_NUDGE_MAX_MS = 250;
const TRANSPOSE_MIN = -16;
const TRANSPOSE_MAX = 16;
const EDIT_STORAGE_PREFIX = "sona.editedContour.v1.";


const PAST_SECONDS = 2.6;
const FUTURE_SECONDS = 4.4;
const PLAYHEAD_FRACTION = PAST_SECONDS / (PAST_SECONDS + FUTURE_SECONDS);
// Full credit within ±0.5 semitone of the wave; credit tapers to zero at ±1.
const INNER_BAND = 0.5;
const INNER_RELEASE = 0.6;
const OUTER_BAND = 1;
const TRAIL_SECONDS = 1.8;
const UI_UPDATE_MS = 200;
// Left gutter reserved for the note axis; scrolling content is clipped to its right.
const AXIS_WIDTH = 46;
const NATURAL_PITCH_CLASSES = new Set([0, 2, 4, 5, 7, 9, 11]);

// Latency model. SoundTouch's `framesBuffered` metric reports what is sitting
// in its output buffer, not pipeline delay, so it must never be used as a
// clock correction: it swings over a huge range and updates only a few times a
// second, which drags the target note around under the singer. Use measured
// constants instead, and bypass the shifter entirely at the original key.
const SHIFTER_LATENCY_S = 0.046;
// MediaStreamTrack.getSettings().latency is not implemented by every browser.
// Use a modest input-path estimate when it is unavailable; unlike
// AudioContext.baseLatency, this represents capture rather than playback.
const DEFAULT_CAPTURE_LATENCY_S = 0.025;
// The 5-frame median in the adaptive filter reports the middle sample, so it
// trails the newest frame by two hops.
const FILTER_MEDIAN_LAG_S = (2 * 512) / 48000;
const DEFAULT_ANALYSIS_LATENCY_S = (1024 + 600 - 512) / 48000;

// Scoring: 9000 points are spread across the song's voiced frames by
// duration; each frame pays out its share times the singer's credit. The
// remaining 1000 come from phrase bonuses (full share for a >=90% clean
// phrase, half for >=75%), for a perfect total of 10000.
const MAX_POINTS = 10000;
const MELODY_POINTS = 9000;
const BONUS_POINTS = MAX_POINTS - MELODY_POINTS;
const PHRASE_CLEAN_RATIO = 0.9;
const PHRASE_GOOD_RATIO = 0.75;
const PHRASE_ELIGIBLE_COVERAGE = 0.7;

type TrailPoint = { t: number; midi: number; inTune: boolean };
type HitInterval = { t0: number; t1: number };
type Segment = { t0: number; t1: number; midi: number[] };
type RecordingStatus = "idle" | "recording" | "paused" | "processing" | "complete";
type RecordingFiles = {
  vocals: string;
  instrumental: string;
  combined: string;
  duration: number;
  songTranspose: number;
};
type PerformanceTotals = {
  earnedSeconds: number;
  targetSeconds: number;
  bonusPoints: number;
  segmentEarned: number;
  segmentTarget: number;
};

function formatSigned(value: number) {
  if (value === 0) return "0";
  return value > 0 ? `+${value}` : String(value);
}

function displayMidiName(midi: number) {
  const rounded = Math.round(midi);
  return rounded < 0 || rounded > 127 ? "…" : midiToName(midi);
}

function sameContour(left: ContourData, right: ContourData) {
  return JSON.stringify(left.segments) === JSON.stringify(right.segments);
}

function segmentIndexAt(segments: Segment[], time: number) {
  let low = 0;
  let high = segments.length - 1;
  while (low <= high) {
    const middle = (low + high) >> 1;
    const segment = segments[middle];
    if (time < segment.t0) high = middle - 1;
    else if (time >= segment.t1) low = middle + 1;
    else return middle;
  }
  return -1;
}

// 100% inside the inner band, then a cosine ease down to zero at the outer
// band: gentle right at the edge, steepest midway, nothing beyond ±1 st.
function creditForDistance(distance: number) {
  if (distance <= INNER_BAND) return 1;
  if (distance >= OUTER_BAND) return 0;
  return 0.5 + 0.5 * Math.cos((Math.PI * (distance - INNER_BAND)) / (OUTER_BAND - INNER_BAND));
}

function rankForAccuracy(accuracy: number) {
  if (accuracy >= 95) return "S";
  if (accuracy >= 90) return "A";
  if (accuracy >= 80) return "B";
  if (accuracy >= 65) return "C";
  return "D";
}

export default function StudioPlayer({ song, onBack }: { song: Song; onBack: () => void }) {
  const [studioMode, setStudioMode] = useState<StudioMode>("practicing");
  const [songTranspose, setSongTranspose] = useState(0);
  const [contourTranspose, setContourTranspose] = useState(0);
  const [transpositionsLinked, setTranspositionsLinked] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(song.contour.duration);
  const [inputMode, setInputMode] = useState<InputMode>("demo");
  const [micBusy, setMicBusy] = useState(false);
  const [micError, setMicError] = useState("");
  const [micNudgeMs, setMicNudgeMs] = useState(0);
  // Practice starts on the instrumental. Editing opts into the original mix;
  // recording can monitor that mix while exporting only the instrumental.
  const [guideVocals, setGuideVocals] = useState(false);
  const [livePitch, setLivePitch] = useState<number | null>(null);
  const [targetPitch, setTargetPitch] = useState<number | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [waveNearby, setWaveNearby] = useState(true);
  const [score, setScore] = useState<number | null>(null);
  const [accuracy, setAccuracy] = useState<number | null>(null);
  const [editedContour, setEditedContour] = useState<ContourData>(song.contour);
  const [savedContour, setSavedContour] = useState<ContourData>(song.contour);
  const [selection, setSelection] = useState<TimeSelection | null>(null);
  const [selecting, setSelecting] = useState(false);
  const [repairSourceResult, setRepairSourceResult] = useState<{
    url: string;
    source: RepairSourceData | null;
    state: "ready" | "missing";
  } | null>(null);
  const repairSource =
    repairSourceResult?.url === song.repairSource ? repairSourceResult.source : null;
  const repairSourceState =
    repairSourceResult?.url === song.repairSource ? repairSourceResult.state : "loading";
  const [editNotice, setEditNotice] = useState("");
  const [recordingStatus, setRecordingStatus] = useState<RecordingStatus>("idle");
  const [recordingFiles, setRecordingFiles] = useState<RecordingFiles | null>(null);
  const [recordingError, setRecordingError] = useState("");
  const [recordedUntil, setRecordedUntil] = useState(0);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const guideAudioRef = useRef<HTMLAudioElement | null>(null);
  const laneRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const laneSizeRef = useRef({ width: 900, height: 440 });
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioInitPromiseRef = useRef<Promise<AudioContext> | null>(null);
  const soundTouchRef = useRef<SoundTouchNodeType | null>(null);
  const guideSoundTouchRef = useRef<SoundTouchNodeType | null>(null);
  const outputLatencyRef = useRef(0);
  const micLatencyRef = useRef(0);
  const captureLatencyRef = useRef(DEFAULT_CAPTURE_LATENCY_S);
  const analysisLatencyRef = useRef(DEFAULT_ANALYSIS_LATENCY_S);
  const micClockOriginRef = useRef<number | null>(null);
  const livePitchSampleTimeRef = useRef<number | null>(null);
  const livePitchEventRef = useRef(0);
  const drawnPitchEventRef = useRef(0);
  const shifterBypassedRef = useRef(true);
  const mediaSourceRef = useRef<MediaElementAudioSourceNode | null>(null);
  const guideMediaSourceRef = useRef<MediaElementAudioSourceNode | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const micSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const micNodeRef = useRef<AudioWorkletNode | null>(null);
  const micSilentGainRef = useRef<GainNode | null>(null);
  const micModuleContextRef = useRef<AudioContext | null>(null);
  const micRequestGenerationRef = useRef(0);
  // Try the microphone automatically on first play; cleared when the user
  // explicitly chooses demo input or when mic access fails.
  const autoMicRef = useRef(true);
  const adaptiveFilterRef = useRef(createAdaptivePitchFilter());
  const songTransposeRef = useRef(songTranspose);
  const contourTransposeRef = useRef(contourTranspose);
  const guideVocalsRef = useRef(guideVocals);
  const micNudgeSecondsRef = useRef(0);
  const contourScaleRef = useRef(1);
  const currentTimeRef = useRef(0);
  const livePitchRef = useRef<number | null>(null);
  const lastUiUpdateRef = useRef(0);
  const lastScoreTimeRef = useRef(0);
  const hitIntervalsRef = useRef(new Map<number, HitInterval[]>());
  const performanceRef = useRef<PerformanceTotals>({
    earnedSeconds: 0,
    targetSeconds: 0,
    bonusPoints: 0,
    segmentEarned: 0,
    segmentTarget: 0,
  });
  const scoringSegmentRef = useRef(-1);
  const hitActiveRef = useRef(false);
  const trailRef = useRef<TrailPoint[]>([]);
  const rafRef = useRef<number | null>(null);
  const studioModeRef = useRef<StudioMode>(studioMode);
  const selectionRef = useRef<TimeSelection | null>(selection);
  const selectionAnchorRef = useRef<number | null>(null);
  const instrumentCaptureRef = useRef<ScriptProcessorNode | null>(null);
  const vocalCaptureRef = useRef<ScriptProcessorNode | null>(null);
  const captureSilentGainRef = useRef<GainNode | null>(null);
  const recordingActiveRef = useRef(false);
  const instrumentalBlocksRef = useRef<PcmBlock[]>([]);
  const vocalBlocksRef = useRef<PcmBlock[]>([]);
  const recordingFilesRef = useRef<RecordingFiles | null>(null);

  // The edited contour is flattened for O(1) target lookup by frame index.
  const melody = useMemo(() => {
    const { hop, duration: contourDuration, segments: raw, range } = editedContour;
    const frames = new Float32Array(Math.ceil(contourDuration / hop) + 2).fill(Number.NaN);
    const voiced: number[] = [];
    const segments: Segment[] = raw.map((segment) => {
      const startIndex = Math.round(segment.t0 / hop);
      for (let index = 0; index < segment.midi.length; index += 1) {
        frames[startIndex + index] = segment.midi[index];
      }
      voiced.push(...segment.midi);
      return {
        t0: segment.t0,
        t1: segment.t0 + segment.midi.length * hop,
        midi: segment.midi,
      };
    });
    const low = range[0];
    const high = range[1];
    voiced.sort((a, b) => a - b);
    const tessituraLow = percentile(voiced, 0.05);
    const tessituraHigh = percentile(voiced, 0.95);
    return {
      hop,
      duration: contourDuration,
      frames,
      segments,
      totalVoicedSeconds: segments.reduce((total, segment) => total + (segment.t1 - segment.t0), 0),
      low,
      high,
      tessituraLow: Math.round(Number.isFinite(tessituraLow) ? tessituraLow : low),
      tessituraHigh: Math.round(Number.isFinite(tessituraHigh) ? tessituraHigh : high),
    };
  }, [editedContour]);
  const melodyRef = useRef(melody);
  useEffect(() => {
    melodyRef.current = melody;
  }, [melody]);

  const targetMidiAt = useCallback((contourTime: number) => {
    const data = melodyRef.current;
    const position = contourTime / data.hop;
    const leftIndex = Math.floor(position);
    const rightIndex = Math.ceil(position);
    if (leftIndex < 0 || leftIndex >= data.frames.length) return null;
    const left = data.frames[leftIndex];
    const right = data.frames[Math.min(rightIndex, data.frames.length - 1)];
    if (!Number.isFinite(left)) return null;
    if (!Number.isFinite(right) || rightIndex === leftIndex) return left;
    return left + (right - left) * (position - leftIndex);
  }, []);

  const transposedLow = melody.tessituraLow + contourTranspose;
  const transposedHigh = melody.tessituraHigh + contourTranspose;
  const voiceFit = useMemo(
    () => classifyVoiceRange(transposedLow, transposedHigh),
    [transposedHigh, transposedLow],
  );
  const targetDisplay = targetPitch === null ? null : targetPitch + contourTranspose;
  // Signed: positive when the singer is sharp (above center), negative when flat.
  const pitchOffset =
    livePitch !== null && targetDisplay !== null ? livePitch - targetDisplay : null;
  const pitchDistance = pitchOffset === null ? null : Math.abs(pitchOffset);
  const isInside = pitchDistance !== null && pitchDistance <= INNER_BAND;
  const primaryGuideVocals = studioMode !== "recording" && guideVocals;

  const startAudioTime = useCallback(() => {
    const firstSegment = melodyRef.current.segments[0];
    const contourStart = Math.max(0, (firstSegment?.t0 ?? 0) - 1.2);
    return contourStart / (contourScaleRef.current || 1);
  }, []);

  const resetPerformance = useCallback((audioTime: number) => {
    hitIntervalsRef.current.clear();
    performanceRef.current = {
      earnedSeconds: 0,
      targetSeconds: 0,
      bonusPoints: 0,
      segmentEarned: 0,
      segmentTarget: 0,
    };
    trailRef.current = [];
    livePitchRef.current = null;
    livePitchSampleTimeRef.current = null;
    drawnPitchEventRef.current = livePitchEventRef.current;
    hitActiveRef.current = false;
    scoringSegmentRef.current = -1;
    lastScoreTimeRef.current = audioTime;
    setScore(null);
    setAccuracy(null);
    setLivePitch(null);
    setTargetPitch(null);
  }, []);

  const setMicNudge = useCallback((valueMs: number) => {
    const next = clamp(Math.round(valueMs), MIC_NUDGE_MIN_MS, MIC_NUDGE_MAX_MS);
    micNudgeSecondsRef.current = next / 1000;
    setMicNudgeMs(next);
    try {
      window.localStorage.setItem(MIC_NUDGE_STORAGE_KEY, String(next));
    } catch {
      // Private mode or blocked storage: the nudge still applies this session.
    }
  }, []);

  // Restore the saved nudge after mount (not in the initializer: this page is
  // server-rendered first, and reading storage there would desync hydration —
  // the effect-based restore is React's documented pattern for local storage).
  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(MIC_NUDGE_STORAGE_KEY);
      // eslint-disable-next-line react-hooks/set-state-in-effect -- one-shot client-only state restore; a render pass cannot know this value
      if (stored !== null && Number.isFinite(Number(stored))) setMicNudge(Number(stored));
    } catch {
      // Unreadable storage: keep the default of 0 ms.
    }
  }, [setMicNudge]);

  useEffect(() => {
    studioModeRef.current = studioMode;
  }, [studioMode]);

  useEffect(() => {
    selectionRef.current = selection;
  }, [selection]);

  useEffect(() => {
    recordingFilesRef.current = recordingFiles;
  }, [recordingFiles]);

  useEffect(() => {
    const storageKey = EDIT_STORAGE_PREFIX + song.id;
    try {
      const stored = window.localStorage.getItem(storageKey);
      if (stored) {
        const parsed = JSON.parse(stored) as ContourData;
        if (
          parsed &&
          parsed.hop === song.contour.hop &&
          parsed.duration === song.contour.duration &&
          Array.isArray(parsed.segments)
        ) {
          // eslint-disable-next-line react-hooks/set-state-in-effect -- client-only saved contour
          setEditedContour(parsed);
          setSavedContour(parsed);
        }
      }
    } catch {
      // Invalid or unavailable local storage: use the immutable shipped source.
    }
  }, [song]);

  useEffect(() => {
    let cancelled = false;
    fetch(song.repairSource, { cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) throw new Error("repair source unavailable");
        return (await response.json()) as RepairSourceData;
      })
      .then((source) => {
        if (cancelled) return;
        setRepairSourceResult({ url: song.repairSource, source, state: "ready" });
      })
      .catch(() => {
        if (cancelled) return;
        setRepairSourceResult({ url: song.repairSource, source: null, state: "missing" });
      });
    return () => {
      cancelled = true;
    };
  }, [song.repairSource]);

  const updateScore = useCallback(() => {
    const totals = performanceRef.current;
    if (totals.targetSeconds <= 0) {
      setScore(null);
      setAccuracy(null);
      return;
    }
    const melodyPoints =
      (totals.earnedSeconds / Math.max(melodyRef.current.totalVoicedSeconds, 0.001)) *
      MELODY_POINTS;
    setScore(Math.round(clamp(melodyPoints + totals.bonusPoints, 0, MAX_POINTS)));
    setAccuracy(Math.round((totals.earnedSeconds / totals.targetSeconds) * 100));
  }, []);

  const finalizeScoringSegment = useCallback((segmentIndex: number) => {
    const totals = performanceRef.current;
    const segment = melodyRef.current.segments[segmentIndex];
    if (segmentIndex < 0 || !segment) {
      totals.segmentEarned = 0;
      totals.segmentTarget = 0;
      return;
    }
    const segmentDuration = segment.t1 - segment.t0;
    const phraseAccuracy = totals.segmentTarget > 0 ? totals.segmentEarned / totals.segmentTarget : 0;
    // A phrase only counts once most of it was actually traversed, so a seek
    // through the last beat of a phrase can't bank its bonus.
    const eligible = totals.segmentTarget >= PHRASE_ELIGIBLE_COVERAGE * segmentDuration;
    if (eligible && phraseAccuracy >= PHRASE_GOOD_RATIO) {
      const share =
        (segmentDuration / Math.max(melodyRef.current.totalVoicedSeconds, 0.001)) * BONUS_POINTS;
      totals.bonusPoints += phraseAccuracy >= PHRASE_CLEAN_RATIO ? share : share / 2;
    }
    totals.segmentEarned = 0;
    totals.segmentTarget = 0;
  }, []);

  // What the singer hears is delayed by the output chain; what we hear back
  // from them is delayed by the independent capture path. baseLatency belongs
  // only to the former. Pitch-message timestamps account for analysis and
  // scheduling delay frame by frame; this static value is the fallback until
  // the worklet clock is available.
  const refreshLatencies = useCallback(() => {
    const context = audioContextRef.current;
    const base = context ? context.baseLatency || 0 : 0;
    const output = context && Number.isFinite(context.outputLatency) ? context.outputLatency : 0;
    outputLatencyRef.current =
      base + output + (shifterBypassedRef.current ? 0 : SHIFTER_LATENCY_S);
    micLatencyRef.current =
      captureLatencyRef.current + analysisLatencyRef.current + FILTER_MEDIAN_LAG_S;
  }, []);

  // In recording mode the instrumental remains the capture source. Optional
  // guide vocals replace only the monitored output with the full mix.
  const applyAudioRouting = useCallback(
    (useShifter: boolean) => {
      const context = audioContextRef.current;
      const source = mediaSourceRef.current;
      const shifter = soundTouchRef.current;
      const guideSource = guideMediaSourceRef.current;
      const guideShifter = guideSoundTouchRef.current;
      if (!context || !source || !shifter || !guideSource || !guideShifter) return;
      for (const node of [source, shifter, guideSource, guideShifter]) {
        try {
          node.disconnect();
        } catch {
          // A node that was never connected has nothing to unwind.
        }
      }
      const monitorGuide =
        studioModeRef.current === "recording" && guideVocalsRef.current;
      if (useShifter) {
        source.connect(shifter);
        if (!monitorGuide) shifter.connect(context.destination);
        if (instrumentCaptureRef.current) shifter.connect(instrumentCaptureRef.current);
      } else {
        if (!monitorGuide) source.connect(context.destination);
        if (instrumentCaptureRef.current) source.connect(instrumentCaptureRef.current);
      }
      if (monitorGuide) {
        if (useShifter) {
          guideSource.connect(guideShifter);
          guideShifter.connect(context.destination);
        } else {
          guideSource.connect(context.destination);
        }
      }
      shifterBypassedRef.current = !useShifter;
      refreshLatencies();
    },
    [refreshLatencies],
  );

  const ensureAudioGraph = useCallback(async () => {
    const audio = audioRef.current;
    const guideAudio = guideAudioRef.current;
    if (!audio || !guideAudio) throw new Error("The song player is not ready yet.");

    if (
      audioContextRef.current &&
      soundTouchRef.current &&
      guideSoundTouchRef.current
    ) {
      if (audioContextRef.current.state === "suspended") await audioContextRef.current.resume();
      return audioContextRef.current;
    }
    if (audioInitPromiseRef.current) return audioInitPromiseRef.current;

    const initialize = async () => {
      const context = new AudioContext({ latencyHint: "interactive" });
      try {
        if (context.state === "suspended") await context.resume();
        const { SoundTouchNode } = await import("@soundtouchjs/audio-worklet");
        await SoundTouchNode.register(context, "/soundtouch-processor.js");
        const shifter = new SoundTouchNode({ context });
        const guideShifter = new SoundTouchNode({ context });
        const source = context.createMediaElementSource(audio);
        const guideSource = context.createMediaElementSource(guideAudio);
        for (const node of [shifter, guideShifter]) {
          node.playbackRate.value = 1;
          node.pitch.value = 1;
          node.pitchSemitones.value = songTransposeRef.current;
        }
        audio.preservesPitch = false;
        audio.playbackRate = 1;
        guideAudio.preservesPitch = false;
        guideAudio.playbackRate = 1;
        mediaSourceRef.current = source;
        soundTouchRef.current = shifter;
        guideMediaSourceRef.current = guideSource;
        guideSoundTouchRef.current = guideShifter;
        audioContextRef.current = context;
        applyAudioRouting(songTransposeRef.current !== 0);
        return context;
      } catch (error) {
        await context.close().catch(() => undefined);
        throw error;
      }
    };

    audioInitPromiseRef.current = initialize();
    try {
      return await audioInitPromiseRef.current;
    } finally {
      audioInitPromiseRef.current = null;
    }
  }, [applyAudioRouting]);

  const clearRecordingFiles = useCallback(() => {
    const files = recordingFilesRef.current;
    if (files) {
      URL.revokeObjectURL(files.vocals);
      URL.revokeObjectURL(files.instrumental);
      URL.revokeObjectURL(files.combined);
    }
    recordingFilesRef.current = null;
    setRecordingFiles(null);
  }, []);

  const ensureRecordingCapture = useCallback(async () => {
    const context = await ensureAudioGraph();
    if (!captureSilentGainRef.current) {
      const silent = context.createGain();
      silent.gain.value = 0;
      silent.connect(context.destination);
      captureSilentGainRef.current = silent;
    }
    const captureBlock = (
      event: AudioProcessingEvent,
      channels: number,
      destination: React.MutableRefObject<PcmBlock[]>,
      latencySeconds: number,
    ) => {
      if (!recordingActiveRef.current) return;
      const length = event.inputBuffer.length;
      const blockDuration = length / context.sampleRate;
      const copied = Array.from({ length: channels }, (_, channel) => {
        const availableChannel = Math.min(channel, event.inputBuffer.numberOfChannels - 1);
        return new Float32Array(event.inputBuffer.getChannelData(availableChannel));
      });
      destination.current.push({
        start: Math.max(0, currentTimeRef.current - blockDuration - latencySeconds),
        sampleRate: context.sampleRate,
        channels: copied,
      });
      setRecordedUntil((current) => Math.max(current, currentTimeRef.current));
    };
    if (!instrumentCaptureRef.current) {
      const node = context.createScriptProcessor(4096, 2, 2);
      node.onaudioprocess = (event) => {
        captureBlock(event, 2, instrumentalBlocksRef, 0);
      };
      node.connect(captureSilentGainRef.current);
      instrumentCaptureRef.current = node;
      applyAudioRouting(songTransposeRef.current !== 0);
    }
    if (!vocalCaptureRef.current) {
      const node = context.createScriptProcessor(4096, 1, 1);
      node.onaudioprocess = (event) => {
        captureBlock(event, 1, vocalBlocksRef, captureLatencyRef.current);
      };
      node.connect(captureSilentGainRef.current);
      vocalCaptureRef.current = node;
      micSourceRef.current?.connect(node);
    }
    return context;
  }, [applyAudioRouting, ensureAudioGraph]);

  const seekTo = useCallback((time: number, clear = false) => {
    const audio = audioRef.current;
    if (!audio) return;
    const nextTime = clamp(time, 0, Math.max(0, audio.duration || duration));
    audio.currentTime = clamp(nextTime + outputLatencyRef.current, 0, Math.max(0, audio.duration || duration));
    const guideAudio = guideAudioRef.current;
    if (guideAudio && Number.isFinite(guideAudio.duration)) {
      guideAudio.currentTime = clamp(audio.currentTime, 0, guideAudio.duration);
    }
    currentTimeRef.current = nextTime;
    setCurrentTime(nextTime);
    lastScoreTimeRef.current = nextTime;
    if (clear) resetPerformance(nextTime);
  }, [duration, resetPerformance]);

  const restart = useCallback(() => {
    seekTo(startAudioTime(), true);
  }, [seekTo, startAudioTime]);

  // Scrub relative to the live playhead, keeping the accumulated score (a skip
  // is navigation, not a fresh take).
  const skipBy = useCallback((delta: number) => {
    if (recordingActiveRef.current) {
      recordingActiveRef.current = false;
      audioRef.current?.pause();
      guideAudioRef.current?.pause();
      setRecordingStatus("paused");
    }
    seekTo(currentTimeRef.current + delta, false);
  }, [seekTo]);

  const stopMicrophone = useCallback(() => {
    micRequestGenerationRef.current += 1;
    if (micNodeRef.current) {
      micNodeRef.current.port.onmessage = null;
      micNodeRef.current.port.close();
      micNodeRef.current.disconnect();
    }
    micSourceRef.current?.disconnect();
    micSilentGainRef.current?.disconnect();
    micStreamRef.current?.getTracks().forEach((track) => {
      track.onended = null;
      track.stop();
    });
    micNodeRef.current = null;
    micSourceRef.current = null;
    micSilentGainRef.current = null;
    micStreamRef.current = null;
    livePitchRef.current = null;
    livePitchSampleTimeRef.current = null;
    micClockOriginRef.current = null;
    adaptiveFilterRef.current.reset();
  }, []);

  const startMicrophone = useCallback(async (forceOn = false): Promise<boolean> => {
    if (inputMode === "microphone") {
      if (forceOn) return true;
      autoMicRef.current = false;
      stopMicrophone();
      setInputMode("demo");
      setMicError("");
      return false;
    }
    const generation = ++micRequestGenerationRef.current;
    setMicBusy(true);
    setMicError("");
    try {
      const context = await ensureAudioGraph();
      refreshLatencies();
      if (generation !== micRequestGenerationRef.current) return false;
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Microphone access is not supported in this browser.");
      if (micModuleContextRef.current !== context) {
        await context.audioWorklet.addModule("/pitch-worklet.js?v=mic-fix-2");
        micModuleContextRef.current = context;
      }
      if (generation !== micRequestGenerationRef.current) return false;
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          // All three route the mic through the browser's voice-call
          // processing, which gates and ducks a real condenser mic and adds
          // its own delay. The backing track carries no lead vocal, so there
          // is nothing to cancel: take the raw signal.
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
          // `latency` is in the Media Capture spec but not the DOM lib types.
          latency: { ideal: 0 },
        } as MediaTrackConstraints & { latency?: ConstrainDouble },
      });
      if (generation !== micRequestGenerationRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return false;
      }
      micStreamRef.current = stream;
      const track = stream.getAudioTracks()[0];
      const reportedCaptureLatency = (
        track?.getSettings() as (MediaTrackSettings & { latency?: number }) | undefined
      )?.latency;
      captureLatencyRef.current =
        Number.isFinite(reportedCaptureLatency) && (reportedCaptureLatency as number) > 0
          ? (reportedCaptureLatency as number)
          : DEFAULT_CAPTURE_LATENCY_S;
      refreshLatencies();
      const source = context.createMediaStreamSource(stream);
      micSourceRef.current = source;
      const node = new AudioWorkletNode(context, "sona-pitch-processor", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
      });
      micNodeRef.current = node;
      const silentGain = context.createGain();
      micSilentGainRef.current = silentGain;
      silentGain.gain.value = 0;
      source.connect(node);
      if (vocalCaptureRef.current) source.connect(vocalCaptureRef.current);
      node.connect(silentGain);
      silentGain.connect(context.destination);
      adaptiveFilterRef.current.reset();
      node.port.onmessage = (event: MessageEvent) => {
        const data = event.data as {
          type: string;
          f0?: number | null;
          clarity?: number;
          rms?: number;
          t?: number;
          analysisLatency?: number;
          contextTimeOrigin?: number;
        };
        if (data.type === "init") {
          if (Number.isFinite(data.analysisLatency)) {
            analysisLatencyRef.current = data.analysisLatency as number;
            refreshLatencies();
          }
          return;
        }
        if (data.type === "clock") {
          if (Number.isFinite(data.contextTimeOrigin)) {
            micClockOriginRef.current = data.contextTimeOrigin as number;
          }
          return;
        }
        if (data.type !== "pitch") return;
        const rms = data.rms ?? 0;
        const clarity = data.clarity ?? 0;
        const filtered = adaptiveFilterRef.current.process({
          f0: data.f0 ?? null,
          clarity,
          rms,
          timeMs: (data.t ?? context.currentTime) * 1000,
        });
        livePitchRef.current = filtered.midi;
        livePitchSampleTimeRef.current =
          micClockOriginRef.current !== null && Number.isFinite(data.t)
            ? micClockOriginRef.current + (data.t as number) - FILTER_MEDIAN_LAG_S
            : null;
        livePitchEventRef.current += 1;
      };
      if (track) {
        track.onended = () => {
          if (micStreamRef.current !== stream) return;
          if (recordingActiveRef.current) {
            recordingActiveRef.current = false;
            audioRef.current?.pause();
            guideAudioRef.current?.pause();
            setRecordingStatus("paused");
            setRecordingError("Microphone disconnected; the take was paused.");
          }
          stopMicrophone();
          setInputMode("demo");
          setMicError("Microphone disconnected; demo input is active.");
        };
      }
      setInputMode("microphone");
      return true;
    } catch (error) {
      if (generation !== micRequestGenerationRef.current) return false;
      autoMicRef.current = false;
      stopMicrophone();
      setInputMode("demo");
      setMicError(error instanceof Error ? error.message : "Microphone access could not start.");
      setMicBusy(false);
      return false;
    } finally {
      if (generation === micRequestGenerationRef.current) setMicBusy(false);
    }
  }, [ensureAudioGraph, inputMode, refreshLatencies, stopMicrophone]);

  const setSongKey = useCallback((value: number) => {
    const next = clamp(Math.round(value), TRANSPOSE_MIN, TRANSPOSE_MAX);
    songTransposeRef.current = next;
    setSongTranspose(next);
    if (transpositionsLinked) {
      contourTransposeRef.current = next;
      setContourTranspose(next);
    }
  }, [transpositionsLinked]);

  const setContourKey = useCallback((value: number) => {
    const next = clamp(Math.round(value), TRANSPOSE_MIN, TRANSPOSE_MAX);
    contourTransposeRef.current = next;
    setContourTranspose(next);
    if (transpositionsLinked) {
      songTransposeRef.current = next;
      setSongTranspose(next);
    }
  }, [transpositionsLinked]);

  const toggleKeyLink = useCallback(() => {
    if (transpositionsLinked) {
      setTranspositionsLinked(false);
      return;
    }
    contourTransposeRef.current = songTranspose;
    setContourTranspose(songTranspose);
    setTranspositionsLinked(true);
  }, [songTranspose, transpositionsLinked]);

  const pauseRecording = useCallback(() => {
    recordingActiveRef.current = false;
    audioRef.current?.pause();
    guideAudioRef.current?.pause();
    if (recordingStatus === "recording") setRecordingStatus("paused");
  }, [recordingStatus]);

  const beginRecording = useCallback(async (reset = false) => {
    const audio = audioRef.current;
    const guideAudio = guideAudioRef.current;
    if (!audio || !guideAudio) return;
    setRecordingError("");
    const micReady = inputMode === "microphone" || await startMicrophone(true);
    if (!micReady || !micStreamRef.current) {
      setRecordingError("Microphone access is required to record a take.");
      return;
    }
    try {
      await ensureRecordingCapture();
      clearRecordingFiles();
      if (reset) {
        instrumentalBlocksRef.current = [];
        vocalBlocksRef.current = [];
        setRecordedUntil(0);
        seekTo(0, true);
      }
      recordingActiveRef.current = true;
      setRecordingStatus("recording");
      applyAudioRouting(songTransposeRef.current !== 0);
      if (guideVocals) {
        if (guideAudio.readyState < HTMLMediaElement.HAVE_METADATA) {
          await new Promise<void>((resolve, reject) => {
            guideAudio.addEventListener("loadedmetadata", () => resolve(), { once: true });
            guideAudio.addEventListener("error", () => reject(new Error("Guide vocals could not load.")), { once: true });
          });
        }
        guideAudio.currentTime = clamp(
          audio.currentTime,
          0,
          Math.max(0, guideAudio.duration || duration),
        );
        await Promise.all([audio.play(), guideAudio.play()]);
      } else {
        guideAudio.pause();
        await audio.play();
      }
    } catch (error) {
      recordingActiveRef.current = false;
      setRecordingStatus("paused");
      setRecordingError(error instanceof Error ? error.message : "Recording could not start.");
    }
  }, [
    clearRecordingFiles,
    ensureRecordingCapture,
    guideVocals,
    inputMode,
    applyAudioRouting,
    duration,
    seekTo,
    startMicrophone,
  ]);

  const restartRecording = useCallback(async () => {
    recordingActiveRef.current = false;
    audioRef.current?.pause();
    guideAudioRef.current?.pause();
    instrumentalBlocksRef.current = [];
    vocalBlocksRef.current = [];
    setRecordedUntil(0);
    clearRecordingFiles();
    seekTo(0, true);
    setRecordingStatus("idle");
    await beginRecording(false);
  }, [beginRecording, clearRecordingFiles, seekTo]);

  const rewindRecording = useCallback(() => {
    recordingActiveRef.current = false;
    audioRef.current?.pause();
    guideAudioRef.current?.pause();
    const cutoff = clamp(currentTimeRef.current, 0, duration);
    instrumentalBlocksRef.current = trimPcmBlocks(instrumentalBlocksRef.current, cutoff);
    vocalBlocksRef.current = trimPcmBlocks(vocalBlocksRef.current, cutoff);
    setRecordedUntil((current) => Math.min(current, cutoff));
    clearRecordingFiles();
    setRecordingStatus("paused");
    seekTo(cutoff, false);
    setRecordingError("");
  }, [clearRecordingFiles, duration, seekTo]);

  const endRecording = useCallback(() => {
    recordingActiveRef.current = false;
    audioRef.current?.pause();
    guideAudioRef.current?.pause();
    setRecordingStatus("processing");
    setRecordingError("");
    const context = audioContextRef.current;
    const sampleRate = context?.sampleRate ?? 48000;
    const blocks = [...instrumentalBlocksRef.current, ...vocalBlocksRef.current];
    const exportDuration = blocks.reduce(
      (maximum, block) => Math.max(maximum, pcmBlockEnd(block)),
      0,
    );
    if (exportDuration <= 0) {
      setRecordingStatus("idle");
      setRecordingError("Record some audio before ending the take.");
      return;
    }
    clearRecordingFiles();
    const instrumental = renderPcmTrack(
      instrumentalBlocksRef.current,
      2,
      sampleRate,
      exportDuration,
    );
    const vocals = renderPcmTrack(vocalBlocksRef.current, 1, sampleRate, exportDuration);
    const combined = mixPcm(instrumental, vocals);
    const files: RecordingFiles = {
      vocals: URL.createObjectURL(encodeWav(vocals, sampleRate)),
      instrumental: URL.createObjectURL(encodeWav(instrumental, sampleRate)),
      combined: URL.createObjectURL(encodeWav(combined, sampleRate)),
      duration: exportDuration,
      songTranspose: songTransposeRef.current,
    };
    recordingFilesRef.current = files;
    setRecordingFiles(files);
    setRecordedUntil(exportDuration);
    setRecordingStatus("complete");
    // Object URLs now own the encoded WAV data; release the much larger
    // floating-point working buffers. A completed take is intentionally final.
    instrumentalBlocksRef.current = [];
    vocalBlocksRef.current = [];
  }, [clearRecordingFiles]);

  const switchStudioMode = useCallback((nextMode: StudioMode) => {
    if (nextMode === studioModeRef.current) return;
    recordingActiveRef.current = false;
    audioRef.current?.pause();
    guideAudioRef.current?.pause();
    if (recordingStatus === "recording") setRecordingStatus("paused");
    selectionAnchorRef.current = null;
    selectionRef.current = null;
    setSelecting(false);
    setSelection(null);
    setEditNotice("");
    setRecordingError("");
    studioModeRef.current = nextMode;
    if (nextMode === "editing") {
      guideVocalsRef.current = true;
      setGuideVocals(true);
    }
    if (nextMode === "recording") {
      guideVocalsRef.current = false;
      setGuideVocals(false);
    }
    setStudioMode(nextMode);
    applyAudioRouting(songTransposeRef.current !== 0);
  }, [applyAudioRouting, recordingStatus]);

  const selectionTimeAt = useCallback((clientX: number) => {
    const lane = laneRef.current;
    if (!lane) return 0;
    const rect = lane.getBoundingClientRect();
    const pixelsPerSecond = rect.width / (PAST_SECONDS + FUTURE_SECONDS);
    const playheadX = rect.width * PLAYHEAD_FRACTION;
    const contourNow = currentTimeRef.current * (contourScaleRef.current || 1);
    return clamp(
      contourNow + (clientX - rect.left - playheadX) / pixelsPerSecond,
      0,
      melodyRef.current.duration,
    );
  }, []);

  const beginContourSelection = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (studioModeRef.current !== "editing" || event.button !== 0) return;
    event.preventDefault();
    audioRef.current?.pause();
    const time = selectionTimeAt(event.clientX);
    selectionAnchorRef.current = time;
    const next = { t0: time, t1: time };
    selectionRef.current = next;
    setSelection(next);
    setSelecting(true);
    event.currentTarget.setPointerCapture(event.pointerId);
  }, [selectionTimeAt]);

  const moveContourSelection = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const anchor = selectionAnchorRef.current;
    if (studioModeRef.current !== "editing" || anchor === null) return;
    const next = normalizeSelection({ t0: anchor, t1: selectionTimeAt(event.clientX) });
    selectionRef.current = next;
    setSelection(next);
  }, [selectionTimeAt]);

  const finishContourSelection = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const anchor = selectionAnchorRef.current;
    if (anchor === null) return;
    const next = normalizeSelection({ t0: anchor, t1: selectionTimeAt(event.clientX) });
    const minimum = melodyRef.current.hop;
    const finalSelection =
      next.t1 - next.t0 < minimum
        ? { t0: Math.max(0, next.t0 - minimum), t1: Math.min(melodyRef.current.duration, next.t1 + minimum) }
        : next;
    selectionAnchorRef.current = null;
    selectionRef.current = finalSelection;
    setSelection(finalSelection);
    setSelecting(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }, [selectionTimeAt]);

  const applyContourEdit = useCallback((action: "remove" | "smooth" | "repair") => {
    const selected = selectionRef.current;
    if (!selected) return;
    if (action === "repair" && !repairSource) {
      setEditNotice("Lead pYIN is not available for this song.");
      return;
    }
    setEditedContour((current) => editContour(current, selected, action, repairSource));
    setSelection(null);
    selectionRef.current = null;
    resetPerformance(currentTimeRef.current);
    setEditNotice(
      action === "remove"
        ? "Selection removed."
        : action === "smooth"
          ? "Selection smoothed from its neighboring notes."
          : "Selection replaced directly from lead pYIN.",
    );
  }, [repairSource, resetPerformance]);

  const saveEditedDefault = useCallback(() => {
    try {
      window.localStorage.setItem(EDIT_STORAGE_PREFIX + song.id, JSON.stringify(editedContour));
      setSavedContour(editedContour);
      setEditNotice("Edited contour saved as the default on this device.");
    } catch {
      setEditNotice("This browser could not save the edited contour.");
    }
  }, [editedContour, song.id]);

  const restoreOriginalContour = useCallback(() => {
    try {
      window.localStorage.removeItem(EDIT_STORAGE_PREFIX + song.id);
    } catch {
      // The shipped contour can still be restored for this session.
    }
    setEditedContour(song.contour);
    setSavedContour(song.contour);
    setSelection(null);
    selectionRef.current = null;
    resetPerformance(currentTimeRef.current);
    setEditNotice("Original shipped contour restored.");
  }, [resetPerformance, song]);

  const togglePlayback = useCallback(async () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (studioModeRef.current === "recording") {
      if (recordingActiveRef.current) pauseRecording();
      else await beginRecording(false);
      return;
    }
    try {
      await ensureAudioGraph();
      if (audio.paused) {
        if (inputMode === "demo" && autoMicRef.current) void startMicrophone();
        await audio.play();
        setPlaying(true);
      } else {
        audio.pause();
        setPlaying(false);
      }
    } catch (error) {
      setMicError(error instanceof Error ? error.message : "Playback could not start.");
    }
  }, [beginRecording, ensureAudioGraph, inputMode, pauseRecording, startMicrophone]);

  const scrubTo = useCallback((time: number) => {
    if (recordingActiveRef.current) {
      recordingActiveRef.current = false;
      audioRef.current?.pause();
      guideAudioRef.current?.pause();
      setRecordingStatus("paused");
    }
    seekTo(time, studioModeRef.current !== "recording");
  }, [seekTo]);

  // Transport keyboard shortcuts: Space toggles playback, ←/→ scrub 5 s. Skip
  // when a form control has focus so the keys still drive sliders and inputs.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || target?.isContentEditable) return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.code === "Space") {
        event.preventDefault();
        void togglePlayback();
      } else if (event.code === "ArrowRight") {
        event.preventDefault();
        skipBy(5);
      } else if (event.code === "ArrowLeft") {
        event.preventDefault();
        skipBy(-5);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [skipBy, togglePlayback]);

  useEffect(() => {
    const lane = laneRef.current;
    const canvas = canvasRef.current;
    if (!lane || !canvas) return;
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      laneSizeRef.current = { width, height };
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(width * ratio));
      canvas.height = Math.max(1, Math.round(height * ratio));
    });
    observer.observe(lane);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    songTransposeRef.current = songTranspose;
    const shifter = soundTouchRef.current;
    const guideShifter = guideSoundTouchRef.current;
    if (shifter) {
      shifter.pitchSemitones.setTargetAtTime(
        songTranspose,
        audioContextRef.current?.currentTime ?? 0,
        0.025,
      );
    }
    if (guideShifter) {
      guideShifter.pitchSemitones.setTargetAtTime(
        songTranspose,
        audioContextRef.current?.currentTime ?? 0,
        0.025,
      );
    }
    applyAudioRouting(songTranspose !== 0);
  }, [applyAudioRouting, songTranspose]);

  useEffect(() => {
    contourTransposeRef.current = contourTranspose;
    resetPerformance(currentTimeRef.current);
  }, [contourTranspose, resetPerformance]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.pause();
    guideAudioRef.current?.pause();
    setPlaying(false);
    audio.src = primaryGuideVocals ? song.audio : song.instrumental;
    audio.load();
    contourScaleRef.current = 1;
    const startTime = startAudioTime();
    currentTimeRef.current = startTime;
    setCurrentTime(startTime);
    setDuration(song.contour.duration);
    resetPerformance(startTime);
  }, [primaryGuideVocals, resetPerformance, song, startAudioTime]);

  useEffect(() => {
    guideVocalsRef.current = guideVocals;
    applyAudioRouting(songTransposeRef.current !== 0);
    const audio = audioRef.current;
    const guideAudio = guideAudioRef.current;
    if (
      studioMode === "recording" &&
      guideVocals &&
      audio &&
      guideAudio &&
      !audio.paused
    ) {
      if (guideAudio.readyState >= HTMLMediaElement.HAVE_METADATA) {
        guideAudio.currentTime = clamp(
          audio.currentTime,
          0,
          Math.max(0, guideAudio.duration || duration),
        );
      }
      void guideAudio.play().catch(() => {
        setRecordingError("Guide vocals could not start; the instrumental is still recording.");
        guideVocalsRef.current = false;
        setGuideVocals(false);
      });
    } else {
      guideAudio?.pause();
    }
  }, [applyAudioRouting, duration, guideVocals, studioMode]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const handleLoaded = () => {
      // The contour was measured on the vocal stem; this scale keeps tiny
      // decoder/stem duration differences from accumulating as drift.
      contourScaleRef.current =
        audio.duration > 0 ? melodyRef.current.duration / audio.duration : 1;
      const startTime = startAudioTime();
      audio.currentTime = clamp(startTime + outputLatencyRef.current, 0, audio.duration);
      currentTimeRef.current = startTime;
      setCurrentTime(startTime);
      setDuration(audio.duration);
    };
    const handleEnded = () => {
      setPlaying(false);
      guideAudioRef.current?.pause();
      if (recordingActiveRef.current) {
        recordingActiveRef.current = false;
        setRecordingStatus("paused");
      }
    };
    const handlePlay = () => setPlaying(true);
    const handlePause = () => setPlaying(false);
    audio.addEventListener("loadedmetadata", handleLoaded);
    audio.addEventListener("ended", handleEnded);
    audio.addEventListener("play", handlePlay);
    audio.addEventListener("pause", handlePause);
    return () => {
      audio.removeEventListener("loadedmetadata", handleLoaded);
      audio.removeEventListener("ended", handleEnded);
      audio.removeEventListener("play", handlePlay);
      audio.removeEventListener("pause", handlePause);
    };
  }, [song, startAudioTime]);

  const drawLane = useCallback(
    (contourNow: number, nowMs: number, detected: number | null) => {
      const canvas = canvasRef.current;
      const context = canvas?.getContext("2d");
      if (!canvas || !context) return;
      const { width, height } = laneSizeRef.current;
      if (width < 10 || height < 10) return;
      const ratio = window.devicePixelRatio || 1;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);

      const data = melodyRef.current;
      const shift = contourTransposeRef.current;
      const mode = studioModeRef.current;
      const palette =
        mode === "editing"
          ? {
              outer: "rgba(142, 74, 74, 0.18)",
              band: "rgba(188, 102, 102, 0.5)",
              line: "rgba(239, 154, 154, 0.8)",
              active: "rgba(239, 154, 154, 0.96)",
              bright: "#ffc5c5",
              glow: "rgba(239, 154, 154, 0.82)",
            }
          : mode === "recording"
            ? {
                outer: "rgba(66, 111, 145, 0.18)",
                band: "rgba(104, 164, 207, 0.5)",
                line: "rgba(134, 197, 244, 0.8)",
                active: "rgba(134, 197, 244, 0.96)",
                bright: "#c5e9ff",
                glow: "rgba(134, 197, 244, 0.82)",
              }
            : {
                outer: "rgba(122, 106, 48, 0.18)",
                band: "rgba(122, 106, 48, 0.55)",
                line: "rgba(244, 200, 77, 0.55)",
                active: "rgba(244, 200, 77, 0.95)",
                bright: "#ffe47c",
                glow: "rgba(244, 200, 77, 0.9)",
              };
      const laneLow = data.low + shift - 1.5;
      const laneHigh = data.high + shift + 1.5;
      const span = laneHigh - laneLow;
      const pixelsPerSemitone = height / span;
      const pixelsPerSecond = width / (PAST_SECONDS + FUTURE_SECONDS);
      const playheadX = width * PLAYHEAD_FRACTION;
      const windowStart = contourNow - PAST_SECONDS - 0.3;
      const windowEnd = contourNow + FUTURE_SECONDS + 0.3;
      const yOf = (midi: number) => height * (1 - (midi - laneLow) / span);
      const xOf = (time: number) => playheadX + (time - contourNow) * pixelsPerSecond;
      const bandWidth = clamp(2 * INNER_BAND * pixelsPerSemitone, 6, 30);
      const outerWidth = clamp(2 * OUTER_BAND * pixelsPerSemitone, bandWidth + 4, 60);

      // Pitch grid: a line at every whole semitone.
      const rowHeight = pixelsPerSemitone;
      for (let midi = Math.ceil(laneLow); midi <= Math.floor(laneHigh); midi += 1) {
        const y = yOf(midi);
        const pitchClass = ((midi % 12) + 12) % 12;
        const isC = pitchClass === 0;
        const isNatural = NATURAL_PITCH_CLASSES.has(pitchClass);
        context.strokeStyle = isC
          ? "rgba(244, 242, 232, 0.1)"
          : isNatural
            ? "rgba(244, 242, 232, 0.05)"
            : "rgba(244, 242, 232, 0.028)";
        context.lineWidth = 1;
        context.beginPath();
        context.moveTo(AXIS_WIDTH, y);
        context.lineTo(width, y);
        context.stroke();
      }

      // Time grid, one line per second.
      context.strokeStyle = "rgba(244, 242, 232, 0.035)";
      for (let second = Math.ceil(windowStart); second <= windowEnd; second += 1) {
        const x = xOf(second);
        if (x < AXIS_WIDTH) continue;
        context.beginPath();
        context.moveTo(x, 0);
        context.lineTo(x, height);
        context.stroke();
      }

      // Everything that scrolls stays right of the axis gutter.
      context.save();
      context.beginPath();
      context.rect(AXIS_WIDTH, 0, Math.max(0, width - AXIS_WIDTH), height);
      context.clip();

      // One pass of the wave between two contour times, as a stroked path.
      const strokeWave = (
        segment: Segment,
        from: number,
        to: number,
        style: string,
        lineWidth: number,
        glow = 0,
      ) => {
        const first = Math.max(0, Math.floor((from - segment.t0) / data.hop));
        const last = Math.min(
          segment.midi.length - 1,
          Math.ceil((to - segment.t0) / data.hop),
        );
        if (last < first) return;
        context.strokeStyle = style;
        context.lineWidth = lineWidth;
        context.lineCap = "round";
        context.lineJoin = "round";
        context.shadowBlur = glow;
        context.shadowColor = glow ? palette.glow : "transparent";
        context.beginPath();
        for (let index = first; index <= last; index += 1) {
          const x = xOf(segment.t0 + index * data.hop);
          const y = yOf(segment.midi[index] + shift);
          if (index === first) context.moveTo(x, y);
          else context.lineTo(x, y);
        }
        if (first === last) context.lineTo(xOf(segment.t0 + first * data.hop) + 0.6, yOf(segment.midi[first] + shift));
        context.stroke();
        context.shadowBlur = 0;
      };

      for (let segmentIndex = 0; segmentIndex < data.segments.length; segmentIndex += 1) {
        const segment = data.segments[segmentIndex];
        if (segment.t1 < windowStart || segment.t0 > windowEnd) continue;

        // Partial-credit outer band (±1 st) ahead of the playhead, then the
        // inner corridor, then the elapsed part grayed out over it.
        if (segment.t1 > contourNow) {
          strokeWave(segment, Math.max(contourNow, windowStart), windowEnd, palette.outer, outerWidth);
        }
        strokeWave(segment, windowStart, windowEnd, palette.band, bandWidth);
        strokeWave(segment, windowStart, windowEnd, palette.line, 1.6);
        if (segment.t0 < contourNow) {
          strokeWave(segment, windowStart, Math.min(contourNow, segment.t1), "rgba(56, 58, 53, 0.92)", bandWidth);
          strokeWave(segment, windowStart, Math.min(contourNow, segment.t1), "rgba(128, 131, 122, 0.55)", 1.4);
        }

        // Sections sung inside the inner band ignite yellow.
        const intervals = hitIntervalsRef.current.get(segmentIndex);
        if (intervals) {
          for (const interval of intervals) {
            if (interval.t1 < windowStart || interval.t0 > windowEnd) continue;
            strokeWave(segment, interval.t0, interval.t1, palette.active, bandWidth, 13);
            strokeWave(segment, interval.t0, interval.t1, palette.bright, 2, 0);
          }
        }
      }

      const selected = selectionRef.current;
      if (mode === "editing" && selected) {
        const normalized = normalizeSelection(selected);
        const left = Math.max(AXIS_WIDTH, xOf(normalized.t0));
        const right = Math.min(width, xOf(normalized.t1));
        if (right >= AXIS_WIDTH && left <= width && right > left) {
          context.fillStyle = "rgba(239, 154, 154, 0.12)";
          context.fillRect(left, 0, right - left, height);
          context.strokeStyle = "rgba(255, 197, 197, 0.88)";
          context.lineWidth = 1.5;
          context.setLineDash([5, 4]);
          context.strokeRect(left + 0.75, 0.75, Math.max(0, right - left - 1.5), height - 1.5);
          context.setLineDash([]);
        }
      }

      // Playhead.
      context.strokeStyle = "rgba(244, 242, 232, 0.65)";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(playheadX, 0);
      context.lineTo(playheadX, height);
      context.stroke();

      // The singer's trace.
      const trail = trailRef.current;
      if (trail.length) {
        context.strokeStyle = "rgba(246, 242, 222, 0.9)";
        context.lineWidth = 2.2;
        context.lineCap = "round";
        context.lineJoin = "round";
        context.beginPath();
        let previousTime = Number.NEGATIVE_INFINITY;
        let previousMidi = Number.NaN;
        for (const point of trail) {
          if (point.t < windowStart || point.t > windowEnd) continue;
          const x = xOf(point.t);
          const y = clamp(yOf(point.midi), -20, height + 20);
          // Break the line on silence gaps and on octave-scale jumps so the
          // trace never draws tall vertical connectors.
          if (point.t - previousTime > 0.12 || Math.abs(point.midi - previousMidi) > 5) {
            context.moveTo(x, y);
          } else context.lineTo(x, y);
          previousTime = point.t;
          previousMidi = point.midi;
        }
        context.stroke();
      }

      if (detected !== null) {
        // The dot rides the playhead: a dot drawn at the (older) voice time
        // sits visibly behind the notes and reads as lag. The trail keeps the
        // true timing, so history still lines up with the wave it was sung
        // against.
        const dotX = playheadX;
        const y = clamp(yOf(detected), 6, height - 6);
        const inside = hitActiveRef.current;
        if (inside) {
          const pulse = 10 + Math.sin(nowMs / 110) * 2.5;
          context.fillStyle = palette.band;
          context.beginPath();
          context.arc(dotX, y, pulse, 0, Math.PI * 2);
          context.fill();
        }
        context.fillStyle = inside ? palette.bright : "#b9bcb2";
        context.strokeStyle = inside ? "#fff8d6" : "#f8f5e9";
        context.lineWidth = 2.4;
        context.shadowBlur = inside ? 16 : 0;
        context.shadowColor = palette.glow;
        context.beginPath();
        context.arc(dotX, y, 6, 0, Math.PI * 2);
        context.fill();
        context.stroke();
        context.shadowBlur = 0;
      }

      context.restore();

      // Axis labels last, so the scrolling wave never covers them. They ride
      // just left of the playhead and thin out only when rows get too tight,
      // keeping C and naturals longest.
      const fontSize = rowHeight < 11 ? 9 : rowHeight < 14 ? 10 : 12;
      context.font = `500 ${fontSize}px ui-sans-serif, system-ui, sans-serif`;
      context.textBaseline = "middle";
      context.textAlign = "right";
      const labelX = playheadX - 6;
      for (let midi = Math.ceil(laneLow); midi <= Math.floor(laneHigh); midi += 1) {
        const pitchClass = ((midi % 12) + 12) % 12;
        const isC = pitchClass === 0;
        const isNatural = NATURAL_PITCH_CLASSES.has(pitchClass);
        if (!(rowHeight >= 9 || (rowHeight >= 6 && isNatural) || isC)) continue;
        context.fillStyle = isC
          ? "rgba(190, 193, 183, 0.95)"
          : isNatural
            ? "rgba(152, 155, 146, 0.78)"
            : "rgba(132, 135, 127, 0.6)";
        context.fillText(displayMidiName(midi), labelX, yOf(midi));
      }
      context.textAlign = "left";
    },
    [],
  );

  useEffect(() => {
    const animate = (nowMs: number) => {
      const audio = audioRef.current;
      // Freeze the clock while paused; audio.currentTime is only meaningful
      // as a song position when the element is actually running.
      const audioTime =
        playing && audio && Number.isFinite(audio.currentTime)
          ? Math.max(0, audio.currentTime - outputLatencyRef.current)
          : currentTimeRef.current;
      const guideAudio = guideAudioRef.current;
      if (
        playing &&
        audio &&
        guideAudio &&
        studioModeRef.current === "recording" &&
        guideVocalsRef.current &&
        Math.abs(guideAudio.currentTime - audio.currentTime) > 0.12
      ) {
        guideAudio.currentTime = clamp(
          audio.currentTime,
          0,
          Math.max(0, guideAudio.duration || audio.duration),
        );
      }
      currentTimeRef.current = audioTime;
      const contourScale = contourScaleRef.current || 1;
      const contourNow = audioTime * contourScale;

      // A pitch arriving now was sung micLatency ago, against the note that
      // was sounding then. Score and place it at that moment, not at the
      // playhead, or every fast phrase reads as flat-then-sharp. The user's
      // saved nudge trims the unmeasurable device-specific capture latency.
      let voiceNow = contourNow;
      if (inputMode === "microphone") {
        const context = audioContextRef.current;
        const sampleTime = livePitchSampleTimeRef.current;
        const measuredAge =
          context && sampleTime !== null
            ? context.currentTime - sampleTime + captureLatencyRef.current
            : micLatencyRef.current;
        voiceNow = Math.max(
          0,
          contourNow - Math.max(0, measuredAge + micNudgeSecondsRef.current) * contourScale,
        );
      }
      const segments = melodyRef.current.segments;
      const segmentIndex = segmentIndexAt(segments, voiceNow);
      const rawTarget = targetMidiAt(voiceNow);
      const target =
        rawTarget === null ? null : rawTarget + contourTransposeRef.current;

      let detected = livePitchRef.current;
      if (inputMode === "demo") {
        // Demo is a literal contour preview, so its trace should stay exactly
        // on the interpolated target rather than adding synthetic misses.
        detected = target;
        livePitchRef.current = detected;
      }

      const distance = detected !== null && target !== null ? Math.abs(detected - target) : null;
      if (distance !== null) {
        hitActiveRef.current = hitActiveRef.current
          ? distance <= INNER_RELEASE
          : distance <= INNER_BAND;
      } else hitActiveRef.current = false;

      if (playing && segmentIndex >= 0 && target !== null) {
        const elapsed = clamp(audioTime - lastScoreTimeRef.current, 0, 0.08);
        if (elapsed > 0) {
          const totals = performanceRef.current;
          const credit = detected === null ? 0 : creditForDistance(Math.abs(detected - target));
          totals.targetSeconds += elapsed;
          totals.earnedSeconds += credit * elapsed;
          totals.segmentTarget += elapsed;
          totals.segmentEarned += credit * elapsed;
          if (detected !== null && hitActiveRef.current) {
            const intervals = hitIntervalsRef.current.get(segmentIndex) ?? [];
            const last = intervals[intervals.length - 1];
            if (last && voiceNow - last.t1 <= 0.09 && voiceNow >= last.t0) {
              last.t1 = voiceNow;
            } else {
              intervals.push({ t0: voiceNow, t1: voiceNow + melodyRef.current.hop });
            }
            hitIntervalsRef.current.set(segmentIndex, intervals);
          }
        }
      }
      lastScoreTimeRef.current = audioTime;

      if (scoringSegmentRef.current !== segmentIndex) {
        if (playing) finalizeScoringSegment(scoringSegmentRef.current);
        scoringSegmentRef.current = segmentIndex;
      }

      if (
        playing &&
        detected !== null &&
        (inputMode !== "microphone" || drawnPitchEventRef.current !== livePitchEventRef.current)
      ) {
        trailRef.current.push({ t: voiceNow, midi: detected, inTune: hitActiveRef.current });
        drawnPitchEventRef.current = livePitchEventRef.current;
      }
      if (trailRef.current.length) {
        trailRef.current = trailRef.current
          .filter((item) => item.t >= contourNow - TRAIL_SECONDS && item.t <= contourNow + 0.05)
          .slice(-220);
      }

      drawLane(contourNow, nowMs, detected);

      if (nowMs - lastUiUpdateRef.current > UI_UPDATE_MS) {
        lastUiUpdateRef.current = nowMs;
        setCurrentTime(audioTime);
        setLivePitch(detected);
        setTargetPitch(rawTarget);
        setWaveNearby(
          segments.some(
            (segment) =>
              segment.t1 >= contourNow - PAST_SECONDS && segment.t0 <= contourNow + FUTURE_SECONDS,
          ),
        );
        updateScore();
      }
      rafRef.current = requestAnimationFrame(animate);
    };
    rafRef.current = requestAnimationFrame(animate);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [drawLane, finalizeScoringSegment, inputMode, playing, targetMidiAt, updateScore]);

  useEffect(() => () => {
    recordingActiveRef.current = false;
    const files = recordingFilesRef.current;
    if (files) {
      URL.revokeObjectURL(files.vocals);
      URL.revokeObjectURL(files.instrumental);
      URL.revokeObjectURL(files.combined);
    }
    stopMicrophone();
    guideAudioRef.current?.pause();
    instrumentCaptureRef.current?.disconnect();
    vocalCaptureRef.current?.disconnect();
    captureSilentGainRef.current?.disconnect();
    guideMediaSourceRef.current?.disconnect();
    guideSoundTouchRef.current?.disconnect();
    audioContextRef.current?.close();
  }, [stopMicrophone]);

  const hasUnsavedEdits = !sameContour(editedContour, savedContour);
  const hasSavedDefault = !sameContour(savedContour, song.contour);
  const selectedRange = selection ? normalizeSelection(selection) : null;
  const takeLocksSongKey =
    studioMode === "recording" &&
    recordedUntil > 0 &&
    recordingStatus !== "complete";
  const modeLabel =
    studioMode === "editing"
      ? "Editing contour"
      : studioMode === "recording"
        ? "Recording take"
        : "Now practicing";

  return (
    <main className={`appShell studioShell mode-${studioMode}`}>
      <audio ref={audioRef} preload="metadata" aria-label={`${song.title} backing track`} />
      <audio ref={guideAudioRef} src={song.audio} preload="metadata" aria-hidden="true" />
      <header className="topbar">
        <a className="brand" href="#studio" aria-label="Sona home">
          so<span>na</span>
        </a>
        <nav className="studioModeSwitch" aria-label="Studio mode">
          {(["editing", "practicing", "recording"] as StudioMode[]).map((mode) => (
            <button
              key={mode}
              type="button"
              className={studioMode === mode ? "isActive" : ""}
              aria-pressed={studioMode === mode}
              onClick={() => switchStudioMode(mode)}
            >
              {mode[0].toUpperCase() + mode.slice(1)}
            </button>
          ))}
        </nav>
      </header>

      <section className="songStrip" aria-label="Current song">
        <button className="backButton" type="button" onClick={onBack}>
          <span className="backArrow" aria-hidden="true" />
          Library
        </button>
        <div className="albumDisc" aria-hidden="true"><span>♪</span></div>
        <div className="songIdentity">
          <span className="eyebrow">{modeLabel}</span>
          <strong className="songTitle">{song.title}</strong>
          <span className="songArtist">{song.artist}</span>
        </div>
        <div className="heroStats" aria-label="Session metrics">
          <div className="heroStat"><span>Score</span><strong>{score === null ? "—" : score.toLocaleString("en-US")}{score !== null && <small>/10k</small>}</strong></div>
          <div className="heroStat"><span>Accuracy</span><strong>{accuracy === null ? "—" : accuracy}{accuracy !== null && <small>% · {rankForAccuracy(accuracy)}</small>}</strong></div>
        </div>
      </section>

      <div className="studioLayout" id="studio">
        <section className="performanceCard">
          <div className="phraseHeader">
            <h1 className="srOnly">{song.title} by {song.artist}</h1>
            <div className="phraseMeta">
              <span>{targetDisplay === null ? "Listening for the next phrase" : `Target ${displayMidiName(targetDisplay)}`}</span>
              <span>{livePitch === null ? "Mic —" : `Mic ${displayMidiName(livePitch)}`}</span>
              <span className={isInside ? "insideLabel isInside" : "insideLabel"}>
                <i aria-hidden="true" />
                {isInside
                  ? "On pitch"
                  : pitchOffset === null
                    ? "Awaiting voice"
                    : `${pitchOffset > 0 ? "+" : ""}${Math.round(pitchOffset * 100)}¢ from center`}
              </span>
            </div>
          </div>

          <div
            className="pitchLane"
            ref={laneRef}
            role={studioMode === "editing" ? "region" : "img"}
            onPointerDown={beginContourSelection}
            onPointerMove={moveContourSelection}
            onPointerUp={finishContourSelection}
            onPointerCancel={finishContourSelection}
            aria-label={
              studioMode === "editing"
                ? `Editable pitch contour for ${song.title}. Drag horizontally to select a region.`
                : `Scrolling pitch wave for ${song.title}, transcribed from the vocal audio. The elapsed wave fades gray; accurately sung inner-band sections glow yellow.`
            }
          >
            <canvas className="pitchCanvas" ref={canvasRef} aria-hidden="true" />
            {!waveNearby && <div className="countIn">The next phrase is approaching</div>}
            {studioMode === "editing" && selectedRange && !selecting && (
              <div
                className="editPopover"
                role="toolbar"
                aria-label={`Edit contour from ${formatTime(selectedRange.t0)} to ${formatTime(selectedRange.t1)}`}
                onPointerDown={(event) => event.stopPropagation()}
              >
                <span>{formatTime(selectedRange.t0)}–{formatTime(selectedRange.t1)}</span>
                <button type="button" onClick={() => applyContourEdit("remove")}>Remove</button>
                <button type="button" onClick={() => applyContourEdit("smooth")}>Smooth</button>
                <button
                  type="button"
                  onClick={() => applyContourEdit("repair")}
                  disabled={repairSourceState !== "ready"}
                  title={
                    repairSourceState === "ready"
                      ? "Replace this region directly from lead pYIN"
                      : "Lead pYIN is not available yet"
                  }
                >
                  Repair vocals
                </button>
              </div>
            )}
          </div>

          <div className="transportBar">
            <div className="scrubRow">
              <span className="timecode">{formatTime(currentTime)}</span>
              <input
                className="timeline"
                type="range"
                min={0}
                max={Math.max(duration, 1)}
                step={0.01}
                value={clamp(currentTime, 0, Math.max(duration, 1))}
                onChange={(event) => scrubTo(Number(event.target.value))}
                aria-label="Song position"
                style={{ "--timeline-progress": `${clamp(currentTime / Math.max(duration, 1), 0, 1) * 100}%` } as React.CSSProperties}
              />
              <span className="timecode">{formatTime(duration)}</span>
            </div>
            <div className="controlRow">
              <span className="controlEdge" aria-hidden="true" />
              <div className="controlCenter">
                <button className="skipButton" type="button" onClick={() => skipBy(-5)} title="Back 5 seconds (←)" aria-label="Back 5 seconds">
                  <span className="skipGlyph back" aria-hidden="true" />
                  <span className="skipNum">5s</span>
                </button>
                <button
                  className="playButton"
                  type="button"
                  onClick={togglePlayback}
                  title={studioMode === "recording" ? "Record / Pause (Space)" : "Play / Pause (Space)"}
                  aria-label={
                    studioMode === "recording"
                      ? recordingStatus === "recording"
                        ? "Pause recording"
                        : "Start or resume recording"
                      : playing
                        ? "Pause song"
                        : "Play song"
                  }
                >
                  <span className={playing ? "pauseIcon" : "playIcon"} aria-hidden="true" />
                </button>
                <button className="skipButton" type="button" onClick={() => skipBy(5)} title="Forward 5 seconds (→)" aria-label="Forward 5 seconds">
                  <span className="skipNum">5s</span>
                  <span className="skipGlyph fwd" aria-hidden="true" />
                </button>
              </div>
              <div className="controlEdge controlRight">
                {studioMode !== "recording" && (
                  <button className="restartButton" type="button" onClick={restart}>Restart</button>
                )}
                {studioMode === "practicing" && (
                  <button
                    className={`micButton ${inputMode === "microphone" ? "isOn" : ""}`}
                    type="button"
                    onClick={() => void startMicrophone()}
                    disabled={micBusy}
                    aria-pressed={inputMode === "microphone"}
                  >
                    <span className="micGlyph" aria-hidden="true" />
                    {micBusy ? "Connecting…" : inputMode === "microphone" ? "Disable mic" : "Enable mic"}
                  </button>
                )}
              </div>
            </div>
            <p className="controlHints">
              <kbd>Space</kbd> {studioMode === "recording" ? "record/pause" : "play/pause"} · <kbd>←</kbd> <kbd>→</kbd> skip 5s
            </p>
          </div>
          {micError && <p className="errorMessage" role="alert">{micError}</p>}
        </section>

        <aside className="coachRail" aria-label={`${studioMode} controls`}>
          {studioMode === "editing" && (
            <section className="railCard modeActionCard">
              <div className="cardHeading">
                <div><span className="eyebrow">Contour editor</span><h2>Shape the target</h2></div>
                <span className="modeStatusDot" aria-hidden="true" />
              </div>
              <p className="rangeNote">
                Drag across the pitch lane, then remove, smooth, or repair the selected region.
                Repair copies lead pYIN exactly, including its empty frames.
              </p>
              <div className="editSaveActions">
                <button
                  className="primaryRailButton"
                  type="button"
                  onClick={saveEditedDefault}
                  disabled={!hasUnsavedEdits}
                >
                  Save as new default
                </button>
                <button
                  className="secondaryRailButton"
                  type="button"
                  onClick={restoreOriginalContour}
                  disabled={!hasSavedDefault && !hasUnsavedEdits}
                >
                  Restore original
                </button>
              </div>
              <p className="railStatus" aria-live="polite">
                {editNotice ||
                  (repairSourceState === "loading"
                    ? "Loading lead pYIN repair source…"
                    : repairSourceState === "missing"
                      ? "Lead pYIN repair is unavailable for this song."
                      : hasUnsavedEdits
                        ? "Unsaved contour changes."
                        : hasSavedDefault
                          ? "Your edited default is active."
                          : "The shipped contour is active.")}
              </p>
            </section>
          )}

          {studioMode === "recording" && (
            <section className="railCard modeActionCard recordingCard">
              <div className="cardHeading">
                <div><span className="eyebrow">Mic recorder</span><h2>Capture a take</h2></div>
                <span className={`recordingIndicator is-${recordingStatus}`} aria-hidden="true" />
              </div>
              <p className="rangeNote">
                The exported music follows Song key. Contour key only changes the target you sing.
              </p>
              <div className="recordingReadout">
                <strong>
                  {recordingStatus === "recording"
                    ? "Recording"
                    : recordingStatus === "paused"
                      ? "Take paused"
                      : recordingStatus === "processing"
                        ? "Preparing files"
                        : recordingStatus === "complete"
                          ? "Take complete"
                          : "Ready"}
                </strong>
                <span>{formatTime(recordedUntil)}</span>
              </div>
              <div className="recordingActions">
                <button
                  className="primaryRailButton"
                  type="button"
                  onClick={() =>
                    recordingStatus === "recording"
                      ? pauseRecording()
                      : recordingStatus === "complete"
                        ? void restartRecording()
                      : void beginRecording(false)
                  }
                  disabled={recordingStatus === "processing" || micBusy}
                >
                  {micBusy
                    ? "Connecting mic…"
                    : recordingStatus === "recording"
                      ? "Pause recording"
                      : recordingStatus === "complete"
                        ? "Start new take"
                      : recordedUntil > 0
                        ? "Resume recording"
                        : "Start recording"}
                </button>
                <button
                  className="secondaryRailButton"
                  type="button"
                  onClick={() => void restartRecording()}
                  disabled={recordingStatus === "processing"}
                >
                  Restart from beginning
                </button>
                <button
                  className="secondaryRailButton"
                  type="button"
                  onClick={rewindRecording}
                  disabled={recordedUntil <= 0 || recordingStatus === "processing" || recordingStatus === "complete"}
                >
                  Rewind to playhead
                </button>
                <button
                  className="secondaryRailButton endTakeButton"
                  type="button"
                  onClick={endRecording}
                  disabled={recordedUntil <= 0 || recordingStatus === "processing" || recordingStatus === "complete"}
                >
                  End recording
                </button>
              </div>
              <p className="railStatus" aria-live="polite">
                {recordingError ||
                  (recordedUntil > 0
                    ? "Move the playhead, then rewind to discard everything after it."
                    : "Restart deletes the current take and records again from 0:00.")}
              </p>
              {recordingFiles && (
                <div className="recordingDownloads" aria-label="Recording downloads">
                  <a href={recordingFiles.vocals} download={`${song.id}-raw-vocals.wav`}>Raw vocals</a>
                  <a href={recordingFiles.instrumental} download={`${song.id}-instrumental-${formatSigned(recordingFiles.songTranspose)}st.wav`}>Instrumental</a>
                  <a href={recordingFiles.combined} download={`${song.id}-combined-${formatSigned(recordingFiles.songTranspose)}st.wav`}>Combined mix</a>
                </div>
              )}
            </section>
          )}

          <section className="railCard transposeCard dualTransposeCard">
            <div className="cardHeading transposeHeading">
              <div><span className="eyebrow">Transpose & fit</span><h2>Music and target keys</h2></div>
              <button
                className={`keyLinkButton ${transpositionsLinked ? "isLinked" : ""}`}
                type="button"
                aria-pressed={transpositionsLinked}
                onClick={toggleKeyLink}
                disabled={takeLocksSongKey}
                title={transpositionsLinked ? "Unlock song and contour keys" : "Link song and contour keys"}
              >
                <span className="lockGlyph" aria-hidden="true" />
                {transpositionsLinked ? "Linked" : "Independent"}
              </button>
            </div>

            <div className="transposeControl">
              <label htmlFor="songTransposeSlider">
                <span>Song key</span>
                <output>{formatSigned(songTranspose)} st</output>
              </label>
              <input
                id="songTransposeSlider"
                className="transposeSlider"
                type="range"
                min={TRANSPOSE_MIN}
                max={TRANSPOSE_MAX}
                step={1}
                value={songTranspose}
                onChange={(event) => setSongKey(Number(event.target.value))}
                disabled={takeLocksSongKey}
                aria-label="Transpose song audio in semitones"
                style={{ "--transpose-progress": `${((songTranspose - TRANSPOSE_MIN) / (TRANSPOSE_MAX - TRANSPOSE_MIN)) * 100}%` } as React.CSSProperties}
              />
            </div>

            <div className="transposeControl">
              <label htmlFor="contourTransposeSlider">
                <span>Contour key</span>
                <output>{formatSigned(contourTranspose)} st</output>
              </label>
              <input
                id="contourTransposeSlider"
                className="transposeSlider"
                type="range"
                min={TRANSPOSE_MIN}
                max={TRANSPOSE_MAX}
                step={1}
                value={contourTranspose}
                onChange={(event) => setContourKey(Number(event.target.value))}
                disabled={takeLocksSongKey && transpositionsLinked}
                aria-label="Transpose target contour in semitones"
                style={{ "--transpose-progress": `${((contourTranspose - TRANSPOSE_MIN) / (TRANSPOSE_MAX - TRANSPOSE_MIN)) * 100}%` } as React.CSSProperties}
              />
            </div>
            <div className="sliderTicks"><span>−16</span><span>Original key</span><span>+16</span></div>
            <div className="rangeResult">
              <strong>{displayMidiName(transposedLow)} — {displayMidiName(transposedHigh)}</strong>
              <span className="voiceBadge">{voiceFit.approximateLabel} fit</span>
            </div>
            <p className="rangeNote">
              Song key changes playback and exports. Contour key changes targets, labels, and scoring only.
            </p>
          </section>

          {(studioMode === "practicing" || studioMode === "recording") && (
            <>
              <section className="railCard signalCard">
                <div className="cardHeading signalHeading">
                  <div>
                    <h2>Guide vocals</h2>
                    {studioMode === "recording" && (
                      <p className="rangeNote">
                        Monitoring only—the exported backing track remains instrumental.
                      </p>
                    )}
                  </div>
                  <label className="switch">
                    <input
                      type="checkbox"
                      checked={guideVocals}
                      onChange={(event) => setGuideVocals(event.target.checked)}
                      aria-label="Play the original recording with its lead vocal"
                    />
                    <span aria-hidden="true" />
                  </label>
                </div>
              </section>

              <section className="railCard transposeCard">
                <div className="cardHeading">
                  <div><span className="eyebrow">Mic timing</span><h2>Sync your voice</h2></div>
                  <output className="transposeValue" htmlFor="micNudgeSlider">{formatSigned(micNudgeMs)} ms</output>
                </div>
                <input
                  id="micNudgeSlider"
                  className="transposeSlider"
                  type="range"
                  min={MIC_NUDGE_MIN_MS}
                  max={MIC_NUDGE_MAX_MS}
                  step={5}
                  value={micNudgeMs}
                  onChange={(event) => setMicNudge(Number(event.target.value))}
                  aria-label="Microphone timing adjustment in milliseconds"
                  style={{
                    "--transpose-progress": `${((micNudgeMs - MIC_NUDGE_MIN_MS) / (MIC_NUDGE_MAX_MS - MIC_NUDGE_MIN_MS)) * 100}%`,
                  } as React.CSSProperties}
                />
                <p className="rangeNote">
                  If your trace lands after the wave even when you sing on time, raise this
                  until they line up. Saved on this device.
                </p>
              </section>
            </>
          )}
        </aside>
      </div>
    </main>
  );
}
