"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { hzToMidi, midiToName, type VoiceLabel } from "./pitch-utils";
import {
  exercisesForModule,
  TRAINING_EXERCISES,
  TRAINING_MODULES,
  type TrainingExercise,
  type TrackingProfile,
} from "./training-data";

type View = { kind: "home" } | { kind: "module"; moduleId: number } | { kind: "exercise"; exercise: TrainingExercise };
type MicFrame = { midi: number | null; rms: number; clarity: number; at: number };
type SavedProgress = Record<string, { best: number; takes: number }>;

const PROGRESS_KEY = "sona.training.progress.v1";
const PROFILE_COPY: Record<TrackingProfile, string> = {
  standard: "Balanced tracking",
  sovt: "SOVT · permissive",
  quiet: "Quiet signal boost",
  agility: "Fast-note detail",
  unvoiced: "Airflow envelope",
  guided: "Guided only",
};

function moduleFor(id: number) {
  return TRAINING_MODULES.find((module) => module.id === id) ?? TRAINING_MODULES[0];
}

function playPianoNote(context: AudioContext, midi: number, at: number, duration = .55, gain = .13) {
  const sources: AudioScheduledSourceNode[] = [];
  const frequency = 440 * 2 ** ((midi - 69) / 12);
  const tone = context.createBiquadFilter();
  tone.type = "highshelf";
  tone.frequency.value = 1800;
  tone.gain.value = 4.5;
  const master = context.createGain();
  master.gain.setValueAtTime(.0001, at);
  master.gain.exponentialRampToValueAtTime(gain, at + .004);
  master.gain.exponentialRampToValueAtTime(gain * .42, at + .075);
  master.gain.exponentialRampToValueAtTime(gain * .16, at + Math.min(.42, duration * .58));
  master.gain.exponentialRampToValueAtTime(.0001, at + Math.max(.28, duration + .26));
  master.connect(tone).connect(context.destination);
  [
    { ratio: 1, level: 1, type: "triangle" as OscillatorType, detune: -2 },
    { ratio: 1.003, level: .48, type: "sine" as OscillatorType, detune: 2 },
    { ratio: 2.006, level: .34, type: "sine" as OscillatorType, detune: 0 },
    { ratio: 3.018, level: .17, type: "sine" as OscillatorType, detune: 0 },
    { ratio: 4.04, level: .09, type: "sine" as OscillatorType, detune: 0 },
    { ratio: 5.08, level: .045, type: "sine" as OscillatorType, detune: 0 },
  ].forEach(({ ratio, level, type, detune }) => {
    const oscillator = context.createOscillator();
    const voice = context.createGain();
    oscillator.type = type;
    oscillator.frequency.value = frequency * ratio;
    oscillator.detune.value = detune;
    voice.gain.value = level;
    oscillator.connect(voice).connect(master);
    oscillator.start(at);
    oscillator.stop(at + Math.max(.3, duration) + .32);
    sources.push(oscillator);
  });

  const hammerLength = Math.max(1, Math.floor(context.sampleRate * .018));
  const hammerBuffer = context.createBuffer(1, hammerLength, context.sampleRate);
  const hammerData = hammerBuffer.getChannelData(0);
  for (let index = 0; index < hammerLength; index += 1) {
    hammerData[index] = (Math.random() * 2 - 1) * (1 - index / hammerLength);
  }
  const hammer = context.createBufferSource();
  const hammerFilter = context.createBiquadFilter();
  const hammerGain = context.createGain();
  hammer.buffer = hammerBuffer;
  hammerFilter.type = "bandpass";
  hammerFilter.frequency.value = Math.min(4200, 1800 + frequency * 3);
  hammerFilter.Q.value = .7;
  hammerGain.gain.value = gain * .28;
  hammer.connect(hammerFilter).connect(hammerGain).connect(context.destination);
  hammer.start(at);
  sources.push(hammer);
  return sources;
}

const ROUND_COUNT = 5;
const PREP_SECONDS = 3.2;
const PLAYHEAD_FRACTION = 2.6 / 7;
const SCROLL_PIXELS_PER_SECOND = 92;
const VOICE_TYPE_KEY = "sona.training.voiceType.v1";

const VOICE_PROFILES: ReadonlyArray<{ label: VoiceLabel; low: number; high: number }> = [
  { label: "Bass", low: 40, high: 64 },
  { label: "Baritone", low: 43, high: 67 },
  { label: "Tenor", low: 48, high: 72 },
  { label: "Alto/Contralto", low: 53, high: 77 },
  { label: "Mezzo-soprano", low: 57, high: 81 },
  { label: "Soprano", low: 60, high: 84 },
];

function pitchRiseFor(exercise: TrainingExercise) {
  if (!exercise.pattern.length) return 0;
  if (exercise.kind === "pitch") return 1;
  if (exercise.kind === "glide" || exercise.kind === "sustain") return .5;
  return 0;
}

function pitchForRound(baseMidi: number, risePerRound: number, round: number) {
  return baseMidi + Math.round(risePerRound * round);
}

function startingMidiForVoice(
  exercise: TrainingExercise,
  voice: { low: number; high: number },
  risePerRound: number,
) {
  if (!exercise.pattern.length) return Math.round(voice.low + (voice.high - voice.low) * .55);
  const patternLow = Math.min(...exercise.pattern);
  const patternHigh = Math.max(...exercise.pattern);
  const patternCenter = (patternLow + patternHigh) / 2;
  const placement = exercise.module === 5
    ? .72
    : exercise.module === 2
      ? .66
      : exercise.module === 6
        ? .56
        : exercise.kind === "sustain"
          ? .58
          : .57;
  const desired = Math.round(voice.low + (voice.high - voice.low) * placement - patternCenter);
  const lowestSafeStart = voice.low + 1 - patternLow;
  const highestSafeStart = voice.high - 1 - patternHigh - risePerRound * (ROUND_COUNT - 1);
  return Math.max(lowestSafeStart, Math.min(highestSafeStart, desired));
}

function soundInstruction(exercise: TrainingExercise) {
  if (exercise.id === "box-breathing") return "Follow the breath phases silently—no vocal sound.";
  if (exercise.id === "jaw-release") return "Move gently and breathe normally—no singing yet.";
  if (exercise.kind === "breath") return `Make a quiet, unvoiced “${exercise.syllable}” with no pitch.`;
  if (exercise.id === "panting") return "Make short, unvoiced “HA” air pulses—do not sing the note.";
  if (exercise.id === "lip-trill") return "Bubble the lips on “BRR” and follow the full piano shape.";
  if (exercise.id === "tongue-trill") return "Roll a continuous “RRR” and follow the piano shape.";
  if (exercise.id === "humming-glides") return "Hum a closed-mouth “MM”; keep the buzz at the lips.";
  if (exercise.profile === "quiet") return `Use a very soft “${exercise.syllable}”; match the shape, not the volume.`;
  if (exercise.kind === "glide") return `Glide continuously on “${exercise.syllable}”—do not step between notes.`;
  if (exercise.kind === "sustain") return `Hold “${exercise.syllable}” on one steady, connected pitch.`;
  return `Sing “${exercise.syllable}” on every piano note.`;
}

function Header({ section, onNavigate }: { section: "songs" | "training"; onNavigate: (section: "songs" | "training") => void }) {
  return (
    <header className="trainingTopbar">
      <button type="button" className="trainingBrand" onClick={() => onNavigate("songs")} aria-label="Sona home">so<span>na</span></button>
      <nav className="mainNav" aria-label="Main navigation">
        <button type="button" className={section === "songs" ? "active" : ""} onClick={() => onNavigate("songs")}>Songs</button>
        <button type="button" className={section === "training" ? "active" : ""} onClick={() => onNavigate("training")}>Training</button>
      </nav>
      <span className="topbarSpacer" aria-hidden="true" />
    </header>
  );
}

function ModuleCard({ moduleId, progress, onOpen }: { moduleId: number; progress: SavedProgress; onOpen: () => void }) {
  const trainingModule = moduleFor(moduleId);
  const exercises = exercisesForModule(moduleId);
  const completed = exercises.filter((exercise) => (progress[exercise.id]?.takes ?? 0) > 0).length;
  return (
    <button type="button" className={`trainingModuleCard module-${trainingModule.color}`} onClick={onOpen}>
      <span className="moduleNumber">{String(moduleId).padStart(2, "0")}</span>
      <span className="moduleProgress"><i style={{ width: `${(completed / exercises.length) * 100}%` }} /></span>
      <span className="moduleCardCopy">
        <strong>{trainingModule.title}</strong>
        <small>{trainingModule.kicker}</small>
      </span>
      <span className="moduleMeta">{completed}/{exercises.length}<i>→</i></span>
    </button>
  );
}

function TrainingHome({ progress, onOpenModule, onOpenExercise }: { progress: SavedProgress; onOpenModule: (id: number) => void; onOpenExercise: (exercise: TrainingExercise) => void }) {
  const tried = Object.values(progress).filter((item) => item.takes > 0).length;
  const recommended = TRAINING_EXERCISES.find((exercise) => !progress[exercise.id]?.takes) ?? TRAINING_EXERCISES[13];
  return (
    <main className="trainingPage">
      <section className="trainingHero">
        <div>
          <span className="trainingEyebrow">Vocal practice</span>
          <h1>Train the voice.<br /><em>Trust the feeling.</em></h1>
          <p>Purposeful exercises, a responsive piano, and feedback that helps without getting in the way.</p>
        </div>
        <button type="button" className="continuePractice" onClick={() => onOpenExercise(recommended)}>
          <span className="continueIcon">▶</span>
          <span><small>Up next · {PROFILE_COPY[recommended.profile]}</small><strong>{recommended.shortTitle}</strong><em>{recommended.syllable} · {recommended.bpm} BPM</em></span>
          <i>Begin</i>
        </button>
      </section>

      <section className="trainingOverview">
        <div className="trainingSectionHead">
          <div><span className="trainingEyebrow">Curriculum</span><h2>Build your instrument</h2></div>
          <p><strong>{tried}</strong> of 26 exercises practiced</p>
        </div>
        <div className="moduleGrid">
          {TRAINING_MODULES.map((module) => <ModuleCard key={module.id} moduleId={module.id} progress={progress} onOpen={() => onOpenModule(module.id)} />)}
        </div>
      </section>

      <section className="trainingPrinciple">
        <span>Practice principle</span>
        <blockquote>“Easy and accurate beats loud and impressive.”</blockquote>
        <p>Stop when the voice feels tired. These tools measure the sound—not the health of your voice.</p>
      </section>
    </main>
  );
}

function ModuleView({ moduleId, progress, onBack, onOpen }: { moduleId: number; progress: SavedProgress; onBack: () => void; onOpen: (exercise: TrainingExercise) => void }) {
  const trainingModule = moduleFor(moduleId);
  const exercises = exercisesForModule(moduleId);
  return (
    <main className="trainingPage modulePage">
      <button type="button" className="trainingBack" onClick={onBack}>← All modules</button>
      <section className={`moduleBanner module-${trainingModule.color}`}>
        <span className="moduleNumber">{String(trainingModule.id).padStart(2, "0")}</span>
        <div><span className="trainingEyebrow">Module {trainingModule.id} · {exercises.length} exercises</span><h1>{trainingModule.title}</h1><p>{trainingModule.kicker}. Start comfortably, repeat with attention, and leave a little in reserve.</p></div>
      </section>
      <div className="exerciseList">
        {exercises.map((exercise, index) => {
          const record = progress[exercise.id];
          return (
            <button type="button" className="exerciseRow" key={exercise.id} onClick={() => onOpen(exercise)}>
              <span className="exerciseIndex">{String(index + 1).padStart(2, "0")}</span>
              <span className="exerciseCopy"><strong>{exercise.title}</strong><small>{exercise.description}</small></span>
              <span className="exerciseRecord">{record ? <><strong>{record.best}%</strong><small>best</small></> : <small>Not started</small>}</span>
              <span className="exerciseArrow">→</span>
            </button>
          );
        })}
      </div>
    </main>
  );
}

function ExercisePlayer({ exercise, onBack, onComplete }: { exercise: TrainingExercise; onBack: () => void; onComplete: (score: number) => void }) {
  const [voiceType, setVoiceType] = useState<VoiceLabel>("Tenor");
  const [bpm, setBpm] = useState(exercise.bpm);
  const [micState, setMicState] = useState<"off" | "starting" | "on">("off");
  const [micError, setMicError] = useState("");
  const [running, setRunning] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [live, setLive] = useState<MicFrame>({ midi: null, rms: 0, clarity: 0, at: 0 });
  const [score, setScore] = useState<number | null>(null);
  const [rmsHistory, setRmsHistory] = useState<number[]>([]);
  const [recordingUrl, setRecordingUrl] = useState("");
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const nodeRef = useRef<AudioWorkletNode | null>(null);
  const filterHistoryRef = useRef<number[]>([]);
  const samplesRef = useRef<Array<MicFrame & { target: number | null }>>([]);
  const startRef = useRef(0);
  const timerRef = useRef<number | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const liveRef = useRef(live);
  const recordingUrlRef = useRef("");
  const scheduledSourcesRef = useRef<AudioScheduledSourceNode[]>([]);

  const secondsPerBeat = 60 / bpm;
  const noteDurations = exercise.beats.map((beat) => beat * secondsPerBeat);
  const patternDuration = exercise.kind === "guided" || exercise.kind === "breath"
    ? Math.max(8, (exercise.guidedSeconds ?? 20) / (exercise.kind === "guided" ? 4 : 1))
    : noteDurations.reduce((sum, value) => sum + value, 0);
  const roundDuration = PREP_SECONDS + patternDuration;
  const sessionDuration = roundDuration * ROUND_COUNT;
  const risePerRound = useMemo(() => pitchRiseFor(exercise), [exercise]);
  const voiceProfile = useMemo(
    () => VOICE_PROFILES.find((profile) => profile.label === voiceType) ?? VOICE_PROFILES[2],
    [voiceType],
  );
  const targetMidi = useMemo(
    () => startingMidiForVoice(exercise, voiceProfile, risePerRound),
    [exercise, risePerRound, voiceProfile],
  );

  const timelineNotes = useMemo(() => {
    const notes: Array<{ round: number; index: number; start: number; duration: number; midi: number }> = [];
    for (let round = 0; round < ROUND_COUNT; round += 1) {
      let cursor = round * roundDuration + PREP_SECONDS;
      exercise.pattern.forEach((offset, index) => {
        const duration = noteDurations[index] ?? secondsPerBeat;
        notes.push({ round, index, start: cursor, duration, midi: pitchForRound(targetMidi, risePerRound, round) + offset });
        cursor += duration;
      });
    }
    return notes;
  }, [exercise.pattern, noteDurations, risePerRound, roundDuration, secondsPerBeat, targetMidi]);

  const sessionAt = useCallback((time: number) => {
    const safeTime = Math.max(0, Math.min(time, Math.max(0, sessionDuration - .001)));
    const round = Math.min(ROUND_COUNT - 1, Math.floor(safeTime / roundDuration));
    const withinRound = safeTime - round * roundDuration;
    const preparing = withinRound < PREP_SECONDS;
    return {
      round,
      withinRound,
      preparing,
      activeTime: Math.max(0, withinRound - PREP_SECONDS),
      roundPitch: pitchForRound(targetMidi, risePerRound, round),
    };
  }, [risePerRound, roundDuration, sessionDuration, targetMidi]);

  const stopMic = useCallback(() => {
    nodeRef.current?.disconnect();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    nodeRef.current = null;
    streamRef.current = null;
    setMicState("off");
  }, []);

  useEffect(() => { liveRef.current = live; }, [live]);
  useEffect(() => { recordingUrlRef.current = recordingUrl; }, [recordingUrl]);
  useEffect(() => {
    const restore = window.setTimeout(() => {
      try {
        const saved = localStorage.getItem(VOICE_TYPE_KEY) as VoiceLabel | null;
        if (saved && VOICE_PROFILES.some((profile) => profile.label === saved)) setVoiceType(saved);
      } catch {}
    }, 0);
    return () => window.clearTimeout(restore);
  }, []);
  useEffect(() => () => {
    if (timerRef.current) window.clearInterval(timerRef.current);
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
    nodeRef.current?.disconnect();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    scheduledSourcesRef.current.forEach((source) => { try { source.stop(); } catch {} });
    if (recordingUrlRef.current) URL.revokeObjectURL(recordingUrlRef.current);
  }, []);

  const ensureContext = useCallback(async () => {
    let context = audioContextRef.current;
    if (!context) {
      context = new AudioContext({ latencyHint: "interactive" });
      audioContextRef.current = context;
    }
    if (context.state === "suspended") await context.resume();
    return context;
  }, []);

  const startMic = useCallback(async () => {
    if (micState === "on") { stopMic(); return; }
    setMicState("starting");
    setMicError("");
    try {
      const context = await ensureContext();
      await context.audioWorklet.addModule("/pitch-worklet.js?v=training-1");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation:false, noiseSuppression:false, autoGainControl:false, channelCount:1 } });
      const source = context.createMediaStreamSource(stream);
      const node = new AudioWorkletNode(context, "sona-pitch-processor", { numberOfInputs:1, numberOfOutputs:1, outputChannelCount:[1] });
      const silent = context.createGain();
      silent.gain.value = 0;
      source.connect(node).connect(silent).connect(context.destination);
      streamRef.current = stream;
      nodeRef.current = node;
      node.port.onmessage = (event: MessageEvent) => {
        const data = event.data as { type: string; f0?: number | null; clarity?: number; rms?: number };
        if (data.type !== "pitch") return;
        const raw = data.f0 ? hzToMidi(data.f0) : null;
        const clarity = data.clarity ?? 0;
        const rms = data.rms ?? 0;
        const threshold = exercise.profile === "sovt" || exercise.profile === "quiet" ? .28 : exercise.profile === "agility" ? .42 : .36;
        let midi = raw !== null && clarity >= threshold ? raw : null;
        if (midi !== null && exercise.profile !== "agility") {
          const history = filterHistoryRef.current;
          history.push(midi);
          const windowSize = exercise.profile === "sovt" || exercise.profile === "quiet" ? 7 : 3;
          if (history.length > windowSize) history.shift();
          midi = [...history].sort((a,b) => a-b)[Math.floor(history.length / 2)];
        }
        setLive({ midi, rms, clarity, at: performance.now() });
        if (exercise.kind === "breath" || exercise.profile === "unvoiced") {
          setRmsHistory((prev) => {
            const next = [...prev, rms];
            return next.length > 80 ? next.slice(-80) : next;
          });
        }
      };
      setMicState("on");
    } catch (error) {
      setMicError(error instanceof Error ? error.message : "Microphone could not start.");
      setMicState("off");
    }
  }, [ensureContext, exercise.kind, exercise.profile, micState, stopMic]);

  const targetAt = useCallback((time: number) => {
    if (!exercise.pattern.length) return null;
    const position = sessionAt(time);
    if (position.preparing) return null;
    let cursor = 0;
    for (let index = 0; index < exercise.pattern.length; index += 1) {
      cursor += noteDurations[index] ?? secondsPerBeat;
      if (position.activeTime <= cursor) return position.roundPitch + exercise.pattern[index];
    }
    return null;
  }, [exercise.pattern, noteDurations, secondsPerBeat, sessionAt]);

  const finish = useCallback((completed = true) => {
    if (timerRef.current) window.clearInterval(timerRef.current);
    timerRef.current = null;
    setRunning(false);
    scheduledSourcesRef.current.forEach((source) => { try { source.stop(); } catch {} });
    scheduledSourcesRef.current = [];
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
    if (!completed) {
      setScore(null);
      return;
    }
    const samples = samplesRef.current;
    let nextScore = 100;
    if (exercise.kind === "guided") nextScore = 100;
    else if (exercise.kind === "breath" || exercise.profile === "unvoiced") {
      const voiced = samples.map((sample) => sample.rms).filter((value) => value > .001);
      const mean = voiced.reduce((sum, value) => sum + value, 0) / Math.max(1, voiced.length);
      const variance = voiced.reduce((sum, value) => sum + (value - mean) ** 2, 0) / Math.max(1, voiced.length);
      const steadiness = mean ? Math.max(0, 1 - Math.sqrt(variance) / mean) : 0;
      const coverage = voiced.length / Math.max(1, samples.length);
      nextScore = Math.round(100 * (.65 * steadiness + .35 * coverage));
    } else {
      const pitched = samples.filter((sample) => sample.midi !== null && sample.target !== null);
      const credit = pitched.reduce((sum, sample) => {
        const cents = Math.abs((sample.midi! - sample.target!) * 100);
        return sum + Math.max(0, 1 - cents / (exercise.profile === "agility" ? 70 : 90));
      }, 0);
      const coverage = pitched.length / Math.max(1, samples.length);
      nextScore = Math.round(100 * (credit / Math.max(1, pitched.length)) * Math.min(1, coverage / .65));
    }
    setScore(nextScore);
    onComplete(nextScore);
  }, [exercise.kind, exercise.profile, onComplete]);

  const start = useCallback(async () => {
    if (running) { finish(false); return; }
    const context = await ensureContext();
    setHasStarted(true);
    scheduledSourcesRef.current.forEach((source) => { try { source.stop(); } catch {} });
    scheduledSourcesRef.current = [];
    setScore(null);
    setElapsed(0);
    samplesRef.current = [];
    setRmsHistory([]);
    filterHistoryRef.current = [];
    if (recordingUrl) { URL.revokeObjectURL(recordingUrl); setRecordingUrl(""); }
    if (streamRef.current && typeof MediaRecorder !== "undefined") {
      chunksRef.current = [];
      const recorder = new MediaRecorder(streamRef.current);
      recorderRef.current = recorder;
      recorder.ondataavailable = (event) => event.data.size && chunksRef.current.push(event.data);
      recorder.onstop = () => setRecordingUrl(URL.createObjectURL(new Blob(chunksRef.current, { type: recorder.mimeType })));
      recorder.start();
    }
    const now = context.currentTime + .05;
    for (let round = 0; round < ROUND_COUNT; round += 1) {
      const activeStart = round * roundDuration + PREP_SECONDS;
      const roundPitch = pitchForRound(targetMidi, risePerRound, round);
      if (exercise.pattern.length) {
        const chordRoot = roundPitch + exercise.pattern[0];
        const chordAt = now + activeStart - .92;
        scheduledSourcesRef.current.push(...playPianoNote(context, chordRoot, chordAt, .62, .075));
        scheduledSourcesRef.current.push(...playPianoNote(context, chordRoot + 4, chordAt, .62, .052));
        scheduledSourcesRef.current.push(...playPianoNote(context, chordRoot + 7, chordAt, .62, .044));
      }
      if (exercise.pattern.length) {
        let cursor = activeStart;
        exercise.pattern.forEach((offset, index) => {
          const duration = noteDurations[index] ?? secondsPerBeat;
          scheduledSourcesRef.current.push(...playPianoNote(context, roundPitch + offset, now + cursor, Math.min(duration * .9, 1), .145));
          cursor += duration;
        });
      } else if (exercise.kind === "sustain") {
        scheduledSourcesRef.current.push(...playPianoNote(context, roundPitch, now + activeStart, 1.1, .145));
      }
    }
    startRef.current = performance.now();
    setRunning(true);
    timerRef.current = window.setInterval(() => {
      const value = (performance.now() - startRef.current) / 1000;
      setElapsed(Math.min(value, sessionDuration));
      const position = sessionAt(value);
      if (!position.preparing) {
        const target = targetAt(value);
        samplesRef.current.push({ ...liveRef.current, target });
      }
      if (value >= sessionDuration) finish(true);
    }, 70);
  }, [ensureContext, exercise.kind, exercise.pattern, finish, noteDurations, recordingUrl, risePerRound, roundDuration, running, secondsPerBeat, sessionAt, sessionDuration, targetAt, targetMidi]);

  const currentTarget = targetAt(elapsed);
  const cents = live.midi !== null && currentTarget !== null ? Math.round((live.midi - currentTarget) * 100) : null;
  const sessionPosition = sessionAt(elapsed);
  const progress = Math.min(100, (elapsed / sessionDuration) * 100);
  const roundProgress = Math.min(100, (sessionPosition.withinRound / roundDuration) * 100);
  const allPitches = timelineNotes.map((note) => note.midi);
  const pitchLow = allPitches.length ? Math.min(...allPitches) : targetMidi;
  const pitchHigh = allPitches.length ? Math.max(...allPitches) : targetMidi + 1;
  const pitchY = (midi: number) => pitchHigh === pitchLow ? 50 : 17 + ((midi - pitchLow) / (pitchHigh - pitchLow)) * 66;
  const isRhythm = exercise.kind === "rhythm";
  const tracksPitch = exercise.kind === "pitch"
    || exercise.kind === "glide"
    || exercise.kind === "sustain";

  const isBoxBreathing = exercise.id === "box-breathing";
  const isBreathExercise = exercise.kind === "breath";
  const guidePhase = isBoxBreathing
    ? ["Inhale", "Suspend", "Exhale", "Rest"][Math.floor(sessionPosition.activeTime / 4) % 4]
    : exercise.id === "jaw-release"
      ? ["Massage the jaw", "Silent yawn", "Tongue forward", "Easy breath"][sessionPosition.round % 4]
      : isBreathExercise ? "Steady exhale" : null;

  // Box breathing: compute per-phase progress (each phase is 4 seconds)
  const boxPhaseProgress = isBoxBreathing && !sessionPosition.preparing
    ? ((sessionPosition.activeTime % 4) / 4) * 100
    : 0;
  const boxPhaseCountdown = isBoxBreathing && !sessionPosition.preparing
    ? Math.ceil(4 - (sessionPosition.activeTime % 4))
    : null;

  // Sustained hiss / breath: compute live volume consistency
  const liveRmsNorm = Math.min(1, live.rms / 0.06);
  const rmsMean = rmsHistory.length > 0 ? rmsHistory.reduce((s, v) => s + v, 0) / rmsHistory.length : 0;
  const rmsVariance = rmsHistory.length > 1 ? rmsHistory.reduce((s, v) => s + (v - rmsMean) ** 2, 0) / rmsHistory.length : 0;
  const rmsSteadiness = rmsMean > 0.001 ? Math.max(0, 1 - Math.sqrt(rmsVariance) / rmsMean) : 0;
  const steadinessLabel = rmsSteadiness > 0.85 ? "Very steady" : rmsSteadiness > 0.6 ? "Good" : rmsSteadiness > 0.3 ? "Uneven" : rmsHistory.length < 3 ? "Listening…" : "Inconsistent";

  return (
    <main className="practicePage">
      <div className="practiceHeader">
        <button type="button" className="trainingBack" onClick={onBack}>← Module {exercise.module}</button>
        <div className="practiceIdentity"><span className="trainingEyebrow">{moduleFor(exercise.module).title}</span><h1>{exercise.title}</h1></div>
      </div>

      <section className="practiceWorkspace">
        <div className="practiceMain">
          <div className="practiceCue">
            <div className="practiceCueInstruction"><strong>{soundInstruction(exercise)}</strong><p>{exercise.cue}</p></div>
            <div className="roundReadout">
              <span>Round {sessionPosition.round + 1} of {ROUND_COUNT}</span>
              {tracksPitch && <strong>{midiToName(sessionPosition.roundPitch)}</strong>}
            </div>
          </div>
          <div className="pitchStage">
            <div className="stageGrid" aria-hidden="true" />
            {isBoxBreathing ? (
              <div className="guidedStage">
                <div className="breathOrb breathOrb--timer" style={{ "--phase-progress": `${boxPhaseProgress}` } as React.CSSProperties}>
                  <span>{sessionPosition.preparing ? "Get ready" : guidePhase}</span>
                  <strong>{boxPhaseCountdown !== null ? boxPhaseCountdown : `${sessionPosition.round + 1}/${ROUND_COUNT}`}</strong>
                  <small>{sessionPosition.preparing ? "Settle before the next round" : `Round ${sessionPosition.round + 1} of ${ROUND_COUNT}`}</small>
                </div>
              </div>
            ) : isBreathExercise && guidePhase ? (
              <div className="guidedStage">
                <div className="volumeStage">
                  <div className="volumeMeterWrap">
                    <div className="volumeMeterTrack">
                      <div className="volumeMeterFill" style={{ height: `${liveRmsNorm * 100}%` }} />
                      {rmsMean > 0.001 && <div className="volumeMeterTarget" style={{ bottom: `${Math.min(1, rmsMean / 0.06) * 100}%` }} />}
                    </div>
                    <span className="volumeMeterLabel">{live.rms > 0.001 ? Math.round(liveRmsNorm * 100) : "—"}</span>
                  </div>
                  <div className="volumeReadout">
                    <span className="volumePhaseLabel">{sessionPosition.preparing ? "Get ready" : guidePhase}</span>
                    <div className="steadinessDisplay">
                      <span className="steadinessLabel">{steadinessLabel}</span>
                      <div className="steadinessBar"><i style={{ width: `${rmsSteadiness * 100}%` }} /></div>
                      <small>Consistency</small>
                    </div>
                    <div className="volumeRoundInfo"><span>Round {sessionPosition.round + 1} of {ROUND_COUNT}</span></div>
                    <div className="volumeHistory">
                      {rmsHistory.slice(-40).map((v, i) => (
                        <span key={i} className="volumeHistoryBar" style={{ height: `${Math.min(1, v / 0.06) * 100}%` }} />
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ) : guidePhase ? (
              <div className="guidedStage"><div className="breathOrb" style={{ "--breath-progress": `${roundProgress}%` } as React.CSSProperties}><span>{sessionPosition.preparing ? "Get ready" : guidePhase}</span><strong>{sessionPosition.round + 1}/{ROUND_COUNT}</strong><small>{sessionPosition.preparing ? "Settle before the next round" : "rounds"}</small></div></div>
            ) : (
              <>
                <div className="scrollingNoteLane" aria-label="Five-round scrolling piano exercise">
                  {timelineNotes.map((note) => {
                    const active = elapsed >= note.start && elapsed < note.start + note.duration;
                    return <span key={`${note.round}-${note.index}`} className={active ? "active" : ""} style={{ left:`calc(${PLAYHEAD_FRACTION * 100}% + ${(note.start - elapsed) * SCROLL_PIXELS_PER_SECOND}px)`, width:`${Math.max(26, note.duration * SCROLL_PIXELS_PER_SECOND - 6)}px`, bottom:`${pitchY(note.midi)}%` }}>{!isRhythm && <i>{midiToName(note.midi)}</i>}</span>;
                  })}
                  {Array.from({ length: ROUND_COUNT }, (_, round) => {
                    const start = round * roundDuration;
                    return <span key={`round-${round}`} className="roundMarker" style={{ left:`calc(${PLAYHEAD_FRACTION * 100}% + ${(start - elapsed) * SCROLL_PIXELS_PER_SECOND}px)` }}><i>ROUND {round + 1}</i></span>;
                  })}
                </div>
                {!isRhythm && live.midi !== null && <div className={`livePitchMarker ${cents !== null && Math.abs(cents) <= 50 ? "inTune" : ""}`} style={{ left:`${PLAYHEAD_FRACTION * 100}%`, bottom:`${Math.max(8, Math.min(92, pitchY(live.midi)))}%` }}><i /> <span>{midiToName(live.midi)} {cents !== null ? `${cents > 0 ? "+" : ""}${cents}¢` : ""}</span></div>}
              </>
            )}
            {!guidePhase && <div className="songStylePlayhead" style={{ left:`${PLAYHEAD_FRACTION * 100}%` }}><span /></div>}
            <div className="sessionProgress"><i style={{ width:`${progress}%` }} /></div>
          </div>
          <div className="practiceTransport">
            <button type="button" className={`practiceMic ${micState === "on" ? "on" : ""}`} onClick={startMic} disabled={micState === "starting"}><span className="micGlyph" />{micState === "on" ? "Mic on" : micState === "starting" ? "Starting…" : "Enable mic"}</button>
            <button type="button" className={`practiceStart ${running ? "running" : ""}`} onClick={start}><span>{running ? "■" : "▶"}</span>{running ? "Stop take" : score !== null ? "Try again" : hasStarted ? "Restart exercise" : "Start exercise"}</button>
          </div>
          {micError && <p className="practiceError">{micError}</p>}
        </div>

        <aside className="practiceRail">
          <div className={`practiceRailCard accuracyCard ${score !== null ? "complete" : ""}`} aria-live="polite">
            <span className="trainingEyebrow">Accuracy</span>
            <div className="accuracyValue">{score === null ? "—" : score}{score !== null && <small>%</small>}</div>
            <strong>{score === null ? (running ? "Session in progress" : "Complete all five rounds") : score >= 88 ? "Beautifully centered" : score >= 70 ? "Good foundation" : "Keep it easy"}</strong>
            <p>{score === null ? "Your result appears here when the full exercise is complete." : score >= 70 ? "The pattern is settling in. Repeat with the same relaxed feeling." : "Slow the tempo or lower the key for the next take."}</p>
            {score !== null && recordingUrl && <audio controls src={recordingUrl} aria-label="Playback your completed take" />}
          </div>
          <div className="practiceRailCard">
            <span className="trainingEyebrow">5-round session</span>
            <label>Vocal type <strong>{midiToName(voiceProfile.low)}–{midiToName(voiceProfile.high)}</strong>
              <select value={voiceType} disabled={running} onChange={(event) => {
                const next = event.target.value as VoiceLabel;
                setVoiceType(next);
                setScore(null);
                try { localStorage.setItem(VOICE_TYPE_KEY, next); } catch {}
              }}>
                {VOICE_PROFILES.map((profile) => <option key={profile.label} value={profile.label}>{profile.label}</option>)}
              </select>
            </label>
            <label>Tempo <strong>{bpm} BPM</strong><input type="range" min="50" max="140" value={bpm} onChange={(event) => setBpm(Number(event.target.value))} disabled={running || exercise.kind === "guided" || exercise.kind === "breath"} /></label>
            <p className="voiceRangeNote">This exercise starts at {midiToName(targetMidi)} and stays in the intended part of the selected range.</p>
          </div>
          <div className="practiceRailCard quickTips"><span className="trainingEyebrow">Before you begin</span><ul><li>Choose a comfortable key</li><li>Use conversational volume</li><li>Stop if anything feels strained</li></ul></div>
        </aside>
      </section>
    </main>
  );
}

export default function VocalTraining({ onNavigate }: { onNavigate: (section: "songs" | "training") => void }) {
  const [view, setView] = useState<View>({ kind:"home" });
  const [progress, setProgress] = useState<SavedProgress>({});
  useEffect(() => {
    const restore = window.setTimeout(() => {
      try {
        const saved = localStorage.getItem(PROGRESS_KEY);
        if (saved) setProgress(JSON.parse(saved));
      } catch {}
    }, 0);
    return () => window.clearTimeout(restore);
  }, []);
  const complete = useCallback((exercise: TrainingExercise, score: number) => {
    setProgress((current) => {
      const prior = current[exercise.id] ?? { best:0, takes:0 };
      const next = { ...current, [exercise.id]: { best:Math.max(prior.best, score), takes:prior.takes + 1 } };
      try { localStorage.setItem(PROGRESS_KEY, JSON.stringify(next)); } catch {}
      return next;
    });
  }, []);
  const content = useMemo(() => {
    if (view.kind === "home") return <TrainingHome progress={progress} onOpenModule={(moduleId) => setView({ kind:"module", moduleId })} onOpenExercise={(exercise) => setView({ kind:"exercise", exercise })} />;
    if (view.kind === "module") return <ModuleView moduleId={view.moduleId} progress={progress} onBack={() => setView({ kind:"home" })} onOpen={(exercise) => setView({ kind:"exercise", exercise })} />;
    return <ExercisePlayer exercise={view.exercise} onBack={() => setView({ kind:"module", moduleId:view.exercise.module })} onComplete={(score) => complete(view.exercise, score)} />;
  }, [complete, progress, view]);
  return <div className="trainingShell"><Header section="training" onNavigate={onNavigate} />{content}</div>;
}
