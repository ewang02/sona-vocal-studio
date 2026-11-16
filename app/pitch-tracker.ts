/**
 * Microphone pitch tracking shared by the studio and the training exercises.
 *
 * Owns the capture stream, the YIN worklet, and the timing model that says how
 * old a pitch reading is. Callers decide what to do with each reading; nothing
 * here knows about songs, scoring, or React.
 */

import {
  ADAPTIVE_FILTER_LAG_FRAMES,
  createAdaptivePitchFilter,
  hzToMidi,
} from "./pitch-utils";

// MediaStreamTrack.getSettings().latency is not implemented by every browser.
// Use a modest input-path estimate when it is unavailable; unlike
// AudioContext.baseLatency, this represents capture rather than playback.
export const DEFAULT_CAPTURE_LATENCY_S = 0.025;

const WORKLET_URL = "/pitch-worklet.js?v=3";
const WORKLET_NAME = "sona-pitch-processor";
// Used only until the worklet reports its real figures in its init message.
const DEFAULT_ANALYSIS_LATENCY_S = 0.0285;
const DEFAULT_HOP_S = 0.016;

const loadedContexts = new WeakSet<BaseAudioContext>();

export type TrackedPitch = {
  /** Pitch after the adaptive voicing/octave filter; null when unvoiced. */
  midi: number | null;
  /** The detector's own reading, for callers that apply a different gate. */
  rawMidi: number | null;
  clarity: number;
  rms: number;
  /**
   * AudioContext time at which the analysed sound entered the audio graph, or
   * null until the worklet has published its clock. This is the raw analysis
   * time: subtract the lag of whichever filter produced the value you use.
   */
  sampleTime: number | null;
};

export type PitchTrackerOptions = {
  /**
   * The backing track is playing through speakers, so the microphone hears
   * it. Turns on the browser's echo canceller; leave off for headphones,
   * where voice processing only gates and delays a clean signal.
   */
  speakers?: boolean;
  onPitch: (pitch: TrackedPitch) => void;
  /** The capture device went away (unplugged, permission revoked). */
  onEnded?: () => void;
};

export type PitchTracker = {
  readonly stream: MediaStream;
  readonly source: MediaStreamAudioSourceNode;
  /** Stable per-device key for remembering calibration; "" when unknown. */
  readonly deviceKey: string;
  /** Capture-path delay in seconds (reported by the browser, or estimated). */
  readonly captureLatency: number;
  /** Seconds between pitch readings. */
  hopSeconds(): number;
  /** Lag of the adaptive filter's `midi` behind `sampleTime`, in seconds. */
  filterLag(): number;
  /** Age estimate for a reading whose `sampleTime` is not yet known. */
  fallbackAge(): number;
  resetFilter(): void;
  stop(): void;
};

/**
 * Seconds since the sound behind `pitch` was actually sung, given the lag of
 * the filter that produced the value in use.
 */
export function voiceAge(
  context: BaseAudioContext,
  tracker: PitchTracker,
  pitch: Pick<TrackedPitch, "sampleTime">,
  filterLagSeconds: number,
): number {
  if (pitch.sampleTime === null) {
    return tracker.fallbackAge() - tracker.filterLag() + filterLagSeconds;
  }
  return Math.max(
    0,
    context.currentTime - pitch.sampleTime + filterLagSeconds + tracker.captureLatency,
  );
}

export async function startPitchTracker(
  context: AudioContext,
  options: PitchTrackerOptions,
): Promise<PitchTracker> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Microphone access is not supported in this browser.");
  }
  if (!loadedContexts.has(context)) {
    await context.audioWorklet.addModule(WORKLET_URL);
    loadedContexts.add(context);
  }

  const speakers = options.speakers === true;
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      // With headphones the backing track never reaches the microphone, so
      // there is nothing to cancel: take the raw signal. Voice-call processing
      // gates and ducks a real condenser mic and adds its own delay. With
      // speakers the canceller is the lesser evil, because otherwise the
      // detector tracks the song instead of the singer.
      echoCancellation: speakers ? true : false,
      noiseSuppression: false,
      autoGainControl: false,
      channelCount: 1,
      // `latency` is in the Media Capture spec but not the DOM lib types.
      latency: { ideal: 0 },
    } as MediaTrackConstraints & { latency?: ConstrainDouble },
  });

  const track = stream.getAudioTracks()[0];
  const settings = track?.getSettings() as
    | (MediaTrackSettings & { latency?: number })
    | undefined;
  const reportedLatency = settings?.latency;
  const captureLatency =
    typeof reportedLatency === "number" && Number.isFinite(reportedLatency) && reportedLatency > 0
      ? reportedLatency
      : DEFAULT_CAPTURE_LATENCY_S;
  // deviceId is stable per origin once permission is granted; the label is a
  // readable fallback for browsers that rotate or withhold the id.
  const deviceKey = settings?.deviceId || track?.label || "";

  const source = context.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(context, WORKLET_NAME, {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
  });
  // The worklet passes audio through so it stays pull-connected; the zero
  // gain keeps the microphone out of the speakers.
  const silentGain = context.createGain();
  silentGain.gain.value = 0;
  source.connect(node);
  node.connect(silentGain);
  silentGain.connect(context.destination);

  const filter = createAdaptivePitchFilter();
  let analysisLatency = DEFAULT_ANALYSIS_LATENCY_S;
  let hopSeconds = DEFAULT_HOP_S;
  let clockOrigin: number | null = null;
  let stopped = false;

  node.port.onmessage = (event: MessageEvent) => {
    const data = event.data as {
      type: string;
      f0?: number | null;
      clarity?: number;
      rms?: number;
      t?: number;
      analysisLatency?: number;
      hopSeconds?: number;
      contextTimeOrigin?: number;
    };
    if (data.type === "init") {
      if (Number.isFinite(data.analysisLatency)) analysisLatency = data.analysisLatency as number;
      if (Number.isFinite(data.hopSeconds)) hopSeconds = data.hopSeconds as number;
      return;
    }
    if (data.type === "clock") {
      if (Number.isFinite(data.contextTimeOrigin)) clockOrigin = data.contextTimeOrigin as number;
      return;
    }
    if (data.type !== "pitch") return;
    const rms = data.rms ?? 0;
    const clarity = data.clarity ?? 0;
    const f0 = data.f0 ?? null;
    const hasTime = Number.isFinite(data.t);
    const filtered = filter.process({
      f0,
      clarity,
      rms,
      timeMs: (hasTime ? (data.t as number) : context.currentTime) * 1000,
    });
    const rawMidi = f0 === null ? Number.NaN : hzToMidi(f0);
    options.onPitch({
      midi: filtered.midi,
      rawMidi: Number.isFinite(rawMidi) ? rawMidi : null,
      clarity,
      rms,
      sampleTime: clockOrigin !== null && hasTime ? clockOrigin + (data.t as number) : null,
    });
  };

  const stop = () => {
    if (stopped) return;
    stopped = true;
    node.port.onmessage = null;
    node.port.close();
    node.disconnect();
    source.disconnect();
    silentGain.disconnect();
    stream.getTracks().forEach((streamTrack) => {
      streamTrack.onended = null;
      streamTrack.stop();
    });
  };

  if (track && options.onEnded) {
    const onEnded = options.onEnded;
    track.onended = () => {
      if (!stopped) onEnded();
    };
  }

  const filterLag = () => ADAPTIVE_FILTER_LAG_FRAMES * hopSeconds;
  return {
    stream,
    source,
    deviceKey,
    captureLatency,
    hopSeconds: () => hopSeconds,
    filterLag,
    fallbackAge: () => captureLatency + analysisLatency + filterLag(),
    resetFilter: () => filter.reset(),
    stop,
  };
}
