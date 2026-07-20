import type { ContourData, ContourSegmentData } from "./songs";

export type StudioMode = "editing" | "practicing" | "recording";
export type ContourEditAction = "remove" | "smooth" | "repair";
export type TimeSelection = { t0: number; t1: number };

export type RepairSourceData = {
  hop: number;
  duration: number;
  segments: ContourSegmentData[];
};

export type PcmBlock = {
  start: number;
  sampleRate: number;
  channels: Float32Array[];
};

export function normalizeSelection(selection: TimeSelection): TimeSelection {
  return {
    t0: Math.max(0, Math.min(selection.t0, selection.t1)),
    t1: Math.max(0, Math.max(selection.t0, selection.t1)),
  };
}

export function contourToFrames(
  contour: Pick<ContourData, "hop" | "duration" | "segments">,
): Float32Array {
  const frames = new Float32Array(Math.ceil(contour.duration / contour.hop) + 2);
  frames.fill(Number.NaN);
  for (const segment of contour.segments) {
    const start = Math.round(segment.t0 / contour.hop);
    for (let offset = 0; offset < segment.midi.length; offset += 1) {
      if (start + offset >= 0 && start + offset < frames.length) {
        frames[start + offset] = segment.midi[offset];
      }
    }
  }
  return frames;
}

export function framesToContour(
  source: ContourData,
  frames: Float32Array,
): ContourData {
  const segments: ContourSegmentData[] = [];
  const voiced: number[] = [];
  let index = 0;
  while (index < frames.length) {
    while (index < frames.length && !Number.isFinite(frames[index])) index += 1;
    if (index >= frames.length) break;
    const start = index;
    const midi: number[] = [];
    while (index < frames.length && Number.isFinite(frames[index])) {
      const value = Math.round(frames[index] * 100) / 100;
      midi.push(value);
      voiced.push(value);
      index += 1;
    }
    segments.push({ t0: Math.round(start * source.hop * 1000) / 1000, midi });
  }
  const range = voiced.length
    ? [Math.floor(Math.min(...voiced)), Math.ceil(Math.max(...voiced))]
    : [...source.range];
  return { ...source, range, segments };
}

function median(values: number[]): number {
  const finite = values.filter(Number.isFinite).sort((left, right) => left - right);
  if (!finite.length) return Number.NaN;
  const middle = Math.floor(finite.length / 2);
  return finite.length % 2
    ? finite[middle]
    : (finite[middle - 1] + finite[middle]) / 2;
}

export function editContour(
  contour: ContourData,
  selection: TimeSelection,
  action: ContourEditAction,
  repairSource?: RepairSourceData | null,
): ContourData {
  const normalized = normalizeSelection(selection);
  const frames = contourToFrames(contour);
  // Convert the time interval to frame boundaries without letting binary
  // floating-point error pull an exact boundary one frame to the left.
  const boundaryEpsilon = 1e-7;
  const start = Math.max(
    0,
    Math.ceil(normalized.t0 / contour.hop - boundaryEpsilon),
  );
  const end = Math.min(
    frames.length,
    Math.max(
      start + 1,
      Math.ceil(normalized.t1 / contour.hop - boundaryEpsilon),
    ),
  );
  if (end <= start) return contour;

  if (action === "remove") {
    frames.fill(Number.NaN, start, end);
  } else if (action === "smooth") {
    // Pull context only from outside the selected region. A 350 ms shoulder
    // on each side is long enough to represent the settled neighboring pitch
    // without allowing the selected excursion to vote for itself.
    const shoulder = Math.max(
      3,
      Math.round(0.35 / contour.hop + boundaryEpsilon),
    );
    const context = [
      ...frames.slice(Math.max(0, start - shoulder), start),
      ...frames.slice(end, Math.min(frames.length, end + shoulder)),
    ];
    const settled = median(context);
    if (!Number.isFinite(settled)) return contour;
    for (let index = start; index < end; index += 1) {
      // Smoothing changes pitch but preserves the current voiced/unvoiced
      // shape; "repair" is the operation that may create or remove voicing.
      if (Number.isFinite(frames[index])) frames[index] = settled;
    }
  } else {
    if (!repairSource) return contour;
    const repairFrames = contourToFrames(repairSource);
    for (let index = start; index < end; index += 1) {
      const time = index * contour.hop;
      const sourceIndex = Math.round(time / repairSource.hop);
      const replacement = repairFrames[sourceIndex];
      // Deliberately copy NaN too: an empty lead-pYIN frame means the repaired
      // contour should be empty at that point, exactly as requested.
      frames[index] = Number.isFinite(replacement) ? replacement : Number.NaN;
    }
  }
  return framesToContour(contour, frames);
}

export function trimPcmBlocks(blocks: PcmBlock[], cutoff: number): PcmBlock[] {
  const safeCutoff = Math.max(0, cutoff);
  const result: PcmBlock[] = [];
  for (const block of blocks) {
    if (block.start >= safeCutoff) continue;
    const length = block.channels[0]?.length ?? 0;
    const keepSamples = Math.min(
      length,
      Math.max(0, Math.round((safeCutoff - block.start) * block.sampleRate)),
    );
    if (!keepSamples) continue;
    result.push({
      ...block,
      channels: block.channels.map((channel) => channel.slice(0, keepSamples)),
    });
  }
  return result;
}

export function pcmBlockEnd(block: PcmBlock): number {
  return block.start + (block.channels[0]?.length ?? 0) / block.sampleRate;
}

export function renderPcmTrack(
  blocks: PcmBlock[],
  outputChannels: number,
  sampleRate: number,
  duration: number,
): Float32Array[] {
  const sampleCount = Math.max(1, Math.ceil(Math.max(0, duration) * sampleRate));
  const output = Array.from({ length: outputChannels }, () => new Float32Array(sampleCount));
  for (const block of blocks) {
    const destinationStart = Math.max(0, Math.round(block.start * sampleRate));
    const sourceLength = block.channels[0]?.length ?? 0;
    for (let channel = 0; channel < outputChannels; channel += 1) {
      const source = block.channels[Math.min(channel, block.channels.length - 1)];
      if (!source) continue;
      const limit = Math.min(sourceLength, output[channel].length - destinationStart);
      for (let index = 0; index < limit; index += 1) {
        output[channel][destinationStart + index] = source[index];
      }
    }
  }
  return output;
}

export function mixPcm(
  instrumental: Float32Array[],
  vocals: Float32Array[],
): Float32Array[] {
  const channels = Math.max(2, instrumental.length);
  const length = Math.max(
    instrumental[0]?.length ?? 0,
    vocals[0]?.length ?? 0,
  );
  return Array.from({ length: channels }, (_, channel) => {
    const mixed = new Float32Array(length);
    const music = instrumental[Math.min(channel, instrumental.length - 1)];
    const voice = vocals[Math.min(channel, vocals.length - 1)];
    for (let index = 0; index < length; index += 1) {
      mixed[index] = Math.max(
        -1,
        Math.min(1, (music?.[index] ?? 0) + (voice?.[index] ?? 0)),
      );
    }
    return mixed;
  });
}

export function encodeWav(channels: Float32Array[], sampleRate: number): Blob {
  const safeChannels = channels.length ? channels : [new Float32Array(1)];
  const frameCount = safeChannels[0].length;
  const channelCount = safeChannels.length;
  const bytesPerSample = 2;
  const dataSize = frameCount * channelCount * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  const writeAscii = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  };
  writeAscii(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(8, "WAVE");
  writeAscii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channelCount, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channelCount * bytesPerSample, true);
  view.setUint16(32, channelCount * bytesPerSample, true);
  view.setUint16(34, bytesPerSample * 8, true);
  writeAscii(36, "data");
  view.setUint32(40, dataSize, true);
  let offset = 44;
  for (let frame = 0; frame < frameCount; frame += 1) {
    for (let channel = 0; channel < channelCount; channel += 1) {
      const value = Math.max(-1, Math.min(1, safeChannels[channel][frame] ?? 0));
      view.setInt16(
        offset,
        value < 0 ? Math.round(value * 0x8000) : Math.round(value * 0x7fff),
        true,
      );
      offset += bytesPerSample;
    }
  }
  return new Blob([buffer], { type: "audio/wav" });
}
