import assert from "node:assert/strict";
import test from "node:test";
import {
  appendCapture,
  beginCaptureRun,
  CAPTURE_SETTLE_SECONDS,
  createCaptureTimeline,
  editContour,
  encodeWav,
  mixPcm,
  pcmBlockEnd,
  renderPcmTrack,
  renderTake,
  trimPcmBlocks,
  type PcmBlock,
} from "../app/studio-utils.ts";
import { foldToOctave } from "../app/pitch-utils.ts";
import type { ContourData } from "../app/songs.ts";

const contour: ContourData = {
  hop: 0.1,
  duration: 1,
  range: [60, 72],
  segments: [
    { t0: 0, midi: [60, 60, 62, 64, 66, 68, 70, 72, 72, 72] },
  ],
};

test("contour editing removes, smooths, and exactly copies repair voicing", () => {
  const removed = editContour(contour, { t0: 0.3, t1: 0.6 }, "remove");
  assert.deepEqual(removed.segments, [
    { t0: 0, midi: [60, 60, 62] },
    { t0: 0.6, midi: [70, 72, 72, 72] },
  ]);

  const smoothed = editContour(contour, { t0: 0.3, t1: 0.6 }, "smooth");
  assert.deepEqual(smoothed.segments[0].midi.slice(3, 6), [70, 70, 70]);

  const repaired = editContour(
    contour,
    { t0: 0.2, t1: 0.6 },
    "repair",
    {
      hop: 0.1,
      duration: 1,
      segments: [
        { t0: 0.2, midi: [65] },
        { t0: 0.4, midi: [67, 69] },
      ],
    },
  );
  assert.deepEqual(repaired.segments, [
    { t0: 0, midi: [60, 60, 65] },
    { t0: 0.4, midi: [67, 69, 70, 72, 72, 72] },
  ]);
});

test("recording helpers trim at the playhead and produce a valid clipped WAV mix", async () => {
  const block: PcmBlock = {
    start: 0.25,
    sampleRate: 4,
    channels: [new Float32Array([0.25, 0.5, 0.75, 1])],
  };
  const trimmed = trimPcmBlocks([block], 0.75);
  assert.deepEqual(Array.from(trimmed[0].channels[0]), [0.25, 0.5]);

  const instrumental = renderPcmTrack(trimmed, 2, 4, 1);
  const vocals = [new Float32Array([0, 0.75, 0.75, 0])];
  const combined = mixPcm(instrumental, vocals);
  assert.deepEqual(Array.from(combined[0]), [0, 1, 1, 0]);
  assert.deepEqual(Array.from(combined[1]), [0, 1, 1, 0]);

  const wav = encodeWav(combined, 4);
  assert.equal(wav.type, "audio/wav");
  const header = new Uint8Array(await wav.arrayBuffer());
  assert.equal(new TextDecoder().decode(header.slice(0, 4)), "RIFF");
  assert.equal(new TextDecoder().decode(header.slice(8, 12)), "WAVE");
});

test("a capture run stays sample-contiguous whatever the main-thread timing", () => {
  const sampleRate = 48000;
  const chunk = 2048;
  const timeline = createCaptureTimeline();
  // Song position leads the context clock by 12.5 s; the singer is 80 ms late.
  beginCaptureRun(timeline, 0.08);
  const trueOffset = 12.5;
  let frame = 96000;
  for (let index = 0; index < 40; index += 1) {
    // The offset is read on the main thread: early readings are too high
    // because the media element has not started advancing, and every reading
    // carries scheduling jitter.
    const startup = Math.max(0, 0.06 - index * 0.02);
    const jitter = ((index * 7919) % 13) / 1000 - 0.006;
    appendCapture(
      timeline,
      [new Float32Array(chunk).fill(index)],
      sampleRate,
      frame / sampleRate,
      trueOffset + startup + jitter,
    );
    frame += chunk;
  }
  assert.equal(timeline.blocks.length, 40);
  for (let index = 1; index < timeline.blocks.length; index += 1) {
    const gap = timeline.blocks[index].start - pcmBlockEnd(timeline.blocks[index - 1]);
    assert.ok(Math.abs(gap) < 1e-9, `block ${index} is ${gap}s from its neighbour`);
  }
  // The run settles on a reading taken after playback was really running.
  const expectedStart = 96000 / sampleRate + trueOffset - 0.08;
  assert.ok(Math.abs(timeline.blocks[0].start - expectedStart) < 0.007);
  assert.equal(timeline.run?.settled, true);
  assert.ok(40 * chunk / sampleRate > CAPTURE_SETTLE_SECONDS);

  // Rendered back out, the take is one unbroken ramp of block indices.
  const rendered = renderPcmTrack(timeline.blocks, 1, sampleRate, pcmBlockEnd(timeline.blocks[39]))[0];
  const first = Math.round(timeline.blocks[0].start * sampleRate);
  for (let index = 0; index < 40 * chunk; index += 1) {
    assert.equal(rendered[first + index], Math.floor(index / chunk));
  }
});

test("a new capture run after a pause is placed independently", () => {
  const sampleRate = 48000;
  const timeline = createCaptureTimeline();
  beginCaptureRun(timeline, 0);
  appendCapture(timeline, [new Float32Array(2048)], sampleRate, 1, 10);
  // Paused, the singer rewound: the song is now 4 s earlier for the same clock.
  beginCaptureRun(timeline, 0);
  appendCapture(timeline, [new Float32Array(2048)], sampleRate, 5, 6);
  assert.ok(Math.abs(timeline.blocks[0].start - 11) < 1e-9);
  assert.ok(Math.abs(timeline.blocks[1].start - 11) < 1e-9);
});

test("audio placed before the start of the song is cropped, not shifted", () => {
  const block: PcmBlock = {
    start: -0.5,
    sampleRate: 4,
    channels: [new Float32Array([1, 2, 3, 4])],
  };
  const rendered = renderPcmTrack([block], 1, 4, 1);
  assert.deepEqual(Array.from(rendered[0]), [3, 4, 0, 0]);
});

test("a finished take renders to three WAV files of the same length", async () => {
  const sampleRate = 8000;
  const tone = (value: number, frames: number) => new Float32Array(frames).fill(value);
  const result = renderTake({
    sampleRate,
    instrumental: [{ start: 0, sampleRate, channels: [tone(0.25, 8000), tone(0.25, 8000)] }],
    vocals: [{ start: 0.5, sampleRate, channels: [tone(0.5, 4000)] }],
  });
  assert.ok(result);
  assert.equal(result.duration, 1);
  const sizes = [result.vocals.size, result.instrumental.size, result.combined.size];
  assert.deepEqual(sizes, [44 + 8000 * 2, 44 + 8000 * 4, 44 + 8000 * 4]);
  const combined = new DataView(await result.combined.arrayBuffer());
  // Before the vocal enters: instrumental alone. After: the two summed.
  assert.equal(combined.getInt16(44, true), Math.round(0.25 * 0x7fff));
  assert.equal(combined.getInt16(44 + 6000 * 4, true), Math.round(0.75 * 0x7fff));
  assert.equal(renderTake({ sampleRate, instrumental: [], vocals: [] }), null);
});

test("octave folding moves a pitch to the octave nearest its reference", () => {
  assert.equal(foldToOctave(48, 60), 60);
  assert.equal(foldToOctave(73.5, 61), 61.5);
  assert.equal(foldToOctave(62, 60), 62);
  assert.equal(foldToOctave(55, 60), 55);
  assert.ok(Number.isNaN(foldToOctave(Number.NaN, 60)));
});
