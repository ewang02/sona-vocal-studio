import assert from "node:assert/strict";
import test from "node:test";
import {
  editContour,
  encodeWav,
  mixPcm,
  renderPcmTrack,
  trimPcmBlocks,
  type PcmBlock,
} from "../app/studio-utils.ts";
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
