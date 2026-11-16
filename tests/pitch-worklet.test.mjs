import assert from "node:assert/strict";
import test from "node:test";
import { loadWorklet, voiceLike } from "./helpers/worklet-harness.mjs";

const WORKLET = new URL("../public/pitch-worklet.js", import.meta.url);

function detect(f0, sampleRate, options) {
  const worklet = loadWorklet(WORKLET, { sampleRate });
  worklet.run(voiceLike(f0, sampleRate, 1.5, { noise: 0.002, ...options }));
  // Skip the first frames, which straddle the tone's onset.
  const pitched = worklet.messages.filter((message) => message.type === "pitch").slice(5);
  return { worklet, pitched };
}

test("reports its timing so the main thread can date each reading", () => {
  const { worklet } = detect(220, 48000);
  const init = worklet.messages.find((message) => message.type === "init");
  assert.ok(init.analysisLatency > 0.01 && init.analysisLatency < 0.06);
  assert.ok(Math.abs(init.hopSeconds - 0.016) < 0.002);
  assert.ok(worklet.messages.some((message) => message.type === "clock"));
});

for (const sampleRate of [44100, 48000]) {
  test(`tracks sung pitches within two cents at ${sampleRate} Hz`, () => {
    // E2 to E6: bass low notes through soprano high notes.
    for (const f0 of [82.41, 110, 220, 440, 659.26, 880, 1046.5, 1318.51]) {
      const { pitched } = detect(f0, sampleRate);
      assert.ok(pitched.length > 50, `${f0} Hz produced too few frames`);
      for (const message of pitched) {
        assert.ok(message.f0 !== null, `${f0} Hz dropped a frame`);
        const cents = 1200 * Math.log2(message.f0 / f0);
        assert.ok(Math.abs(cents) < 2, `${f0} Hz read ${cents.toFixed(1)} cents off`);
      }
    }
  });
}

test("follows vibrato without octave errors", () => {
  for (const f0 of [196, 523.25, 987.77]) {
    const { pitched } = detect(f0, 48000, { noise: 0.02, vibrato: 0.5 });
    for (const message of pitched) {
      assert.ok(message.f0 !== null);
      // Half a semitone of vibrato either way, plus tracking slack.
      assert.ok(Math.abs(1200 * Math.log2(message.f0 / f0)) < 70);
    }
  }
});

test("reports silence and noise as unvoiced", () => {
  const silent = loadWorklet(WORKLET, { sampleRate: 48000 });
  silent.run(new Float32Array(48000));
  const silentFrames = silent.messages.filter((message) => message.type === "pitch");
  assert.ok(silentFrames.length > 30);
  assert.ok(silentFrames.every((message) => message.f0 === null));

  const noisy = loadWorklet(WORKLET, { sampleRate: 48000 });
  let seed = 7;
  const noise = Float32Array.from({ length: 48000 }, () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return (seed / 4294967296 - 0.5) * 0.2;
  });
  noisy.run(noise);
  const noisyFrames = noisy.messages.filter((message) => message.type === "pitch");
  const voiced = noisyFrames.filter((message) => message.f0 !== null && message.clarity > 0.55);
  assert.ok(voiced.length / noisyFrames.length < 0.05);
});
