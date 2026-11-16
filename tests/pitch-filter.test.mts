import assert from "node:assert/strict";
import test from "node:test";
import {
  createAdaptivePitchFilter,
  type AdaptivePitchFilter,
} from "../app/pitch-utils.ts";

const HOP_MS = 16;
const midiToHz = (midi: number) => 440 * 2 ** ((midi - 69) / 12);

/** Feed `frames` voiced frames of one pitch; return every filtered output. */
function sing(
  filter: AdaptivePitchFilter,
  clock: { ms: number },
  midi: number,
  frames: number,
): Array<number | null> {
  const output: Array<number | null> = [];
  for (let index = 0; index < frames; index += 1) {
    clock.ms += HOP_MS;
    output.push(
      filter.process({ f0: midiToHz(midi), clarity: 0.9, rms: 0.05, timeMs: clock.ms }).midi,
    );
  }
  return output;
}

function silence(filter: AdaptivePitchFilter, clock: { ms: number }, frames: number) {
  for (let index = 0; index < frames; index += 1) {
    clock.ms += HOP_MS;
    filter.process({ f0: null, clarity: 0, rms: 0.0004, timeMs: clock.ms });
  }
}

const last = (values: Array<number | null>) => values[values.length - 1];

test("a held note passes through and silence clears it", () => {
  const filter = createAdaptivePitchFilter();
  const clock = { ms: 0 };
  const held = sing(filter, clock, 60, 20);
  assert.ok(Math.abs((last(held) as number) - 60) < 0.01);
  clock.ms += HOP_MS;
  // One dropped frame inside the bridge window keeps the pitch alive...
  const bridged = filter.process({ f0: null, clarity: 0, rms: 0.0004, timeMs: clock.ms });
  assert.equal(bridged.bridged, true);
  // ...a real gap does not.
  silence(filter, clock, 10);
  clock.ms += HOP_MS;
  assert.equal(
    filter.process({ f0: null, clarity: 0, rms: 0.0004, timeMs: clock.ms }).midi,
    null,
  );
});

test("a brief exact-octave glitch is folded back onto the held note", () => {
  const filter = createAdaptivePitchFilter();
  const clock = { ms: 0 };
  sing(filter, clock, 60, 20);
  for (const glitch of [48, 72]) {
    const during = sing(filter, clock, glitch, 2);
    assert.ok(during.every((value) => value !== null && Math.abs(value - 60) < 0.5));
    sing(filter, clock, 60, 6);
  }
});

test("a sustained upward octave is accepted as a real leap", () => {
  const filter = createAdaptivePitchFilter();
  const clock = { ms: 0 };
  sing(filter, clock, 57, 20);
  const leap = sing(filter, clock, 69, 12);
  assert.ok(Math.abs((last(leap) as number) - 69) < 0.01);
});

test("a large non-octave leap lands only after it is confirmed", () => {
  const filter = createAdaptivePitchFilter();
  const clock = { ms: 0 };
  sing(filter, clock, 55, 20);
  const leap = sing(filter, clock, 64, 4);
  assert.ok(Math.abs((leap[0] as number) - 55) < 0.01);
  assert.ok(Math.abs((last(leap) as number) - 64) < 0.01);
});

test("without register history a downward octave stays folded", () => {
  const filter = createAdaptivePitchFilter();
  const clock = { ms: 0 };
  sing(filter, clock, 67, 20);
  const low = sing(filter, clock, 55, 60);
  assert.ok(Math.abs((last(low) as number) - 67) < 0.5);
});

test("a phrase locked an octave high recovers to the singer's register", () => {
  const filter = createAdaptivePitchFilter();
  const clock = { ms: 0 };
  // Three seconds of ordinary singing around A3 establishes the register.
  for (const midi of [57, 59, 55, 57, 60, 57]) {
    sing(filter, clock, midi, 30);
    silence(filter, clock, 8);
  }
  // The next phrase's onset is misdetected an octave high for a few frames,
  // then the detector reads the true pitch.
  sing(filter, clock, 69, 4);
  const recovering = sing(filter, clock, 57, 40);
  // Held high at first (it looks like a subharmonic glitch)...
  assert.ok(Math.abs((recovering[3] as number) - 69) < 0.5);
  // ...then released once it has persisted and the register agrees.
  assert.ok(Math.abs((last(recovering) as number) - 57) < 0.01);
});

test("a subharmonic away from the singer's register is never accepted", () => {
  const filter = createAdaptivePitchFilter();
  const clock = { ms: 0 };
  for (const midi of [69, 71, 67, 69, 72, 69]) {
    sing(filter, clock, midi, 30);
    silence(filter, clock, 8);
  }
  sing(filter, clock, 69, 10);
  const subharmonic = sing(filter, clock, 57, 80);
  assert.ok(subharmonic.every((value) => value !== null && Math.abs(value - 69) < 0.5));
});
