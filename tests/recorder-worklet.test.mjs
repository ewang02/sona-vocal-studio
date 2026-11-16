import assert from "node:assert/strict";
import test from "node:test";
import { loadWorklet } from "./helpers/worklet-harness.mjs";

const WORKLET = new URL("../public/recorder-worklet.js", import.meta.url);

function ramp(length) {
  return Float32Array.from({ length }, (_, index) => index);
}

test("captures contiguous, frame-stamped chunks only while active", () => {
  const recorder = loadWorklet(WORKLET, { options: { processorOptions: { channelCount: 1 } } });
  const signal = ramp(128 * 100);

  recorder.run(signal, { to: 128 * 10 });
  assert.equal(recorder.messages.length, 0, "idle recorder must not post");

  recorder.post({ active: true, epoch: 3 });
  recorder.run(signal, { from: 128 * 10, to: 128 * 60 });
  recorder.post({ active: false });

  const chunks = recorder.messages;
  assert.ok(chunks.length >= 3);
  assert.ok(chunks.every((chunk) => chunk.type === "chunk" && chunk.epoch === 3));
  // Each chunk starts exactly where the previous one ended, and the samples
  // are the ones that were playing at that frame.
  let expectedFrame = 128 * 10;
  for (const chunk of chunks) {
    assert.equal(chunk.frame, expectedFrame);
    const samples = chunk.channels[0];
    assert.equal(samples[0], expectedFrame);
    assert.equal(samples[samples.length - 1], expectedFrame + samples.length - 1);
    expectedFrame += samples.length;
  }
  // Stopping flushes the partial tail, so nothing recorded is lost.
  assert.equal(expectedFrame, 128 * 60);
});

test("numbers chunks with the epoch of the start that produced them", () => {
  const recorder = loadWorklet(WORKLET, { options: { processorOptions: { channelCount: 2 } } });
  const left = ramp(128 * 40);
  const right = left.map((value) => -value);
  recorder.post({ active: true, epoch: 1 });
  recorder.run([left, right], { to: 128 * 8 });
  recorder.post({ active: false });
  recorder.post({ active: true, epoch: 2 });
  recorder.run([left, right], { from: 128 * 8, to: 128 * 40 });
  recorder.post({ active: false });

  const epochs = recorder.messages.map((chunk) => chunk.epoch);
  assert.deepEqual([...new Set(epochs)], [1, 2]);
  const first = recorder.messages[0];
  assert.equal(first.channels.length, 2);
  assert.equal(first.channels[1][5], -5);
});
