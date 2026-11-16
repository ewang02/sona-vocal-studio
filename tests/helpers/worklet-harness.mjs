import { readFileSync } from "node:fs";
import vm from "node:vm";

/**
 * Load an AudioWorklet module into a bare VM context that provides just the
 * worklet globals, and return the registered processor with its messages.
 */
export function loadWorklet(url, { sampleRate = 48000, options } = {}) {
  const messages = [];
  let Processor = null;
  const sandbox = {
    sampleRate,
    currentTime: 0,
    currentFrame: 0,
    AudioWorkletProcessor: class {
      constructor() {
        this.port = {
          onmessage: null,
          postMessage: (message) => messages.push(message),
        };
      }
    },
    registerProcessor: (_name, processor) => {
      Processor = processor;
    },
  };
  vm.createContext(sandbox);
  vm.runInContext(readFileSync(url, "utf8"), sandbox);
  const node = new Processor(options);
  return {
    node,
    messages,
    /** Send a message from the "main thread" to the processor. */
    post(data) {
      node.port.onmessage?.({ data });
    },
    /** Feed mono or multi-channel audio through in 128-frame render quanta. */
    run(channels, { from = 0, to } = {}) {
      const inputChannels = Array.isArray(channels) ? channels : [channels];
      const end = to ?? inputChannels[0].length;
      const output = [new Float32Array(128)];
      for (let index = from; index + 128 <= end; index += 128) {
        sandbox.currentFrame = index;
        sandbox.currentTime = index / sampleRate;
        node.process(
          [inputChannels.map((channel) => channel.subarray(index, index + 128))],
          [output],
        );
      }
    },
  };
}

/** A harmonic, voice-like tone with optional noise and vibrato. */
export function voiceLike(f0, sampleRate, seconds, { noise = 0, vibrato = 0 } = {}) {
  const length = Math.floor(sampleRate * seconds);
  const signal = new Float32Array(length);
  let phase = 0;
  let seed = 1;
  const random = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296) * 2 - 1;
  for (let index = 0; index < length; index += 1) {
    const frequency =
      f0 * 2 ** ((vibrato * Math.sin((2 * Math.PI * 5.5 * index) / sampleRate)) / 12);
    phase += (2 * Math.PI * frequency) / sampleRate;
    let value = 0;
    for (let harmonic = 1; harmonic <= 12 && frequency * harmonic < sampleRate / 2; harmonic += 1) {
      value += Math.sin(phase * harmonic) / harmonic ** 1.2;
    }
    signal[index] = 0.2 * value + noise * random();
  }
  return signal;
}
