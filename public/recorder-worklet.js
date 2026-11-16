// Sample-accurate capture for recorded takes.
//
// Each chunk carries the AudioContext frame of its first sample, stamped on
// the audio thread. The main thread can therefore lay chunks end to end
// exactly, whatever its own scheduling jitter, and a dropped render quantum
// shows up as a correctly sized gap instead of shifting everything after it.
const CHUNK_FRAMES = 2048;

class SonaRecorderProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const requested = options?.processorOptions?.channelCount;
    this.channelCount = requested === 2 ? 2 : 1;
    this.active = false;
    this.buffers = null;
    this.filled = 0;
    this.chunkFrame = 0;
    // The main thread numbers each start. Chunks echo that number back so a
    // tail flushed after a discarded take can be told apart from the new one.
    this.epoch = 0;
    this.port.onmessage = (event) => {
      const next = event.data?.active === true;
      // Hand over whatever was captured before a pause so no tail is lost.
      if (this.active && !next) this.flush();
      this.active = next;
      if (next && Number.isFinite(event.data.epoch)) this.epoch = event.data.epoch;
    };
  }

  allocate() {
    this.buffers = Array.from(
      { length: this.channelCount },
      () => new Float32Array(CHUNK_FRAMES),
    );
    this.filled = 0;
  }

  flush() {
    if (!this.buffers || this.filled === 0) return;
    const channels = this.buffers.map((buffer) =>
      this.filled === CHUNK_FRAMES ? buffer : buffer.slice(0, this.filled),
    );
    this.port.postMessage(
      { type: "chunk", epoch: this.epoch, frame: this.chunkFrame, channels },
      channels.map((channel) => channel.buffer),
    );
    this.buffers = null;
    this.filled = 0;
  }

  process(inputs) {
    if (!this.active) return true;
    const input = inputs[0];
    // An unconnected or silent upstream delivers no channels; record it as
    // silence so the timeline stays continuous.
    const frames = input && input[0] ? input[0].length : 128;
    let offset = 0;
    while (offset < frames) {
      if (!this.buffers) {
        this.allocate();
        this.chunkFrame = currentFrame + offset;
      }
      const count = Math.min(frames - offset, CHUNK_FRAMES - this.filled);
      for (let channel = 0; channel < this.channelCount; channel += 1) {
        const source = input && (input[channel] || input[0]);
        if (source) {
          this.buffers[channel].set(source.subarray(offset, offset + count), this.filled);
        }
      }
      this.filled += count;
      offset += count;
      if (this.filled === CHUNK_FRAMES) this.flush();
    }
    return true;
  }
}

registerProcessor("sona-recorder-processor", SonaRecorderProcessor);
