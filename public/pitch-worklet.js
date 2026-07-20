// Low-latency YIN detector. The browser-facing adaptive filter makes the
// final voiced/unvoiced decision; this processor intentionally stays permissive.
// Analyze a band-limited, averaged signal near 16 kHz. Running the quadratic
// YIN difference loop at a USB device's native 48/96 kHz can miss real-time
// AudioWorklet deadlines and make the visible pitch trace stutter.
const DETECTOR_TARGET_SR = 16000;
const WINDOW = 512;
const HOP = 256;
// 80 Hz (~E2) sits below any sung note but keeps tauMax small: the difference
// function costs tauMax * WINDOW ops and must fit in one render quantum.
const F_MIN = 80;
const F_MAX = 1500;
const RMS_GATE = 0.001;
const CLARITY_GATE = 0.30;
const YIN_THRESHOLD = 0.15;
// Continuity-aware candidate pick: octave errors show up as a second CMND dip
// almost as deep as the true one, and pick-the-first-dip flips between them
// frame to frame. Consider every dip within CANDIDATE_MARGIN of the deepest,
// and prefer the one nearest the recent pitch; CONTINUITY_WEIGHT converts the
// quality deficit into semitones (0.1 CMND ≈ 6 st) so a clearly better dip
// still wins over mere proximity.
const CANDIDATE_MARGIN = 0.12;
const CONTINUITY_WEIGHT = 60;
const CONTINUITY_MEMORY_S = 0.18;

class SonaPitchProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.decimation = Math.max(1, Math.round(sampleRate / DETECTOR_TARGET_SR));
    this.detectorRate = sampleRate / this.decimation;
    this.decimationSum = 0;
    this.decimationCount = 0;
    this.tauMin = Math.max(2, Math.floor(this.detectorRate / F_MAX));
    this.tauMax = Math.floor(this.detectorRate / F_MIN);
    // Keep a full comparison window for every candidate lag. A
    // WINDOW-only buffer starves low notes of comparisons near tauMax.
    this.frameLength = WINDOW + this.tauMax;
    this.ring = new Float32Array(this.frameLength);
    this.frame = new Float32Array(this.frameLength);
    this.cmnd = new Float32Array(this.tauMax + 1);
    this.writeIndex = 0;
    this.filled = 0;
    this.samplesSinceAnalysis = 0;
    this.totalSamples = 0;
    this.contextTimeOrigin = null;
    this.lastF0 = null;
    this.lastF0Time = null;

    this.port.postMessage({
      type: "init",
      sampleRate,
      window: WINDOW,
      hop: HOP,
      // Age of the pitch estimate when it is posted: the buffer must fill
      // before analysis, and the estimate describes its first WINDOW samples.
      analysisLatency: (this.frameLength - WINDOW * 0.5) / this.detectorRate,
      detectorRate: this.detectorRate,
    });
  }

  process(inputs, outputs) {
    const input = inputs[0];
    const samples = input && input[0];
    const output = outputs[0];

    // Pass through so the node remains pull-connected. The main graph routes
    // this output through a zero-gain node, so the microphone is never heard.
    for (let channel = 0; channel < output.length; channel += 1) {
      const destination = output[channel];
      const source = input && (input[channel] || input[0]);
      if (source) destination.set(source);
      else destination.fill(0);
    }

    if (!samples) return true;
    // `t` on pitch messages is relative to the first input sample. Publish the
    // matching AudioContext time once so the main thread can recover the true
    // age of every estimate, including worklet/main-thread scheduling delay.
    if (this.contextTimeOrigin === null) {
      this.contextTimeOrigin = currentTime;
      this.port.postMessage({
        type: "clock",
        contextTimeOrigin: this.contextTimeOrigin,
      });
    }
    for (let index = 0; index < samples.length; index += 1) {
      this.decimationSum += samples[index];
      this.decimationCount += 1;
      if (this.decimationCount === this.decimation) {
        // A block average is a cheap anti-aliasing filter for the voice band.
        this.ring[this.writeIndex] = this.decimationSum / this.decimation;
        this.writeIndex = (this.writeIndex + 1) % this.frameLength;
        this.filled = Math.min(this.frameLength, this.filled + 1);
        this.samplesSinceAnalysis += 1;
        this.totalSamples += 1;
        this.decimationSum = 0;
        this.decimationCount = 0;
      }
    }

    if (this.filled >= this.frameLength && this.samplesSinceAnalysis >= HOP) {
      this.samplesSinceAnalysis %= HOP;
      this.analyze();
    }
    return true;
  }

  analyze() {
    for (let index = 0; index < this.frameLength; index += 1) {
      this.frame[index] = this.ring[(this.writeIndex + index) % this.frameLength];
    }

    // YIN compares frame[i] against frame[i + tau] for i < WINDOW, so the pitch
    // it reports describes the FIRST WINDOW samples of the buffer. Measure RMS
    // over that same span: taking it from the newest samples instead would gate
    // a still-valid pitch the moment a note's level fell away.
    let energy = 0;
    for (let index = 0; index < WINDOW; index += 1) {
      const sample = this.frame[index];
      energy += sample * sample;
    }
    const rms = Math.sqrt(energy / WINDOW);
    // ...and that span is centred WINDOW/2 samples after the buffer start.
    const bufferStart = this.totalSamples - this.frameLength;
    const timestamp = (bufferStart + WINDOW * 0.5) / this.detectorRate;

    if (rms < RMS_GATE) {
      this.postPitch(timestamp, null, 0, rms);
      return;
    }

    const cmnd = this.cmnd;
    cmnd[0] = 1;
    let running = 0;
    for (let tau = 1; tau <= this.tauMax; tau += 1) {
      let difference = 0;
      for (let index = 0; index < WINDOW; index += 1) {
        const delta = this.frame[index] - this.frame[index + tau];
        difference += delta * delta;
      }
      running += difference;
      cmnd[tau] = running > 0 ? (difference * tau) / running : 1;
    }

    // Standard YIN pick first: the first dip below threshold, walked to its
    // local minimum (this guards against the deeper subharmonic dip at twice
    // the period, so it must stay the default).
    let candidate = -1;
    for (let tau = this.tauMin; tau <= this.tauMax; tau += 1) {
      if (cmnd[tau] < YIN_THRESHOLD) {
        candidate = tau;
        while (candidate < this.tauMax && cmnd[candidate + 1] < cmnd[candidate]) {
          candidate += 1;
        }
        break;
      }
    }

    if (candidate < 0) {
      candidate = this.tauMin;
      for (let tau = this.tauMin + 1; tau <= this.tauMax; tau += 1) {
        if (cmnd[tau] < cmnd[candidate]) candidate = tau;
      }
    }

    // Continuity re-pick: among dips nearly as deep as the chosen one, prefer
    // the one closest to the recent pitch. This is what stops frame-to-frame
    // octave flips while the singer holds one note.
    if (this.lastF0 !== null && timestamp - this.lastF0Time <= CONTINUITY_MEMORY_S) {
      const baseQuality = cmnd[candidate];
      const ceiling = baseQuality + CANDIDATE_MARGIN;
      let bestScore =
        Math.abs(12 * Math.log2(this.detectorRate / candidate / this.lastF0));
      for (let tau = this.tauMin; tau <= this.tauMax; tau += 1) {
        if (cmnd[tau] > ceiling) continue;
        const isDip =
          cmnd[tau] <= cmnd[tau - 1] && (tau === this.tauMax || cmnd[tau] <= cmnd[tau + 1]);
        if (!isDip) continue;
        const semitones = Math.abs(12 * Math.log2(this.detectorRate / tau / this.lastF0));
        const score = semitones + Math.max(0, cmnd[tau] - baseQuality) * CONTINUITY_WEIGHT;
        if (score < bestScore) {
          bestScore = score;
          candidate = tau;
        }
      }
    }

    const clarity = Math.max(0, Math.min(1, 1 - cmnd[candidate]));
    let refinedTau = candidate;
    if (candidate > this.tauMin && candidate < this.tauMax) {
      const left = cmnd[candidate - 1];
      const center = cmnd[candidate];
      const right = cmnd[candidate + 1];
      const denominator = left - 2 * center + right;
      if (Math.abs(denominator) > 1e-12) {
        refinedTau += Math.max(-1, Math.min(1, 0.5 * (left - right) / denominator));
      }
    }

    const frequency = this.detectorRate / refinedTau;
    const valid =
      clarity >= CLARITY_GATE && frequency >= F_MIN && frequency <= F_MAX;
    if (valid) {
      this.lastF0 = frequency;
      this.lastF0Time = timestamp;
    }
    this.postPitch(timestamp, valid ? frequency : null, clarity, rms);
  }

  postPitch(t, f0, clarity, rms) {
    this.port.postMessage({ type: "pitch", t, f0, clarity, rms });
  }
}

registerProcessor("sona-pitch-processor", SonaPitchProcessor);
