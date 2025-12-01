/**
 * AudioWorklet processor for low-latency audio capture
 *
 * Features:
 * - Runs on a separate thread (not main thread)
 * - Low-latency processing
 * - Proper resampling with anti-aliasing
 * - Outputs 20ms frames at 16kHz (320 samples = 640 bytes)
 */

class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Target: 16kHz mono, 20ms frames = 320 samples
    this.targetSampleRate = 16000;
    this.frameSize = 320; // 20ms at 16kHz

    // Buffer for accumulating samples before sending
    this.sampleBuffer = [];

    // Resampling state
    this.resamplingBuffer = [];

    // Simple low-pass filter state for anti-aliasing
    this.filterState = 0;

    // VAD state
    this.voiceThreshold = 0.005;
    this.isSpeaking = false;
    this.silenceFrameCount = 0;
    this.maxSilenceFrames = 15; // ~300ms at 20ms frames

    // Listen for messages from main thread
    this.port.onmessage = (event) => {
      if (event.data.type === "setVoiceThreshold") {
        this.voiceThreshold = event.data.value;
      } else if (event.data.type === "setSilenceFrames") {
        this.maxSilenceFrames = event.data.value;
      }
    };
  }

  /**
   * Simple low-pass filter for anti-aliasing before downsampling
   * Cutoff at ~7kHz (Nyquist for 16kHz)
   */
  lowPassFilter(samples, cutoffRatio) {
    const alpha = cutoffRatio; // Simple RC filter coefficient
    const filtered = new Float32Array(samples.length);
    let lastValue = this.filterState;

    for (let i = 0; i < samples.length; i++) {
      lastValue = lastValue + alpha * (samples[i] - lastValue);
      filtered[i] = lastValue;
    }

    this.filterState = lastValue;
    return filtered;
  }

  /**
   * High-quality resampling with anti-aliasing
   */
  resample(inputData, sourceRate, targetRate) {
    if (sourceRate === targetRate) {
      return inputData;
    }

    // Apply low-pass filter before downsampling (anti-aliasing)
    // Cutoff at ~80% of target Nyquist to avoid aliasing
    const cutoffRatio = Math.min(1.0, (targetRate / sourceRate) * 0.8);
    const filtered = this.lowPassFilter(inputData, cutoffRatio);

    // Resample using linear interpolation
    const ratio = sourceRate / targetRate;
    const outputLength = Math.floor(inputData.length / ratio);
    const output = new Float32Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
      const srcIndex = i * ratio;
      const srcIndexFloor = Math.floor(srcIndex);
      const srcIndexCeil = Math.min(srcIndexFloor + 1, filtered.length - 1);
      const t = srcIndex - srcIndexFloor;

      // Linear interpolation
      output[i] =
        filtered[srcIndexFloor] * (1 - t) + filtered[srcIndexCeil] * t;
    }

    return output;
  }

  /**
   * Convert Float32 samples to Int16 PCM (little-endian)
   */
  float32ToInt16(float32Array) {
    const int16Array = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      // Clamp to [-1, 1]
      const val = Math.max(-1, Math.min(1, float32Array[i]));
      // Convert to 16-bit signed integer
      int16Array[i] = val < 0 ? val * 0x8000 : val * 0x7fff;
    }
    return int16Array;
  }

  /**
   * Calculate RMS energy for voice activity detection
   */
  calculateRMS(samples) {
    let sum = 0;
    for (let i = 0; i < samples.length; i++) {
      sum += samples[i] * samples[i];
    }
    return Math.sqrt(sum / samples.length);
  }

  /**
   * Process audio - called by the audio system
   * @param {Float32Array[][]} inputs - Input audio buffers
   * @param {Float32Array[][]} outputs - Output audio buffers (passthrough)
   * @param {Object} parameters - Audio parameters
   */
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) {
      return true;
    }

    const inputData = input[0]; // Mono channel
    const sourceSampleRate = sampleRate; // Global from AudioWorklet

    // Resample to 16kHz with anti-aliasing
    const resampled = this.resample(
      inputData,
      sourceSampleRate,
      this.targetSampleRate
    );

    // Add to buffer
    for (let i = 0; i < resampled.length; i++) {
      this.sampleBuffer.push(resampled[i]);
    }

    // Process complete frames (320 samples = 20ms at 16kHz)
    while (this.sampleBuffer.length >= this.frameSize) {
      // Extract frame
      const frame = new Float32Array(this.frameSize);
      for (let i = 0; i < this.frameSize; i++) {
        frame[i] = this.sampleBuffer.shift();
      }

      // Voice activity detection
      const rms = this.calculateRMS(frame);
      const hasVoice = rms > this.voiceThreshold;

      // Track speaking state with hysteresis
      if (hasVoice) {
        this.silenceFrameCount = 0;
        if (!this.isSpeaking) {
          this.isSpeaking = true;
          this.port.postMessage({ type: "voiceStart" });
        }
      } else if (this.isSpeaking) {
        this.silenceFrameCount++;
        if (this.silenceFrameCount >= this.maxSilenceFrames) {
          this.isSpeaking = false;
          this.port.postMessage({ type: "voiceEnd" });
        }
      }

      // Send audio data if speaking (or in grace period)
      if (this.isSpeaking || hasVoice) {
        // Convert to Int16 PCM
        const pcm16 = this.float32ToInt16(frame);

        // Send to main thread
        this.port.postMessage(
          {
            type: "audioData",
            audio: pcm16.buffer,
            rms: rms,
            hasVoice: hasVoice,
          },
          [pcm16.buffer]
        ); // Transfer buffer for performance
      }

      // Send RMS for visualization (throttled)
      if (Math.random() < 0.2) {
        // ~20% of frames
        this.port.postMessage({
          type: "rms",
          rms: rms,
          hasVoice: hasVoice,
        });
      }
    }

    return true; // Keep processor alive
  }
}

// Register the processor
registerProcessor("audio-processor", AudioProcessor);
