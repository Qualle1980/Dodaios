/** Duration of each microphone packet sent from the audio render thread. */
export const CAPTURE_PACKET_SECONDS = 0.02;

/**
 * Maximum playback backlog retained inside the AudioWorklet.
 *
 * Sized for a long spoken Live answer that arrives faster than realtime.
 * Measured 2026-11-13: Gemini Live delivered 6.8 s of PCM in 0.99 s of wall
 * clock; a ~27 s answer therefore outruns a ten-second ring. Dropping the
 * oldest samples to stay inside that ring jumped the playhead mid-word and
 * listeners heard words on top of each other. Sixty seconds holds a long
 * answer (~5.8 MiB PCM16 at 48 kHz) without that jump; a still-larger burst
 * refuses the newest samples instead (see BoundedPcm16Queue.enqueue).
 */
export const MAX_PLAYBACK_BUFFER_SECONDS = 60;

/**
 * Jitter reserve banked before playback starts and after a true underrun.
 *
 * Provider PCM arrives over a WebSocket in bursts, is resampled on the main
 * thread and is then read by the render thread once per 128-sample quantum
 * (~2.7 ms at 48 kHz). Without a reserve, any scheduling or network gap shorter
 * than one packet is rendered as digital silence mid-waveform — a hard
 * discontinuity, audible as a chop plus a click. Mirrors the desktop surface's
 * proven 180 ms reserve (`jarvis/realtime/desktop.py::DEFAULT_PREBUFFER_MS`).
 */
export const PLAYBACK_PREBUFFER_SECONDS = 0.15;

/**
 * Longest the reserve is allowed to hold audio back.
 *
 * A short reply ("yes.") can be the whole burst, so waiting for a full reserve
 * that will never arrive would swallow it. Once audio has been queued for this
 * long, playback starts under-filled rather than never.
 */
export const PLAYBACK_PREBUFFER_TIMEOUT_SECONDS = 0.24;

/** Ramp length applied at a resume edge and at an underrun edge. */
const PLAYBACK_DECLICK_SECONDS = 0.001;

function validatedSampleCount(sampleRate: number, seconds: number): number {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new RangeError("sampleRate must be a positive finite number");
  }
  return Math.max(1, Math.round(sampleRate * seconds));
}

export function capturePacketSampleCount(sampleRate: number): number {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new RangeError("sampleRate must be a positive finite number");
  }
  // Round upward so even an unusual non-divisible context rate never crosses
  // the 50 messages/s ceiling.
  return Math.max(1, Math.ceil(sampleRate * CAPTURE_PACKET_SECONDS));
}

export function playbackBufferSampleCount(sampleRate: number): number {
  return validatedSampleCount(sampleRate, MAX_PLAYBACK_BUFFER_SECONDS);
}

export function playbackPrebufferSampleCount(sampleRate: number): number {
  return validatedSampleCount(sampleRate, PLAYBACK_PREBUFFER_SECONDS);
}

function floatToPcm16(sample: number): number {
  const clamped = Math.max(-1, Math.min(1, sample));
  return clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
}

/** Coalesces arbitrary render quanta into fixed-duration PCM16 packets. */
export class Pcm16Packetizer {
  readonly packetSamples: number;
  private packet: Int16Array;
  private used = 0;

  constructor(sampleRate: number) {
    this.packetSamples = capturePacketSampleCount(sampleRate);
    this.packet = new Int16Array(this.packetSamples);
  }

  push(samples: Float32Array, emit: (packet: ArrayBuffer) => void): void {
    let sourceOffset = 0;
    while (sourceOffset < samples.length) {
      const copyCount = Math.min(
        this.packetSamples - this.used,
        samples.length - sourceOffset,
      );
      for (let i = 0; i < copyCount; i++) {
        this.packet[this.used + i] = floatToPcm16(samples[sourceOffset + i]);
      }
      this.used += copyCount;
      sourceOffset += copyCount;

      if (this.used === this.packetSamples) {
        const completed = this.packet;
        this.packet = new Int16Array(this.packetSamples);
        this.used = 0;
        // Every packet is allocated above with `new Int16Array(length)`, so
        // its backing store is always a transferable ArrayBuffer.
        emit(completed.buffer as ArrayBuffer);
      }
    }
  }
}

/**
 * Fixed-capacity PCM16 ring used by the playback render thread.
 *
 * On overrun, enqueue() keeps the samples already queued (and the prefix of
 * the new chunk that still fits) and returns how many NEW samples were
 * refused. Advancing the read cursor used to drop the oldest audio instead;
 * that jump was the live overlap of 2026-11-13. Refusing the newest audio
 * leaves a contiguous answer on the speaker; barge-in still clears the ring
 * through the existing flush path.
 */
export class BoundedPcm16Queue {
  private readonly samples: Int16Array;
  private readIndex = 0;
  private writeIndex = 0;
  private queued = 0;

  constructor(readonly capacity: number) {
    if (!Number.isInteger(capacity) || capacity <= 0) {
      throw new RangeError("capacity must be a positive integer");
    }
    this.samples = new Int16Array(capacity);
  }

  get length(): number {
    return this.queued;
  }

  clear(): void {
    this.readIndex = 0;
    this.writeIndex = 0;
    this.queued = 0;
  }

  enqueue(input: Int16Array): number {
    if (input.length === 0) return 0;

    const free = this.capacity - this.queued;
    if (free <= 0) return input.length;

    const accepted = Math.min(input.length, free);
    const refused = input.length - accepted;
    const chunk = accepted === input.length ? input : input.subarray(0, accepted);

    const firstCopy = Math.min(chunk.length, this.capacity - this.writeIndex);
    this.samples.set(chunk.subarray(0, firstCopy), this.writeIndex);
    if (firstCopy < chunk.length) {
      this.samples.set(chunk.subarray(firstCopy), 0);
    }
    this.writeIndex = (this.writeIndex + chunk.length) % this.capacity;
    this.queued += chunk.length;
    return refused;
  }

  dequeueInto(output: Float32Array): number {
    output.fill(0);
    const readCount = Math.min(output.length, this.queued);
    const firstRead = Math.min(readCount, this.capacity - this.readIndex);

    for (let i = 0; i < firstRead; i++) {
      output[i] = this.samples[this.readIndex + i] / 0x8000;
    }
    const secondRead = readCount - firstRead;
    for (let i = 0; i < secondRead; i++) {
      output[firstRead + i] = this.samples[i] / 0x8000;
    }

    this.readIndex = (this.readIndex + readCount) % this.capacity;
    this.queued -= readCount;
    return readCount;
  }
}

/**
 * Playback ring with a jitter reserve — the browser surface's equivalent of
 * `DesktopRealtimePlayback`'s prebuffer.
 *
 * The reserve is banked in exactly two places and nowhere else, so it never
 * becomes permanently added latency: at the start of a burst, and after a true
 * underrun (the render thread asked for more than the ring held). While
 * playing, samples are handed out at wall clock with no delay whatsoever.
 * Both edges are ramped, because jumping between a non-zero sample and silence
 * is itself an audible click even when the gap is later filled.
 */
export class JitterBufferedPcm16Queue {
  private readonly queue: BoundedPcm16Queue;
  private readonly prebufferSamples: number;
  private readonly prebufferTimeoutSamples: number;
  private readonly declickSamples: number;
  private priming = true;
  private primingSamples = 0;
  private resuming = false;

  constructor(sampleRate: number) {
    this.queue = new BoundedPcm16Queue(playbackBufferSampleCount(sampleRate));
    this.prebufferSamples = playbackPrebufferSampleCount(sampleRate);
    this.prebufferTimeoutSamples = validatedSampleCount(
      sampleRate,
      PLAYBACK_PREBUFFER_TIMEOUT_SECONDS,
    );
    this.declickSamples = validatedSampleCount(sampleRate, PLAYBACK_DECLICK_SECONDS);
  }

  /** Samples currently banked, whether or not playback has started. */
  get length(): number {
    return this.queue.length;
  }

  /** True while the reserve is still being banked. */
  get isPriming(): boolean {
    return this.priming;
  }

  clear(): void {
    this.queue.clear();
    this.priming = true;
    this.primingSamples = 0;
    this.resuming = false;
  }

  enqueue(input: Int16Array): number {
    return this.queue.enqueue(input);
  }

  /** Fill one render quantum. Returns the number of real samples written. */
  render(output: Float32Array): number {
    if (this.priming) {
      if (this.queue.length === 0) {
        // Silence between turns must not burn the timeout: the clock starts
        // when audio actually exists to be held back.
        this.primingSamples = 0;
        output.fill(0);
        return 0;
      }
      if (
        this.queue.length < this.prebufferSamples &&
        this.primingSamples < this.prebufferTimeoutSamples
      ) {
        this.primingSamples += output.length;
        output.fill(0);
        return 0;
      }
      this.priming = false;
      this.primingSamples = 0;
      this.resuming = true;
    }

    const rendered = this.queue.dequeueInto(output);
    if (this.resuming) {
      this.resuming = false;
      this.rampIn(output, rendered);
    }
    if (rendered < output.length) {
      this.rampOut(output, rendered);
      this.priming = true;
      this.primingSamples = 0;
    }
    return rendered;
  }

  private rampIn(output: Float32Array, rendered: number): void {
    const ramp = Math.min(rendered, this.declickSamples);
    for (let i = 0; i < ramp; i++) output[i] *= (i + 1) / (ramp + 1);
  }

  private rampOut(output: Float32Array, rendered: number): void {
    if (rendered === 0) return;
    const last = output[rendered - 1];
    if (last === 0) return;
    // `dequeueInto` already zeroed the tail; overwrite the first slice of it
    // with a ramp down from the last real sample.
    const ramp = Math.min(output.length - rendered, this.declickSamples);
    for (let i = 0; i < ramp; i++) {
      output[rendered + i] = last * (1 - (i + 1) / (ramp + 1));
    }
  }
}
