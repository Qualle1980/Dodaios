import { describe, expect, it } from "vitest";

import {
  BoundedPcm16Queue,
  Pcm16Packetizer,
  capturePacketSampleCount,
  playbackBufferSampleCount,
} from "./pcmWorkletBuffer";

function pcm(values: number[]): Int16Array {
  return Int16Array.from(values);
}

function decoded(values: number[]): number[] {
  return values.map((value) => value / 0x8000);
}

describe("Pcm16Packetizer", () => {
  it.each([
    [48_000, 960],
    [44_100, 882],
    [16_000, 320],
  ])("emits exactly 50 twenty-millisecond packets at %i Hz", (rate, packetSize) => {
    const packetizer = new Pcm16Packetizer(rate);
    const packets: ArrayBuffer[] = [];
    let remaining = rate;

    while (remaining > 0) {
      const frameSamples = Math.min(128, remaining);
      packetizer.push(new Float32Array(frameSamples).fill(0.5), (packet) => {
        packets.push(packet);
      });
      remaining -= frameSamples;
    }

    expect(capturePacketSampleCount(rate)).toBe(packetSize);
    expect(packets).toHaveLength(50);
    expect(packets.every((packet) => packet.byteLength === packetSize * 2)).toBe(true);
  });

  it("does not exceed 50 packets per second at an unusual context rate", () => {
    const rate = 48_001;
    const packetizer = new Pcm16Packetizer(rate);
    let packetCount = 0;
    let remaining = rate;

    while (remaining > 0) {
      const frameSamples = Math.min(128, remaining);
      packetizer.push(new Float32Array(frameSamples), () => {
        packetCount += 1;
      });
      remaining -= frameSamples;
    }

    expect(packetCount).toBeLessThanOrEqual(50);
    expect(capturePacketSampleCount(rate)).toBe(961);
  });
});

describe("BoundedPcm16Queue", () => {
  it("preserves sample order across ring wraparound", () => {
    const queue = new BoundedPcm16Queue(5);
    queue.enqueue(pcm([1_000, 2_000, 3_000]));

    const prefix = new Float32Array(2);
    expect(queue.dequeueInto(prefix)).toBe(2);
    expect(Array.from(prefix)).toEqual(decoded([1_000, 2_000]));

    expect(queue.enqueue(pcm([4_000, 5_000, 6_000, 7_000]))).toBe(0);
    const rest = new Float32Array(5);
    expect(queue.dequeueInto(rest)).toBe(5);
    expect(Array.from(rest)).toEqual(decoded([3_000, 4_000, 5_000, 6_000, 7_000]));
  });

  it("refuses the newest samples on overrun and keeps the playhead contiguous", () => {
    // Measured 2026-11-13: Gemini Live can deliver ~6.8 s of PCM in <1 s of
    // wall clock. Dropping the OLDEST samples to make room jumped the
    // playhead mid-word; listeners heard that as words running over each
    // other. Refusing what does not fit preserves the audio already queued.
    const queue = new BoundedPcm16Queue(4);
    queue.enqueue(pcm([1_000, 2_000, 3_000]));

    expect(queue.enqueue(pcm([4_000, 5_000, 6_000]))).toBe(2);
    const output = new Float32Array(4);
    queue.dequeueInto(output);
    expect(Array.from(output)).toEqual(decoded([1_000, 2_000, 3_000, 4_000]));
  });

  it("keeps the oldest prefix of an oversized chunk so playback stays contiguous", () => {
    const queue = new BoundedPcm16Queue(4);
    queue.enqueue(pcm([8_000, 9_000]));

    expect(queue.enqueue(pcm([1_000, 2_000, 3_000, 4_000, 5_000, 6_000]))).toBe(4);
    const output = new Float32Array(4);
    queue.dequeueInto(output);
    expect(Array.from(output)).toEqual(decoded([8_000, 9_000, 1_000, 2_000]));
  });

  it("a burst longer than the ring must not jump the playhead mid-stream", () => {
    // Repro shape for the live overlap: play a little, then a burst that
    // exceeds free space arrives. Advancing readIndex (drop-oldest) made the
    // next quantum start hundreds of milliseconds later in the answer.
    const capacity = 1_000;
    const queue = new BoundedPcm16Queue(capacity);
    queue.enqueue(pcm(Array.from({ length: capacity }, (_, i) => i + 1)));

    const first = new Float32Array(100);
    expect(queue.dequeueInto(first)).toBe(100);
    expect(Array.from(first).map((value) => Math.round(value * 0x8000))).toEqual(
      Array.from({ length: 100 }, (_, i) => i + 1),
    );

    expect(queue.enqueue(pcm(Array.from({ length: 500 }, (_, i) => i + 1_001)))).toBe(400);

    const second = new Float32Array(100);
    expect(queue.dequeueInto(second)).toBe(100);
    expect(Array.from(second).map((value) => Math.round(value * 0x8000))).toEqual(
      Array.from({ length: 100 }, (_, i) => i + 101),
    );
  });

  it("flushes queued audio and writes silence on underrun", () => {
    const queue = new BoundedPcm16Queue(4);
    queue.enqueue(pcm([10_000, 20_000]));
    queue.clear();

    const output = new Float32Array([1, 1, 1]);
    expect(queue.dequeueInto(output)).toBe(0);
    expect(Array.from(output)).toEqual([0, 0, 0]);
    expect(queue.length).toBe(0);
  });

  it("bounds a 48 kHz queue to sixty seconds of PCM16", () => {
    expect(playbackBufferSampleCount(48_000)).toBe(2_880_000);
  });
});
