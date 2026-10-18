/**
 * The JARVIS HUD emblem as the voice bubble's call button.
 *
 * One transparent image, animated with CSS only. It never rotates (the
 * wordmark sits in its centre); instead its glow pulses whenever a session is
 * live, and the pulse is quicker while the assistant is busy. Reduced motion
 * holds it still.
 */
import { cn } from "@/lib/utils";
import type { VoiceState } from "@/store/events";

const PULSE_SECONDS: Partial<Record<VoiceState, number>> = {
  connecting: 1,
  listening: 2,
  thinking: 1,
  speaking: 1.4,
};

export function JarvisHudOrb({ state, size }: { state: VoiceState; size: number }) {
  const pulse = PULSE_SECONDS[state];
  return (
    <span
      data-testid="jarvis-hud-orb"
      data-state={state}
      className="relative block"
      style={{ width: size, height: size }}
    >
      <img
        src="/jarvis-hud.webp"
        alt=""
        draggable={false}
        className={cn(
          "block h-full w-full select-none [filter:drop-shadow(0_0_12px_rgba(56,189,248,0.55))]",
          pulse && "motion-safe:animate-pulse",
        )}
        style={pulse ? { animationDuration: `${pulse}s` } : undefined}
      />
    </span>
  );
}
