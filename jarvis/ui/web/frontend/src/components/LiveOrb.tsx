import { cn } from "@/lib/utils";

/**
 * The live-conversation mark: the looping JARVIS clip cropped to a circle,
 * with a soft pulse ring. Decorative; the surrounding button carries the label.
 * Reduced motion pauses nothing but the ring, since the clip has no flashes.
 */
export function LiveOrb({ className }: { className?: string }) {
  return (
    <span aria-hidden className={cn("relative inline-flex shrink-0", className)}>
      <span className="absolute inset-0 rounded-full bg-accent/50 motion-safe:animate-ping" />
      <video
        src="/jarvis-live.mp4"
        autoPlay
        loop
        muted
        playsInline
        disablePictureInPicture
        className="relative h-full w-full rounded-full object-cover ring-1 ring-border"
      />
    </span>
  );
}
