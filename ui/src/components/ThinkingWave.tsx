import React from "react";

type Props = {
  active: boolean;
  className?: string;
  strokeWidth?: number;
  height?: number;
};

/**
 * A simple SVG sine-wave line used as a "thinking" indicator.
 * Render it as an overlay bar (e.g., at the top edge of the composer).
 */
export function ThinkingWave({
  active,
  className,
  strokeWidth = 4,
  height = 12,
}: Props) {
  return (
    <svg
      className={className ?? ""}
      viewBox="0 0 800 20"
      preserveAspectRatio="none"
      style={{ height, width: "100%", opacity: active ? 1 : 0 }}
      aria-hidden="true"
    >
      <path
        d="M0 10 C 40 2, 80 18, 120 10 S 200 2, 240 10 S 320 18, 360 10 S 440 2, 480 10 S 560 18, 600 10 S 680 2, 720 10 S 780 18, 800 10"
        fill="none"
        stroke="currentColor"
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        className={active ? "wavePath" : ""}
      />
    </svg>
  );
}
