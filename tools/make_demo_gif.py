#!/usr/bin/env python3
"""Convert a short MP4 excerpt into a GitHub-friendly animated GIF."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--fps", type=float, default=6.0)
    args = parser.parse_args()

    import cv2
    from PIL import Image

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.input}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    frame_limit = max(1, int(args.seconds * source_fps))
    stride = max(1, int(round(source_fps / args.fps)))
    frames = []
    for source_index in range(frame_limit):
        ok, frame = capture.read()
        if not ok:
            break
        if source_index % stride:
            continue
        height, width = frame.shape[:2]
        target_height = max(1, round(height * args.width / width))
        frame = cv2.resize(frame, (args.width, target_height), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame).convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
    capture.release()
    if not frames:
        raise RuntimeError(f"Video contains no readable frames: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, round(1000 / args.fps))
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"Wrote {args.output} ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
