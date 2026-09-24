#!/usr/bin/env python3
"""Extract a real DROID seed frame from the DreamZero research clone's bundled
debug episode into the ``--obs-dir`` PNG layout ``dreamzero_pipeline.py``'s
``_load_grid_frame`` reads.

The research repo (``dreamzero0/dreamzero``) ships a real 419-frame DROID
episode as ``debug_image/{exterior_image_1_left,exterior_image_2_left,
wrist_image_left}.mp4`` (320x180, RGB) — used by its own ``test_client_AR.py``
to drive the reference server. This kills the "every DreamZero run so far
used random-pixel PNGs" caveat (docs/DREAMZERO_PORT_PLAN.md §2c) without
needing an external DROID dataset download: frame 0 of this same episode is
exactly what the reference's own test client sends as its step-0 observation.

    python scripts/extract_droid_debug_frames.py \
        --dreamzero-src /workspace/dreamzero --out-dir /workspace/obs/droid_ep0
"""

from __future__ import annotations

import argparse
import os

_CAMERA_FILES = {
    "exterior_image_1_left": "exterior_image_1_left.mp4",
    "exterior_image_2_left": "exterior_image_2_left.mp4",
    "wrist_image_left": "wrist_image_left.mp4",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dreamzero-src", required=True, help="research clone root (has debug_image/)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="matches the reference client's step-0 frame",
    )
    args = ap.parse_args()

    import cv2

    os.makedirs(args.out_dir, exist_ok=True)
    video_dir = os.path.join(args.dreamzero_src, "debug_image")
    for cam_key, fname in _CAMERA_FILES.items():
        path = os.path.join(video_dir, fname)
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not 0 <= args.frame_index < total:
            raise ValueError(f"{fname}: frame_index {args.frame_index} out of range [0, {total})")
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)
        ok, frame_bgr = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"failed to read frame {args.frame_index} from {path}")
        out_path = os.path.join(args.out_dir, f"{cam_key}.png")
        cv2.imwrite(out_path, frame_bgr)  # imwrite expects BGR -- no conversion needed
        print(f"wrote {out_path} ({frame_bgr.shape[1]}x{frame_bgr.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
