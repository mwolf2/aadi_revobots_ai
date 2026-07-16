# -*- coding: utf-8 -*-
"""
Apply semantic segmentation masks to every session_<i>.mp4 produced by
split_session_agv.py, in place or into a mirror directory.

Sits between step 1 (split) and step 2 (convert) in data-pipeline.py.

The .jsonl files are copied through untouched so frame indices stay aligned
with the masked video frames — data_convert_agv.py continues to work with
zero changes.

Standalone usage:
    python mask_sessions_agv.py <split_session_dir> \
        --seg-mode semantic \
        --seg-weights yolo26n-sem.pt \
        --det-weights yolo26n.pt

Importable:
    from mask_sessions_agv import run_mask
    masked_dir = run_mask(parent_dir, seg_mode="semantic", ...)
"""

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.plotting import colors


# ---------------------------------------------------------------------------
# Masker — models loaded ONCE per run, not per frame.
# ---------------------------------------------------------------------------

class SemanticMasker:
    def __init__(
        self,
        seg_mode: str = "semantic",
        seg_weights: str = "yolo26n-sem.pt",
        det_weights: str = "yolo26n.pt",
        det_classes_to_keep=(0, 1, 9, 10, 11, 12),
        seg_conf: float = 0.25,
        overlay_alpha: float = 0.0,   # 0.0 = pure black bg (original behavior)
    ):
        assert seg_mode in ("semantic", "segmentation")
        self.seg_mode = seg_mode
        self.det_classes_to_keep = list(det_classes_to_keep)
        self.seg_conf = seg_conf
        self.overlay_alpha = float(overlay_alpha)

        print(f"[masker] loading det weights: {det_weights}")
        self.det_model = YOLO(det_weights)
        print(f"[masker] loading seg weights ({seg_mode}): {seg_weights}")
        self.seg_model = YOLO(seg_weights)

        if seg_mode == "semantic":
            self.ignore_class = self.seg_model.model.nc - 1
        else:
            self.ignore_class = None

    # ---- drawing ----------------------------------------------------------

    @staticmethod
    def _darker(color, factor=0.5):
        return tuple(int(c * factor) for c in color)

    def _draw_semantic(self, sem_result, canvas):
        if sem_result.semantic_mask is None:
            return canvas
        class_map = sem_result.semantic_mask.data.cpu().numpy()
        for cls in np.unique(class_map):
            if cls == self.ignore_class:
                continue
            color = (0, 255, 255) if cls == 3 else colors(int(cls), bgr=True)
            canvas[class_map == cls] = color
        return canvas

    def _draw_instance(self, frame, seg_result, canvas):
        if seg_result.masks is None:
            return canvas
        masks = seg_result.masks.data.cpu().numpy()
        classes = seg_result.boxes.cls.cpu().numpy().astype(int)
        for mask, cls in zip(masks, classes):
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            binary = (mask > 0.5).astype(np.uint8)
            color = (0, 255, 255) if cls == 3 else colors(cls, True)
            canvas[binary == 1] = color
            if cls != 3:
                contours, _ = cv2.findContours(
                    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(canvas, contours, -1, colors(cls, True), 2)
        return canvas

    def _draw_detection(self, det_result, canvas):
        if det_result.boxes is None:
            return canvas
        boxes = det_result.boxes.xyxy.cpu().numpy()
        classes = det_result.boxes.cls.cpu().numpy().astype(int)
        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            h = y2 - y1
            y1_new = int(y2 - 0.25 * h)
            color = (0, 0, 255) if cls in (0, 1) else colors(cls, True)
            border = self._darker(color)
            cv2.rectangle(canvas, (x1, y1_new), (x2, y2), color, -1)
            cv2.rectangle(canvas, (x1, y1_new), (x2, y2), border, 3)
        return canvas

    # ---- public -----------------------------------------------------------

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.seg_mode == "semantic":
            seg_results = self.seg_model.predict(
                frame_bgr, task="semantic", conf=self.seg_conf, verbose=False
            )
        else:
            seg_results = self.seg_model(frame_bgr, verbose=False)
        det_results = self.det_model(
            frame_bgr, classes=self.det_classes_to_keep, verbose=False
        )

        # start from black canvas OR faded original, per overlay_alpha
        if self.overlay_alpha <= 0.0:
            canvas = np.zeros_like(frame_bgr)
        else:
            canvas = (frame_bgr.astype(np.float32) * self.overlay_alpha).astype(np.uint8)

        if self.seg_mode == "semantic":
            canvas = self._draw_semantic(seg_results[0], canvas)
        else:
            canvas = self._draw_instance(frame_bgr, seg_results[0], canvas)
        canvas = self._draw_detection(det_results[0], canvas)
        return canvas


# ---------------------------------------------------------------------------
# Session-level driver
# ---------------------------------------------------------------------------

def _find_sessions(parent_dir: Path):
    """Return sorted list of session_<i>/ dirs that have both .mp4 and .jsonl."""
    out = []
    for sub in sorted(parent_dir.iterdir()):
        if not sub.is_dir():
            continue
        mp4 = sub / f"{sub.name}.mp4"
        jsonl = sub / f"{sub.name}.jsonl"
        if mp4.is_file() and jsonl.is_file():
            out.append(sub)
    return out


def _mask_one_video(masker: SemanticMasker, src_mp4: Path, dst_mp4: Path,
                    fps_override=None):
    cap = cv2.VideoCapture(str(src_mp4))
    if not cap.isOpened():
        print(f"[!] Could not open {src_mp4}, skipping.")
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    # OpenCV returns 0.0 for CAP_PROP_FPS on some mp4s; a zero/invalid fps makes
    # VideoWriter silently refuse to open and write nothing. Prefer the fps the
    # pipeline was configured with, then the source's, then a sane fallback.
    fps = fps_override or src_fps
    if not fps or fps <= 0:
        fps = 30.0
        print(f"[!] {src_mp4.name}: invalid source fps ({src_fps!r}); "
              f"falling back to {fps}.")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Write to a temp path so a crash mid-encode never leaves a half-written
    # file at the final path (which convert would happily consume). The temp
    # name MUST keep the .mp4 extension: OpenCV's FFmpeg backend picks the
    # output container from the extension, and a ".tmp" suffix leaves it unable
    # to infer a muxer, so VideoWriter silently fails to open.
    tmp = dst_mp4.with_name(dst_mp4.stem + ".partial" + dst_mp4.suffix)
    writer = cv2.VideoWriter(str(tmp), fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(
            f"[!] cv2.VideoWriter could not open an encoder for {tmp} "
            f"(fps={fps}, size={w}x{h}, fourcc=mp4v). The source clip may be "
            f"invalid, or this OpenCV build lacks an mp4v encoder."
        )

    pbar = tqdm(total=total, desc=f"  {src_mp4.parent.name}", unit="f",
                leave=True, dynamic_ncols=True)
    n = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            masked = masker.process_frame(frame)
            writer.write(masked)
            n += 1
            pbar.update(1)
    finally:
        cap.release()
        writer.release()
        pbar.close()

    if not tmp.exists():
        raise SystemExit(
            f"[!] Masking wrote no output for {src_mp4} ({n} frame(s) read); "
            f"leaving {dst_mp4} untouched."
        )
    os.replace(tmp, dst_mp4)
    return n


def run_mask(
    parent_dir,
    output_dir=None,
    seg_mode: str = "semantic",
    seg_weights: str = "yolo26n-sem.pt",
    det_weights: str = "yolo26n.pt",
    det_classes_to_keep=(0, 1, 9, 10, 11, 12),
    seg_conf: float = 0.25,
    overlay_alpha: float = 0.0,
    in_place: bool = False,
    fps=None,
):
    """Mask every session_<i>.mp4 under `parent_dir`.

    - If `in_place=True`, overwrites the mp4s in `parent_dir` (jsonls
      unchanged). Original videos are backed up as session_<i>.raw.mp4.
    - Else writes a mirror tree at `output_dir` (or `<parent_dir>_masked/`
      if not given) containing masked mp4s + copied jsonls, and returns
      that path so it can be fed straight to run_convert().
    """
    parent = Path(os.path.expanduser(parent_dir)).resolve()
    if not parent.is_dir():
        raise SystemExit(f"[!] Not a directory: {parent}")

    sessions = _find_sessions(parent)
    if not sessions:
        raise SystemExit(f"[!] No session_<i>/ folders in {parent}")

    if in_place:
        out_root = parent
    else:
        out_root = (Path(os.path.expanduser(output_dir)).resolve()
                    if output_dir else parent.with_name(parent.name + "_masked"))
        out_root.mkdir(parents=True, exist_ok=True)

    print(f"[mask] input:   {parent}")
    print(f"[mask] output:  {out_root}  {'(in place)' if in_place else ''}")
    print(f"[mask] sessions: {len(sessions)}")

    masker = SemanticMasker(
        seg_mode=seg_mode,
        seg_weights=seg_weights,
        det_weights=det_weights,
        det_classes_to_keep=det_classes_to_keep,
        seg_conf=seg_conf,
        overlay_alpha=overlay_alpha,
    )

    total_frames = 0
    outer = tqdm(total=len(sessions), desc="Sessions", unit="s", position=0,
                 dynamic_ncols=True)
    try:
        for sess in sessions:
            name = sess.name
            src_mp4 = sess / f"{name}.mp4"
            src_jsonl = sess / f"{name}.jsonl"

            if in_place:
                # keep a copy of the raw video before overwriting
                backup = sess / f"{name}.raw.mp4"
                if not backup.exists():
                    shutil.copy2(src_mp4, backup)
                dst_dir = sess
            else:
                dst_dir = out_root / name
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_jsonl, dst_dir / f"{name}.jsonl")

            dst_mp4 = dst_dir / f"{name}.mp4"
            total_frames += _mask_one_video(masker, src_mp4, dst_mp4,
                                             fps_override=fps)
            outer.set_postfix(frames=total_frames)
            outer.update(1)
    finally:
        outer.close()

    print(f"\n[✓] Masked {total_frames} frames across {len(sessions)} session(s).")
    print(f"[✓] Output: {out_root}")
    return str(out_root)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("parent_dir",
                   help="split_session_<...>/ folder containing session_<i>/ dirs")
    p.add_argument("--output-dir", default=None,
                   help="Where to write masked mirror tree "
                        "(default: <parent_dir>_masked/).")
    p.add_argument("--in-place", action="store_true",
                   help="Overwrite mp4s inside parent_dir. Originals are "
                        "backed up as session_<i>.raw.mp4.")
    p.add_argument("--seg-mode", choices=["semantic", "segmentation"],
                   default="semantic")
    p.add_argument("--seg-weights", default="yolo26n-sem.pt")
    p.add_argument("--det-weights", default="yolo26n.pt")
    p.add_argument("--seg-conf", type=float, default=0.25)
    p.add_argument("--overlay-alpha", type=float, default=0.0,
                   help="0.0 = pure black bg (default, matches original "
                        "video_mask_batch.py). >0 fades the raw frame in "
                        "behind the mask (0.3 is a reasonable overlay).")
    p.add_argument("--det-classes", type=int, nargs="*",
                   default=[0, 1, 9, 10, 11, 12])
    return p.parse_args()


def main():
    args = parse_args()
    run_mask(
        parent_dir=args.parent_dir,
        output_dir=args.output_dir,
        seg_mode=args.seg_mode,
        seg_weights=args.seg_weights,
        det_weights=args.det_weights,
        det_classes_to_keep=tuple(args.det_classes),
        seg_conf=args.seg_conf,
        overlay_alpha=args.overlay_alpha,
        in_place=args.in_place,
    )


if __name__ == "__main__":
    main()