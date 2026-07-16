# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 22:10:49 2026

@author: Aadi
"""

"""
Build a LeRobot HuggingFace dataset for `revobots_agv_follower` from pre-recorded
split sessions (output of split_session_agv.py) — WITHOUT using the physical robot.

This script intentionally mirrors lerobot's official `lerobot_record.py` flow as
closely as possible. The ONLY thing that changes is the data source: instead of
pulling observations from a live `Robot` and actions from a live `Teleoperator`,
we pull them from (video frame + JSONL row) pairs sitting on disk.

Specifically, we use the same helpers / context manager that the production
recorder uses:
    - hw_to_dataset_features    (feature schema)
    - make_default_processors   (observation processor — applied here to keep
                                 training data symmetric with what inference
                                 feeds into the policy. See [SYMMETRY] notes.)
    - build_dataset_frame       (per-frame packing — guarantees key/shape match)
    - VideoEncodingManager      (proper video encoding lifecycle around the loop)
    - dataset.add_frame / save_episode / finalize

[SYMMETRY]
    inference_api.run_inference() calls
        obs_processed = robot_observation_processor(robot.get_observation())
        frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
    before handing the frame to the policy. To avoid a silent train/inference
    distribution gap if the default processor ever stops being identity (key
    rename, dtype coercion, etc.), we apply the same processor here. The
    action processor in inference is policy-output → robot-input, which has
    no analog in the dataset, so actions stay raw — matching what
    lerobot_record.py does in the live recording path.

Input layout (produced by split_session_agv.py):

    <parent-dir>/
        session_0/  session_0.mp4  session_0.jsonl
        session_1/  session_1.mp4  session_1.jsonl
        ...

Each session_<i>/ folder becomes ONE episode.

Expected JSONL row format (one row per video frame, same order as frames):

    {
      "frame_index": 0,
      "linear_velocity": 0.0,
      "angular_velocity": 0.0,
      "gps_latitude":  45.45799...,
      "gps_longitude": -122.85448...,
      ... (any other fields are ignored)
    }
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import make_default_processors          # [SYMMETRY] new import
from lerobot.utils.constants import ACTION, OBS_STR


# ---------------------------------------------------------------------------
# Feature schema — mirrors RevobotsAGVFollower.observation_features /
# action_features. Defined here so we DON'T need to import the robot class
# (which would pull in ROS, GpsReader, cameras, etc.).
# ---------------------------------------------------------------------------

CAMERA_KEY = "front"
CAMERA_HEIGHT = 480
CAMERA_WIDTH = 640

ROBOT_ACTION_FEATURES = {
    "lin_x": float,
    "ang_z": float,
}

ROBOT_OBSERVATION_FEATURES = {
    "lin_x": float,
    "ang_z": float,
    "lat":   float,
    "long":  float,
    "orientation": float,
    CAMERA_KEY: (CAMERA_HEIGHT, CAMERA_WIDTH, 3),
}


def build_dataset_features() -> dict:
    """Build the LeRobotDataset feature schema using the same helper the official
    record script uses. With use_videos=True at create time, the camera tuple
    becomes a 'video' feature; scalar floats get packed into 'observation.state'
    and 'action' vectors."""
    action_features = hw_to_dataset_features(ROBOT_ACTION_FEATURES, ACTION)
    obs_features = hw_to_dataset_features(ROBOT_OBSERVATION_FEATURES, OBS_STR)
    return {**action_features, **obs_features}


# ---------------------------------------------------------------------------
# JSONL row → raw observation dict + raw action dict
# These are the SAME shape that RevobotsAGVFollower.get_observation() and
# the teleop would produce on a live robot. build_dataset_frame then handles
# the packing into dataset keys.
# Edit this function if your JSONL uses a different layout.
# ---------------------------------------------------------------------------

def row_to_raw_obs_and_action(row: dict, image_rgb: np.ndarray) -> tuple[dict, dict]:
    """Return (raw_observation, raw_action) in the robot's native key space.

    Schema produced by the upstream recorder is flat. Action and observation
    share the same velocity fields (commanded == measured for this dataset).
    """
    lin_x = float(row.get("linear_velocity", 0.0))
    ang_z = float(row.get("angular_velocity", 0.0))
    lat   = float(row.get("gps_latitude",  0.0))
    lon   = float(row.get("gps_longitude", 0.0))
    ori   = float(row.get("orientation",    0.0))

    raw_action = {
        "lin_x": lin_x,
        "ang_z": ang_z,
    }
    raw_observation = {
        "lin_x":       lin_x,
        "ang_z":       ang_z,
        "lat":         lat,
        "long":        lon,
        "orientation": ori,
        CAMERA_KEY:    image_rgb,
    }
    return raw_observation, raw_action


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------

def find_session_folders(input_root: Path) -> list[Path]:
    """Return all session_<i>/ folders that contain a matching mp4 + jsonl pair."""
    sessions = []
    for sub in sorted(input_root.iterdir()):
        if not sub.is_dir():
            continue
        name = sub.name
        mp4 = sub / f"{name}.mp4"
        jsonl = sub / f"{name}.jsonl"
        if mp4.is_file() and jsonl.is_file():
            sessions.append(sub)
    return sessions


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Parquet footer verification / recovery
# ---------------------------------------------------------------------------

PARQUET_MAGIC = b"PAR1"


def _has_parquet_footer(path: Path) -> bool:
    """A valid parquet file ends with the 4-byte magic 'PAR1'. If it doesn't,
    the writer's close() never wrote the footer and the file is unreadable."""
    if not path.is_file() or path.stat().st_size < 8:
        return False
    with open(path, "rb") as f:
        f.seek(-4, 2)
        return f.read(4) == PARQUET_MAGIC


def _try_recover_parquet(path: Path) -> None:
    """Attempt to recover a footer-less parquet file.

    Strategy: streaming pq.ParquetWriter writes row groups sequentially with
    'PAR1' at the start but never appended the footer. We can scan the file
    for the row groups using pyarrow's lower-level interface, but the simpler
    practical approach is to re-read the underlying buffer via pyarrow's
    BufferReader with explicit handling. In practice the cleanest recovery
    is to fail loud here and tell the user what to do.
    """
    print(f"      [recover] {path.name}: footer missing; cannot auto-recover "
          f"a streamed parquet without the row-group index. The episode data "
          f"was streamed but never committed.")
    try:
        import pyarrow.parquet as pq
        pq.read_table(str(path))
        print(f"      [recover] {path.name}: unexpectedly readable, no action needed.")
    except Exception as e:
        print(f"      [recover] {path.name}: confirmed unreadable ({type(e).__name__}).")


# ---------------------------------------------------------------------------
# Per-session → one episode
# ---------------------------------------------------------------------------

def add_session_as_episode(
    session_dir: Path,
    dataset: LeRobotDataset,
    task: str,
    robot_observation_processor,                          # [SYMMETRY] new arg
) -> int:
    """Read one session_<i>/ folder and append it as a single episode.
    Mirrors the inner body of record_loop() — just without the live robot.

    [SYMMETRY] raw_obs is run through `robot_observation_processor` exactly
    the way `inference_api.inference_loop()` does, so the dataset stores the
    same shape the policy will be queried with at inference time.
    """
    name = session_dir.name
    mp4_path = session_dir / f"{name}.mp4"
    jsonl_path = session_dir / f"{name}.jsonl"

    rows = load_jsonl(jsonl_path)

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        print(f"[!] Could not open {mp4_path}, skipping.")
        return 0

    total_frames_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(total_frames_vid, len(rows))

    pbar = tqdm(
        total=total_frames,
        desc=f"  {name}",
        unit="f",
        leave=True,
        dynamic_ncols=True,
    )
    n_written = 0
    try:
        for i in range(total_frames):
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # cv2 reads BGR; cameras in lerobot return RGB.
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            raw_obs, raw_action = row_to_raw_obs_and_action(rows[i], frame_rgb)

            # [SYMMETRY] Apply the same observation processor that the
            # inference loop applies before build_dataset_frame. If the
            # default processor is identity this is a no-op; if it ever
            # gains logic (key renames, dtype coercion, etc.) the dataset
            # and inference input stay aligned automatically.
            obs_processed = robot_observation_processor(raw_obs)

            # Same packing the official record_loop uses.
            observation_frame = build_dataset_frame(
                dataset.features, obs_processed, prefix=OBS_STR
            )
            action_frame = build_dataset_frame(
                dataset.features, raw_action, prefix=ACTION
            )
            frame = {**observation_frame, **action_frame, "task": task}
            dataset.add_frame(frame)
            n_written += 1

            if (i & 0x1F) == 0:
                pbar.set_postfix(
                    lin=f"{raw_action['lin_x']:+.2f}",
                    ang=f"{raw_action['ang_z']:+.2f}",
                )
            pbar.update(1)
    finally:
        cap.release()
        pbar.close()

    # Same call the official record() makes after each episode.
    dataset.save_episode()
    return n_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--parent-dir", required=True, dest="parent_dir",
                   help="Parent folder containing session_0/, session_1/, ...")
    p.add_argument("--repo-id", required=True,
                   help="HuggingFace dataset repo id, e.g. user/revobots_agv_v1")
    p.add_argument("--task", default="Drive the AGV.",
                   help="Task description string stored with every frame.")
    p.add_argument("--fps", type=int, default=15,
                   help="Recording fps (must match split_session_agv.py; default 15).")
    p.add_argument("--robot-type", default="revobots_agv_follower")
    p.add_argument("--image-writer-threads", type=int, default=4)
    p.add_argument("--video-encoding-batch-size", type=int, default=1,
                   help="Encode N episodes' videos in one batch. Default 1 = "
                        "encode per episode (matches lerobot_record default).")
    p.add_argument("--resume", action="store_true",
                   help="Resume into an existing dataset; skip session_<i>/ "
                        "folders whose episode index is already saved.")
    p.add_argument("--push-to-hub", action="store_true",
                   help="Push to HuggingFace Hub after writing all episodes.")
    p.add_argument("--no-videos", action="store_true",
                   help="Store images as PNG instead of encoded video (larger).")
    p.add_argument("--no-obs-processor", action="store_true",
                   help="Skip the observation processor — write raw obs straight "
                        "to the dataset. Use only if you also patch inference to "
                        "skip its observation processor. Default OFF (i.e. apply).")
    return p.parse_args()


def run_convert(
    parent_dir,
    repo_id,
    task="Drive the AGV.",
    fps=15,
    robot_type="revobots_agv_follower",
    image_writer_threads=4,
    video_encoding_batch_size=1,
    resume=False,
    push_to_hub=False,
    no_videos=False,
    no_obs_processor=False,
):
    """Build a LeRobot dataset from split sessions under `parent_dir`.

    Mirrors main() but takes plain kwargs so it can be called from a pipeline.
    Returns the dataset root path.
    """
    input_root = Path(os.path.expanduser(parent_dir)).resolve()
    if not input_root.is_dir():
        raise SystemExit(f"[!] Not a directory: {input_root}")

    sessions = find_session_folders(input_root)
    if not sessions:
        raise SystemExit(f"[!] No session_<i>/ folders found in {input_root}")

    print(f"[*] Input root:    {input_root}")
    print(f"[*] Sessions:      {len(sessions)}")
    print(f"[*] Repo id:       {repo_id}")
    print(f"[*] FPS:           {fps}")
    print(f"[*] Resume:        {resume}")
    print(f"[*] Obs processor: {'DISABLED' if no_obs_processor else 'ENABLED (symmetric with inference)'}")

    features = build_dataset_features()

    # ── [SYMMETRY] Build the observation processor exactly the way
    # `build_policy_pipeline` does. The first return value (the env-side
    # processor) and the action processor aren't relevant here — only the
    # observation processor is, because that's what mutates obs *before*
    # build_dataset_frame on the inference side.
    _, _, robot_observation_processor = make_default_processors()
    if no_obs_processor:
        # Identity stand-in so the rest of the loop doesn't branch.
        robot_observation_processor = lambda obs: obs   # noqa: E731

    # Show the resolved schema so we can sanity-check it before writing.
    print("\n[*] Resolved dataset feature schema:")
    for k, v in features.items():
        dtype = v.get("dtype", "?")
        shape = tuple(v.get("shape", ())) if "shape" in v else ""
        names = v.get("names", "")
        print(f"      {k:35s}  dtype={dtype:8s}  shape={shape}  names={names}")
    print()

    # --- Open or create the dataset (same calls as lerobot_record.record) --
    if resume:
        try:
            dataset = LeRobotDataset(
                repo_id=repo_id,
                batch_encoding_size=video_encoding_batch_size,
            )
            start_idx = dataset.num_episodes
            print(f"[*] Resuming dataset at episode index {start_idx}")
        except Exception as e:
            print(f"[!] --resume failed to load existing dataset ({e}); "
                  f"creating new one.")
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                features=features,
                robot_type=robot_type,
                use_videos=not no_videos,
                image_writer_threads=image_writer_threads,
                batch_encoding_size=video_encoding_batch_size,
            )
            start_idx = 0
    else:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            robot_type=robot_type,
            use_videos=not no_videos,
            image_writer_threads=image_writer_threads,
            batch_encoding_size=video_encoding_batch_size,
        )
        start_idx = 0

    # --- Process each session as one episode -------------------------------
    todo = sessions[start_idx:]
    print(f"[*] Processing {len(todo)} episode(s)...\n")

    total_frames = 0

    try:
        with VideoEncodingManager(dataset):
            outer = tqdm(
                total=len(todo),
                desc="Episodes",
                unit="ep",
                position=0,
                dynamic_ncols=True,
            )
            try:
                for offset, session_dir in enumerate(todo):
                    ep_idx = start_idx + offset
                    outer.set_description(f"Episode {ep_idx} ({session_dir.name})")
                    n = add_session_as_episode(
                        session_dir,
                        dataset,
                        task,
                        robot_observation_processor,           # [SYMMETRY] pass it down
                    )
                    total_frames += n
                    outer.set_postfix(frames=total_frames)
                    outer.update(1)
            finally:
                outer.close()
    finally:
        print("[*] Finalizing dataset...")
        try:
            dataset.finalize()
        except Exception as e:
            print(f"[!] finalize() raised: {e}")

    # --- Verify every parquet on disk has a footer -------------------------
    print("\n[*] Verifying parquet files...")
    bad = []
    for pq_path in sorted(Path(dataset.root).rglob("*.parquet")):
        ok = _has_parquet_footer(pq_path)
        rel = pq_path.relative_to(dataset.root)
        print(f"      {'OK ' if ok else 'BAD'}  {rel}")
        if not ok:
            bad.append(pq_path)

    if bad:
        print(f"\n[!] {len(bad)} parquet file(s) are missing footers — the close()"
              f" call did not flush. Attempting recovery by rewriting them...")
        for bad_path in bad:
            _try_recover_parquet(bad_path)
        print("\n[*] Re-verifying after recovery...")
        still_bad = [p for p in bad if not _has_parquet_footer(p)]
        if still_bad:
            print(f"[!] Could not recover: {still_bad}")
        else:
            print("[✓] All parquets now have valid footers.")

    print(f"\n[✓] Done. Wrote {total_frames} new frames across "
          f"{len(todo)} new episode(s).")
    print(f"[✓] Dataset lives at: {dataset.root}")

    if push_to_hub:
        print("[*] Pushing to HuggingFace Hub...")
        dataset.push_to_hub()
        print("[✓] Pushed.")

    return dataset.root


def main():
    args = parse_args()
    run_convert(
        parent_dir=args.parent_dir,
        repo_id=args.repo_id,
        task=args.task,
        fps=args.fps,
        robot_type=args.robot_type,
        image_writer_threads=args.image_writer_threads,
        video_encoding_batch_size=args.video_encoding_batch_size,
        resume=args.resume,
        push_to_hub=args.push_to_hub,
        no_videos=args.no_videos,
        no_obs_processor=args.no_obs_processor,
    )


if __name__ == "__main__":
    main()