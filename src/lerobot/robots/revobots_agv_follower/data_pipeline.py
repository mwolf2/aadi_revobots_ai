# -*- coding: utf-8 -*-
"""
End-to-end AGV data pipeline: pull raw sessions from Azure Blob Storage, split
them, optionally semantic-mask the video frames, build a LeRobot dataset, then
upload the dataset back to Azure.

Lives at: src/lerobot/robots/revobots_agv_follower/data_pipeline.py

Runs, in order:
    0.   download          — pull every session_YYYYMMDD_123456/ folder from the
                             input blob container (default "raw") to a local
                             work dir.
    1.   split_session_agv — chop raw session(s) (video.mp4 + data.jsonl) into
                             fixed-length session_<i>/ chunks.
    1.5  mask_sessions_agv — (optional) replace each session_<i>.mp4 with a
                             semantic-segmentation-masked version. .jsonl files
                             pass through untouched so frame indices stay
                             aligned with the masked frames.
    2.   data_convert_agv  — turn the chunks into a LeRobot HuggingFace dataset.
    3.   upload            — push the finished dataset to the output blob
                             container (default "datasets").

Everything is driven by a TOML config file (default: pipeline.toml next to this
script); the only command-line argument is --config. Credentials are read from
the AZURE_STORAGE_CONNECTION_STRING environment variable — never from the TOML.

Example:
    export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=...;"
    python data_pipeline.py                       # uses ./pipeline.toml
    python data_pipeline.py --config other.toml   # uses a different config
"""

import argparse
import os
import sys
from datetime import datetime

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))                 # .../robots/revobots_agv_follower
_REPO_SRC = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))  # .../src
_MASK_DIR = os.path.join(_REPO_SRC, "lerobot", "cameras", "image_detection_tracking")

# The step modules live inside the robot package dir; importing them via the
# package (lerobot.robots.revobots_agv_follower.*) would run its __init__.py,
# which pulls in ROS. Load them as flat top-level modules from their sibling
# directories. azure_blob_io lives alongside this script.
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _MASK_DIR)

from split_session_agv import run_split      # noqa: E402
from data_convert_agv import run_convert     # noqa: E402
from mask_sessions_agv import run_mask       # noqa: E402

import azure_blob_io                          # noqa: E402
import azure_keyvault                          # noqa: E402
import azure_ml_models                         # noqa: E402


DEFAULT_CONFIG = os.path.join(_THIS_DIR, "pipeline.toml")


def load_config(path):
    """Read the pipeline TOML config into a nested dict, filling in defaults
    for any missing key so a minimal config still works."""
    if not os.path.isfile(path):
        raise SystemExit(f"[pipeline] Config file not found: {path}")
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    keyvault = raw.get("keyvault", {})
    azure = raw.get("azure", {})
    source = raw.get("source", {})
    split = raw.get("split", {})
    mask = raw.get("mask", {})
    models = raw.get("models", {})
    convert = raw.get("convert", {})

    def opt(value):
        """Treat empty strings in the TOML as 'not set'."""
        return value if value else None

    return {
        "keyvault": {
            "enabled": keyvault.get("enabled", False),
            "vault_url": opt(keyvault.get("vault_url", "")),
            "connection_string_secret": opt(
                keyvault.get("connection_string_secret", "")),
            "hf_token_secret": opt(keyvault.get("hf_token_secret", "")),
        },
        "azure": {
            "input_container": azure.get("input_container", "raw"),
            "output_container": azure.get("output_container", "datasets"),
            "session_pattern": azure.get("session_pattern",
                                         azure_blob_io.DEFAULT_SESSION_PATTERN),
            "account_url": opt(azure.get("account_url", "")),
            "upload": azure.get("upload", True),
        },
        "source": {
            "input_path": opt(source.get("input_path", "")),
            "split_dir": opt(source.get("split_dir", "")),
            "work_dir": opt(source.get("work_dir", "")),
        },
        "split": {
            "segment_minutes": split.get("segment_minutes", 2),
            "split_out": opt(split.get("split_out", "")),
        },
        "models": {
            "enabled": models.get("enabled", False),
            "registry_name": opt(models.get("registry_name", "")),
            "subscription_id": opt(models.get("subscription_id", "")),
            "resource_group": opt(models.get("resource_group", "")),
            "workspace_name": opt(models.get("workspace_name", "")),
            "download_dir": opt(models.get("download_dir", "")),
            "seg_model_name": opt(models.get("seg_model_name", "")),
            "seg_model_version": opt(models.get("seg_model_version", "")),
            "det_model_name": opt(models.get("det_model_name", "")),
            "det_model_version": opt(models.get("det_model_version", "")),
        },
        "mask": {
            "apply_mask": mask.get("apply_mask", False),
            "mask_in_place": mask.get("mask_in_place", False),
            "mask_out": opt(mask.get("mask_out", "")),
            "seg_mode": mask.get("seg_mode", "semantic"),
            "seg_weights": mask.get("seg_weights", "yolo26n-sem.pt"),
            "det_weights": mask.get("det_weights", "yolo26n.pt"),
            "seg_conf": mask.get("seg_conf", 0.25),
            "overlay_alpha": mask.get("overlay_alpha", 0.0),
            "det_classes": mask.get("det_classes", [0, 1, 9, 10, 11, 12]),
        },
        "convert": {
            "repo_id": convert.get("repo_id"),
            "task": convert.get("task", "Drive the AGV."),
            "fps": convert.get("fps", 15),
            "robot_type": convert.get("robot_type", "revobots_agv_follower"),
            "image_writer_threads": convert.get("image_writer_threads", 4),
            "video_encoding_batch_size": convert.get("video_encoding_batch_size", 1),
            "resume": convert.get("resume", False),
            "push_to_hub": convert.get("push_to_hub", False),
            "no_videos": convert.get("no_videos", False),
            "no_obs_processor": convert.get("no_obs_processor", False),
        },
    }


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help=f"Path to the pipeline TOML config (default: {DEFAULT_CONFIG}).")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    keyvault = cfg["keyvault"]
    azure = cfg["azure"]
    source = cfg["source"]
    split = cfg["split"]
    mask = cfg["mask"]
    models = cfg["models"]
    convert = cfg["convert"]

    # Pull secrets (Azure connection string, HF token) from Key Vault and
    # inject them as env vars before any Azure/HF access. Done first so a
    # missing/misconfigured vault fails fast, before any downloading.
    if keyvault["enabled"]:
        if not keyvault["vault_url"]:
            raise SystemExit("[pipeline] keyvault.enabled is true but "
                             "keyvault.vault_url is empty.")
        print(f"[pipeline] Loading secrets from Key Vault: {keyvault['vault_url']}")
        azure_keyvault.load_secrets(
            keyvault["vault_url"],
            connection_string_secret=keyvault["connection_string_secret"],
            hf_token_secret=keyvault["hf_token_secret"],
        )

    if not convert["repo_id"]:
        raise SystemExit("[pipeline] convert.repo_id is required in the config.")

    # Stamp the dataset name with the current date (MMDDYYYY) so each run is
    # uniquely identifiable both on disk/hub and in the upload prefix.
    convert["repo_id"] = f"{convert['repo_id']}_{datetime.now().strftime('%m%d%Y')}"
    print(f"[pipeline] Dataset repo_id: {convert['repo_id']}")

    if source["input_path"] and source["split_dir"]:
        raise SystemExit("[pipeline] Set only one of source.input_path / "
                         "source.split_dir in the config.")

    fps = convert["fps"]

    # --- Step 0: obtain raw sessions ---------------------------------------
    if source["split_dir"]:
        parent_dir = os.path.expanduser(source["split_dir"])
        print(f"[pipeline] Skipping download & split; using existing chunks: {parent_dir}\n")
    else:
        if source["input_path"]:
            raw_dir = os.path.expanduser(source["input_path"])
            print(f"[pipeline] === Step 0/4: using local raw folder {raw_dir} ===\n")
        else:
            if source["work_dir"]:
                raw_dir = os.path.expanduser(source["work_dir"])
            else:
                timestamp = datetime.now().strftime("%d%m%Y%H%M%S")
                raw_dir = os.path.expanduser(
                    f"~/.cache/scout/lab/raw_download/raw_{timestamp}"
                )
            print(f"[pipeline] === Step 0/4: downloading raw sessions from "
                  f"container '{azure['input_container']}' ===")
            print(f"[pipeline] Staging to: {raw_dir}")
            sessions = azure_blob_io.download_sessions(
                azure["input_container"], raw_dir,
                pattern=azure["session_pattern"],
                account_url=azure["account_url"],
            )
            if not sessions:
                raise SystemExit(
                    f"[pipeline] No session folders matching "
                    f"{azure['session_pattern']!r} found in container "
                    f"'{azure['input_container']}'; aborting."
                )
            print(f"[pipeline] Sessions: {', '.join(sessions)}\n")

        # --- Step 1: split --------------------------------------------------
        print("[pipeline] === Step 1/4: splitting sessions ===")
        parent_dir = run_split(
            raw_dir,
            output_root=split["split_out"],
            fps=fps,
            segment_minutes=split["segment_minutes"],
        )
        if not parent_dir:
            raise SystemExit("[pipeline] Split step produced no output; aborting.")

    # --- Step 1.5: mask (optional) -----------------------------------------
    if mask["apply_mask"]:
        seg_weights = mask["seg_weights"]
        det_weights = mask["det_weights"]

        # Optionally pull the YOLO weights from Azure ML (workspace/registry)
        # and use the downloaded local paths instead of mask.*_weights.
        if models["enabled"]:
            print("\n[pipeline] Fetching YOLO weights from Azure ML...")
            client = azure_ml_models.get_client(
                subscription_id=models["subscription_id"],
                resource_group=models["resource_group"],
                workspace_name=models["workspace_name"],
                registry_name=models["registry_name"],
            )
            dl_dir = models["download_dir"] or os.path.expanduser(
                "~/.cache/scout/lab/ml_models")
            if models["seg_model_name"]:
                seg_weights = azure_ml_models.download_weight(
                    client, models["seg_model_name"], dl_dir,
                    version=models["seg_model_version"],
                    weight_filename=os.path.basename(mask["seg_weights"] or ""),
                )
            if models["det_model_name"]:
                det_weights = azure_ml_models.download_weight(
                    client, models["det_model_name"], dl_dir,
                    version=models["det_model_version"],
                    weight_filename=os.path.basename(mask["det_weights"] or ""),
                )

        print("\n[pipeline] === Step 2/4: masking session videos ===")
        parent_dir = run_mask(
            parent_dir=parent_dir,
            output_dir=mask["mask_out"],
            seg_mode=mask["seg_mode"],
            seg_weights=seg_weights,
            det_weights=det_weights,
            det_classes_to_keep=tuple(mask["det_classes"]),
            seg_conf=mask["seg_conf"],
            overlay_alpha=mask["overlay_alpha"],
            in_place=mask["mask_in_place"],
            fps=fps,
        )
    else:
        print("\n[pipeline] Skipping mask step (mask.apply_mask is false).")

    # --- Step 2: convert ---------------------------------------------------
    print("\n[pipeline] === Step 3/4: building LeRobot dataset ===")
    dataset_root = run_convert(
        parent_dir=parent_dir,
        repo_id=convert["repo_id"],
        task=convert["task"],
        fps=fps,
        robot_type=convert["robot_type"],
        image_writer_threads=convert["image_writer_threads"],
        video_encoding_batch_size=convert["video_encoding_batch_size"],
        resume=convert["resume"],
        push_to_hub=convert["push_to_hub"],
        no_videos=convert["no_videos"],
        no_obs_processor=convert["no_obs_processor"],
    )
    print(f"\n[pipeline] Dataset written to: {dataset_root}")

    # --- Step 3: upload ----------------------------------------------------
    if not azure["upload"]:
        print("[pipeline] azure.upload is false; skipping dataset upload.")
    else:
        print(f"\n[pipeline] === Step 4/4: uploading dataset to container "
              f"'{azure['output_container']}' ===")
        azure_blob_io.upload_directory(
            azure["output_container"], dataset_root,
            dest_prefix=convert["repo_id"],
            account_url=azure["account_url"],
        )

    print(f"\n[pipeline] Done. Dataset at: {dataset_root}")


if __name__ == "__main__":
    main()
