# -*- coding: utf-8 -*-
"""
End-to-end AGV data pipeline: pull raw sessions from Azure Blob Storage, split
them, optionally semantic-mask the video frames, build a LeRobot dataset, then
upload the dataset back to Azure.

Lives at: src/lerobot/robots/revobots_agv_follower/data_pipeline.py

Runs, in order:
    0.   download          — pull every session_YYYYMMDD_123456/ folder from the
                             input blob container (default "raw") to a local
                             work dir. azure.input_prefix scopes this to one
                             folder inside that container (e.g. a route), so a
                             run can train on one route's sessions only.
    1.   split_session_agv — chop raw session(s) (video.mp4 + data.jsonl) into
                             fixed-length session_<i>/ chunks.
    1.5  mask_sessions_agv — (optional) replace each session_<i>.mp4 with a
                             semantic-segmentation-masked version. .jsonl files
                             pass through untouched so frame indices stay
                             aligned with the masked frames.
    2.   data_convert_agv  — turn the chunks into a LeRobot HuggingFace dataset.
    3.   upload            — push the finished dataset to the output blob
                             container (default "datasets").

With [train_test_split].enabled the raw sessions are partitioned into a train
group and a test group BEFORE step 1, and steps 1-3 then run once per group,
producing two independent datasets: <repo_id>_train and <repo_id>_test. The
partition is at the session level, never the chunk/episode level — every chunk
cut out of one raw session stays on the same side — and is seeded, so the same
set of session names plus the same seed reproduces the same split exactly.

Manifest / append mode (opt-in, used by revobots-ml's ctcd pipeline):
    SESSION_MANIFEST=<path to manifest.json> downloads exactly the sessions it
    lists (catalog raw_path values) instead of prefix+pattern matching, and takes
    the dataset repo_id from it. APPEND=true adds the converted sessions as new
    episodes after those already in that Hub dataset, instead of building a new
    dataset. Neither is set by the other pipelines, which behave as before.

Everything is driven by a TOML config file (default: pipeline.toml next to this
script); the only command-line argument is --config. Credentials are read from
the AZURE_STORAGE_CONNECTION_STRING environment variable — never from the TOML.

Example:
    export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=...;"
    python data_pipeline.py                       # uses ./pipeline.toml
    python data_pipeline.py --config other.toml   # uses a different config
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))                 # .../robots/revobots_agv_follower
_REPO_SRC = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))  # .../src
_MASK_DIR = os.path.join(_REPO_SRC, "lerobot", "cameras", "image_detection_tracking")
_CICD_DIR = os.path.join(_REPO_SRC, "lerobot", "CICD")

# The step modules live inside the robot package dir; importing them via the
# package (lerobot.robots.revobots_agv_follower.*) would run its __init__.py,
# which pulls in ROS. Load them as flat top-level modules from their sibling
# directories. The Azure helpers live in lerobot/CICD and are loaded the same
# way (its directory is put on sys.path below).
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _MASK_DIR)
sys.path.insert(0, _CICD_DIR)

from split_session_agv import find_sessions, run_split   # noqa: E402
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
    train_test = raw.get("train_test_split", {})
    mask = raw.get("mask", {})
    models = raw.get("models", {})
    convert = raw.get("convert", {})

    def opt(value):
        """Treat empty strings in the TOML as 'not set'."""
        return value if value else None

    def env_bool(name, default):
        """Read a boolean from an env var (docker -e), falling back to the TOML
        value. Accepts 1/true/yes/on (case-insensitive) as true."""
        v = os.environ.get(name)
        if not v:
            return default
        return v.strip().lower() in ("1", "true", "yes", "on")

    def env_number(name, default, cast):
        """Read a number from an env var (docker -e), falling back to the TOML
        value. A malformed value is a hard error rather than a silent fallback:
        a typo'd TEST_RATIO would otherwise quietly produce a differently-sized
        test set than the run was asked for."""
        v = os.environ.get(name)
        if not v:
            return default
        try:
            return cast(v)
        except ValueError:
            raise SystemExit(f"[pipeline] {name}={v!r} is not a valid "
                             f"{cast.__name__}.")

    return {
        "keyvault": {
            "enabled": keyvault.get("enabled", False),
            "vault_url": opt(keyvault.get("vault_url", "")),
            "connection_string_secret": opt(
                keyvault.get("connection_string_secret", "")),
            "hf_token_secret": opt(keyvault.get("hf_token_secret", "")),
            "hf_user_secret": opt(keyvault.get("hf_user_secret", "")),
        },
        "azure": {
            # Env vars (passed via docker -e) override the TOML when set.
            "input_container": (os.environ.get("INPUT_CONTAINER")
                                or azure.get("input_container", "raw")),
            "output_container": (os.environ.get("OUTPUT_CONTAINER")
                                 or azure.get("output_container", "datasets")),
            # `is not None`, not `or`, unlike the two above: "" is a real value
            # here ("the container root"), so INPUT_PREFIX="" has to be able to
            # cancel a prefix set in the TOML rather than fall back to it.
            "input_prefix": (os.environ["INPUT_PREFIX"]
                             if os.environ.get("INPUT_PREFIX") is not None
                             else azure.get("input_prefix", "")),
            "session_pattern": azure.get("session_pattern",
                                         azure_blob_io.DEFAULT_SESSION_PATTERN),
            "account_url": opt(azure.get("account_url", "")),
            "upload": azure.get("upload", True),
        },
        "source": {
            # Env only: a manifest is per-run, written by the caller's own
            # session-selection step, never a static config value.
            "manifest": opt(os.environ.get("SESSION_MANIFEST", "")),
            "input_path": opt(source.get("input_path", "")),
            "split_dir": opt(source.get("split_dir", "")),
            "work_dir": opt(source.get("work_dir", "")),
        },
        "split": {
            "segment_minutes": split.get("segment_minutes", 2),
            "split_out": opt(split.get("split_out", "")),
        },
        "train_test_split": {
            # Env vars (passed via docker -e) override the TOML when set.
            "enabled": env_bool("TRAIN_TEST_SPLIT",
                                train_test.get("enabled", False)),
            "test_ratio": env_number("TEST_RATIO",
                                     train_test.get("test_ratio", 0.2), float),
            "seed": env_number("TRAIN_TEST_SEED",
                               train_test.get("seed", 42), int),
            # Appended to convert.repo_id for each side. Not opt()-ed: an empty
            # suffix would collide the two datasets onto one repo id, so it is
            # validated as non-empty and distinct before any work starts.
            "train_suffix": train_test.get("train_suffix", "_train"),
            "test_suffix": train_test.get("test_suffix", "_test"),
        },
        "models": {
            "enabled": models.get("enabled", False),
            "registry_name": opt(models.get("registry_name", "")),
            "subscription_id": opt(models.get("subscription_id", "")),
            "resource_group": opt(models.get("resource_group", "")),
            "workspace_name": opt(models.get("workspace_name", "")),
            "download_dir": opt(models.get("download_dir", "")),
            # Env vars (passed via docker -e) override the TOML when set.
            "seg_model_name": (os.environ.get("SEG_MODEL_NAME")
                               or opt(models.get("seg_model_name", ""))),
            "seg_model_version": opt(models.get("seg_model_version", "")),
            "det_model_name": (os.environ.get("DET_MODEL_NAME")
                               or opt(models.get("det_model_name", ""))),
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
            # Env var (passed via docker -e) overrides the TOML when set.
            "repo_id": os.environ.get("REPO_ID") or convert.get("repo_id"),
            "task": convert.get("task", "Drive the AGV."),
            "fps": convert.get("fps", 15),
            "robot_type": convert.get("robot_type", "revobots_agv_follower"),
            "image_writer_threads": convert.get("image_writer_threads", 4),
            "video_encoding_batch_size": convert.get("video_encoding_batch_size", 1),
            # Env var (passed via docker -e) overrides the TOML when set.
            "resume": env_bool("RESUME", convert.get("resume", False)),
            "append": env_bool("APPEND", convert.get("append", False)),
            "push_to_hub": convert.get("push_to_hub", False),
            "no_videos": convert.get("no_videos", False),
            "no_obs_processor": convert.get("no_obs_processor", False),
        },
    }


def partition_sessions(session_names, test_ratio, seed):
    """Deterministically partition raw session folder names into (train, test).

    The partition is at the SESSION level, never the chunk/episode level: every
    fixed-length chunk that split_session_agv later cuts out of one raw session
    lands on the same side. A test session is therefore a genuinely unseen
    recording, not an interleaved sample from a recording the policy already
    trained on.

    The names are sorted before shuffling, so the result depends only on the
    SET of session names and the seed — not on the order the blob listing or
    the local directory walk happened to return them in. Same corpus + same
    seed reproduces the same split exactly, on any machine.
    """
    ordered = sorted(session_names)
    if len(ordered) < 2:
        raise SystemExit(
            f"[pipeline] train_test_split.enabled is true but only "
            f"{len(ordered)} session(s) were found; at least 2 are needed to "
            f"put a whole session on each side."
        )
    if not 0.0 < test_ratio < 1.0:
        raise SystemExit(
            f"[pipeline] train_test_split.test_ratio must be strictly between "
            f"0 and 1; got {test_ratio}."
        )

    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)

    # Clamped to leave at least one session on each side: on a small corpus
    # round() can land on 0 (or on everything), which would produce an empty
    # dataset that only fails much later, in the convert step.
    n_test = int(round(len(shuffled) * test_ratio))
    n_test = min(max(n_test, 1), len(shuffled) - 1)

    return sorted(shuffled[n_test:]), sorted(shuffled[:n_test])


def build_dataset(cfg, *, label, suffix, only_sessions, raw_dir, pre_split_dir,
                  seg_weights, det_weights, run_stamp):
    """Run split -> mask -> convert -> upload over one side of the corpus.

    `suffix` is appended to convert.repo_id and to every configured
    intermediate output dir, so the two sides of a train/test split never write
    over each other. With suffix="" / only_sessions=None this is exactly the
    original single-dataset run.

    `seg_weights`/`det_weights` are resolved by the caller rather than here, so
    a two-sided run downloads the YOLO weights from Azure ML once, not twice.

    Returns the dataset root path.
    """
    azure = cfg["azure"]
    split = cfg["split"]
    mask = cfg["mask"]
    convert = cfg["convert"]

    tag = f"[pipeline] [{label}]" if label else "[pipeline]"
    repo_id = convert["repo_id"] + suffix

    # --- Step 1: split ------------------------------------------------------
    if pre_split_dir:
        parent_dir = pre_split_dir
    else:
        if split["split_out"]:
            split_out = split["split_out"] + suffix
        elif suffix:
            # run_split's own default names its dir by the current minute, so
            # two sides submitted back to back would land in the same folder
            # and interleave their session_<i>/ numbering. Name them here
            # instead, off one stamp taken before either side started.
            split_out = os.path.expanduser(
                f"~/.cache/scout/lab/split_session/split_session_{run_stamp}{suffix}"
            )
        else:
            split_out = None

        print(f"\n{tag} === Step 1/4: splitting sessions ===")
        if only_sessions is not None:
            print(f"{tag} Sessions in this split ({len(only_sessions)}): "
                  f"{', '.join(only_sessions)}")
        parent_dir = run_split(
            raw_dir,
            output_root=split_out,
            fps=convert["fps"],
            segment_minutes=split["segment_minutes"],
            only_sessions=only_sessions,
        )
        if not parent_dir:
            raise SystemExit(f"{tag} Split step produced no output; aborting.")

    # --- Step 2: mask (optional) -------------------------------------------
    if mask["apply_mask"]:
        print(f"\n{tag} === Step 2/4: masking session videos ===")
        parent_dir = run_mask(
            parent_dir=parent_dir,
            # Empty mask_out falls through to run_mask's <parent_dir>_masked
            # default, which is already per-side because parent_dir is.
            output_dir=(mask["mask_out"] + suffix) if mask["mask_out"] else None,
            seg_mode=mask["seg_mode"],
            seg_weights=seg_weights,
            det_weights=det_weights,
            det_classes_to_keep=tuple(mask["det_classes"]),
            seg_conf=mask["seg_conf"],
            overlay_alpha=mask["overlay_alpha"],
            in_place=mask["mask_in_place"],
            fps=convert["fps"],
        )
    else:
        print(f"\n{tag} Skipping mask step (mask.apply_mask is false).")

    # --- Step 3: convert ---------------------------------------------------
    print(f"\n{tag} === Step 3/4: building LeRobot dataset '{repo_id}' ===")
    dataset_root = run_convert(
        parent_dir=parent_dir,
        repo_id=repo_id,
        task=convert["task"],
        fps=convert["fps"],
        robot_type=convert["robot_type"],
        image_writer_threads=convert["image_writer_threads"],
        video_encoding_batch_size=convert["video_encoding_batch_size"],
        resume=convert["resume"],
        append=convert["append"],
        push_to_hub=convert["push_to_hub"],
        no_videos=convert["no_videos"],
        no_obs_processor=convert["no_obs_processor"],
    )
    print(f"\n{tag} Dataset written to: {dataset_root}")

    # --- Step 4: upload ----------------------------------------------------
    if not azure["upload"]:
        print(f"{tag} azure.upload is false; skipping dataset upload.")
    else:
        print(f"\n{tag} === Step 4/4: uploading dataset to container "
              f"'{azure['output_container']}' ===")
        azure_blob_io.upload_directory(
            azure["output_container"], dataset_root,
            dest_prefix=repo_id,
            account_url=azure["account_url"],
        )

    return dataset_root


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
    train_test = cfg["train_test_split"]
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
            hf_user_secret=keyvault["hf_user_secret"],
        )

    manifest = None
    if source["manifest"]:
        if source["input_path"] or source["split_dir"]:
            raise SystemExit("[pipeline] SESSION_MANIFEST is set; source.input_path "
                             "and source.split_dir must both be empty.")
        with open(source["manifest"], encoding="utf-8") as fh:
            manifest = json.load(fh)
        if not manifest.get("sessions"):
            raise SystemExit(f"[pipeline] Manifest {source['manifest']} lists no sessions.")
        # The manifest names the dataset its sessions belong in; a repo_id from
        # the TOML or REPO_ID could only disagree with it.
        convert["repo_id"] = manifest["repo_id"]
        print(f"[pipeline] Manifest: {len(manifest['sessions'])} session(s) "
              f"for robot {manifest.get('robot_id')!r} -> {convert['repo_id']}")

    if not convert["repo_id"]:
        raise SystemExit("[pipeline] convert.repo_id is required in the config.")
    if convert["append"] and convert["resume"]:
        raise SystemExit("[pipeline] APPEND and RESUME are mutually exclusive.")
    if convert["append"] and not convert["push_to_hub"]:
        # Appending builds a local root holding only the NEW episodes; without
        # the push there's no complete dataset anywhere.
        raise SystemExit("[pipeline] APPEND requires convert.push_to_hub = true.")

    # Stamp the dataset name with the current date (MMDDYYYY) so each run is
    # uniquely identifiable both on disk/hub and in the upload prefix.
    # Disabled for now:
    # convert["repo_id"] = f"{convert['repo_id']}_{datetime.now().strftime('%m%d%Y')}"
    print(f"[pipeline] Dataset repo_id: {convert['repo_id']}")

    if source["input_path"] and source["split_dir"]:
        raise SystemExit("[pipeline] Set only one of source.input_path / "
                         "source.split_dir in the config.")

    # Validated before the download, not after: a bad suffix pair would collide
    # both sides onto one repo id, and there is no point discovering that after
    # pulling the whole corpus down.
    if train_test["enabled"] and (manifest is not None or convert["append"]):
        # A split partitions whatever sessions this run was given; appending
        # would partition only the new ones, independently of every earlier run.
        raise SystemExit("[pipeline] train_test_split.enabled can't be combined "
                         "with SESSION_MANIFEST or APPEND.")
    if train_test["enabled"]:
        if source["split_dir"]:
            raise SystemExit(
                "[pipeline] train_test_split.enabled is true, but source.split_dir "
                "points at an already-chopped split_session_<...>/ dir. Its "
                "session_<i>/ chunks are numbered continuously across the raw "
                "sessions they came from, so the raw-session boundaries a "
                "session-level split needs are no longer recoverable. Use "
                "source.input_path (or the blob container) instead."
            )
        if not train_test["train_suffix"] or not train_test["test_suffix"]:
            raise SystemExit("[pipeline] train_test_split.train_suffix and "
                             ".test_suffix must both be non-empty.")
        if train_test["train_suffix"] == train_test["test_suffix"]:
            raise SystemExit("[pipeline] train_test_split.train_suffix and "
                             ".test_suffix must differ.")

    # --- Step 0: obtain raw sessions ---------------------------------------
    raw_dir = None
    pre_split_dir = None
    session_names = []

    if manifest is not None:
        if source["work_dir"]:
            raw_dir = os.path.expanduser(source["work_dir"])
        else:
            timestamp = datetime.now().strftime("%d%m%Y%H%M%S")
            raw_dir = os.path.expanduser(
                f"~/.cache/scout/lab/raw_download/raw_{timestamp}"
            )
        print("[pipeline] === Step 0/4: downloading the manifest's sessions ===")
        print(f"[pipeline] Staging to: {raw_dir}")
        session_names = azure_blob_io.download_session_paths(
            [s["raw_path"] for s in manifest["sessions"]], raw_dir,
            account_url=azure["account_url"],
        )
        print(f"[pipeline] Sessions: {', '.join(session_names)}\n")
    elif source["split_dir"]:
        pre_split_dir = os.path.expanduser(source["split_dir"])
        print(f"[pipeline] Skipping download & split; using existing chunks: "
              f"{pre_split_dir}\n")
    elif source["input_path"]:
        raw_dir = os.path.expanduser(source["input_path"])
        print(f"[pipeline] === Step 0/4: using local raw folder {raw_dir} ===\n")
        session_names = [name for name, _ in find_sessions(raw_dir)]
    else:
        if source["work_dir"]:
            raw_dir = os.path.expanduser(source["work_dir"])
        else:
            timestamp = datetime.now().strftime("%d%m%Y%H%M%S")
            raw_dir = os.path.expanduser(
                f"~/.cache/scout/lab/raw_download/raw_{timestamp}"
            )
        # Printed as container/prefix so the log says exactly which slice of
        # the corpus a run covered — the whole container, or one route.
        source_desc = azure["input_container"]
        if azure["input_prefix"]:
            source_desc += "/" + azure["input_prefix"].strip("/")
        print(f"[pipeline] === Step 0/4: downloading raw sessions from "
              f"'{source_desc}' ===")
        print(f"[pipeline] Staging to: {raw_dir}")
        session_names = azure_blob_io.download_sessions(
            azure["input_container"], raw_dir,
            pattern=azure["session_pattern"],
            account_url=azure["account_url"],
            prefix=azure["input_prefix"],
        )
        if not session_names:
            raise SystemExit(
                f"[pipeline] No session folders matching "
                f"{azure['session_pattern']!r} found in "
                f"'{source_desc}'; aborting."
            )
        print(f"[pipeline] Sessions: {', '.join(session_names)}\n")

    # --- Step 0.5: partition sessions into train/test ----------------------
    # (label, repo-id suffix, session names) per dataset to build. The
    # single-entry default is the original, unsplit behaviour.
    groups = [(None, "", None)]
    if train_test["enabled"]:
        train_sessions, test_sessions = partition_sessions(
            session_names, train_test["test_ratio"], train_test["seed"],
        )
        print(f"[pipeline] === Train/test split (seed={train_test['seed']}, "
              f"test_ratio={train_test['test_ratio']}) ===")
        print(f"[pipeline] train ({len(train_sessions)} sessions): "
              f"{', '.join(train_sessions)}")
        print(f"[pipeline] test  ({len(test_sessions)} sessions): "
              f"{', '.join(test_sessions)}")
        groups = [
            ("train", train_test["train_suffix"], train_sessions),
            ("test", train_test["test_suffix"], test_sessions),
        ]
    else:
        print("[pipeline] Train/test split disabled; building one dataset.")

    # --- Resolve YOLO weights once, ahead of every group -------------------
    seg_weights = mask["seg_weights"]
    det_weights = mask["det_weights"]
    if mask["apply_mask"] and models["enabled"]:
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

    # --- Steps 1-4, once per group -----------------------------------------
    # One stamp for the whole run so each group's default split dir is named
    # off the same moment, differing only by suffix.
    run_stamp = datetime.now().strftime("%d%m%Y%H%M%S")
    built = []
    for label, suffix, only_sessions in groups:
        dataset_root = build_dataset(
            cfg,
            label=label,
            suffix=suffix,
            only_sessions=only_sessions,
            raw_dir=raw_dir,
            pre_split_dir=pre_split_dir,
            seg_weights=seg_weights,
            det_weights=det_weights,
            run_stamp=run_stamp,
        )
        built.append((convert["repo_id"] + suffix, dataset_root))

    print("\n[pipeline] Done.")
    for repo_id, dataset_root in built:
        print(f"[pipeline]   {repo_id}: {dataset_root}")


if __name__ == "__main__":
    main()
