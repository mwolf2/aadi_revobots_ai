# -*- coding: utf-8 -*-
"""Azure Blob Storage helpers for the AGV data pipeline.

Auth: the connection string is read from the AZURE_STORAGE_CONNECTION_STRING
environment variable (Azure Portal -> Storage account -> Access keys ->
Connection string). Nothing sensitive is ever read from the TOML config.

Requires:  pip install azure-storage-blob
"""

import os
import re
import sys
from pathlib import Path

from azure.storage.blob import BlobServiceClient

# Raw session folders are named session_YYYYMMDD_123456, where the trailing
# six digits vary. Blob names are flat, so the "folder" is just the first
# path segment of the blob name.
DEFAULT_SESSION_PATTERN = r"^session_\d{8}_\d{6}$"

CONN_STR_ENV = "AZURE_STORAGE_CONNECTION_STRING"


def _service() -> BlobServiceClient:
    conn_str = os.environ.get(CONN_STR_ENV)
    if not conn_str:
        sys.exit(f"Set {CONN_STR_ENV} first (Azure connection string).")
    return BlobServiceClient.from_connection_string(conn_str)


def _top_folder(blob_name: str) -> str:
    """First path segment of a blob name — the virtual top-level folder."""
    return blob_name.split("/", 1)[0]


def _is_dir_marker(blob) -> bool:
    """True for the 0-byte placeholder blobs Azure uses to represent a folder
    (e.g. a blob literally named 'session_X' or one ending in '/'). These are
    not real files — downloading them would create a file where we later need
    a directory."""
    name = blob.name
    if name.endswith("/") or "/" not in name:
        return True
    # ADLS Gen2 hierarchical-namespace folder markers.
    meta = getattr(blob, "metadata", None) or {}
    return meta.get("hdi_isfolder") == "true"


def list_session_folders(container_name: str, pattern: str = DEFAULT_SESSION_PATTERN) -> list[str]:
    """Return the sorted set of top-level folder names in `container_name`
    that match `pattern` (e.g. session_20260714_123456)."""
    regex = re.compile(pattern)
    container = _service().get_container_client(container_name)
    names: set[str] = set()
    for blob in container.list_blobs(include=["metadata"]):
        top = _top_folder(blob.name)
        if regex.match(top):
            names.add(top)
    return sorted(names)


def download_sessions(
    container_name: str,
    dest_dir: str | os.PathLike,
    pattern: str = DEFAULT_SESSION_PATTERN,
) -> list[str]:
    """Download every blob whose top-level folder matches `pattern` from
    `container_name` into `dest_dir`, preserving the folder layout.

    Returns the sorted list of session folder names that were downloaded.
    """
    regex = re.compile(pattern)
    container = _service().get_container_client(container_name)
    dest_dir = Path(dest_dir)

    downloaded: set[str] = set()
    n_blobs = 0
    for blob in container.list_blobs(include=["metadata"]):
        top = _top_folder(blob.name)
        if not regex.match(top):
            continue
        if _is_dir_marker(blob):
            continue

        target = dest_dir / blob.name
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"      [download] {blob.name} ({blob.size} bytes)")
        with open(target, "wb") as fh:
            container.download_blob(blob.name).readinto(fh)

        downloaded.add(top)
        n_blobs += 1

    print(f"[*] Downloaded {n_blobs} blob(s) across {len(downloaded)} session(s).")
    return sorted(downloaded)


def upload_directory(
    container_name: str,
    local_dir: str | os.PathLike,
    dest_prefix: str = "",
) -> int:
    """Recursively upload every file under `local_dir` to `container_name`,
    preserving the relative layout under `dest_prefix`. Overwrites existing
    blobs. Returns the number of files uploaded."""
    service = _service()
    container = service.get_container_client(container_name)

    # Create the container if it doesn't exist yet.
    try:
        container.create_container()
        print(f"[*] Created container '{container_name}'.")
    except Exception:
        pass  # Already exists (or we lack create perms — upload will surface it).

    local_dir = Path(local_dir)
    prefix = dest_prefix.strip("/")

    count = 0
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        blob_name = f"{prefix}/{rel}" if prefix else rel
        print(f"      [upload] {blob_name} ({path.stat().st_size} bytes)")
        with open(path, "rb") as fh:
            container.upload_blob(name=blob_name, data=fh, overwrite=True)
        count += 1

    print(f"[*] Uploaded {count} file(s) to '{container_name}/{prefix}'.")
    return count
