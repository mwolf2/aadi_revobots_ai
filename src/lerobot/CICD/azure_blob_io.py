# -*- coding: utf-8 -*-
"""Azure Blob Storage helpers for the AGV data pipeline.

Auth, in order of preference:
  * If an `account_url` is supplied (from azure.account_url in the TOML), use
    DefaultAzureCredential — a Managed Identity when running on Azure, falling
    back to `az login` / env vars locally. No account key involved.
  * Otherwise read the connection string from the AZURE_STORAGE_CONNECTION_STRING
    environment variable (Azure Portal -> Storage account -> Access keys).

Nothing sensitive is ever read from the TOML config — only the account URL.

Requires:  pip install azure-storage-blob
           pip install azure-identity   (only for the managed-identity path)
"""

import os
import re
import sys
from pathlib import Path

from azure.storage.blob import BlobServiceClient

# Raw session folders are named session_YYYYMMDD_123456, where the trailing
# six digits vary. Blob names are flat, so the "folder" is just the first
# path segment of the blob name — or of what's left of it after an
# `input_prefix` (e.g. a route folder) is stripped off the front.
DEFAULT_SESSION_PATTERN = r"^session_\d{8}_\d{6}$"

CONN_STR_ENV = "AZURE_STORAGE_CONNECTION_STRING"


def _service(account_url=None) -> BlobServiceClient:
    """Return a BlobServiceClient.

    With `account_url` set, authenticate via DefaultAzureCredential (Managed
    Identity on Azure; az login / env vars locally) — no key required. Without
    it, fall back to the connection string in AZURE_STORAGE_CONNECTION_STRING.
    """
    if account_url:
        # Imported lazily so connection-string users don't need azure-identity.
        from azure.identity import DefaultAzureCredential
        return BlobServiceClient(account_url=account_url,
                                 credential=DefaultAzureCredential())
    conn_str = os.environ.get(CONN_STR_ENV)
    if not conn_str:
        sys.exit(f"Set {CONN_STR_ENV} (connection string), or configure "
                 f"azure.account_url for managed-identity auth.")
    return BlobServiceClient.from_connection_string(conn_str)


def _normalize_prefix(prefix):
    """Return `prefix` as a bare `folder/` (or `a/b/`), or "" when unset.

    Accepts "route_a", "route_a/", "/route_a/" alike so a config value doesn't
    have to be written in one exact shape.
    """
    if not prefix:
        return ""
    return prefix.strip("/") + "/"


def _relative_name(blob_name: str, prefix: str):
    """Strip `prefix` off `blob_name`, or return None if the blob is outside it.

    Everything downstream — the session-folder match and the local download
    path — works on this relative name, so a run scoped to a route folder sees
    exactly what an unscoped run over that same container root would see.
    """
    if not prefix:
        return blob_name
    if not blob_name.startswith(prefix):
        return None
    return blob_name[len(prefix):]


def _top_folder(blob_name: str) -> str:
    """First path segment of a blob name — the virtual top-level folder."""
    return blob_name.split("/", 1)[0]


def _is_dir_marker(blob, rel_name: str) -> bool:
    """True for the 0-byte placeholder blobs Azure uses to represent a folder
    (e.g. a blob literally named 'session_X' or one ending in '/'). These are
    not real files — downloading them would create a file where we later need
    a directory.

    Judged on the *prefix-relative* name: under a route folder the marker for
    session_X is named 'route_a/session_X', which does contain a slash and
    would slip past the check if the raw blob name were used.
    """
    if rel_name.endswith("/") or "/" not in rel_name:
        return True
    # ADLS Gen2 hierarchical-namespace folder markers.
    meta = getattr(blob, "metadata", None) or {}
    return meta.get("hdi_isfolder") == "true"


def list_session_folders(container_name: str, pattern: str = DEFAULT_SESSION_PATTERN,
                         account_url: str = None, prefix: str = "") -> list[str]:
    """Return the sorted set of session folder names in `container_name` that
    match `pattern` (e.g. session_20260714_123456).

    With `prefix` set (e.g. "route_a"), only that folder is searched and the
    names returned are relative to it — the route segment is not part of them.
    """
    regex = re.compile(pattern)
    prefix = _normalize_prefix(prefix)
    container = _service(account_url).get_container_client(container_name)
    names: set[str] = set()
    for blob in container.list_blobs(name_starts_with=prefix, include=["metadata"]):
        rel = _relative_name(blob.name, prefix)
        if rel is None:
            continue
        top = _top_folder(rel)
        if regex.match(top):
            names.add(top)
    return sorted(names)


def download_sessions(
    container_name: str,
    dest_dir: str | os.PathLike,
    pattern: str = DEFAULT_SESSION_PATTERN,
    account_url: str = None,
    prefix: str = "",
) -> list[str]:
    """Download every blob whose session folder matches `pattern` from
    `container_name` into `dest_dir`, preserving the folder layout.

    `prefix` scopes the download to one folder inside the container (e.g. a
    route: "route_a"), and is stripped from the local paths — sessions land
    directly under `dest_dir` either way, so the split step downstream sees the
    same layout whether or not the corpus is organized into route folders.

    Returns the sorted list of session folder names that were downloaded.
    """
    regex = re.compile(pattern)
    prefix = _normalize_prefix(prefix)
    container = _service(account_url).get_container_client(container_name)
    dest_dir = Path(dest_dir)

    downloaded: set[str] = set()
    n_blobs = 0
    # name_starts_with keeps a prefixed run from paging through every other
    # route's blobs server-side, rather than listing the whole container and
    # discarding most of it here.
    for blob in container.list_blobs(name_starts_with=prefix, include=["metadata"]):
        rel = _relative_name(blob.name, prefix)
        if rel is None:
            continue
        top = _top_folder(rel)
        if not regex.match(top):
            continue
        if _is_dir_marker(blob, rel):
            continue

        target = dest_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"      [download] {blob.name} ({blob.size} bytes)")
        with open(target, "wb") as fh:
            container.download_blob(blob.name).readinto(fh)

        downloaded.add(top)
        n_blobs += 1

    print(f"[*] Downloaded {n_blobs} blob(s) across {len(downloaded)} session(s).")
    return sorted(downloaded)


def split_raw_path(raw_path: str) -> tuple[str, str]:
    """Split a catalog raw_path ("/raw/route_a/session_X/") into
    (container, blob prefix) — ("raw", "route_a/session_X/").

    The first path segment is the container, the same convention the session
    catalog's raw_path column uses.
    """
    parts = raw_path.strip("/").split("/", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError(f"raw_path {raw_path!r} is not /<container>/<folder>/...")
    return parts[0], parts[1].rstrip("/") + "/"


def download_session_paths(
    raw_paths,
    dest_dir: str | os.PathLike,
    account_url: str = None,
) -> list[str]:
    """Download exactly the session folders named by `raw_paths` — catalog
    raw_path values like "/raw/route_a/session_X/" — into `dest_dir`.

    The explicit-list counterpart to download_sessions(), for a caller that
    already knows which sessions it wants (ctcd's manifest) instead of matching
    every folder under a prefix. Each folder lands as dest_dir/<leaf>, the same
    flat layout download_sessions() produces, so the split step can't tell the
    two apart.

    A path with no blobs under it is a hard error: the caller asked for that
    session specifically, and silently building a dataset without it would
    look like success.

    Returns the sorted list of session folder names (leaves) downloaded.
    """
    service = _service(account_url)
    dest_dir = Path(dest_dir)

    downloaded: list[str] = []
    n_blobs = 0
    for raw_path in raw_paths:
        container_name, prefix = split_raw_path(raw_path)
        session = prefix.rstrip("/").rsplit("/", 1)[-1]
        if session in downloaded:
            # Two different folders with the same leaf name would land on top
            # of each other in dest_dir.
            raise SystemExit(f"[pipeline] Two sessions share the folder name "
                             f"{session!r}; can't stage both under {dest_dir}.")
        container = service.get_container_client(container_name)
        found = 0
        for blob in container.list_blobs(name_starts_with=prefix, include=["metadata"]):
            rel = blob.name[len(prefix):]
            if not rel or rel.endswith("/"):
                continue
            meta = getattr(blob, "metadata", None) or {}
            if meta.get("hdi_isfolder") == "true":
                continue
            target = dest_dir / session / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"      [download] {container_name}/{blob.name} ({blob.size} bytes)")
            with open(target, "wb") as fh:
                container.download_blob(blob.name).readinto(fh)
            found += 1
        if not found:
            raise SystemExit(f"[pipeline] No blobs under {container_name}/{prefix} "
                             f"(from raw_path {raw_path!r}).")
        downloaded.append(session)
        n_blobs += found

    print(f"[*] Downloaded {n_blobs} blob(s) across {len(downloaded)} session(s).")
    return sorted(downloaded)


def upload_directory(
    container_name: str,
    local_dir: str | os.PathLike,
    dest_prefix: str = "",
    account_url: str = None,
) -> int:
    """Recursively upload every file under `local_dir` to `container_name`,
    preserving the relative layout under `dest_prefix`. Overwrites existing
    blobs. Returns the number of files uploaded."""
    service = _service(account_url)
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
