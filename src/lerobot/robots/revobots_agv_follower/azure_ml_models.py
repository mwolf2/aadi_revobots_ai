# -*- coding: utf-8 -*-
"""Download registered model weights from Azure ML via MLClient.

Pulls YOLO detection/segmentation `.pt` weights from either an Azure ML
workspace or an Azure ML registry, so the pipeline doesn't depend on the
weights being baked into the image or present on local disk.

Auth uses DefaultAzureCredential, which works automatically with a Managed
Identity when running in Azure, and falls back to `az login`, environment
variables, VS Code, etc. for local development. Nothing sensitive lives in the
TOML — only the workspace/registry coordinates, model names and versions.

Requires:  pip install azure-ai-ml azure-identity
"""

import os
from pathlib import Path


def get_client(subscription_id=None, resource_group=None, workspace_name=None,
               registry_name=None):
    """Return an MLClient bound to a registry (if `registry_name` is given) or
    a workspace (subscription_id + resource_group + workspace_name)."""
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    credential = DefaultAzureCredential()
    if registry_name:
        return MLClient(credential=credential, registry_name=registry_name)
    if not (subscription_id and resource_group and workspace_name):
        raise SystemExit(
            "[mlmodels] Provide either models.registry_name, or all of "
            "models.subscription_id / resource_group / workspace_name."
        )
    return MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )


def _resolve_latest_version(client, name):
    """Return the highest version string for model `name` (numeric-aware)."""
    versions = [m.version for m in client.models.list(name=name)]
    if not versions:
        raise SystemExit(f"[mlmodels] No versions found for model '{name}'.")
    return max(versions, key=lambda v: (int(v), v) if str(v).isdigit() else (-1, v))


def _find_weight(root, weight_filename=None):
    """Locate the .pt weight inside a downloaded model artifact tree."""
    root = Path(root)
    if weight_filename:
        matches = list(root.rglob(weight_filename))
        if matches:
            return matches[0]
        print(f"[mlmodels] '{weight_filename}' not found under {root}; "
              f"falling back to first *.pt.")
    pts = sorted(root.rglob("*.pt"))
    if not pts:
        raise SystemExit(f"[mlmodels] No .pt weight found under {root}.")
    return pts[0]


def download_weight(client, name, dest_dir, version=None, weight_filename=None):
    """Download registered model `name` (given version, or latest) into
    `dest_dir` and return the local path to its .pt weight file."""
    dest = Path(os.path.expanduser(dest_dir))
    dest.mkdir(parents=True, exist_ok=True)

    if not version:
        version = _resolve_latest_version(client, name)

    print(f"[mlmodels] Downloading model '{name}' v{version} -> {dest}")
    client.models.download(name=name, version=str(version),
                           download_path=str(dest))

    # MLClient writes the artifact under dest/<name>/.
    artifact_root = dest / name
    weight = _find_weight(artifact_root if artifact_root.is_dir() else dest,
                          weight_filename)
    print(f"[mlmodels] Resolved weight: {weight}")
    return str(weight)
