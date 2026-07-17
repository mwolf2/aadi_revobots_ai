# -*- coding: utf-8 -*-
"""CICD helpers for the AGV data pipeline.

Azure integration modules used by
``lerobot/robots/revobots_agv_follower/data_pipeline.py``:

    * azure_blob_io    — download raw sessions / upload the built dataset
    * azure_keyvault   — inject secrets (HF token, connection string) as env vars
    * azure_ml_models  — pull YOLO weights from an Azure ML workspace/registry

These are loaded as flat top-level modules (the pipeline adds this directory to
sys.path), so importing this package is not required to use them.
"""
