# -*- coding: utf-8 -*-
"""Pull secrets from Azure Key Vault and inject them as environment variables.

Auth uses DefaultAzureCredential, which works automatically with a Managed
Identity when running in Azure, and falls back to `az login`, environment
variables, VS Code, etc. for local development. Nothing sensitive is stored in
the TOML config — only the vault URL and the secret *names*.

The connection-string secret is injected under AZURE_STORAGE_CONNECTION_STRING,
which is exactly the variable azure_blob_io reads (keep the two in sync). The
HuggingFace token is injected under HF_TOKEN, which huggingface_hub picks up
ambiently during push_to_hub().

Requires:  pip install azure-identity azure-keyvault-secrets
"""

import os

# Must match azure_blob_io.CONN_STR_ENV.
CONN_STR_ENV = "AZURE_STORAGE_CONNECTION_STRING"
HF_TOKEN_ENV = "HF_TOKEN"


def load_secrets(vault_url, connection_string_secret=None, hf_token_secret=None):
    """Fetch the named secrets from `vault_url` and export them as env vars.

    Only secrets whose name is provided (non-empty) are fetched; pass None/""
    to skip either one. Returns the list of environment variable names that
    were set. The azure SDK imports happen here (not at module import time) so
    the pipeline can run without these packages when Key Vault is disabled.
    """
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient

    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=vault_url, credential=credential)

    injected = []
    if connection_string_secret:
        os.environ[CONN_STR_ENV] = client.get_secret(connection_string_secret).value
        injected.append(CONN_STR_ENV)
        print(f"[keyvault] Loaded {CONN_STR_ENV} from secret "
              f"'{connection_string_secret}'.")
    if hf_token_secret:
        os.environ[HF_TOKEN_ENV] = client.get_secret(hf_token_secret).value
        injected.append(HF_TOKEN_ENV)
        print(f"[keyvault] Loaded {HF_TOKEN_ENV} from secret '{hf_token_secret}'.")

    return injected
