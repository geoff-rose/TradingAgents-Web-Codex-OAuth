"""Hermes-backed Anthropic Claude client.

Reads the Anthropic OAuth access token from the Hermes credential pool
(``~/.hermes/auth.json``, ``credential_pool.anthropic[].access_token``)
rather than from an environment variable.

OAuth tokens (``sk-ant-oat01-*``) must be sent as ``Authorization: Bearer``
not ``x-api-key``. This client overrides ChatAnthropic._client_params to
swap ``api_key`` for ``auth_token`` so the Anthropic SDK uses the correct
header transparently.

Add credentials once with:
    hermes auth add anthropic --type oauth

Then use provider ``hermes-claude`` in TradingAgents config.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from langchain_anthropic import ChatAnthropic

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "max_tokens",
    "callbacks",
)


def _read_hermes_anthropic_token() -> Optional[str]:
    """Return the Anthropic OAuth access token from Hermes credential pool.

    Resolution order:
    1. TRADINGAGENTS_ANTHROPIC_API_KEY env var
    2. ANTHROPIC_API_KEY env var
    3. ~/.hermes/auth.json credential_pool.anthropic[].access_token
    """
    env_key = os.environ.get("TRADINGAGENTS_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        return env_key

    auth_path = Path(os.environ.get("HERMES_AUTH_FILE", "~/.hermes/auth.json")).expanduser()
    try:
        data = json.loads(auth_path.read_text())
    except Exception:
        return None

    for entry in data.get("credential_pool", {}).get("anthropic") or []:
        token = entry.get("access_token")
        if token:
            return token

    # Also accept a top-level providers entry (future Hermes schema)
    token = (
        data.get("providers", {})
        .get("anthropic", {})
        .get("tokens", {})
        .get("access_token")
    )
    return token if isinstance(token, str) and token else None


class _OAuthChatAnthropic(ChatAnthropic):
    """ChatAnthropic that sends OAuth bearer auth instead of x-api-key.

    Overrides _client_params to replace ``api_key`` with ``auth_token`` so
    the underlying Anthropic SDK sends ``Authorization: Bearer <token>``
    rather than ``x-api-key: <token>``.
    """

    _oauth_token: str = ""

    @property
    def _client_params(self) -> dict[str, Any]:
        params = super()._client_params
        # Swap api_key → auth_token so the SDK uses the correct header
        params.pop("api_key", None)
        params["auth_token"] = self._oauth_token
        return params

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


class HermesClaudeClient(BaseLLMClient):
    """Anthropic Claude client backed by Hermes OAuth credential pool."""

    provider = "hermes-claude"

    def get_llm(self) -> Any:
        token = _read_hermes_anthropic_token()
        if not token:
            raise ValueError(
                "Anthropic OAuth token not found in Hermes credential pool. "
                "Run: hermes auth add anthropic --type oauth"
            )

        self.warn_if_unknown_model()

        llm_kwargs: dict[str, Any] = {
            "model": self.model,
            # Dummy key satisfies LangChain's required-field check; the real
            # OAuth token is injected via _client_params override below.
            "anthropic_api_key": "sk-ant-placeholder",
        }
        if self.base_url:
            llm_kwargs["anthropic_api_url"] = self.base_url

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        llm = _OAuthChatAnthropic(**llm_kwargs)
        llm._oauth_token = token
        return llm

    def validate_model(self) -> bool:
        return validate_model("anthropic", self.model)
