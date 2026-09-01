"""xAI Grok OAuth client backed by grok-cli credentials.

Reads tokens from ~/.grok-cli/auth.json (written by `grok-cli login`).
Refreshes automatically on expiry using the xAI OAuth endpoint.
Uses the xAI /responses API (OpenAI-compatible, same wire format as Codex).

Only meaningful when the user has authenticated via `grok-cli login` with
a SuperGrok or X Premium+ subscription.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from openai import AuthenticationError, OpenAI
from pydantic import BaseModel

from .base_client import BaseLLMClient
from .openai_codex_client import (
    _StructuredOutputWrapper,
    _content_to_responses,
    _extract_json,
    _messages_to_responses,
)

_XAI_BASE_URL = "https://api.x.ai/v1"
_XAI_TOKEN_URL = "https://auth.x.ai/oauth2/token"
_XAI_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"

_AUTH_FILE = Path(os.environ.get("GROK_CLI_AUTH_FILE", "~/.grok-cli/auth.json")).expanduser()


_KEYCHAIN_KEY = f"https://auth.x.ai::{_XAI_CLIENT_ID}"


def _load_auth_entry() -> Optional[dict]:
    """Return the auth entry dict regardless of which format auth.json uses."""
    try:
        data = json.loads(_AUTH_FILE.read_text())
    except Exception:
        return None
    # Keychain format (Windows grok-cli): top-level key is "<issuer>::<client_id>"
    if _KEYCHAIN_KEY in data:
        return data[_KEYCHAIN_KEY], "keychain", data
    # grok-cli Linux/manual format: {"tokens": {...}, "base_url": ...}
    if "tokens" in data:
        return data["tokens"], "tokens", data
    return None


def _read_grok_token() -> Optional[str]:
    env_token = os.environ.get("TRADINGAGENTS_XAI_ACCESS_TOKEN")
    if env_token:
        return env_token
    result = _load_auth_entry()
    if result is None:
        return None
    entry, fmt, _ = result
    # Keychain format uses "key" for the access token; tokens format uses "access_token"
    token = entry.get("key") if fmt == "keychain" else entry.get("access_token")
    return token if isinstance(token, str) and token else None


def _token_expiry() -> Optional[float]:
    """Return the expiry timestamp of the current access token, or None."""
    import base64 as _b64
    token = _read_grok_token()
    if not token or "." not in token:
        return None
    try:
        payload = token.split(".")[1]
        payload += "=" * (4 - len(payload) % 4)
        claims = json.loads(_b64.urlsafe_b64decode(payload))
        exp = claims.get("exp")
        return float(exp) if exp else None
    except Exception:
        return None


def _refresh_grok_token() -> Optional[str]:
    """Exchange the stored refresh_token for a new access_token and persist it."""
    import httpx

    result = _load_auth_entry()
    if result is None:
        return None
    entry, fmt, data = result

    refresh_token = entry.get("refresh_token")
    if not refresh_token:
        return None

    try:
        resp = httpx.post(
            _XAI_TOKEN_URL,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": _XAI_CLIENT_ID,
            },
            timeout=20.0,
        )
        resp.raise_for_status()
        tokens = resp.json()
    except Exception:
        return None

    new_access = tokens.get("access_token")
    if not new_access:
        return None

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if fmt == "keychain":
        data[_KEYCHAIN_KEY]["key"] = new_access
        data[_KEYCHAIN_KEY]["refresh_token"] = tokens.get("refresh_token", refresh_token)
        data[_KEYCHAIN_KEY]["expires_at"] = now
    else:
        data.setdefault("tokens", {})["access_token"] = new_access
        data["tokens"]["refresh_token"] = tokens.get("refresh_token", refresh_token)
        data["last_refresh"] = now

    try:
        _AUTH_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass

    return new_access


def _convert_tools(tools: Optional[Iterable[Any]]) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    converted: List[Dict[str, Any]] = []
    for tool in tools:
        spec = convert_to_openai_tool(tool)
        fn = spec.get("function", {})
        name = fn.get("name")
        if not name:
            continue
        converted.append({
            "type": "function",
            "name": name,
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        })
    return converted or None


class GrokChatModel(BaseChatModel):
    model_name: str
    base_url: str = _XAI_BASE_URL
    tools: Optional[List[Dict[str, Any]]] = None
    timeout: Optional[float] = None

    @property
    def _llm_type(self) -> str:
        return "xai-grok"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "base_url": self.base_url}

    def bind_tools(
        self,
        tools: Iterable[Dict[str, Any] | type | BaseTool | Any],
        *,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> "GrokChatModel":
        return self.model_copy(update={"tools": _convert_tools(tools)})

    def with_structured_output(
        self, schema: Any, *, method: Optional[str] = None, **kwargs: Any
    ) -> "_StructuredOutputWrapper":
        return _StructuredOutputWrapper(self, schema)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        token = _read_grok_token()
        if not token:
            raise ValueError(
                "xAI Grok OAuth token not found. Run `grok-cli login` or set "
                "TRADINGAGENTS_XAI_ACCESS_TOKEN."
            )

        instructions, input_items = _messages_to_responses(messages)
        client = OpenAI(api_key=token, base_url=self.base_url or _XAI_BASE_URL)

        request: Dict[str, Any] = {
            "model": self.model_name,
            "instructions": instructions,
            "input": input_items,
            "store": False,
        }
        if self.timeout is not None:
            request["timeout"] = self.timeout
        if self.tools:
            request["tools"] = self.tools

        def _do_stream(c: OpenAI) -> tuple:
            texts: List[str] = []
            items: List[Any] = []
            with c.responses.stream(**request) as stream:
                for event in stream:
                    event_type = getattr(event, "type", "")
                    if event_type == "response.output_item.done":
                        item = getattr(event, "item", None)
                        if item is not None:
                            items.append(item)
                    elif "output_text.delta" in event_type:
                        delta = getattr(event, "delta", "")
                        if delta:
                            texts.append(delta)
                return texts, items, stream.get_final_response()

        try:
            collected_text, collected_items, response = _do_stream(client)
        except AuthenticationError as exc:
            if "token_expired" not in str(exc) and "401" not in str(exc):
                raise
            new_token = _refresh_grok_token()
            if not new_token:
                raise
            client = OpenAI(api_key=new_token, base_url=self.base_url or _XAI_BASE_URL)
            collected_text, collected_items, response = _do_stream(client)

        if not (getattr(response, "output", None) or None) and collected_items:
            response.output = collected_items

        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        for item in getattr(response, "output", []) or []:
            item_type = getattr(item, "type", None)
            if item_type == "message":
                for part in getattr(item, "content", []) or []:
                    part_type = getattr(part, "type", None)
                    if part_type in {"output_text", "text"}:
                        text_parts.append(getattr(part, "text", "") or "")
            elif item_type == "function_call":
                name = getattr(item, "name", "") or ""
                arguments = getattr(item, "arguments", "{}") or "{}"
                try:
                    args = json.loads(arguments)
                except Exception:
                    args = {}
                tool_calls.append({
                    "id": getattr(item, "call_id", None) or name,
                    "name": name,
                    "args": args,
                })

        content = ("".join(text_parts) or "".join(collected_text)).strip()
        message = AIMessage(content=content, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=message)])


class XAIGrokClient(BaseLLMClient):
    """TradingAgents client for xAI Grok via grok-cli OAuth."""

    provider = "xai-grok"

    def get_llm(self) -> Any:
        return GrokChatModel(
            model_name=self.model,
            base_url=self.base_url or _XAI_BASE_URL,
            timeout=self.kwargs.get("timeout"),
        )

    def validate_model(self) -> bool:
        return True
