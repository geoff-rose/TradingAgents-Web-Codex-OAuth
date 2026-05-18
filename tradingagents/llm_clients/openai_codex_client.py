"""OpenAI Codex OAuth client backed by Hermes credentials.

This provider is intentionally separate from the normal ``openai`` provider.
TradingAgents' ``openai`` path uses a public OpenAI API key.  ``openai-codex``
uses the ChatGPT/Codex OAuth backend that Hermes stores in
``~/.hermes/auth.json``.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from openai import OpenAI

from .base_client import BaseLLMClient


_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"


def _decode_jwt_payload(token: str) -> Dict[str, Any]:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return {}


def _codex_headers(access_token: str) -> Dict[str, str]:
    headers = {
        "User-Agent": "codex_cli_rs/0.0.0 (TradingAgents)",
        "originator": "codex_cli_rs",
    }
    claims = _decode_jwt_payload(access_token)
    account_id = claims.get("https://api.openai.com/auth", {}).get("chatgpt_account_id")
    if isinstance(account_id, str) and account_id:
        headers["ChatGPT-Account-ID"] = account_id
    return headers


def _read_hermes_codex_token() -> Optional[str]:
    env_token = os.environ.get("TRADINGAGENTS_CODEX_ACCESS_TOKEN")
    if env_token:
        return env_token

    auth_path = Path(os.environ.get("HERMES_AUTH_FILE", "~/.hermes/auth.json")).expanduser()
    try:
        data = json.loads(auth_path.read_text())
    except Exception:
        return None

    pool = data.get("credential_pool", {}).get("openai-codex") or []
    for entry in pool:
        token = entry.get("access_token")
        if token:
            return token

    token = data.get("providers", {}).get("openai-codex", {}).get("tokens", {}).get("access_token")
    return token if isinstance(token, str) and token else None


def _content_to_responses(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)

    converted: List[Dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype in {"text", "input_text"}:
            converted.append({"type": "input_text", "text": part.get("text", "")})
        elif ptype in {"image_url", "input_image"}:
            image = part.get("image_url", "")
            if isinstance(image, dict):
                image = image.get("url", "")
            converted.append({"type": "input_image", "image_url": image})
    return converted or ""


def _messages_to_responses(messages: List[BaseMessage]) -> tuple[str, List[Dict[str, Any]]]:
    instructions = "You are a helpful financial trading research assistant."
    inputs: List[Dict[str, Any]] = []

    for message in messages:
        if isinstance(message, SystemMessage):
            instructions = str(message.content)
            continue
        if isinstance(message, HumanMessage):
            inputs.append({"role": "user", "content": _content_to_responses(message.content)})
            continue
        if isinstance(message, ToolMessage):
            inputs.append({
                "type": "function_call_output",
                "call_id": message.tool_call_id,
                "output": str(message.content),
            })
            continue
        if isinstance(message, AIMessage):
            if message.content:
                inputs.append({"role": "assistant", "content": str(message.content)})
            for call in message.tool_calls or []:
                inputs.append({
                    "type": "function_call",
                    "call_id": call.get("id") or call.get("name") or "call",
                    "name": call.get("name", ""),
                    "arguments": json.dumps(call.get("args", {})),
                })
            continue
        inputs.append({"role": "user", "content": str(message.content)})

    return instructions, inputs or [{"role": "user", "content": ""}]


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


class CodexChatModel(BaseChatModel):
    model_name: str
    base_url: str = _CODEX_BASE_URL
    tools: Optional[List[Dict[str, Any]]] = None
    timeout: Optional[float] = None

    @property
    def _llm_type(self) -> str:
        return "openai-codex"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "base_url": self.base_url}

    def bind_tools(
        self,
        tools: Iterable[Dict[str, Any] | type | BaseTool | Any],
        *,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> "CodexChatModel":
        return self.model_copy(update={"tools": _convert_tools(tools)})

    def with_structured_output(self, schema: Any, *, method: Optional[str] = None, **kwargs: Any) -> Any:
        raise NotImplementedError("openai-codex structured output is not implemented; free-text fallback will be used")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        token = _read_hermes_codex_token()
        if not token:
            raise ValueError(
                "OpenAI Codex OAuth token not found. Run `hermes auth add openai-codex` "
                "or set TRADINGAGENTS_CODEX_ACCESS_TOKEN."
            )

        instructions, input_items = _messages_to_responses(messages)
        client = OpenAI(
            api_key=token,
            base_url=self.base_url or _CODEX_BASE_URL,
            default_headers=_codex_headers(token),
        )

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
        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort:
            request["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
            request["include"] = ["reasoning.encrypted_content"]

        collected_text: List[str] = []
        collected_items: List[Any] = []
        with client.responses.stream(**request) as stream:
            for event in stream:
                event_type = getattr(event, "type", "")
                if event_type == "response.output_item.done":
                    item = getattr(event, "item", None)
                    if item is not None:
                        collected_items.append(item)
                elif "output_text.delta" in event_type:
                    delta = getattr(event, "delta", "")
                    if delta:
                        collected_text.append(delta)
            response = stream.get_final_response()
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


class OpenAICodexClient(BaseLLMClient):
    """TradingAgents client for Hermes' OpenAI Codex OAuth provider."""

    provider = "openai-codex"

    def get_llm(self) -> Any:
        return CodexChatModel(
            model_name=self.model,
            base_url=self.base_url or _CODEX_BASE_URL,
            timeout=self.kwargs.get("timeout"),
        )

    def validate_model(self) -> bool:
        # The Codex backend's model allow-list is account-dependent and moves
        # faster than this project. Accept user-provided model IDs.
        return True
