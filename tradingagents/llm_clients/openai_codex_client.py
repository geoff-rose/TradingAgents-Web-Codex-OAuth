"""OpenAI Codex OAuth client.

Uses the ChatGPT/Codex OAuth backend.  TradingAgents stores its own copy of
the Codex credentials in ``~/.tradingagents/codex_auth.json`` so it never
touches Hermes's ``~/.hermes/auth.json``.

Override with the ``TRADINGAGENTS_CODEX_AUTH_FILE`` env var if needed.
"""

from __future__ import annotations

import base64
import json
import os
import re
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


_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
_CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"


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

    auth_path = Path(os.environ.get(
        "TRADINGAGENTS_CODEX_AUTH_FILE",
        "~/.tradingagents/codex_auth.json",
    )).expanduser()
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


def _refresh_codex_token() -> Optional[str]:
    """Exchange the stored refresh_token for a new access_token and persist it."""
    import httpx

    auth_path = Path(os.environ.get(
        "TRADINGAGENTS_CODEX_AUTH_FILE",
        "~/.tradingagents/codex_auth.json",
    )).expanduser()
    try:
        data = json.loads(auth_path.read_text())
    except Exception:
        return None

    pool = data.get("credential_pool", {}).get("openai-codex") or []
    entry = next((e for e in pool if e.get("refresh_token")), None)
    if not entry:
        return None

    try:
        resp = httpx.post(
            _CODEX_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
            data={
                "grant_type": "refresh_token",
                "refresh_token": entry["refresh_token"],
                "client_id": _CODEX_CLIENT_ID,
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

    entry["access_token"] = new_access
    entry["refresh_token"] = tokens.get("refresh_token", entry["refresh_token"])
    entry["last_refresh"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        auth_path.parent.mkdir(parents=True, exist_ok=True)
        auth_path.write_text(json.dumps(data, indent=2))
    except Exception:
        pass

    return new_access


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


def _extract_json(text: str) -> str:
    """Strip markdown code fences and return the outermost JSON object."""
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()
    # Walk from the first '{' to its matching '}' so we capture the whole object
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


class _StructuredOutputWrapper:
    """Wraps CodexChatModel to produce a parsed Pydantic instance.

    Injects a compact JSON-output instruction into the system message so the
    model returns a JSON object, then parses and validates it into ``schema``.
    """

    def __init__(self, llm: "CodexChatModel", schema: Type[BaseModel]) -> None:
        self._llm = llm
        self._schema = schema
        # Build a compact schema description: field name → type hint + description
        field_descs = []
        for name, f in schema.model_fields.items():
            ann = f.annotation.__name__ if hasattr(f.annotation, "__name__") else str(f.annotation)
            desc = ""
            if f.description:
                # Keep just the first sentence so the instruction stays concise
                first = f.description.split(". ")[0].strip()
                desc = f" — {first}"
            field_descs.append(f'"{name}" ({ann}){desc}')
        schema_hint = "; ".join(field_descs)
        self._instruction = (
            f"\n\nRespond ONLY with a valid JSON object. Required keys:\n{schema_hint}\n"
            "Output nothing except the JSON object — no markdown fences, no explanations."
        )

    def invoke(self, messages: List[BaseMessage], **kwargs: Any) -> BaseModel:
        modified = self._inject_instruction(messages)
        result = self._llm.invoke(modified, **kwargs)
        text = _extract_json(result.content)
        data = json.loads(text)
        # model_validate allows extra fields; strict=False lets int/str coercion work
        return self._schema.model_validate(data)

    def _inject_instruction(self, messages) -> List[BaseMessage]:
        """Append the JSON instruction to the system message (or the last human message)."""
        # Plain string prompt (common from agent factories) — wrap it properly
        if isinstance(messages, str):
            return [
                SystemMessage(content=self._instruction.strip()),
                HumanMessage(content=messages),
            ]
        # PromptValue (e.g. ChatPromptTemplate output)
        if hasattr(messages, "to_messages"):
            messages = messages.to_messages()
        out = list(messages)
        for i, msg in enumerate(out):
            if isinstance(msg, SystemMessage):
                out[i] = SystemMessage(content=str(msg.content) + self._instruction)
                return out
        # No system message — prepend one
        out.insert(0, SystemMessage(content=self._instruction.strip()))
        return out


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

    def with_structured_output(self, schema: Any, *, method: Optional[str] = None, **kwargs: Any) -> "_StructuredOutputWrapper":
        """Return a wrapper that requests JSON from the model and parses it into ``schema``.

        The Codex backend does not support native JSON-schema mode, so we
        inject a JSON instruction into the system message and extract the
        response with a lightweight parser.  This gives the same Pydantic
        instance that callers expect from any other provider.
        """
        return _StructuredOutputWrapper(self, schema)

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
            if "token_expired" not in str(exc):
                raise
            new_token = _refresh_codex_token()
            if not new_token:
                raise
            client = OpenAI(
                api_key=new_token,
                base_url=self.base_url or _CODEX_BASE_URL,
                default_headers=_codex_headers(new_token),
            )
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
