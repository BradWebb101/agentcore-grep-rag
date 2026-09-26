"""Call Kimi K3 with Bedrock InvokeModel.

Strands' Bedrock provider uses the Converse API. Kimi K3's InvokeModel body is
the chat-completions message format, including tool calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import boto3
from botocore.exceptions import ClientError
from strands.models.model import Model
from strands.types.exceptions import ModelThrottledException
from strands.types.streaming import StreamEvent

log = logging.getLogger("grep_rag")

MAX_TOKENS = 8192


def _text_from_blocks(blocks: list) -> str:
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def to_kimi_messages(messages: list, system_prompt: str | None) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    if system_prompt:
        converted.append({"role": "system", "content": system_prompt})
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content") or []
        if not isinstance(content, list):
            continue
        if message.get("role") == "assistant":
            tool_calls = []
            for block in content:
                tool_use = block.get("toolUse") if isinstance(block, dict) else None
                if not isinstance(tool_use, dict):
                    continue
                tool_calls.append({
                    "id": tool_use.get("toolUseId"),
                    "type": "function",
                    "function": {
                        "name": tool_use.get("name"),
                        "arguments": json.dumps(tool_use.get("input") or {}),
                    },
                })
            item: dict[str, Any] = {
                "role": "assistant",
                "content": _text_from_blocks(content) or None,
            }
            if tool_calls:
                item["tool_calls"] = tool_calls
            converted.append(item)
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            result = block.get("toolResult")
            if isinstance(result, dict):
                converted.append({
                    "role": "tool",
                    "tool_call_id": result.get("toolUseId"),
                    "content": _text_from_blocks(result.get("content") or []) or str(result.get("status", "")),
                })
        text = _text_from_blocks(content)
        if text:
            converted.append({"role": "user", "content": text})
    return converted


def tool_specs_to_kimi(tool_specs: list | None) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for spec in tool_specs or []:
        schema = spec.get("inputSchema", {})
        if isinstance(schema, dict) and "json" in schema:
            schema = schema["json"]
        tools.append({
            "type": "function",
            "function": {
                "name": spec.get("name"),
                "description": spec.get("description", ""),
                "parameters": schema or {"type": "object", "properties": {}},
            },
        })
    return tools


def _message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _text_from_blocks(content)
    return ""


def _tool_arguments(function: dict) -> str:
    arguments = function.get("arguments", "{}")
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments)


def events_from_response(payload: dict) -> list[StreamEvent]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError(f"invoke_model response has no choices: {sorted(payload)}")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    finish = choice.get("finish_reason") or "stop"
    tool_calls = message.get("tool_calls") or []
    text = _message_text(message)
    events: list[StreamEvent] = [{"messageStart": {"role": "assistant"}}]
    index = 0
    if text:
        events.append({"contentBlockDelta": {"contentBlockIndex": index, "delta": {"text": text}}})
        events.append({"contentBlockStop": {"contentBlockIndex": index}})
        index += 1
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        events.append({
            "contentBlockStart": {
                "contentBlockIndex": index,
                "start": {"toolUse": {"name": function.get("name", ""), "toolUseId": call.get("id", "")}},
            },
        })
        events.append({
            "contentBlockDelta": {
                "contentBlockIndex": index,
                "delta": {"toolUse": {"input": _tool_arguments(function)}},
            },
        })
        events.append({"contentBlockStop": {"contentBlockIndex": index}})
        index += 1
    stop_reason = "tool_use" if finish == "tool_calls" or tool_calls else "end_turn"
    if finish == "length":
        stop_reason = "max_tokens"
    events.append({"messageStop": {"stopReason": stop_reason}})
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    events.append({
        "metadata": {
            "usage": {
                "inputTokens": input_tokens,
                "outputTokens": output_tokens,
                "totalTokens": input_tokens + output_tokens,
            },
            "metrics": {"latencyMs": 0},
        },
    })
    return events


class KimiInvokeModel(Model):
    """Bedrock InvokeModel client for the Kimi K3 chat-completions body."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._config: dict[str, Any] = {"model_id": model_id}
        self._client = None

    def update_config(self, **model_config: Any) -> None:
        self._config.update(model_config)
        if "model_id" in model_config:
            self.model_id = model_config["model_id"]

    def get_config(self) -> dict[str, Any]:
        return self._config

    def _runtime_client(self):
        if self._client is None:
            self._client = boto3.client("bedrock-runtime")
        return self._client

    def _invoke(self, body: dict[str, Any]) -> dict:
        try:
            response = self._runtime_client().invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body).encode("utf-8"),
            )
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"ThrottlingException", "TooManyRequestsException", "ServiceQuotaExceededException"}:
                raise ModelThrottledException(str(exc)) from exc
            raise
        return json.loads(response["body"].read())

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError("KimiInvokeModel does not implement structured output")
        yield {}

    async def stream(
        self,
        messages,
        tool_specs=None,
        system_prompt=None,
        *,
        tool_choice=None,
        system_prompt_content=None,
        invocation_state=None,
        cancel_signal=None,
        agent_metadata=None,
        **kwargs,
    ):
        del tool_choice, system_prompt_content, invocation_state, cancel_signal, agent_metadata, kwargs
        kimi_messages = to_kimi_messages(messages, system_prompt)
        tools = tool_specs_to_kimi(tool_specs)
        body: dict[str, Any] = {
            "messages": kimi_messages,
            "max_tokens": MAX_TOKENS,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        log.info("%s", json.dumps({
            "step": "invoke_model.request",
            "api": "InvokeModel",
            "model_id": self.model_id,
            "messages": len(kimi_messages),
            "tools": [tool["function"]["name"] for tool in tools],
        }))
        payload = await asyncio.to_thread(self._invoke, body)
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") if isinstance(choice, dict) else {}
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        log.info("%s", json.dumps({
            "step": "invoke_model.response",
            "api": "InvokeModel",
            "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
            "tool_calls": [
                call.get("function", {}).get("name")
                for call in tool_calls or []
                if isinstance(call, dict)
            ],
            "content_chars": len(_message_text(message) if isinstance(message, dict) else ""),
        }, default=str))
        for event in events_from_response(payload):
            yield event
