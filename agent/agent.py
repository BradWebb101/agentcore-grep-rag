"""Answer from an S3 corpus by copying it into session storage and grepping it.

The AWS CLI and grep run inside the AgentCore session. Files land on the
managed session mount, so the same runtime session id keeps them across
stop and resume. There is no vector index.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent, tool
from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import (
    AfterInvocationEvent,
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    MessageAddedEvent,
)
from kimi_invoke import KimiInvokeModel

LOG_PREVIEW_CHARS = 4_000


def _configure_logging() -> logging.Logger:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    for name in ("grep_rag", "strands", "bedrock_agentcore"):
        logging.getLogger(name).setLevel(level)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return logging.getLogger("grep_rag")


log = _configure_logging()

SESSION_MOUNT = Path(os.environ.get("SESSION_MOUNT", "/mnt/workspace")).resolve()
CORPUS_BUCKET = os.environ["CORPUS_BUCKET"]
CORPUS_PREFIX = os.environ.get("CORPUS_PREFIX", "corpus").strip("/")
MODEL_ID = os.environ.get("MODEL_ID", "global.moonshotai.kimi-k3")
MAX_OUTPUT_CHARS = 16_000
COMMAND_TIMEOUT_SECONDS = 180

app = BedrockAgentCoreApp(debug=True)


def _clip(value: object, limit: int = LOG_PREVIEW_CHARS) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated {len(text) - limit} chars]"


def _step(step: str, **fields: object) -> None:
    log.info("%s", json.dumps({"step": step, **fields}, default=str, ensure_ascii=False))


def _message_summary(message: object) -> str:
    if not isinstance(message, dict):
        return _clip(message)
    parts: list[str] = [f"role={message.get('role', '')}"]
    blocks = message.get("content", [])
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if isinstance(block.get("text"), str):
                parts.append(f"text={_clip(block['text'])}")
            tool_use = block.get("toolUse")
            if isinstance(tool_use, dict):
                parts.append(
                    "toolUse "
                    + _clip({
                        "name": tool_use.get("name"),
                        "toolUseId": tool_use.get("toolUseId"),
                        "input": tool_use.get("input"),
                    })
                )
            tool_result = block.get("toolResult")
            if isinstance(tool_result, dict):
                texts = [
                    item.get("text", "")
                    for item in tool_result.get("content", [])
                    if isinstance(item, dict)
                ]
                parts.append(
                    "toolResult "
                    + _clip({
                        "toolUseId": tool_result.get("toolUseId"),
                        "status": tool_result.get("status"),
                        "text": "".join(texts),
                    })
                )
    return " | ".join(parts)


class StepLogger(HookProvider):
    """Log each agent-loop step to stdout so AgentCore ships it to CloudWatch."""

    def __init__(self) -> None:
        self._started = 0.0
        self._model_calls = 0

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeInvocationEvent, self.on_invoke_start)
        registry.add_callback(AfterInvocationEvent, self.on_invoke_end)
        registry.add_callback(BeforeModelCallEvent, self.on_model_start)
        registry.add_callback(AfterModelCallEvent, self.on_model_end)
        registry.add_callback(BeforeToolCallEvent, self.on_tool_start)
        registry.add_callback(AfterToolCallEvent, self.on_tool_end)
        registry.add_callback(MessageAddedEvent, self.on_message)

    def on_invoke_start(self, event: BeforeInvocationEvent) -> None:
        self._started = time.perf_counter()
        self._model_calls = 0
        _step(
            "invoke.start",
            messages=[_message_summary(message) for message in (event.messages or [])],
        )

    def on_invoke_end(self, event: AfterInvocationEvent) -> None:
        result = event.result
        message = getattr(result, "message", None)
        _step(
            "invoke.end",
            elapsed_s=round(time.perf_counter() - self._started, 3),
            model_calls=self._model_calls,
            stop_reason=getattr(result, "stop_reason", None),
            answer=_message_text(message) if message is not None else "",
        )

    def on_model_start(self, event: BeforeModelCallEvent) -> None:
        self._model_calls += 1
        removed = _strip_reasoning_blocks(event.agent.messages)
        _step(
            "model.start",
            call=self._model_calls,
            model_id=MODEL_ID,
            projected_input_tokens=event.projected_input_tokens,
            reasoning_blocks_removed=removed,
        )

    def on_model_end(self, event: AfterModelCallEvent) -> None:
        response = event.stop_response
        _step(
            "model.end",
            call=self._model_calls,
            stop_reason=response.stop_reason if response else None,
            message=_message_summary(response.message) if response else None,
            error=repr(event.exception) if event.exception else None,
        )

    def on_tool_start(self, event: BeforeToolCallEvent) -> None:
        tool_use = event.tool_use
        _step(
            "tool.start",
            name=tool_use.get("name"),
            tool_use_id=tool_use.get("toolUseId"),
            input=tool_use.get("input"),
        )

    def on_tool_end(self, event: AfterToolCallEvent) -> None:
        result: Any = event.result
        texts: list[str] = []
        status = None
        if isinstance(result, dict):
            status = result.get("status")
            texts = [
                item.get("text", "")
                for item in result.get("content", [])
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ]
        _step(
            "tool.end",
            name=event.tool_use.get("name"),
            tool_use_id=event.tool_use.get("toolUseId"),
            status=status,
            duration_s=None if event.duration is None else round(event.duration, 3),
            output=_clip("".join(texts)),
            error=repr(event.exception) if event.exception else None,
            cancelled=event.cancel_message,
        )

    def on_message(self, event: MessageAddedEvent) -> None:
        _step("message.added", message=_message_summary(event.message))


def _strip_reasoning_blocks(messages: list) -> int:
    """Drop prior reasoning blocks before Converse.

    Kimi K3 returns InternalServerException on Converse when an earlier turn's
    reasoning content is sent again. Strands keeps those blocks by default.
    """
    removed = 0
    kept_messages = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            kept_messages.append(message)
            continue
        kept = [
            block for block in content
            if not (isinstance(block, dict) and "reasoningContent" in block)
        ]
        removed += len(content) - len(kept)
        if not kept:
            continue
        message["content"] = kept
        kept_messages.append(message)
    if len(kept_messages) != len(messages):
        messages[:] = kept_messages
    return removed


def _under_session(relative_path: str) -> Path:
    if "\x00" in relative_path:
        raise ValueError("path contains a null byte")
    candidate = (SESSION_MOUNT / relative_path).resolve()
    if candidate != SESSION_MOUNT and SESSION_MOUNT not in candidate.parents:
        raise ValueError(f"path must stay under {SESSION_MOUNT}")
    return candidate


def _validate_s3_uri(s3_uri: str) -> str:
    allowed = f"s3://{CORPUS_BUCKET}"
    if s3_uri != allowed and not s3_uri.startswith(allowed + "/"):
        raise ValueError(f"s3_uri must be {allowed} or {allowed}/<key>")
    if any(char in s3_uri for char in ("\n", "\r", "\x00")):
        raise ValueError("s3_uri contains a control character")
    return s3_uri


def _run(argv: list[str]) -> str:
    started = time.perf_counter()
    _step("command.start", argv=argv)
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        _step(
            "command.timeout",
            argv=argv,
            timeout_s=COMMAND_TIMEOUT_SECONDS,
            stdout=_clip(exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")),
            stderr=_clip(exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")),
        )
        return f"command timed out after {COMMAND_TIMEOUT_SECONDS}s"
    output = (completed.stdout or "") + (completed.stderr or "")
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + "\n...[truncated]"
    no_matches = argv[0] == "grep" and completed.returncode == 1
    returned = "no matches" if no_matches else (
        f"exit {completed.returncode}\n{output}".strip() if completed.returncode != 0
        else output.strip() or "(no output)"
    )
    _step(
        "command.end",
        argv=argv,
        returncode=completed.returncode,
        elapsed_s=round(time.perf_counter() - started, 3),
        output_chars=len(returned),
        output=_clip(returned),
    )
    return returned


@tool
def s3_cp(s3_uri: str, destination: str = ".") -> str:
    """Copy objects from the corpus bucket into session storage with the AWS CLI.

    A URI that is the bucket root or ends with / is copied recursively.
    A URI of a single object is copied into the destination directory.
    Credentials are the AgentCore runtime execution role.

    Args:
        s3_uri: Source, for example s3://the-corpus-bucket/docs/.
        destination: Directory under the session mount. "." is the mount root.
    """
    _step("s3_cp.requested", s3_uri=s3_uri, destination=destination)
    try:
        source = _validate_s3_uri(s3_uri)
        dest = _under_session(destination)
        dest.mkdir(parents=True, exist_ok=True)
        bucket_root = f"s3://{CORPUS_BUCKET}"
        recursive = source.endswith("/") or source == bucket_root
        if source == bucket_root:
            source = bucket_root + "/"
        argv = ["aws", "s3", "cp", source, f"{dest}/", "--only-show-errors"]
        if recursive:
            argv.append("--recursive")
        _step("s3_cp.planned", source=source, destination=str(dest), recursive=recursive)
        return _run(argv)
    except Exception as exc:
        _step("s3_cp.failed", error=repr(exc), traceback=traceback.format_exc())
        return f"s3_cp failed: {exc}"


@tool
def grep_files(pattern: str, path: str = ".") -> str:
    """Search files already copied into session storage.

    Uses grep -n -R. List the session directory first. Copy from S3 only when
    the files are not already there.

    Args:
        pattern: Basic regular expression passed to grep -e.
        path: Directory or file under the session mount. "." is the mount root.
    """
    _step("grep.requested", pattern=pattern, path=path)
    try:
        if not pattern or len(pattern) > 500 or "\x00" in pattern:
            raise ValueError("pattern must be 1-500 characters")
        root = _under_session(path)
        _step("grep.planned", pattern=pattern, root=str(root))
        argv = [
            "grep",
            "-n",
            "-R",
            "-m",
            "40",
            "--binary-files=without-match",
            "-e",
            pattern,
            str(root),
        ]
        return _run(argv)
    except Exception as exc:
        _step("grep.failed", error=repr(exc), traceback=traceback.format_exc())
        return f"grep_files failed: {exc}"


@tool
def list_session(path: str = ".") -> str:
    """List files already stored for this runtime session.

    Call this before s3_cp. If the repository files are listed, grep them
    and do not copy them again.

    Args:
        path: Directory under the session mount. "." is the mount root.
            Use an owner directory such as "aleju" to see that owner's files.
    """
    _step("list_session.requested", path=path)
    try:
        root = _under_session(path)
        if not root.exists():
            _step("list_session.empty", path=str(root))
            return f"{root} does not exist"
        completed = subprocess.run(
            ["find", str(root), "-maxdepth", "2", "-type", "f"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        files = [line for line in (completed.stdout or "").splitlines() if line]
        preview = "\n".join(files[:80])
        if len(files) > 80:
            preview += f"\n...[{len(files) - 80} more files]"
        summary = preview or "(empty)"
        _step("list_session.result", path=str(root), file_count=len(files))
        return f"{len(files)} files under {root}\n{summary}"
    except Exception as exc:
        _step("list_session.failed", error=repr(exc), traceback=traceback.format_exc())
        return f"list_session failed: {exc}"


def _message_text(message: object) -> str:
    if isinstance(message, dict):
        blocks = message.get("content", [])
        if isinstance(blocks, list):
            parts = [
                block["text"]
                for block in blocks
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            ]
            if parts:
                return "\n".join(parts)
    return str(message)


SYSTEM_PROMPT = f"""You answer questions about GitHub READMEs. You do not have a vector database.
The person will not tell you whether the files are already on disk. You decide.

Corpus, if you need to copy: s3://{CORPUS_BUCKET}/{CORPUS_PREFIX}/
Files are named <owner>/<repo>.md.
Session storage: {SESSION_MOUNT}
Anything written there stays for this runtime session.

To answer:
1. Call list_session for the owner or repository the question names. If you do not know the directory, list ".".
2. If those files are already listed, do not copy them. Call grep_files on that directory.
3. If the directory is missing or empty, call s3_cp for that prefix, then grep_files.
4. Answer only with lines grep returned. Include the file path and line number. If grep finds nothing, say so.

Use only list_session, s3_cp, and grep_files.
"""

agent = Agent(
    model=KimiInvokeModel(MODEL_ID),
    system_prompt=SYSTEM_PROMPT,
    tools=[list_session, s3_cp, grep_files],
    hooks=[StepLogger()],
)


def _prompt_from_payload(payload: object) -> str:
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        return ""
    prompt = payload.get("prompt")
    nested = payload.get("input")
    if prompt is None and isinstance(nested, dict):
        prompt = nested.get("prompt")
    return prompt if isinstance(prompt, str) else ""


@app.entrypoint
def invoke(payload: object) -> dict:
    started = time.perf_counter()
    prompt = _prompt_from_payload(payload).strip()
    _step(
        "request.received",
        prompt=prompt,
        payload_type=type(payload).__name__,
        model_id=MODEL_ID,
        corpus=f"s3://{CORPUS_BUCKET}/{CORPUS_PREFIX}/",
        session_mount=str(SESSION_MOUNT),
    )
    if not prompt:
        _step("request.rejected", reason="prompt is required")
        return {"error": "prompt is required"}
    try:
        result = agent(prompt)
        text = _message_text(result.message)
        _step(
            "request.completed",
            elapsed_s=round(time.perf_counter() - started, 3),
            answer=text,
        )
        return {"answer": text}
    except Exception as exc:
        _step(
            "request.failed",
            elapsed_s=round(time.perf_counter() - started, 3),
            error=repr(exc),
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    SESSION_MOUNT.mkdir(parents=True, exist_ok=True)
    _step(
        "runtime.ready",
        model_id=MODEL_ID,
        corpus_bucket=CORPUS_BUCKET,
        corpus_prefix=CORPUS_PREFIX,
        session_mount=str(SESSION_MOUNT),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )
    app.run(host="0.0.0.0", port=8080)
