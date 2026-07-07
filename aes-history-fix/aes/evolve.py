"""Minimal advisor-worker evolutionary search loop."""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic


# ---------------------------------------------------------------------------
# Configuration and result records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentConfig:
    prompt_file: Path
    timeout_seconds: int
    model: str | None = None
    command: tuple[str, ...] | None = None
    max_tokens: int = 8096
    temperature: float | None = None
    reasoning_effort: str | None = None

    @property
    def is_subprocess(self) -> bool:
        return self.command is not None


@dataclass(frozen=True)
class AppConfig:
    kind: str
    workspace: Path
    editable_files: tuple[str, ...]
    evaluation_command: tuple[str, ...]
    metric_pattern: str
    metric_group: int
    direction: str
    evaluation_timeout_seconds: int
    environment: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    name: str
    iterations: int
    runs_dir: Path
    improvement_tolerance: float
    advisor: AgentConfig
    worker: AgentConfig
    app: AppConfig
    path: Path
    use_advisor: bool = True


@dataclass(frozen=True)
class AgentResult:
    text: str
    stderr: str
    returncode: int
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def success(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True)
class Evaluation:
    success: bool
    metric: float | None
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


@dataclass
class Experiment:
    number: int
    status: str
    metric: float | None
    best_metric: float
    plan: str
    report: str
    changed_files: list[str]
    patch_file: str | None
    candidate_files: list[str]
    evaluation_file: str
    error: str = ""
    advisor_input_tokens: int = 0
    advisor_output_tokens: int = 0
    worker_input_tokens: int = 0
    worker_output_tokens: int = 0


ADVISOR_STATE_VERSION = 1
ADVISOR_RECENT_EXPERIMENTS = 8
ADVISOR_TOOL_MAX_CHARS = 20_000
INFRASTRUCTURE_ERROR_LIMIT = 3


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path if path.is_absolute() else (base / path).resolve()


def _table(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing [{name}] table")
    return value


def _agent_config(base: Path, data: dict[str, Any], name: str) -> AgentConfig:
    table = _table(data, name)
    model = table.get("model")
    command = table.get("command")
    prompt_file = table.get("prompt_file")

    has_model = isinstance(model, str) and bool(model)
    has_command = (
        isinstance(command, list)
        and bool(command)
        and all(isinstance(x, str) for x in command)
    )
    if has_model and has_command:
        raise ValueError(f"[{name}] must set either model or command, not both")
    if not has_model and not has_command:
        raise ValueError(f"[{name}] must set either model (string) or command (string array)")
    if not isinstance(prompt_file, str):
        raise ValueError(f"[{name}].prompt_file is required")

    return AgentConfig(
        prompt_file=_resolve(base, prompt_file),
        timeout_seconds=int(table.get("timeout_seconds", 600)),
        model=model if has_model else None,
        command=tuple(command) if has_command else None,
        max_tokens=int(table.get("max_tokens", 8096)),
        temperature=(float(table["temperature"]) if "temperature" in table else None),
        reasoning_effort=table.get("reasoning_effort"),
    )


def load_config(path: str | Path, workspace_override: str | Path | None = None) -> Config:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    base = config_path.parent
    run = _table(data, "run")
    app = _table(data, "application")
    workspace = workspace_override or app.get("workspace")
    editable = app.get("editable_files")
    evaluation_command = app.get("evaluation_command")
    environment = app.get("environment", {})

    if not isinstance(workspace, (str, Path)):
        raise ValueError("[application].workspace is required")
    if not isinstance(editable, list) or not editable or not all(isinstance(x, str) for x in editable):
        raise ValueError("[application].editable_files must be a non-empty string array")
    if (
        not isinstance(evaluation_command, list)
        or not evaluation_command
        or not all(isinstance(x, str) for x in evaluation_command)
    ):
        raise ValueError("[application].evaluation_command must be a non-empty string array")
    if not isinstance(environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in environment.items()
    ):
        raise ValueError("[application].environment must contain only strings")

    direction = str(app.get("direction", "minimize"))
    if direction not in {"minimize", "maximize"}:
        raise ValueError("[application].direction must be 'minimize' or 'maximize'")

    app_config = AppConfig(
        kind=str(app.get("kind", "command")),
        workspace=_resolve(base, workspace),
        editable_files=tuple(editable),
        evaluation_command=tuple(evaluation_command),
        metric_pattern=str(app["metric_pattern"]),
        metric_group=int(app.get("metric_group", 1)),
        direction=direction,
        evaluation_timeout_seconds=int(app.get("evaluation_timeout_seconds", 1200)),
        environment=dict(environment),
    )
    return Config(
        name=str(run.get("name", app_config.kind)),
        iterations=int(run.get("iterations", 10)),
        runs_dir=_resolve(base, str(run.get("runs_dir", "runs"))),
        improvement_tolerance=float(run.get("improvement_tolerance", 0.0)),
        advisor=_agent_config(base, data, "advisor"),
        worker=_agent_config(base, data, "worker"),
        app=app_config,
        path=config_path,
        use_advisor=bool(run.get("use_advisor", True)),
    )


def validate_config(config: Config) -> None:
    if not config.app.workspace.is_dir():
        raise ValueError(f"workspace does not exist: {config.app.workspace}")
    for relative in config.app.editable_files:
        path = config.app.workspace / relative
        if not path.is_file():
            raise ValueError(f"editable file does not exist: {path}")
    for prompt in (config.advisor.prompt_file, config.worker.prompt_file):
        if not prompt.is_file():
            raise ValueError(f"prompt file does not exist: {prompt}")


# ---------------------------------------------------------------------------
# Evaluator subprocess
# ---------------------------------------------------------------------------


def evaluate(config: Config) -> Evaluation:
    env = os.environ.copy()
    env.update(config.app.environment)
    try:
        completed = subprocess.run(
            list(config.app.evaluation_command),
            cwd=config.app.workspace,
            env=env,
            text=True,
            capture_output=True,
            timeout=config.app.evaluation_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return Evaluation(False, None, stdout, f"{stderr}\nevaluation timed out".strip(), 124)

    combined = f"{completed.stdout}\n{completed.stderr}"
    match = re.search(config.app.metric_pattern, combined, re.MULTILINE)
    metric = None
    parse_error = ""
    if match:
        try:
            metric = float(match.group(config.app.metric_group))
        except (IndexError, ValueError) as exc:
            parse_error = f"could not parse metric: {exc}"
    elif completed.returncode == 0:
        parse_error = "evaluation succeeded but metric pattern did not match"

    stderr = completed.stderr
    if parse_error:
        stderr = f"{stderr}\n{parse_error}".strip()
    return Evaluation(
        success=completed.returncode == 0 and metric is not None,
        metric=metric,
        stdout=completed.stdout,
        stderr=stderr,
        returncode=completed.returncode,
    )


# ---------------------------------------------------------------------------
# Editable-file snapshot, diff, and restoration
# ---------------------------------------------------------------------------


Snapshot = dict[str, bytes]


def snapshot(config: Config) -> Snapshot:
    return {
        relative: (config.app.workspace / relative).read_bytes()
        for relative in config.app.editable_files
    }


def changed_files(config: Config, before: Snapshot) -> list[str]:
    return [
        relative
        for relative, content in before.items()
        if not (config.app.workspace / relative).exists()
        or (config.app.workspace / relative).read_bytes() != content
    ]


def restore(config: Config, before: Snapshot) -> None:
    for relative, content in before.items():
        path = config.app.workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def make_patch(config: Config, before: Snapshot) -> str:
    chunks: list[str] = []
    for relative in changed_files(config, before):
        path = config.app.workspace / relative
        old = before[relative].decode("utf-8", errors="replace").splitlines(keepends=True)
        new = (
            path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            if path.exists()
            else []
        )
        chunks.extend(
            difflib.unified_diff(old, new, fromfile=f"a/{relative}", tofile=f"b/{relative}")
        )
    return "".join(chunks)


# ---------------------------------------------------------------------------
# Tool schemas and dispatch
# ---------------------------------------------------------------------------


ADVISOR_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_experiment",
        "description": (
            "Read one prior experiment by number, including its proposal, report, "
            "error, evaluation, and candidate patch. Use only when the bounded "
            "advisor state does not contain enough detail."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"number": {"type": "integer", "minimum": 1}},
            "required": ["number"],
        },
    },
    {
        "name": "get_best_candidate",
        "description": (
            "Read the current best editable files. Output is bounded; request this "
            "only when source details are necessary for the proposal."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_candidate_diff",
        "description": "Read the bounded candidate patch for one prior experiment.",
        "input_schema": {
            "type": "object",
            "properties": {"number": {"type": "integer", "minimum": 1}},
            "required": ["number"],
        },
    },
]

WORKER_TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read any file in the workspace by path (relative to workspace root or absolute).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (relative to workspace root or absolute).",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write content to an editable file, replacing it entirely. "
            "Only files listed in editable_files may be written."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path of the editable file to write (must be in editable_files).",
                },
                "content": {
                    "type": "string",
                    "description": "Complete new content for the file.",
                },
            },
            "required": ["path", "content"],
        },
    },
]


def _bounded_text(text: str, max_chars: int = ADVISOR_TOOL_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[truncated at {max_chars} characters]"


def _experiment_by_number(
    experiments: list[Experiment], number: int
) -> Experiment | None:
    return next((item for item in experiments if item.number == number), None)


def _dispatch_advisor_tool(
    name: str,
    args: dict[str, Any],
    run_dir: Path,
    experiments: list[Experiment],
) -> str:
    if name in {"get_experiment", "get_candidate_diff"}:
        try:
            number = int(args.get("number"))
        except (TypeError, ValueError):
            return "Error: number must be an integer."
        item = _experiment_by_number(experiments, number)
        if item is None:
            return f"Error: experiment #{number} does not exist."
        if name == "get_candidate_diff":
            if not item.patch_file:
                return f"Experiment #{number} has no candidate patch."
            path = run_dir / item.patch_file
            return _bounded_text(path.read_text(errors="replace"))
        evaluation = (run_dir / item.evaluation_file).read_text(errors="replace")
        detail = {
            "number": item.number,
            "status": item.status,
            "metric": item.metric,
            "best_metric": item.best_metric,
            "proposal": item.plan,
            "worker_report": item.report,
            "changed_files": item.changed_files,
            "error_kind": classify_experiment_error(item),
            "error": item.error,
            "evaluation": evaluation,
        }
        return _bounded_text(json.dumps(detail, indent=2, sort_keys=True))
    if name == "get_best_candidate":
        sections = []
        for path in sorted((run_dir / "best").rglob("*")):
            if path.is_file():
                relative = path.relative_to(run_dir / "best")
                sections.append(f"## {relative}\n\n{path.read_text(errors='replace')}")
        return _bounded_text("\n\n".join(sections) or "No best candidate files found.")
    return f"Unknown tool: {name}"


def _dispatch_worker_tool(name: str, args: dict[str, Any], config: Config) -> str:
    if name == "read_file":
        path_str = args.get("path", "")
        path = Path(path_str) if Path(path_str).is_absolute() else config.app.workspace / path_str
        try:
            return path.read_text(errors="replace")
        except Exception as exc:
            return f"Error reading {path}: {exc}"
    elif name == "write_file":
        relative = args.get("path", "")
        if relative not in config.app.editable_files:
            return (
                f"Error: '{relative}' is not in the list of editable files. "
                f"Editable files: {list(config.app.editable_files)}"
            )
        content = args.get("content", "")
        try:
            target = config.app.workspace / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            return f"{relative} written successfully."
        except Exception as exc:
            return f"Error writing {relative}: {exc}"
    return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# Subprocess coding-agent runner
# ---------------------------------------------------------------------------


def run_agent(agent: AgentConfig, prompt: str, config: Config, role: str = "agent") -> AgentResult:
    """Run a CLI coding agent as a subprocess.

    If any element of command equals '{prompt}', the prompt is substituted
    there as a CLI argument (e.g. codex). Otherwise the prompt is written to
    stdin (e.g. claude --print).
    """
    assert agent.command is not None
    env = os.environ.copy()
    env.update(config.app.environment)
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    prompt_in_args = "{prompt}" in agent.command
    cmd = [prompt if part == "{prompt}" else part for part in agent.command]

    def pump(stream: Any, sink: Any, chunks: list[str]) -> None:
        try:
            for line in iter(stream.readline, ""):
                chunks.append(line)
                sink.write(line)
                sink.flush()
        finally:
            stream.close()

    print(f"[{role}] starting", flush=True)
    try:
        process = subprocess.Popen(
            cmd,
            cwd=config.app.workspace,
            env=env,
            stdin=None if prompt_in_args else subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        if not prompt_in_args:
            assert process.stdin is not None
            process.stdin.write(prompt)
            process.stdin.close()

        stdout_thread = threading.Thread(target=pump, args=(process.stdout, sys.stdout, stdout_chunks), daemon=True)
        stderr_thread = threading.Thread(target=pump, args=(process.stderr, sys.stderr, stderr_chunks), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        timed_out = False
        try:
            returncode = process.wait(timeout=agent.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait()
            returncode = 124

        stdout_thread.join()
        stderr_thread.join()
        stdout = "".join(stdout_chunks)
        stderr = "".join(stderr_chunks)
        if timed_out:
            stderr = f"{stderr}\nagent timed out after {agent.timeout_seconds}s".strip()
        if stdout and not stdout.endswith("\n"):
            print()
        print(f"[{role}] completed (exit {returncode})", flush=True)
        return AgentResult(text=stdout.strip(), stderr=stderr, returncode=returncode)
    except OSError as exc:
        print(f"[{role}] failed to start: {exc}", file=sys.stderr, flush=True)
        return AgentResult(text="", stderr=str(exc), returncode=127)


# ---------------------------------------------------------------------------
# Provider detection and OpenAI helpers
# ---------------------------------------------------------------------------


def _detect_provider(model: str) -> str:
    if model.startswith("claude-"):
        return "anthropic"
    if re.match(r"^(gpt-|o1-|o3-|o4-|chatgpt-)", model):
        return "openai"
    raise NotImplementedError(
        f"Cannot detect provider for model {model!r}. "
        "Supported prefixes: claude- (Anthropic), gpt-/o1-/o3-/o4- (OpenAI)."
    )


def _make_client(model: str) -> Any:
    provider = _detect_provider(model)
    if provider == "anthropic":
        return anthropic.Anthropic()
    try:
        import openai  # noqa: PLC0415
    except ImportError:
        raise RuntimeError(
            "openai package is required for OpenAI models: pip install openai"
        ) from None
    return openai.OpenAI()


def _to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def _history_to_openai(
    history: list[dict[str, Any]], system: str
) -> list[dict[str, Any]]:
    """Convert Anthropic-format history to OpenAI messages, including the system prompt."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        # list content
        if role == "user" and all(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        ):
            # Tool results become separate "tool" role messages in OpenAI format
            for block in content:
                messages.append({
                    "role": "tool",
                    "tool_call_id": block["tool_use_id"],
                    "content": str(block["content"]),
                })
        elif role == "assistant":
            text = " ".join(
                b["text"] for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
            tool_calls = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {
                        "name": b["name"],
                        "arguments": json.dumps(b["input"]),
                    },
                }
                for b in content
                if isinstance(b, dict) and b.get("type") == "tool_use"
            ]
            # Some OpenAI-compatible endpoints reject null content even for a
            # tool-only assistant turn. An empty string is valid for both cases.
            oai_msg: dict[str, Any] = {"role": "assistant", "content": text}
            if tool_calls:
                oai_msg["tool_calls"] = tool_calls
            messages.append(oai_msg)
        else:
            text = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
            messages.append({"role": role, "content": text})
    return messages


def _to_responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Responses-API tool format (flat, not nested under 'function')."""
    return [
        {
            "type": "function",
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t["input_schema"],
        }
        for t in tools
    ]


def _history_to_responses(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic-format history to Responses-API input items.

    The system prompt is passed separately via `instructions`, not here.
    """
    items: list[dict[str, Any]] = []
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            items.append({"role": role, "content": content})
            continue
        if role == "user" and all(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        ):
            for block in content:
                items.append({
                    "type": "function_call_output",
                    "call_id": block["tool_use_id"],
                    "output": str(block["content"]),
                })
        elif role == "assistant":
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and block.get("text", "").strip():
                    items.append({"role": "assistant", "content": block["text"]})
                elif block.get("type") == "tool_use":
                    items.append({
                        "type": "function_call",
                        "call_id": block["id"],
                        "name": block["name"],
                        "arguments": json.dumps(block["input"]),
                    })
        else:
            text = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
            items.append({"role": role, "content": text})
    return items


# ---------------------------------------------------------------------------
# Agent turn runner (SDK-based)
# ---------------------------------------------------------------------------


def run_agent_turn(
    client: Any,
    history: list[dict[str, Any]],
    system: str,
    tools: list[dict[str, Any]],
    agent_config: AgentConfig,
    label: str,
    dispatch: Any,
) -> tuple[str, int, int, int]:
    """Run one agent turn to completion with a tool-calling loop.

    Appends all messages to history in place.
    Returns (final_text, n_llm_calls, input_tokens, output_tokens).
    """
    n_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    provider = _detect_provider(agent_config.model)

    while True:
        content_blocks: list[dict[str, Any]] = []
        stop_reason: str

        if provider == "anthropic":
            response = client.messages.create(
                model=agent_config.model,
                max_tokens=agent_config.max_tokens,
                system=system,
                tools=tools,
                messages=history,
                timeout=agent_config.timeout_seconds,
            )
            for block in response.content:
                if block.type == "text":
                    content_blocks.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    content_blocks.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
            stop_reason = response.stop_reason
            in_tok = response.usage.input_tokens
            out_tok = response.usage.output_tokens

        else:  # openai — Responses API (required for reasoning_effort + tools)
            resp_params: dict[str, Any] = dict(
                model=agent_config.model,
                instructions=system,
                input=_history_to_responses(history),
                tools=_to_responses_tools(tools),
                max_output_tokens=agent_config.max_tokens,
                timeout=agent_config.timeout_seconds,
            )
            if agent_config.temperature is not None:
                resp_params["temperature"] = agent_config.temperature
            if agent_config.reasoning_effort is not None:
                resp_params["reasoning"] = {"effort": agent_config.reasoning_effort}
            response = client.responses.create(**resp_params)
            had_tool_call = False
            for item in response.output:
                itype = getattr(item, "type", None)
                if itype == "message":
                    for c in item.content:
                        if getattr(c, "type", None) == "output_text":
                            content_blocks.append({"type": "text", "text": c.text})
                elif itype == "function_call":
                    had_tool_call = True
                    content_blocks.append({
                        "type": "tool_use",
                        "id": item.call_id,
                        "name": item.name,
                        "input": json.loads(item.arguments) if item.arguments else {},
                    })
            if had_tool_call:
                stop_reason = "tool_use"
            elif getattr(response, "status", "") == "incomplete":
                stop_reason = "max_tokens"
            else:
                stop_reason = "end_turn"
            in_tok = response.usage.input_tokens if response.usage else 0
            out_tok = response.usage.output_tokens if response.usage else 0

        n_calls += 1
        total_input_tokens += in_tok
        total_output_tokens += out_tok

        for block in content_blocks:
            if block["type"] == "text" and block["text"].strip():
                print(f"  [{label}] {block['text'][:1000]}", flush=True)
            elif block["type"] == "tool_use":
                print(f"  [{label}] {block['name']}({str(block['input'])[:120]})", flush=True)

        history.append({"role": "assistant", "content": content_blocks})

        if stop_reason == "end_turn":
            text = " ".join(
                b["text"] for b in content_blocks if b["type"] == "text"
            ).strip()
            return text, n_calls, total_input_tokens, total_output_tokens

        if stop_reason == "max_tokens":
            text = " ".join(
                b["text"] for b in content_blocks if b["type"] == "text"
            ).strip()
            if text:
                print(f"  [{label}] stopped after reaching max_tokens", flush=True)
                return text, n_calls, total_input_tokens, total_output_tokens
            break

        if stop_reason == "tool_use":
            tool_results = []
            for block in content_blocks:
                if block["type"] == "tool_use":
                    result = dispatch(block["name"], block["input"])
                    print(f"  [{label}] → {str(result)[:1000]}", flush=True)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": str(result),
                    })
            history.append({"role": "user", "content": tool_results})
        else:
            break

    return "", n_calls, total_input_tokens, total_output_tokens


def _is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, (anthropic.APITimeoutError, anthropic.APIConnectionError)):
        return True
    # Cover OpenAI SDK errors without a hard import dependency
    if type(exc).__name__ in ("APITimeoutError", "APIConnectionError"):
        return True
    msg = str(exc).lower()
    return any(kw in msg for kw in ("timeout", "timed out", "connection reset"))


def run_agent_turn_retrying(
    client: Any,
    history: list[dict[str, Any]],
    system: str,
    tools: list[dict[str, Any]],
    agent_config: AgentConfig,
    label: str,
    dispatch: Any,
    max_attempts: int = 3,
    base_delay: float = 15.0,
) -> tuple[str, int, int, int]:
    """Like run_agent_turn but retries transient API errors, restoring history on failure."""
    last_exc: Exception | None = None
    total_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for attempt in range(max_attempts):
        if attempt > 0:
            delay = base_delay * (2 ** (attempt - 1))
            print(
                f"  [{label}] Retrying (attempt {attempt + 1}/{max_attempts}) in {delay:.0f}s...",
                flush=True,
            )
            time.sleep(delay)

        history_snapshot = copy.deepcopy(history)
        try:
            text, n, in_tok, out_tok = run_agent_turn(client, history, system, tools, agent_config, label, dispatch)
            return text, total_calls + n, total_input_tokens + in_tok, total_output_tokens + out_tok
        except Exception as exc:
            total_calls += 1
            history.clear()
            history.extend(history_snapshot)
            if _is_transient_error(exc) and attempt < max_attempts - 1:
                print(
                    f"  [{label}] Transient error on attempt {attempt + 1}: "
                    f"{type(exc).__name__}: {str(exc)[:150]}",
                    flush=True,
                )
                last_exc = exc
            else:
                raise

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Advisor and worker runners
# ---------------------------------------------------------------------------


def run_advisor(
    client: Any,
    config: Config,
    run_dir: Path,
    iteration: int,
    baseline_metric: float,
    best: float,
    experiments: list[Experiment],
) -> AgentResult:
    """Run one stateless advisor turn from a bounded, harness-owned state."""
    system = config.advisor.prompt_file.read_text().rstrip()

    if config.advisor.is_subprocess:
        prompt = system + "\n\n" + advisor_prompt(
            config, run_dir, iteration, baseline_metric, best, experiments, subprocess_mode=True
        )
        return run_agent(config.advisor, prompt, config, role="advisor")

    advisor_history: list[dict[str, Any]] = [{
        "role": "user",
        "content": advisor_prompt(config, run_dir, iteration, baseline_metric, best, experiments),
    }]
    print("[advisor] starting", flush=True)
    try:
        text, n_calls, in_tok, out_tok = run_agent_turn_retrying(
            client,
            advisor_history,
            system,
            ADVISOR_TOOLS,
            config.advisor,
            label="advisor",
            dispatch=lambda name, args: _dispatch_advisor_tool(
                name, args, run_dir, experiments
            ),
        )
        print(f"[advisor] completed ({n_calls} LLM call(s), {in_tok}+{out_tok} tokens)", flush=True)
        return AgentResult(text=text, stderr="", returncode=0 if text else 1,
                           input_tokens=in_tok, output_tokens=out_tok)
    except Exception as exc:
        print(f"[advisor] failed: {exc}", file=sys.stderr, flush=True)
        return AgentResult(text="", stderr=str(exc), returncode=1)


def run_worker(
    client: Any,
    config: Config,
    worker_history: list[dict[str, Any]],
    iteration: int,
    best: float,
    proposal: str,
    experiments: list[Experiment],
) -> AgentResult:
    """Run the worker. Subprocess agents get the prompt via stdin; SDK agents use persistent history."""
    system = config.worker.prompt_file.read_text().rstrip()

    if config.worker.is_subprocess:
        prompt = system + "\n\n" + worker_prompt(config, iteration, best, proposal, experiments)
        return run_agent(config.worker, prompt, config, role="worker")

    history_len_before = len(worker_history)
    worker_history.append({
        "role": "user",
        "content": worker_prompt(config, iteration, best, proposal, experiments),
    })
    print("[worker] starting", flush=True)
    try:
        text, n_calls, in_tok, out_tok = run_agent_turn_retrying(
            client,
            worker_history,
            system,
            WORKER_TOOLS,
            config.worker,
            label="worker",
            dispatch=lambda name, args: _dispatch_worker_tool(name, args, config),
        )
        print(f"[worker] completed ({n_calls} LLM call(s), {in_tok}+{out_tok} tokens)", flush=True)
        return AgentResult(text=text, stderr="", returncode=0 if text else 1,
                           input_tokens=in_tok, output_tokens=out_tok)
    except Exception as exc:
        del worker_history[history_len_before:]
        print(f"[worker] failed: {exc}", file=sys.stderr, flush=True)
        return AgentResult(text="", stderr=str(exc), returncode=1)


# ---------------------------------------------------------------------------
# Prompts and run artifacts
# ---------------------------------------------------------------------------


def _history_path(run_dir: Path) -> Path:
    return run_dir / "experiment_history.md"


def _candidate_sections(run_dir: Path, candidate_files: list[str]) -> str:
    sections = []
    for relative in candidate_files:
        path = run_dir / relative
        if path.exists():
            sections.append(
                f"### {path.name}\n\n```\n{path.read_text(errors='replace')}\n```"
            )
    return "\n\n".join(sections) or "(no candidate files)"


def append_baseline_history(
    run_dir: Path,
    metric: float,
    evaluation_file: str,
    candidate_files: list[str],
) -> None:
    evaluation = (run_dir / evaluation_file).read_text(errors="replace")
    _history_path(run_dir).write_text(
        f"""# Experiment History

Tracks every attempted candidate, proposal, implementation report, and result.

## Baseline

Status: keep
Metric: {metric}

### Candidate files

{_candidate_sections(run_dir, candidate_files)}

### Evaluation

```json
{evaluation}
```

"""
    )


def append_experiment_history(run_dir: Path, experiment: Experiment) -> None:
    evaluation = (run_dir / experiment.evaluation_file).read_text(errors="replace")
    patch = (
        (run_dir / experiment.patch_file).read_text(errors="replace")
        if experiment.patch_file
        else "(no patch)"
    )
    entry = f"""---

## Experiment #{experiment.number}

Status: {experiment.status}
Metric: {experiment.metric}
Best metric after experiment: {experiment.best_metric}

### Advisor proposal

{experiment.plan or "(no proposal)"}

### Worker report

{experiment.report or "(no report)"}

### Candidate patch

```diff
{patch}
```

### Candidate files

{_candidate_sections(run_dir, experiment.candidate_files)}

### Evaluation

```json
{evaluation}
```

"""
    with _history_path(run_dir).open("a", encoding="utf-8") as handle:
        handle.write(entry)


def read_experiment_history(run_dir: Path, max_chars: int = 50_000) -> str:
    """Legacy artifact reader; advisor prompts no longer call this automatically."""
    path = _history_path(run_dir)
    if not path.exists():
        return "No experiment history yet."
    text = path.read_text(errors="replace")
    if len(text) <= max_chars:
        return text
    return "# Experiment History (recent tail)\n\n" + text[-max_chars:]


def results_summary(experiments: list[Experiment], best: float, limit: int = 10) -> str:
    if not experiments:
        return f"Best metric: {best}\nNo optimization experiments completed yet."
    counts: dict[str, int] = {}
    for item in experiments:
        counts[item.status] = counts.get(item.status, 0) + 1
    count_text = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    recent = ["Recent experiments:"]
    for item in experiments[-limit:]:
        report = " ".join(item.report.split())[:160] or "(no report)"
        recent.append(
            f"- #{item.number}: {item.status}, metric={item.metric}, "
            f"best={item.best_metric}, change={report}"
        )
    return f"Best metric: {best}\nOutcomes: {count_text}\n" + "\n".join(recent)


def classify_experiment_error(experiment: Experiment) -> str | None:
    """Classify failures without asking the advisor to infer them from prose."""
    if experiment.status != "error":
        return None
    error = experiment.error.lower()
    infrastructure_markers = (
        "invalid_request_error",
        "rate limit",
        "status code: 429",
        "error code: 429",
        "api timeout",
        "connection reset",
        "connection error",
        "timed out",
        "timeout",
    )
    agent_failure = error.startswith(("worker failed:", "advisor failed:"))
    if agent_failure and any(marker in error for marker in infrastructure_markers):
        return "infrastructure"
    if agent_failure:
        return "agent"
    return "evaluation"


def _one_line(text: str, limit: int) -> str:
    return " ".join(text.split())[:limit]


def build_advisor_state(
    config: Config,
    iteration: int,
    baseline_metric: float,
    best: float,
    experiments: list[Experiment],
) -> dict[str, Any]:
    """Reduce the raw ledger to bounded, deterministic advisor state."""
    outcome_counts: dict[str, int] = {}
    error_counts: dict[str, int] = {}
    for item in experiments:
        outcome_counts[item.status] = outcome_counts.get(item.status, 0) + 1
        error_kind = classify_experiment_error(item)
        if error_kind:
            error_counts[error_kind] = error_counts.get(error_kind, 0) + 1

    best_experiment = next(
        (item.number for item in reversed(experiments)
         if item.status == "keep" and item.metric == best),
        0,
    )
    recent = []
    for item in experiments[-ADVISOR_RECENT_EXPERIMENTS:]:
        recent.append({
            "number": item.number,
            "status": item.status,
            "metric": item.metric,
            "best_metric": item.best_metric,
            "proposal": _one_line(extract_proposal(item.plan), 400),
            "worker_report": _one_line(item.report, 300),
            "error_kind": classify_experiment_error(item),
            "error": _one_line(item.error, 1200),
        })
    kept = [
        {
            "number": item.number,
            "metric": item.metric,
            "proposal": _one_line(extract_proposal(item.plan), 400),
        }
        for item in experiments
        if item.status == "keep"
    ][-10:]

    return {
        "version": ADVISOR_STATE_VERSION,
        "run": {
            "application": config.app.kind,
            "direction": config.app.direction,
            "iteration": iteration,
            "iterations": config.iterations,
            "remaining_after_this": config.iterations - iteration,
        },
        "metrics": {
            "baseline": baseline_metric,
            "best": best,
            "best_experiment": best_experiment,
        },
        "outcomes": outcome_counts,
        "errors": error_counts,
        "kept_experiments": kept,
        "recent_experiments": recent,
        "available_tools": [tool["name"] for tool in ADVISOR_TOOLS],
    }


def should_stop_for_infrastructure_errors(
    experiments: list[Experiment], limit: int = INFRASTRUCTURE_ERROR_LIMIT
) -> bool:
    if len(experiments) < limit:
        return False
    recent = experiments[-limit:]
    if not all(classify_experiment_error(item) == "infrastructure" for item in recent):
        return False
    fingerprints = {
        re.sub(r"\d+", "#", " ".join(item.error.lower().split()))
        for item in recent
    }
    return len(fingerprints) == 1


def advisor_prompt(
    config: Config,
    run_dir: Path,
    iteration: int,
    baseline_metric: float,
    best: float,
    experiments: list[Experiment],
    *,
    subprocess_mode: bool = False,
) -> str:
    state = build_advisor_state(config, iteration, baseline_metric, best, experiments)
    history_instruction = "Use the bounded state below as the source of truth. "
    if subprocess_mode:
        history_instruction += (
            "Do not load the raw history artifact; it exists only for debugging. "
            "Base the proposal on this bounded state and cite supporting experiment numbers."
        )
    else:
        history_instruction += (
            "Do not request the full experiment history. Use a targeted tool only when a "
            "specific experiment, diff, or best-candidate detail is necessary. Cite "
            "supporting experiment numbers."
        )
    history_instruction += (
        " Treat infrastructure errors as harness failures, not experimental evidence."
    )
    return f"""## Current Run

Application: {config.app.kind}
Iteration: {iteration}/{config.iterations}
Baseline metric: {baseline_metric}

## Bounded Advisor State

```json
{json.dumps(state, indent=2, sort_keys=True)}
```

{history_instruction}
Propose one strategic, targeted experiment. Do not edit files or run the evaluator.
"""


def worker_prompt(
    config: Config,
    iteration: int,
    best: float,
    plan: str,
    experiments: list[Experiment] | None = None,
) -> str:
    experiments = experiments or []
    files = "\n".join(f"- {name}" for name in config.app.editable_files)
    command = " ".join(config.app.evaluation_command)
    return f"""## Assignment

Application: {config.app.kind}
Iteration: {iteration}/{config.iterations}
Current global best metric: {best}

## Advisor Proposal

{plan}

## Compact Search Summary

{results_summary(experiments, best, limit=5)}

## Task

Read the current state of the editable files, implement the advisor's proposal,
and write back the updated file(s). Output your implementation report and stop.

AES, not you, runs `{command}` and records the measured result after you finish.

## Editable Files

{files}
"""


def extract_proposal(advisor_text: str) -> str:
    """Pass only a structured PROPOSAL section when the advisor provides one."""
    match = re.search(
        r"(?ms)^##\s+PROPOSAL\s*\n(.*?)(?=^##\s+|\Z)",
        advisor_text,
    )
    proposal = match.group(1).strip() if match else advisor_text.strip()
    return proposal or advisor_text.strip()


def append_event(run_dir: Path, event_type: str, payload: dict[str, Any]) -> None:
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
        "payload": payload,
    }
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def write_evaluation(run_dir: Path, number: int, result: Evaluation) -> str:
    directory = run_dir / "experiments" / f"{number:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "evaluation.json"
    path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")
    return str(path.relative_to(run_dir))


def write_agent_output(run_dir: Path, number: int, role: str, result: AgentResult) -> None:
    directory = run_dir / "experiments" / f"{number:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{role}.txt").write_text(result.text)
    if result.stderr:
        (directory / f"{role}.stderr.txt").write_text(result.stderr)


def write_advisor_state(run_dir: Path, number: int, state: dict[str, Any]) -> None:
    directory = run_dir / "experiments" / f"{number:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "advisor_state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n"
    )


def write_patch(run_dir: Path, number: int, patch: str) -> str | None:
    if not patch:
        return None
    path = run_dir / "experiments" / f"{number:04d}" / "candidate.patch"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(patch)
    return str(path.relative_to(run_dir))


def write_candidate_files(run_dir: Path, number: int, config: Config) -> list[str]:
    saved = []
    root = run_dir / "experiments" / f"{number:04d}" / "candidate"
    for relative in config.app.editable_files:
        source = config.app.workspace / relative
        if not source.exists():
            continue
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        saved.append(str(destination.relative_to(run_dir)))
    return saved


def write_best_files(run_dir: Path, config: Config) -> None:
    root = run_dir / "best"
    for relative in config.app.editable_files:
        source = config.app.workspace / relative
        if not source.exists():
            continue
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def write_summary(run_dir: Path, config: Config, baseline: Evaluation, experiments: list[Experiment]) -> None:
    total_in = sum(e.advisor_input_tokens + e.worker_input_tokens for e in experiments)
    total_out = sum(e.advisor_output_tokens + e.worker_output_tokens for e in experiments)
    token_line = f"- Total tokens: `{total_in:,}` input, `{total_out:,}` output" if total_in or total_out else ""
    lines = [
        f"# AES Run: {config.name}",
        "",
        f"- Baseline metric: `{baseline.metric}`",
        f"- Experiments: `{len(experiments)}`",
    ]
    if token_line:
        lines.append(token_line)
    lines += [
        "",
        "| # | Status | Metric | Best | Tokens (in+out) | Changed files | Report |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for item in experiments:
        changed = ", ".join(item.changed_files) or "-"
        report = item.report.replace("|", "\\|").replace("\n", " ")[:120]
        tok = item.advisor_input_tokens + item.worker_input_tokens
        tok_out = item.advisor_output_tokens + item.worker_output_tokens
        tok_str = f"{tok:,}+{tok_out:,}" if tok or tok_out else "-"
        lines.append(
            f"| {item.number} | {item.status} | "
            f"{item.metric if item.metric is not None else '-'} | {item.best_metric} | "
            f"{tok_str} | {changed} | {report} |"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Search loop
# ---------------------------------------------------------------------------


def is_better(config: Config, candidate: float, best: float) -> bool:
    tolerance = config.improvement_tolerance
    if config.app.direction == "minimize":
        return candidate < best - tolerance
    return candidate > best + tolerance


def create_run_dir(config: Config) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    name = "".join(char if char.isalnum() or char in "-_" else "-" for char in config.name)
    run_dir = config.runs_dir / f"{timestamp}_{name}"
    (run_dir / "experiments").mkdir(parents=True)
    shutil.copy2(config.path, run_dir / "config.toml")
    return run_dir


def evolve(config: Config) -> Path:
    validate_config(config)
    run_dir = create_run_dir(config)
    print(f"run directory: {run_dir}", flush=True)
    append_event(
        run_dir,
        "run_started",
        {
            "name": config.name,
            "application": config.app.kind,
            "workspace": str(config.app.workspace),
            "iterations": config.iterations,
        },
    )

    print("[baseline] evaluating starting point", flush=True)
    baseline = evaluate(config)
    baseline_evaluation_file = write_evaluation(run_dir, 0, baseline)
    baseline_candidate_files = write_candidate_files(run_dir, 0, config)
    append_event(run_dir, "baseline_evaluated", asdict(baseline))
    if not baseline.success or baseline.metric is None:
        write_summary(run_dir, config, baseline, [])
        raise RuntimeError(f"baseline evaluation failed: {baseline.stderr}")
    print(f"[baseline] metric={baseline.metric}", flush=True)
    append_baseline_history(
        run_dir,
        baseline.metric,
        baseline_evaluation_file,
        baseline_candidate_files,
    )

    best = baseline.metric
    write_best_files(run_dir, config)
    best_snapshot = snapshot(config)
    experiments: list[Experiment] = []

    advisor_client = _make_client(config.advisor.model) if not config.advisor.is_subprocess else None
    worker_client = _make_client(config.worker.model) if not config.worker.is_subprocess else None
    worker_history: list[dict[str, Any]] = []

    for number in range(1, config.iterations + 1):
        print(f"\n=== iteration {number}/{config.iterations} (best={best}) ===", flush=True)
        if config.use_advisor:
            advisor_state = build_advisor_state(
                config, number, baseline.metric, best, experiments
            )
            write_advisor_state(run_dir, number, advisor_state)
            advisor = run_advisor(
                advisor_client,
                config,
                run_dir,
                number,
                baseline.metric,
                best,
                experiments,
            )
            write_agent_output(run_dir, number, "advisor", advisor)
            append_event(
                run_dir,
                "advisor_completed",
                {
                    "experiment": number,
                    "success": advisor.success,
                    "plan": advisor.text,
                    "state_version": ADVISOR_STATE_VERSION,
                    "state_chars": len(json.dumps(advisor_state)),
                },
            )
            advisor_failed = not advisor.success or not advisor.text
            advisor_text = advisor.text
            advisor_stderr = advisor.stderr
            advisor_in_tok = advisor.input_tokens
            advisor_out_tok = advisor.output_tokens
            proposal = "" if advisor_failed else extract_proposal(advisor.text)
        else:
            # No-advisor ablation: skip the advisor; the worker already receives
            # the full experiment history and self-directs.
            advisor_failed = False
            advisor_text = "(no advisor)"
            advisor_stderr = ""
            advisor_in_tok = 0
            advisor_out_tok = 0
            proposal = (
                "(no advisor mode) There is no advisor. Independently analyze the "
                "current editable file(s) and the experiment history, then implement "
                "the single highest-value improvement you can justify."
            )
            append_event(
                run_dir, "advisor_skipped", {"experiment": number, "use_advisor": False}
            )

        if advisor_failed:
            evaluation = Evaluation(
                False,
                None,
                stderr=f"advisor failed: {advisor_stderr}",
                returncode=1,
            )
            evaluation_file = write_evaluation(run_dir, number, evaluation)
            experiment = Experiment(
                number,
                "error",
                None,
                best,
                advisor_text,
                "",
                [],
                None,
                [],
                evaluation_file,
                evaluation.stderr,
                advisor_in_tok,
                advisor_out_tok,
            )
        else:
            before = snapshot(config)
            worker = run_worker(
                worker_client,
                config,
                worker_history,
                number,
                best,
                proposal,
                experiments,
            )
            write_agent_output(run_dir, number, "worker", worker)
            changed = changed_files(config, before)
            patch_file = write_patch(run_dir, number, make_patch(config, before))
            candidate_files = write_candidate_files(run_dir, number, config)

            if not worker.success:
                restore(config, best_snapshot)
                evaluation = Evaluation(
                    False,
                    None,
                    stderr=f"worker failed: {worker.stderr}",
                    returncode=worker.returncode,
                )
                status = "error"
            else:
                target = ", ".join(changed) if changed else "unchanged candidate"
                print(f"[evaluator] testing {target}", flush=True)
                evaluation = evaluate(config)
                keep = (
                    evaluation.success
                    and evaluation.metric is not None
                    and is_better(config, evaluation.metric, best)
                )
                status = "keep" if keep else ("discard" if evaluation.success else "error")
                if keep:
                    best = evaluation.metric  # type: ignore[assignment]
                    best_snapshot = snapshot(config)
                    write_best_files(run_dir, config)
                elif not evaluation.success:
                    restore(config, best_snapshot)

            evaluation_file = write_evaluation(run_dir, number, evaluation)
            experiment = Experiment(
                number=number,
                status=status,
                metric=evaluation.metric,
                best_metric=best,
                plan=advisor_text,
                report=worker.text,
                changed_files=changed,
                patch_file=patch_file,
                candidate_files=candidate_files,
                evaluation_file=evaluation_file,
                error="" if evaluation.success else evaluation.stderr,
                advisor_input_tokens=advisor_in_tok,
                advisor_output_tokens=advisor_out_tok,
                worker_input_tokens=worker.input_tokens,
                worker_output_tokens=worker.output_tokens,
            )

        metric = experiment.metric if experiment.metric is not None else "unavailable"
        print(
            f"[iteration {number}] status={experiment.status} metric={metric} "
            f"best={experiment.best_metric}",
            flush=True,
        )
        if experiment.error:
            print(f"[iteration {number}] error: {experiment.error}", file=sys.stderr, flush=True)
        experiments.append(experiment)
        append_experiment_history(run_dir, experiment)
        append_event(run_dir, "experiment_completed", asdict(experiment))
        write_summary(run_dir, config, baseline, experiments)
        if should_stop_for_infrastructure_errors(experiments):
            message = (
                f"stopping after {INFRASTRUCTURE_ERROR_LIMIT} consecutive "
                "infrastructure failures"
            )
            print(f"[run] {message}", file=sys.stderr, flush=True)
            append_event(run_dir, "run_aborted", {"reason": message})
            break

    restore(config, best_snapshot)
    append_event(run_dir, "run_completed", {"best_metric": best, "experiments": len(experiments)})
    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aes", description="Agentic Evolutionary Search")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("run", "run an advisor-worker search"),
        ("validate", "validate a run configuration"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("config", help="path to a TOML run configuration")
        command.add_argument("--workspace", help="override the application workspace")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config, args.workspace)
        validate_config(config)
        if args.command == "validate":
            print(f"valid: {config.path}")
            print(f"application: {config.app.kind}")
            print(f"workspace: {config.app.workspace}")
            return 0

        run_dir = evolve(config)
        print(f"run complete: {run_dir}")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"aes: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
