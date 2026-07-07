"""Minimal advisor-worker evolutionary search loop."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Configuration and result records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentConfig:
    command: tuple[str, ...]
    prompt_file: Path
    timeout_seconds: int


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
    command = table.get("command")
    prompt_file = table.get("prompt_file")
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise ValueError(f"[{name}].command must be a non-empty string array")
    if not isinstance(prompt_file, str):
        raise ValueError(f"[{name}].prompt_file is required")
    return AgentConfig(
        command=tuple(command),
        prompt_file=_resolve(base, prompt_file),
        timeout_seconds=int(table.get("timeout_seconds", 600)),
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
# Agent and evaluator subprocesses
# ---------------------------------------------------------------------------


def run_agent(agent: AgentConfig, prompt: str, config: Config, role: str = "agent") -> AgentResult:
    env = os.environ.copy()
    env.update(config.app.environment)
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

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
            list(agent.command),
            cwd=config.app.workspace,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        process.stdin.write(prompt)
        process.stdin.close()

        stdout_thread = threading.Thread(
            target=pump,
            args=(process.stdout, sys.stdout, stdout_chunks),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=pump,
            args=(process.stderr, sys.stderr, stderr_chunks),
            daemon=True,
        )
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
# Prompts and run artifacts
# ---------------------------------------------------------------------------


def experiment_history(
    run_dir: Path,
    config: Config,
    baseline_metric: float,
    experiments: list[Experiment],
) -> str:
    baseline_evaluation = (run_dir / "experiments/0000/evaluation.json").read_text()
    baseline_files = []
    for relative in config.app.editable_files:
        path = run_dir / "experiments/0000/candidate" / relative
        if path.exists():
            baseline_files.append(
                f"### {relative}\n\n```\n{path.read_text(errors='replace')}\n```"
            )
    sections = [
        f"""## Baseline

Status: accepted starting point
Metric: {baseline_metric}

### Baseline files

{chr(10).join(baseline_files) or "(no baseline files)"}

### Baseline evaluation

```json
{baseline_evaluation}
```"""
    ]
    for item in experiments:
        evaluation = (run_dir / item.evaluation_file).read_text()
        patch = (run_dir / item.patch_file).read_text() if item.patch_file else "(no patch)"
        candidate = []
        for relative in item.candidate_files:
            path = run_dir / relative
            candidate.append(f"### {path.name}\n\n```\n{path.read_text(errors='replace')}\n```")
        sections.append(
            f"""## Experiment #{item.number}

Status: {item.status}
Metric: {item.metric}
Best metric after experiment: {item.best_metric}

### Advisor plan

{item.plan}

### Worker report

{item.report or "(no report)"}

### Candidate patch

```diff
{patch}
```

### Candidate files

{chr(10).join(candidate) or "(no candidate files)"}

### Evaluation

```json
{evaluation}
```"""
        )
    return "\n\n---\n\n".join(sections)


def advisor_prompt(
    config: Config,
    run_dir: Path,
    iteration: int,
    baseline_metric: float,
    best: float,
    experiments: list[Experiment],
) -> str:
    base = config.advisor.prompt_file.read_text().rstrip()
    return f"""{base}

## Current Run

Application: {config.app.kind}
Iteration: {iteration}
Current best metric: {best}

## Experiment History

{experiment_history(run_dir, config, baseline_metric, experiments)}

Follow the output format in your role prompt. Do not edit files or run the evaluator.
"""


def worker_prompt(
    config: Config, iteration: int, best: float, plan: str, history: str = ""
) -> str:
    base = config.worker.prompt_file.read_text().rstrip()
    files = "\n".join(f"- {name}" for name in config.app.editable_files)
    command = " ".join(config.app.evaluation_command)
    # In no-advisor mode the worker is the only agent, so it receives the full
    # experiment history directly (the role the advisor would otherwise digest).
    history_section = (
        f"""## Experiment History

{history}

"""
        if history
        else ""
    )
    return f"""{base}

## Assignment

Application: {config.app.kind}
Iteration: {iteration}
Current best metric: {best}

{history_section}## Advisor Proposal

{plan}

## Hard Bounds

You may edit only:
{files}

Follow the mandatory sequence in your role prompt. AES, not you, runs
`{command}` and records the measured result after you finish.
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
    lines = [
        f"# AES Run: {config.name}",
        "",
        f"- Baseline metric: `{baseline.metric}`",
        f"- Experiments: `{len(experiments)}`",
        "",
        "| # | Status | Metric | Best | Changed files | Report |",
        "|---:|---|---:|---:|---|---|",
    ]
    for item in experiments:
        changed = ", ".join(item.changed_files) or "-"
        report = item.report.replace("|", "\\|").replace("\n", " ")[:120]
        lines.append(
            f"| {item.number} | {item.status} | "
            f"{item.metric if item.metric is not None else '-'} | {item.best_metric} | "
            f"{changed} | {report} |"
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
    write_evaluation(run_dir, 0, baseline)
    write_candidate_files(run_dir, 0, config)
    append_event(run_dir, "baseline_evaluated", asdict(baseline))
    if not baseline.success or baseline.metric is None:
        write_summary(run_dir, config, baseline, [])
        raise RuntimeError(f"baseline evaluation failed: {baseline.stderr}")
    print(f"[baseline] metric={baseline.metric}", flush=True)

    best = baseline.metric
    write_best_files(run_dir, config)
    experiments: list[Experiment] = []

    for number in range(1, config.iterations + 1):
        print(f"\n=== iteration {number}/{config.iterations} (best={best}) ===", flush=True)
        if config.use_advisor:
            advisor = run_agent(
                config.advisor,
                advisor_prompt(config, run_dir, number, baseline.metric, best, experiments),
                config,
                role="advisor",
            )
            write_agent_output(run_dir, number, "advisor", advisor)
            append_event(
                run_dir,
                "advisor_completed",
                {"experiment": number, "success": advisor.success, "plan": advisor.text},
            )
            advisor_failed = not advisor.success or not advisor.text
            advisor_text = advisor.text
            advisor_stderr = advisor.stderr
            proposal = "" if advisor_failed else extract_proposal(advisor.text)
            worker_history = ""
        else:
            # No-advisor ablation: skip the advisor agent entirely and hand the
            # worker the full experiment history so it can self-direct.
            advisor_failed = False
            advisor_text = "(no advisor)"
            advisor_stderr = ""
            proposal = (
                "(no advisor mode) There is no advisor. Independently analyze the "
                "current editable file(s) and the experiment history above, then "
                "implement the single highest-value improvement you can justify."
            )
            worker_history = experiment_history(run_dir, config, baseline.metric, experiments)
            append_event(
                run_dir, "advisor_skipped", {"experiment": number, "use_advisor": False}
            )

        if advisor_failed:
            evaluation = Evaluation(False, None, stderr=f"advisor failed: {advisor_stderr}", returncode=1)
            evaluation_file = write_evaluation(run_dir, number, evaluation)
            experiment = Experiment(
                number, "error", None, best, advisor_text, "", [], None, [],
                evaluation_file, evaluation.stderr,
            )
        else:
            before = snapshot(config)
            worker = run_agent(
                config.worker,
                worker_prompt(config, number, best, proposal, worker_history),
                config,
                role="worker",
            )
            write_agent_output(run_dir, number, "worker", worker)
            changed = changed_files(config, before)
            patch_file = write_patch(run_dir, number, make_patch(config, before))
            candidate_files = write_candidate_files(run_dir, number, config)

            if not worker.success:
                restore(config, before)
                evaluation = Evaluation(
                    False, None, stderr=f"worker failed: {worker.stderr}", returncode=worker.returncode
                )
                status = "error"
            elif not changed:
                evaluation = Evaluation(
                    False, None, stderr="worker completed without changing an editable file"
                )
                status = "error"
            else:
                print(f"[evaluator] testing {', '.join(changed)}", flush=True)
                evaluation = evaluate(config)
                keep = (
                    evaluation.success
                    and evaluation.metric is not None
                    and is_better(config, evaluation.metric, best)
                )
                status = "keep" if keep else ("reject" if evaluation.success else "error")
                if keep:
                    best = evaluation.metric  # type: ignore[assignment]
                    write_best_files(run_dir, config)
                else:
                    restore(config, before)

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
        append_event(run_dir, "experiment_completed", asdict(experiment))
        write_summary(run_dir, config, baseline, experiments)

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
