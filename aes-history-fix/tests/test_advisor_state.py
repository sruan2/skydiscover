import tempfile
import unittest
from pathlib import Path

from aes.evolve import (
    ADVISOR_RECENT_EXPERIMENTS,
    ADVISOR_TOOL_MAX_CHARS,
    AgentConfig,
    AppConfig,
    Config,
    Experiment,
    _dispatch_advisor_tool,
    _history_to_openai,
    advisor_prompt,
    build_advisor_state,
    should_stop_for_infrastructure_errors,
)


def make_config(root: Path, iterations: int = 1000) -> Config:
    agent = AgentConfig(root / "prompt.md", 60, model="gpt-5")
    app = AppConfig(
        kind="test",
        workspace=root,
        editable_files=("submission.py",),
        evaluation_command=("true",),
        metric_pattern=r"metric: (.+)",
        metric_group=1,
        direction="maximize",
        evaluation_timeout_seconds=60,
    )
    return Config("test", iterations, root, 0.0, agent, agent, app, root / "config.toml")


def make_experiment(number: int, status: str = "discard", error: str = "") -> Experiment:
    return Experiment(
        number=number,
        status=status,
        metric=float(number) if status != "error" else None,
        best_metric=float(number),
        plan="## PROPOSAL\n" + ("proposal " * 500),
        report="report " * 500,
        changed_files=["submission.py"],
        patch_file=None,
        candidate_files=[],
        evaluation_file=f"experiments/{number:04d}/evaluation.json",
        error=error,
    )


class AdvisorStateTests(unittest.TestCase):
    def test_state_and_prompt_remain_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = make_config(root)
            experiments = [make_experiment(i) for i in range(1, 1001)]
            state = build_advisor_state(config, 1000, 0.0, 1000.0, experiments)
            self.assertEqual(len(state["recent_experiments"]), ADVISOR_RECENT_EXPERIMENTS)
            self.assertEqual(state["recent_experiments"][0]["number"], 993)
            prompt = advisor_prompt(config, root, 1000, 0.0, 1000.0, experiments)
            self.assertLess(len(prompt), 20_000)
            self.assertNotIn("proposal " * 100, prompt)

    def test_tool_results_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            best = root / "best"
            best.mkdir()
            (best / "submission.py").write_text("x" * (ADVISOR_TOOL_MAX_CHARS * 2))
            result = _dispatch_advisor_tool("get_best_candidate", {}, root, [])
            self.assertLessEqual(len(result), ADVISOR_TOOL_MAX_CHARS + 100)
            self.assertIn("truncated", result)

    def test_repeated_infrastructure_errors_trip_circuit_breaker(self) -> None:
        error = "worker failed: invalid_request_error messages.[104].content"
        experiments = [make_experiment(i, "error", error) for i in range(1, 4)]
        self.assertTrue(should_stop_for_infrastructure_errors(experiments))
        experiments[-1].error = "worker failed: API timeout"
        self.assertFalse(should_stop_for_infrastructure_errors(experiments))

    def test_evaluator_timeout_is_not_infrastructure(self) -> None:
        experiments = [
            make_experiment(i, "error", "evaluation timed out after 360 seconds")
            for i in range(1, 4)
        ]
        self.assertFalse(should_stop_for_infrastructure_errors(experiments))

    def test_openai_tool_only_message_uses_string_content(self) -> None:
        history = [{
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "1", "name": "tool", "input": {}}],
        }]
        messages = _history_to_openai(history, "system")
        self.assertEqual(messages[1]["content"], "")
        self.assertIn("tool_calls", messages[1])


if __name__ == "__main__":
    unittest.main()
