"""Tests for baseline evaluation runner."""

from pathlib import Path

from evocode.evaluation.baselines import BaselineResult, BaselineRunner


class TestBaselineResult:
    def test_creation(self) -> None:
        r = BaselineResult(
            baseline_name="fixed_k",
            task_id="t1",
            retrieved_files=["a.py"],
            top_k=3,
            precision_at_k=0.5,
            recall_at_k=0.8,
        )
        assert r.baseline_name == "fixed_k"
        assert r.precision_at_k == 0.5

    def test_defaults(self) -> None:
        r = BaselineResult(
            baseline_name="x",
            task_id="y",
            retrieved_files=[],
            top_k=1,
            precision_at_k=0.0,
            recall_at_k=0.0,
        )
        assert r.task_id == "y"


class TestBaselineRunner:
    def test_run_returns_results(self) -> None:
        runner = BaselineRunner()
        results = runner.run(
            query="test query",
            task_id="t1",
            relevant_files={"a.py", "b.py"},
            candidate_files=["a.py", "b.py", "c.py"],
            top_k=3,
        )
        assert len(results) == 3  # fixed_k, top_k_length, random

    def test_run_baseline_names(self) -> None:
        runner = BaselineRunner()
        results = runner.run(
            query="q",
            task_id="t1",
            relevant_files=set(),
            candidate_files=[],
            top_k=1,
        )
        names = {r.baseline_name for r in results}
        assert names == {"fixed_k", "top_k_length", "random"}

    def test_precision_at_k(self) -> None:
        runner = BaselineRunner()
        results = runner.run(
            query="q",
            task_id="t1",
            relevant_files={"a.py"},
            candidate_files=["a.py", "b.py"],
            top_k=2,
        )
        fixed = [r for r in results if r.baseline_name == "fixed_k"][0]
        assert fixed.precision_at_k == 0.5  # 1/2 relevant
