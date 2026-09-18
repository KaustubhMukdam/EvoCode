"""Evaluation dataset types for EvoCode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class Task:
    """A benchmark task with success criteria."""

    task_id: str
    repository_commit: str
    category: str
    prompt: str
    expected_artifact: str
    success_criteria: list[dict]


@runtime_checkable
class TaskDataset(Protocol):
    """Protocol for loading and querying task datasets."""

    def load(self) -> list[Task]: ...

    def get(self, task_id: str) -> Task | None: ...

    def list_all(self) -> list[Task]: ...


class FakeTaskDataset:
    """In-memory fake task dataset for testing."""

    def __init__(self, tasks: list[Task]) -> None:
        self._tasks = {t.task_id: t for t in tasks}

    def load(self) -> list[Task]:
        return list(self._tasks.values())

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def list_all(self) -> list[Task]:
        return list(self._tasks.values())
