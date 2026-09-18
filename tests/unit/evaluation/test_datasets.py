"""Tests for evaluation datasets."""

from evocode.evaluation.datasets import FakeTaskDataset, Task, TaskDataset


class TestTask:
    """Test Task dataclass."""

    def test_task_creation(self) -> None:
        """Test creating a task with all fields."""
        task = Task(
            task_id="task-001",
            repository_commit="abc123",
            category="bug_localization",
            prompt="Find why login fails",
            expected_artifact="auth.py",
            success_criteria=[
                {"type": "file_contains", "file": "auth.py", "symbol": "login"},
            ],
        )
        assert task.task_id == "task-001"
        assert task.repository_commit == "abc123"
        assert task.category == "bug_localization"
        assert task.prompt == "Find why login fails"
        assert task.expected_artifact == "auth.py"
        assert len(task.success_criteria) == 1

    def test_task_categories(self) -> None:
        """Test valid task categories."""
        valid = [
            "repository_understanding",
            "bug_localization",
            "debugging",
            "test_diagnosis",
            "refactoring",
            "documentation",
            "code_navigation",
        ]
        for cat in valid:
            task = Task(
                task_id="t",
                repository_commit="abc",
                category=cat,
                prompt="q",
                expected_artifact="f.py",
                success_criteria=[],
            )
            assert task.category == cat

    def test_task_empty_criteria(self) -> None:
        """Test task with empty success criteria."""
        task = Task(
            task_id="t",
            repository_commit="abc",
            category="repository_understanding",
            prompt="q",
            expected_artifact="f.py",
            success_criteria=[],
        )
        assert len(task.success_criteria) == 0


class TestTaskDatasetProtocol:
    """Test TaskDataset protocol contract."""

    def test_protocol_has_load(self) -> None:
        """Test protocol defines load method."""
        assert hasattr(TaskDataset, "load")

    def test_protocol_has_get(self) -> None:
        """Test protocol defines get method."""
        assert hasattr(TaskDataset, "get")

    def test_protocol_has_list_all(self) -> None:
        """Test protocol defines list_all method."""
        assert hasattr(TaskDataset, "list_all")

    def test_fake_dataset_satisfies_protocol(self) -> None:
        """Test FakeTaskDataset satisfies TaskDataset protocol."""
        dataset = FakeTaskDataset(tasks=[])
        assert hasattr(dataset, "load")
        assert hasattr(dataset, "get")
        assert hasattr(dataset, "list_all")


class TestFakeTaskDataset:
    """Test FakeTaskDataset implementation."""

    def _make_task(self, task_id: str = "task-001") -> Task:
        return Task(
            task_id=task_id,
            repository_commit="abc123",
            category="bug_localization",
            prompt="test prompt",
            expected_artifact="test.py",
            success_criteria=[],
        )

    def test_load_empty(self) -> None:
        """Test loading empty dataset."""
        dataset = FakeTaskDataset(tasks=[])
        tasks = dataset.load()
        assert tasks == []

    def test_load_returns_all(self) -> None:
        """Test load returns all tasks."""
        tasks = [self._make_task("t1"), self._make_task("t2")]
        dataset = FakeTaskDataset(tasks=tasks)
        result = dataset.load()
        assert len(result) == 2

    def test_get_by_id(self) -> None:
        """Test get retrieves task by ID."""
        tasks = [self._make_task("t1"), self._make_task("t2")]
        dataset = FakeTaskDataset(tasks=tasks)
        task = dataset.get("t1")
        assert task is not None
        assert task.task_id == "t1"

    def test_get_missing_returns_none(self) -> None:
        """Test get returns None for missing task."""
        dataset = FakeTaskDataset(tasks=[])
        assert dataset.get("nonexistent") is None

    def test_list_all_empty(self) -> None:
        """Test list_all returns empty list."""
        dataset = FakeTaskDataset(tasks=[])
        assert dataset.list_all() == []

    def test_list_all_multiple(self) -> None:
        """Test list_all returns all tasks."""
        tasks = [self._make_task(f"t{i}") for i in range(5)]
        dataset = FakeTaskDataset(tasks=tasks)
        assert len(dataset.list_all()) == 5

    def test_get_returns_correct_type(self) -> None:
        """Test get returns Task instance."""
        dataset = FakeTaskDataset(tasks=[self._make_task("t1")])
        task = dataset.get("t1")
        assert isinstance(task, Task)
