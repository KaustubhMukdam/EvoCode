"""Pytest configuration and fixtures."""

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

# Set HuggingFace cache to project directory (not C: drive)
os.environ.setdefault(
    "HF_HOME",
    str(Path(__file__).resolve().parent.parent / "huggingface_cache"),
)

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def toy_repo_dir(temp_dir: Path) -> Path:
    """Create a minimal toy repository for testing."""
    repo = temp_dir / "toy_repo"
    repo.mkdir()

    # Create a simple Python project structure
    (repo / "main.py").write_text("""
def greet(name: str) -> str:
    \"\"\"Return a greeting.\"\"\"
    return f"Hello, {name}!"


def add(a: int, b: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return a + b


if __name__ == "__main__":
    print(greet("World"))
""")

    (repo / "utils.py").write_text("""
import os
from pathlib import Path


def find_files(directory: str, pattern: str) -> list[Path]:
    \"\"\"Find files matching pattern.\"\"\"
    return list(Path(directory).rglob(pattern))


def read_config(path: str) -> dict:
    \"\"\"Read a simple config file.\"\"\"
    config = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                config[key] = value
    return config
""")

    (repo / "config.py").write_text("""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    \"\"\"Application configuration.\"\"\"
    debug: bool = False
    max_retries: int = 3
    timeout: float = 30.0
    api_key: Optional[str] = None


DEFAULT_CONFIG = Config()
""")

    tests_dir = repo / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_main.py").write_text("""
import pytest
from main import greet, add


def test_greet():
    assert greet("Test") == "Hello, Test!"


def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
""")

    (tests_dir / "test_utils.py").write_text("""
import pytest
from utils import find_files, read_config
import tempfile
from pathlib import Path


def test_find_files():
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "a.txt").write_text("a")
        Path(tmp, "b.txt").write_text("b")
        Path(tmp, "sub", "c.txt").mkdir(parents=True, exist_ok=True)
        Path(tmp, "sub", "c.txt").write_text("c")

        files = find_files(tmp, "*.txt")
        assert len(files) == 3
""")

    (repo / "requirements.txt").write_text("""
pytest>=8.0
""")

    (repo / "README.md").write_text("""
# Toy Repository

A minimal Python project for testing EvoCode.
""")

    return repo


@pytest.fixture
def sample_task() -> dict:
    """Provide a sample task fixture."""
    return {
        "task_id": "test-001",
        "repository_commit": "abc123",
        "category": "repository_understanding",
        "prompt": "Find the function that returns a greeting",
        "expected_artifact": "main.py",
        "success_criteria": [
            {"type": "file_contains", "file": "main.py", "symbol": "greet"},
        ],
    }


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton state between tests."""
    yield
    # Add cleanup if needed
