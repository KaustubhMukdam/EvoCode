"""Path validation for sandbox security."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PathValidationResult:
    """Result of path validation."""

    is_valid: bool
    resolved_path: Path | None
    error: str | None


class PathValidator:
    """Validates file paths against sandbox constraints."""

    def __init__(self, sandbox_root: Path, max_file_size_mb: int) -> None:
        self._sandbox_root = sandbox_root.resolve()
        self._max_file_size_bytes = max_file_size_mb * 1024 * 1024

    @property
    def sandbox_root(self) -> Path:
        """Return the configured sandbox root."""
        return self._sandbox_root

    def validate(self, path: str) -> PathValidationResult:
        """Validate a path against sandbox constraints."""
        # Reject absolute paths
        if Path(path).is_absolute():
            return PathValidationResult(
                is_valid=False,
                resolved_path=None,
                error="Absolute paths are not allowed",
            )

        # Resolve the path
        try:
            candidate = (self._sandbox_root / path).resolve()
        except (OSError, ValueError) as e:
            return PathValidationResult(
                is_valid=False,
                resolved_path=None,
                error=f"Invalid path: {e}",
            )

        # Check if path is within sandbox root
        try:
            candidate.relative_to(self._sandbox_root)
        except ValueError:
            return PathValidationResult(
                is_valid=False,
                resolved_path=None,
                error="Path traversal detected: path escapes sandbox root",
            )

        # Check if file exists
        if not candidate.exists():
            return PathValidationResult(
                is_valid=False,
                resolved_path=None,
                error=f"File not found: {path}",
            )

        # Check if it's a symlink that escapes sandbox
        if candidate.is_symlink():
            try:
                target = candidate.resolve()
                target.relative_to(self._sandbox_root)
            except ValueError:
                return PathValidationResult(
                    is_valid=False,
                    resolved_path=None,
                    error="Symlink escapes sandbox root",
                )

        # Check file size
        try:
            size = candidate.stat().st_size
            if size > self._max_file_size_bytes:
                return PathValidationResult(
                    is_valid=False,
                    resolved_path=None,
                    error=f"File size {size} bytes exceeds limit of {self._max_file_size_bytes} bytes",
                )
        except OSError:
            return PathValidationResult(
                is_valid=False,
                resolved_path=None,
                error="Cannot stat file",
            )

        return PathValidationResult(
            is_valid=True,
            resolved_path=candidate,
            error=None,
        )