"""Tests for path validator / sandbox."""

from pathlib import Path

from evocode.tools.path_validator import PathValidationResult, PathValidator


class TestPathValidator:
    """Test path validation for sandbox security."""

    def test_valid_relative_path(self) -> None:
        """A clean relative path inside sandbox passes."""
        validator = PathValidator(sandbox_root=Path("/repo"), max_file_size_mb=5)
        # Create a mock file using a real temp path
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "src").mkdir()
            (Path(d) / "src" / "main.py").write_text("x = 1")
            validator = PathValidator(sandbox_root=Path(d), max_file_size_mb=5)
            result = validator.validate("src/main.py")
            assert result.is_valid
            assert result.resolved_path == Path(d) / "src" / "main.py"

    def test_rejects_absolute_path(self) -> None:
        """Absolute paths are rejected."""
        validator = PathValidator(sandbox_root=Path("/repo"), max_file_size_mb=5)
        # Use a path that's clearly absolute on both platforms
        result = validator.validate("C:/etc/passwd")
        assert not result.is_valid
        # On Windows, C: paths are absolute; on Unix, / paths are absolute
        # The error message may vary
        assert result.error is not None

    def test_rejects_path_traversal(self) -> None:
        """Path traversal like ../../etc/passwd is rejected."""
        validator = PathValidator(sandbox_root=Path("/repo"), max_file_size_mb=5)
        result = validator.validate("../../etc/passwd")
        assert not result.is_valid
        assert "traversal" in result.error.lower() or "outside" in result.error.lower()

    def test_rejects_symlink_escape(self, tmp_path: Path) -> None:
        """Symlink pointing outside sandbox is rejected."""
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret").write_text("x")
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        link = sandbox / "link"
        link.symlink_to(outside / "secret")
        validator = PathValidator(sandbox_root=sandbox, max_file_size_mb=5)
        result = validator.validate("link")
        assert not result.is_valid

    def test_max_file_size_check(self, tmp_path: Path) -> None:
        """File exceeding max_file_size_mb is rejected."""
        big = tmp_path / "big.txt"
        big.write_text("x" * (6 * 1024 * 1024))
        validator = PathValidator(sandbox_root=tmp_path, max_file_size_mb=5)
        result = validator.validate("big.txt")
        assert not result.is_valid
        assert "size" in result.error.lower()

    def test_file_size_under_limit(self, tmp_path: Path) -> None:
        """File under limit passes."""
        small = tmp_path / "small.txt"
        small.write_text("hello")
        validator = PathValidator(sandbox_root=tmp_path, max_file_size_mb=5)
        result = validator.validate("small.txt")
        assert result.is_valid

    def test_missing_file_returns_error(self, tmp_path: Path) -> None:
        """Non-existent file returns structured error."""
        validator = PathValidator(sandbox_root=tmp_path, max_file_size_mb=5)
        result = validator.validate("missing.txt")
        assert not result.is_valid
        assert "not found" in result.error.lower() or "exist" in result.error.lower()

    def test_sandbox_root_configurable(self, tmp_path: Path) -> None:
        """Different sandbox roots work."""
        root1 = tmp_path / "repo1"
        root2 = tmp_path / "repo2"
        root1.mkdir()
        root2.mkdir()
        validator1 = PathValidator(sandbox_root=root1, max_file_size_mb=5)
        validator2 = PathValidator(sandbox_root=root2, max_file_size_mb=5)
        assert validator1._sandbox_root == root1.resolve()
        assert validator2._sandbox_root == root2.resolve()

    def test_file_within_sandbox(self, tmp_path: Path) -> None:
        """File inside sandbox passes."""
        (tmp_path / "sub" / "file.py").parent.mkdir(parents=True)
        (tmp_path / "sub" / "file.py").write_text("x = 1")
        validator = PathValidator(sandbox_root=tmp_path, max_file_size_mb=5)
        result = validator.validate("sub/file.py")
        assert result.is_valid

    def test_file_outside_sandbox(self, tmp_path: Path) -> None:
        """File outside sandbox root is rejected."""
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.py").write_text("x")
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        validator = PathValidator(sandbox_root=sandbox, max_file_size_mb=5)
        result = validator.validate("../outside/secret.py")
        assert not result.is_valid


class TestPathValidationResult:
    """Test PathValidationResult dataclass."""

    def test_creation(self) -> None:
        """Test creating a result."""
        from pathlib import Path
        r = PathValidationResult(is_valid=True, resolved_path=Path("/repo/a.py"), error=None)
        assert r.is_valid
        assert r.resolved_path == Path("/repo/a.py")

    def test_invalid_result(self) -> None:
        """Test invalid result."""
        r = PathValidationResult(is_valid=False, resolved_path=None, error="traversal")
        assert not r.is_valid
        assert r.error == "traversal"