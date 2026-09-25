"""Integration test: Experiment 001 — run full pipeline on toy repo."""

from pathlib import Path

from evocode.retrieval.line_chunker import LineChunker
from evocode.retrieval.embedder import FakeEmbedder
from evocode.retrieval.index import FakeVectorIndex
from evocode.retrieval.semantic_retriever import SemanticRetriever
from evocode.evaluation.baselines import BaselineRunner


def test_experiment_001_pipeline(tmp_path: Path) -> None:
    """End-to-end: chunk -> embed -> index -> retrieve -> evaluate baselines."""
    # Create a minimal toy repo
    (tmp_path / "main.py").write_text(
        "def greet(name):\n    return f'Hello {name}'\n\ndef add(a, b):\n    return a + b\n"
    )
    (tmp_path / "utils.py").write_text(
        "def multiply(a, b):\n    return a * b\n"
    )
    (tmp_path / "test_main.py").write_text(
        "from main import greet\n\nassert greet('world') == 'Hello world'\n"
    )

    # Build retriever
    chunker = LineChunker(max_lines=10)
    embedder = FakeEmbedder(dimension=384)
    index = FakeVectorIndex(dimension=384)
    retriever = SemanticRetriever(chunker=chunker, embedder=embedder, index=index)

    # Index the toy repo
    retriever.index_directory(tmp_path)
    assert index.size > 0

    # Retrieve
    results = retriever.retrieve("greeting function", top_k=3)
    assert len(results) > 0

    # Run baselines
    all_chunks = chunker.chunk_directory(tmp_path)
    candidate_files = list({c.file_path for c in all_chunks})
    runner = BaselineRunner()
    baseline_results = runner.run(
        query="greeting function",
        task_id="experiment_001",
        relevant_files={"main.py"},
        candidate_files=candidate_files,
        top_k=3,
    )
    assert len(baseline_results) == 3
    names = {r.baseline_name for r in baseline_results}
    assert names == {"fixed_k", "top_k_length", "random"}
