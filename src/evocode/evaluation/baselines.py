"""Baseline evaluation runner for Experiment 001."""

from __future__ import annotations

from dataclasses import dataclass

from evocode.retrieval.baselines import FixedKRetriever, RandomRetriever, TopKByLength
from evocode.retrieval.chunker import CodeChunk


@dataclass
class BaselineResult:
    baseline_name: str
    task_id: str
    retrieved_files: list[str]
    top_k: int
    precision_at_k: float
    recall_at_k: float


class BaselineRunner:
    """Runs all static baselines and computes precision/recall@k."""

    def run(
        self,
        query: str,  # noqa: ARG002
        task_id: str,
        relevant_files: set[str],
        candidate_files: list[str],
        top_k: int = 5,
    ) -> list[BaselineResult]:
        chunks = [
            CodeChunk(text=f, file_path=f, start_line=1, end_line=1, language="text", metadata={})
            for f in candidate_files
        ]
        baselines = {
            "fixed_k": FixedKRetriever(chunks=chunks, k=top_k),
            "top_k_length": TopKByLength(chunks=chunks, k=top_k),
            "random": RandomRetriever(chunks=chunks, k=top_k, seed=42),
        }
        results: list[BaselineResult] = []
        for name, retriever in baselines.items():
            retrieved = retriever.retrieve(query, top_k=top_k)
            retrieved_files = [c.file_path for c in retrieved]
            hits = len(set(retrieved_files) & relevant_files)
            precision = hits / top_k if top_k > 0 else 0.0
            recall = hits / len(relevant_files) if relevant_files else 0.0
            results.append(BaselineResult(
                baseline_name=name,
                task_id=task_id,
                retrieved_files=retrieved_files,
                top_k=top_k,
                precision_at_k=precision,
                recall_at_k=recall,
            ))
        return results
