from __future__ import annotations

import typer

from scripts.run_answer import main as run_answer_main
from evaluation.dataset_builder import DatasetBuilder
from evaluation.ragas_evaluator import RagasEvaluator

app = typer.Typer(no_args_is_help=True)


@app.command()
def answer(query: str, mode: str = "direct") -> None:
    """Run retrieval + generation for a single query."""
    import sys
    sys.argv = ["run_answer.py", query, "--mode", mode]
    run_answer_main()


@app.command()
def build_dataset(chunks_path: str, out_path: str, limit: int = 100, dry_run: bool = False) -> None:
    """Build a synthetic QA dataset from chunks."""
    builder = DatasetBuilder()
    builder.build(chunks_path, out_path, limit=limit, dry_run=dry_run)
    builder.close()


@app.command()
def evaluate(dataset_path: str, limit: int = 20) -> None:
    """Run RAGAS evaluation on the pipeline."""
    evaluator = RagasEvaluator()
    result = evaluator.evaluate(dataset_path, limit=limit)
    print(result)


if __name__ == "__main__":
    app()
