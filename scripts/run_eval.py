"""Run named evaluation configurations and log each as an MLflow run.

A configuration is a callable ``(pairs) -> (list[EvalResult], meta)`` (see
``evaluation.ragas_evaluator.PipelineRunner``), so later phases can register
dense-only / +BM25 / +rerank / +graph / full pipelines without touching this
script. With no registered configurations, the built-in pipeline runs as
"default".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.mlflow_logger import MLflowLogger  # noqa: E402
from evaluation.ragas_evaluator import PipelineRunner, RagasConfig, RagasEvaluator  # noqa: E402

# Phase 3/4 can add entries here (or pass their own dict to run_configurations).
CONFIGURATIONS: dict[str, PipelineRunner | None] = {"default": None}


def run_configurations(
    configurations: dict[str, Callable | None],
    dataset_path: str,
    logger: MLflowLogger,
    *,
    limit: int = 20,
    split: str | None = "eval",
    ragas_config: RagasConfig | None = None,
    backend: Any = None,
    repo: str | None = None,
    generation_root: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Run every configuration; each gets its own MLflow run.

    Failures are recorded per configuration (``error``) and never skipped
    silently; the remaining configurations still run.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, runner in configurations.items():
        evaluator = RagasEvaluator(
            config=ragas_config, backend=backend, pipeline_runner=runner, logger=logger,
            repo=repo, generation_root=generation_root,
        )
        try:
            metrics = evaluator.evaluate(
                dataset_path,
                limit=limit,
                run_name=name,
                extra_params={"configuration": name, "split": split or "all"},
                split=split,
            )
            report = evaluator.last_report
            out[name] = {
                "metrics": metrics,
                "mlflow_logged": report.mlflow_logged if report else None,
                "warnings": report.warnings if report else [],
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - recorded, run continues
            out[name] = {"metrics": None, "mlflow_logged": False, "warnings": [], "error": f"{type(exc).__name__}: {exc}"}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evaluation configurations with MLflow logging")
    parser.add_argument("--dataset", required=True, help="QA dataset JSONL from DatasetBuilder")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--split", default="eval", help="dataset split to evaluate ('' for all)")
    parser.add_argument("--experiment", default="gitmind")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--configs", nargs="*", default=None, help="subset of configuration names")
    parser.add_argument("--repo", required=True, help="repository whose active generation is evaluated")
    parser.add_argument("--generation-root", default=None, help="defaults to <data_dir>/generations")
    args = parser.parse_args()

    names = args.configs or list(CONFIGURATIONS)
    unknown = [n for n in names if n not in CONFIGURATIONS]
    if unknown:
        print(f"Unknown configuration(s): {unknown}; available: {list(CONFIGURATIONS)}", file=sys.stderr)
        return 2
    logger = MLflowLogger(args.experiment, tracking_uri=args.tracking_uri)
    results = run_configurations(
        {n: CONFIGURATIONS[n] for n in names},
        args.dataset,
        logger,
        limit=args.limit,
        split=args.split or None,
        repo=args.repo,
        generation_root=args.generation_root,
    )
    print(json.dumps({"results": results, "mlflow": logger.status()}, indent=2, default=str))
    failed = any(r["error"] or r["mlflow_logged"] is False for r in results.values())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
