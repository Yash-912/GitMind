from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunParams:
    embedding_model: str
    top_k: int
    rrf_k: int
    hop_depth: int


class MLflowLogger:
    """Minimal MLflow logger wrapper."""

    def __init__(self, experiment_name: str = "gitmind") -> None:
        try:
            import mlflow  # type: ignore

            self._mlflow = mlflow
            self._mlflow.set_experiment(experiment_name)
        except Exception:
            self._mlflow = None

    def log_run(self, params: RunParams, metrics: dict[str, float]) -> None:
        if self._mlflow is None:
            return
        with self._mlflow.start_run():
            self._mlflow.log_params(params.__dict__)
            self._mlflow.log_metrics(metrics)
