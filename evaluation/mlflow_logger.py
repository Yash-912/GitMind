from __future__ import annotations

import json
import logging
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class RunParams:
    embedding_model: str
    top_k: int
    rrf_k: int
    hop_depth: int


class MLflowLogger:
    """MLflow wrapper that never fails silently.

    If MLflow cannot be imported / configured, or a log call fails, a visible
    warning is emitted and the failure is recorded in ``status()`` so callers
    (and tests) can tell that nothing was logged.

    ``mlflow_module`` lets tests inject a fake instead of the real package.
    """

    def __init__(
        self,
        experiment_name: str = "gitmind",
        tracking_uri: str | None = None,
        mlflow_module: Any = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.error: str | None = None
        self.runs_logged = 0
        self._mlflow = None
        try:
            mlflow = mlflow_module
            if mlflow is None:
                import mlflow  # type: ignore
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._mlflow = mlflow
        except Exception as exc:  # noqa: BLE001 - surfaced, not swallowed
            self._fail(f"MLflow disabled: {type(exc).__name__}: {exc}")

    @property
    def enabled(self) -> bool:
        return self._mlflow is not None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "experiment": self.experiment_name,
            "runs_logged": self.runs_logged,
            "error": self.error,
        }

    def _fail(self, message: str) -> None:
        self.error = message
        log.warning(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)

    def log_run(
        self,
        params: RunParams | dict[str, Any],
        metrics: dict[str, float | None],
        *,
        run_name: str | None = None,
        extra_params: dict[str, Any] | None = None,
        dataset_path: str | None = None,
        per_question: list[dict[str, Any]] | None = None,
        tags: dict[str, str] | None = None,
    ) -> bool:
        """Log one run. Returns True only if everything was logged."""
        if self._mlflow is None:
            self._fail(f"MLflow run not logged (logger disabled: {self.error})")
            return False
        all_params = dict(params if isinstance(params, dict) else params.__dict__)
        all_params.update(extra_params or {})
        # Not-applicable metrics (None) are recorded as params, never as 0 or 1.
        numeric = {k: float(v) for k, v in metrics.items() if v is not None}
        skipped = sorted(k for k, v in metrics.items() if v is None)
        try:
            with self._mlflow.start_run(run_name=run_name):
                self._mlflow.log_params({k: str(v) for k, v in all_params.items()})
                if skipped:
                    self._mlflow.log_param("metrics_not_applicable", ",".join(skipped))
                if tags:
                    self._mlflow.set_tags(tags)
                self._mlflow.log_metrics(numeric)
                if dataset_path and Path(dataset_path).exists():
                    self._mlflow.log_artifact(str(dataset_path), artifact_path="dataset")
                if per_question is not None:
                    with tempfile.TemporaryDirectory() as tmp:
                        out = Path(tmp) / "per_question_results.json"
                        out.write_text(
                            json.dumps(per_question, ensure_ascii=False, indent=2, default=str),
                            encoding="utf-8",
                        )
                        self._mlflow.log_artifact(str(out), artifact_path="results")
        except Exception as exc:  # noqa: BLE001
            self._fail(f"MLflow logging failed: {type(exc).__name__}: {exc}")
            return False
        self.runs_logged += 1
        return True
