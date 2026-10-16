"""interface/api_client.py — thin HTTP client for the GitMind API (no Streamlit import).

Kept separate from the UI so error handling and payload building are testable.
"""
from __future__ import annotations

from typing import Any

import httpx


class ApiError(Exception):
    """A non-2xx API response, normalised for display."""

    def __init__(self, status_code: int, kind: str, message: str, body: dict | None = None) -> None:
        super().__init__(f"{status_code} {kind}: {message}")
        self.status_code = status_code
        self.kind = kind
        self.message = message
        self.body = body or {}


def _error_from(resp: httpx.Response) -> ApiError:
    try:
        body = resp.json()
    except ValueError:
        return ApiError(resp.status_code, "http_error", resp.text[:300])
    if isinstance(body, dict) and "status" in body and "error" in body:
        return ApiError(resp.status_code, str(body["status"]), str(body["error"]), body)
    detail = body.get("detail") if isinstance(body, dict) else body
    if isinstance(detail, list):  # FastAPI validation errors
        detail = "; ".join(
            f"{'.'.join(str(p) for p in d.get('loc', [])[1:])}: {d.get('msg', '')}" for d in detail
        )
    return ApiError(resp.status_code, "http_error", str(detail), body if isinstance(body, dict) else {})


def build_query_payload(
    repo: str,
    query: str,
    mode: str,
    *,
    top_k: int = 12,
    limit: int = 40,
    authors: str = "",
    doc_types: str = "",
    modules: str = "",
    kind: str | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
) -> dict[str, Any]:
    """Build the /query body; comma-separated filter strings become lists."""

    def split(value: str) -> list[str]:
        return [v.strip() for v in value.split(",") if v.strip()]

    payload: dict[str, Any] = {
        "repo": repo.strip(), "query": query.strip(), "mode": mode, "top_k": top_k, "limit": limit,
        "authors": split(authors), "doc_types": split(doc_types), "modules": split(modules),
    }
    if kind and kind != "any":
        payload["kind"] = kind
    if time_start:
        payload["time_start"] = time_start
    if time_end:
        payload["time_end"] = time_end
    return payload


class GitMindClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        *,
        timeout: float = 180.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={"X-API-Key": api_key} if api_key else {},
            transport=transport,
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> dict | list:
        try:
            resp = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError(0, "unreachable", f"cannot reach the API: {exc}") from None
        if resp.status_code >= 400:
            raise _error_from(resp)
        return resp.json()

    def ready(self) -> dict:
        """Readiness report; a 503 'not_ready' body is returned, not raised."""
        try:
            resp = self._client.get("/ready")
        except httpx.HTTPError as exc:
            raise ApiError(0, "unreachable", f"cannot reach the API: {exc}") from None
        if resp.status_code in (200, 503):
            body = resp.json()
            if isinstance(body, dict) and "repos" in body:
                return body
        raise _error_from(resp)

    def query(self, payload: dict) -> dict:
        return self._request("POST", "/api/v1/query", json=payload)  # type: ignore[return-value]

    def start_ingest(self, repo_path: str, github_repo: str = "", max_commits: int = 0) -> dict:
        body: dict[str, Any] = {"repo_path": repo_path or "."}
        if github_repo.strip():
            body["github_repo"] = github_repo.strip()
        if max_commits:
            body["max_commits"] = int(max_commits)
        return self._request("POST", "/api/v1/ingest", json=body)  # type: ignore[return-value]

    def job(self, job_id: str) -> dict:
        return self._request("GET", f"/api/v1/jobs/{job_id}")  # type: ignore[return-value]

    def jobs(self, repo: str | None = None) -> list[dict]:
        params = {"repo": repo} if repo else None
        return self._request("GET", "/api/v1/jobs", params=params)  # type: ignore[return-value]

    def cancel(self, job_id: str) -> dict:
        return self._request("POST", f"/api/v1/jobs/{job_id}/cancel")  # type: ignore[return-value]

    def close(self) -> None:
        self._client.close()
