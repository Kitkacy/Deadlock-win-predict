"""Rate-limited HTTP client for api.deadlock-api.com."""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.deadlock-api.com"

# Bulk metadata OpenAPI limits (IP): 30 req/min. Stay under by default.
DEFAULT_MIN_INTERVAL_S = 2.1  # ~28.5 req/min


class RateLimiter:
    """Simple minimum-interval limiter (thread-safe)."""

    def __init__(self, min_interval_s: float = DEFAULT_MIN_INTERVAL_S) -> None:
        self.min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self.min_interval_s - (now - self._last)
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


class DeadlockAPIClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
        timeout_s: float = 120.0,
        max_retries: int = 8,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.limiter = RateLimiter(min_interval_s)
        self.max_retries = max_retries
        headers = {"User-Agent": "deadlock-win-predict/0.1", "Accept": "application/json"}
        if api_key:
            headers["X-API-KEY"] = api_key
            # Key quota is higher (30/10s); still be polite.
            self.limiter.min_interval_s = min(min_interval_s, 0.4)
        self._client = httpx.Client(timeout=timeout_s, headers=headers, follow_redirects=True)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> DeadlockAPIClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        clean_params = _flatten_params(params or {})
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                resp = self._client.get(url, params=clean_params)
            except httpx.HTTPError as exc:
                last_err = exc
                sleep_s = _backoff(attempt)
                logger.warning("HTTP error %s; retry in %.1fs", exc, sleep_s)
                time.sleep(sleep_s)
                continue

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    sleep_s = float(retry_after) + random.uniform(0, 0.5)
                else:
                    sleep_s = _backoff(attempt)
                logger.warning(
                    "%s %s -> %s; retry in %.1fs (attempt %s)",
                    path,
                    urlencode(clean_params)[:120],
                    resp.status_code,
                    sleep_s,
                    attempt + 1,
                )
                time.sleep(sleep_s)
                last_err = httpx.HTTPStatusError(
                    f"{resp.status_code}", request=resp.request, response=resp
                )
                continue

            resp.raise_for_status()

        raise RuntimeError(f"Failed GET {path} after {self.max_retries} retries: {last_err}")

    def recently_fetched(self) -> list[dict[str, Any]]:
        data = self.get_json("/v1/matches/recently-fetched")
        if not isinstance(data, list):
            raise TypeError("expected list from /v1/matches/recently-fetched")
        return data

    def bulk_metadata(self, **params: Any) -> list[dict[str, Any]]:
        data = self.get_json("/v1/matches/metadata", params=params)
        if not isinstance(data, list):
            raise TypeError("expected list from /v1/matches/metadata")
        return data

    def match_metadata(self, match_id: int, **params: Any) -> dict[str, Any]:
        data = self.get_json(f"/v1/matches/{match_id}/metadata", params=params)
        if not isinstance(data, dict):
            raise TypeError("expected object from match metadata")
        return data


def _flatten_params(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            out[key] = "true" if value else "false"
        elif isinstance(value, (list, tuple)):
            out[key] = ",".join(str(v) for v in value)
        else:
            out[key] = value
    return out


def _backoff(attempt: int) -> float:
    return min(60.0, (2**attempt) + random.uniform(0, 1))
