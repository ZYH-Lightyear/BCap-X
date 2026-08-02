"""HTTP helpers for Cap-X local perception / motion microservices."""

from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import urlparse

import requests

_LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


def ensure_localhost_noproxy() -> None:
    """Ensure loopback hosts bypass HTTP(S)_PROXY (avoids proxy 503s to local APIs)."""
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        parts = [p.strip() for p in current.split(",") if p.strip()]
        changed = False
        for host in _LOCAL_HOSTS:
            if host not in parts:
                parts.append(host)
                changed = True
        if changed or key not in os.environ:
            os.environ[key] = ",".join(parts)


def is_loopback_url(url: str) -> bool:
    """Return True if *url* targets a loopback host."""
    host = (urlparse(url).hostname or "").lower()
    return host in _LOCAL_HOSTS or host.startswith("127.")


def request_proxies(url: str) -> dict[str, str] | None:
    """Disable proxies for loopback URLs; otherwise defer to env (``None``).

    Empty-string proxy values are required: ``None`` still falls through to
    ``HTTP_PROXY`` in requests/urllib3, which caused 503s via corporate proxies.
    """
    if is_loopback_url(url):
        return {"http": "", "https": ""}
    return None


def _loopback_session() -> requests.Session:
    """Session that never consults HTTP(S)_PROXY / NO_PROXY env vars."""
    session = requests.Session()
    session.trust_env = False
    return session


# Apply once on import so any client that pulls in serve_utils is covered.
ensure_localhost_noproxy()


def post_with_retries(
    url: str,
    payload: dict,
    timeout_seconds: float = 120.0,
    retry_interval: float = 1.0,
    max_retries: int = 5,
):
    """
    Retry POST requests with exponential backoff for up to `timeout_seconds` of wall clock time.

    Args:
        url: The URL to POST to.
        payload: JSON payload to send.
        timeout_seconds: Maximum wall clock time before giving up.
        retry_interval: Initial interval between retries (doubles each retry).
        max_retries: Maximum number of retry attempts.

    Raises RuntimeError if the time limit or retry count is exceeded.
    """
    deadline = time.time() + timeout_seconds
    current_interval = retry_interval
    proxies = request_proxies(url)
    post = _loopback_session().post if is_loopback_url(url) else requests.post

    last_err = None
    attempts = 0
    while time.time() < deadline and attempts < max_retries:
        try:
            resp = post(
                url, json=payload, timeout=timeout_seconds, proxies=proxies
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_err = e
            attempts += 1
            time.sleep(min(current_interval, max(0, deadline - time.time())))
            current_interval = min(current_interval * 2, 8.0)

    raise RuntimeError(
        f"Request to {url} failed after {attempts} retries / "
        f"{timeout_seconds:.2f}s. Last error: {last_err}"
    )


def post_with_queue_tolerance(
    url: str,
    payload: dict,
    timeout_seconds: float = 120.0,
    retry_interval: float = 1.0,
    max_retries: int = 5,
):
    """
    POST with tolerance for queued servers (handles 503 gracefully).

    Like `post_with_retries`, but treats HTTP 503 (Service Unavailable) as a
    transient condition (server is busy with other requests) and retries with
    exponential backoff instead of raising immediately.

    Args:
        url: The URL to POST to.
        payload: JSON payload to send.
        timeout_seconds: Maximum wall clock time before giving up.
        retry_interval: Initial interval between retries (doubles each retry).
        max_retries: Maximum number of retry attempts.

    Raises RuntimeError if the time limit or retry count is exceeded.
    """
    deadline = time.time() + timeout_seconds
    current_interval = retry_interval
    proxies = request_proxies(url)
    post = _loopback_session().post if is_loopback_url(url) else requests.post

    last_err = None
    attempts = 0
    while time.time() < deadline and attempts < max_retries:
        try:
            resp = post(
                url, json=payload, timeout=timeout_seconds, proxies=proxies
            )
            if resp.status_code == 503:
                # Server is busy / model not ready -- treat as transient
                last_err = requests.HTTPError(
                    f"503 Service Unavailable: {resp.text}", response=resp
                )
                attempts += 1
                time.sleep(min(current_interval, max(0, deadline - time.time())))
                current_interval = min(current_interval * 2, 8.0)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_err = e
            attempts += 1
            time.sleep(min(current_interval, max(0, deadline - time.time())))
            current_interval = min(current_interval * 2, 8.0)

    raise RuntimeError(
        f"Request to {url} failed after {attempts} retries / "
        f"{timeout_seconds:.2f}s. Last error: {last_err}"
    )


def http_get(url: str, *, timeout: float = 3.0, **kwargs: Any) -> requests.Response:
    """GET with loopback proxy bypass."""
    if is_loopback_url(url):
        kwargs.setdefault("proxies", request_proxies(url))
        return _loopback_session().get(url, timeout=timeout, **kwargs)
    return requests.get(url, timeout=timeout, **kwargs)


def http_post(
    url: str, *, timeout: float = 120.0, **kwargs: Any
) -> requests.Response:
    """POST with loopback proxy bypass."""
    if is_loopback_url(url):
        kwargs.setdefault("proxies", request_proxies(url))
        return _loopback_session().post(url, timeout=timeout, **kwargs)
    return requests.post(url, timeout=timeout, **kwargs)
