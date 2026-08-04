"""Local service preflight for real LIBERO-PRO Context Runtime runs."""

from __future__ import annotations

import socket

REQUIRED_SERVICES = {
    8110: "LLM proxy (vlm_bbox_detection)",
    8114: "SAM3",
    8115: "Contact-GraspNet",
    8116: "PyRoKi IK",
}


def unavailable_services(*, timeout_s: float = 2.0) -> list[str]:
    """Return formatted entries for required localhost services that are down."""

    down: list[str] = []
    for port, name in REQUIRED_SERVICES.items():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout_s)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                down.append(f"  :{port}  {name}")
    return down


def preflight_services() -> None:
    """Fail fast before an episode instead of timing out during a tool call."""

    down = unavailable_services()
    if down:
        raise RuntimeError(
            "required services are not running:\n"
            + "\n".join(down)
            + "\nstart them with: bash scripts/start_libero_services.sh"
        )


__all__ = ["REQUIRED_SERVICES", "preflight_services", "unavailable_services"]
