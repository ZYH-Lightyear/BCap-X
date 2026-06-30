"""Minimal V-API chat proxy that preserves OpenAI tool-call fields.

This server intentionally accepts and returns raw OpenAI-compatible JSON dicts
instead of using a restrictive Pydantic response model. The older
``openrouter_server.py`` normalizes messages to ``role/content`` and therefore
drops ``tools`` / ``tool_calls`` data. This proxy is meant for testing and
running provider-native tool calling with V-API.

Example:
  set -a; source .env; set +a
  .venv-libero/bin/python capx/serving/vapi_tool_server.py --port 8110

Then probe:
  .venv-libero/bin/python scripts/test_vapi_tool_call_support.py --mode proxy --model vapi/gpt-5.5
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import tyro
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


def _load_dotenv(path: str | None) -> None:
    if not path:
        return
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :]
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _resolve_api_key(api_key: str | None, api_key_env: str, key_file: str | None) -> str:
    if api_key:
        return api_key
    for env_name in [x.strip() for x in api_key_env.split(",") if x.strip()]:
        value = os.getenv(env_name)
        if value:
            return value
    if key_file:
        p = Path(key_file)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    return stripped
    raise RuntimeError(
        "No V-API key found. Set V_API_KEY/LLM_API_KEY, pass --api-key, or pass --key-file."
    )


def _proxy_error(exc: Exception, native_model: str | None) -> HTTPException:
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
    body_prefix = ""
    if response is not None:
        try:
            body_prefix = (response.text or "")[:1000]
        except Exception:  # noqa: BLE001 - diagnostic only
            body_prefix = ""
    detail = f"V-API request failed for model={native_model!r}: {type(exc).__name__}: {exc}"
    if status:
        detail += f" (upstream_status={status})"
    if body_prefix:
        detail += f" body_prefix={body_prefix!r}"
    return HTTPException(status_code=502, detail=detail)


def _strip_model_prefix(model: str, route_prefix: str) -> str:
    prefix = route_prefix.strip().strip("/")
    if prefix and model.startswith(prefix + "/"):
        return model[len(prefix) + 1 :]
    return model


def _jsonable_response(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json", exclude_none=False)
    if hasattr(response, "dict"):
        return response.dict()
    if isinstance(response, dict):
        return response
    return json.loads(response.model_dump_json())


def create_app(
    *,
    api_key: str,
    base_url: str,
    route_prefix: str = "vapi",
    timeout_s: float = 600.0,
) -> FastAPI:
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url.rstrip("/") + "/",
        timeout=timeout_s,
        default_headers={
            "HTTP-Referer": "https://github.com/nvidia-gear/CaP-X",
            "X-Title": "CaP-X RoboMEx Tool Proxy",
        },
    )
    app = FastAPI(title="V-API Tool-Call Proxy", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/chat/completions")
    async def chat_completions(request: Request):
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        model = payload.get("model")
        if not isinstance(model, str) or not model:
            raise HTTPException(status_code=400, detail='Request must include string field "model".')

        native_model = _strip_model_prefix(model, route_prefix)
        payload = dict(payload)
        payload["model"] = native_model

        # ``reasoning`` is an OpenRouter extra_body field and should not be sent
        # to OpenAI-compatible V-API unless the caller explicitly embeds it in
        # provider-specific params elsewhere.
        payload.pop("reasoning", None)

        try:
            if payload.get("stream"):
                response = await client.chat.completions.create(**payload)

                async def event_stream():
                    async for chunk in response:
                        if hasattr(chunk, "model_dump_json"):
                            data = chunk.model_dump_json(exclude_none=False)
                        else:
                            data = json.dumps(chunk)
                        yield f"data: {data}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(event_stream(), media_type="text/event-stream")

            payload["stream"] = False
            response = await client.chat.completions.create(**payload)
            return _jsonable_response(response)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("V-API proxy request failed")
            raise _proxy_error(exc, native_model) from exc

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "base_url": base_url,
            "route_prefix": route_prefix,
            "preserves_raw_tool_fields": True,
        }

    return app


def main(
    host: str = "0.0.0.0",
    port: int = 8110,
    base_url: str = "https://api.gpt.ge/v1/",
    api_key: str | None = None,
    api_key_env: str = "V_API_KEY,LLM_API_KEY",
    key_file: str | None = None,
    env_file: str | None = ".env",
    route_prefix: str = "vapi",
    timeout_s: float = 600.0,
    log_level: str = "info",
) -> None:
    """Run the V-API tool-call preserving proxy."""

    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    _load_dotenv(env_file)
    resolved_key = _resolve_api_key(api_key, api_key_env, key_file)
    app = create_app(
        api_key=resolved_key,
        base_url=base_url,
        route_prefix=route_prefix,
        timeout_s=timeout_s,
    )
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    tyro.cli(main)
