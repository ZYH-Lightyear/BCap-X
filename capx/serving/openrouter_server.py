import itertools
import json
import logging
import os
from pathlib import Path
from typing import List, Literal, Optional, Union

import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ImageUrl(BaseModel):
    url: str


class ContentItem(BaseModel):
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageUrl | None = None


class Message(BaseModel):
    role: str
    content: str | list[ContentItem] | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "openrouter/google/gemini-2.5-pro-preview"
    messages: list[Message]
    temperature: float | None = 0.2
    max_tokens: int | None = 256
    stream: bool = False
    top_p: float | None = None
    reasoning_effort: str | None = None
    max_completion_tokens: int | None = None
    # OpenRouter 统一的 reasoning 控制(如 {"effort": "low"} / {"max_tokens": N} /
    # {"exclude": true})。客户端可显式传;不传则用服务端默认(见 create_app)。
    reasoning: dict | None = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]


def _load_dotenv(path: str = ".env") -> None:
    """Best-effort load of a local .env into os.environ (without overriding).

    Keeps the proxy self-sufficient: route api keys referenced via ``api_key_env``
    resolve even when launched in a fresh shell (e.g. a tmux window) that didn't
    inherit our exported secrets. Supports ``KEY=VALUE`` and ``export KEY=VALUE``.
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _load_api_keys(key_file: str) -> list[str]:
    """Load API keys from a file, one key per line. Ignores blank lines and comments."""
    path = Path(key_file)
    if not path.exists():
        raise FileNotFoundError(f"Key file not found: {key_file}")
    keys = []
    for line in path.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            keys.append(line)
    if not keys:
        raise ValueError(f"No API keys found in {key_file}")
    return keys


def _build_extra_body(
    request: "ChatCompletionRequest", default_reasoning_effort: str | None
) -> dict | None:
    """组装转发给上游的 extra_body(主要是 reasoning 控制)。

    优先用请求里显式传的 ``reasoning``;否则在配了该路由默认 effort 时注入
    ``{"effort": <default>}`` 以统一降低 thinking 强度。``reasoning`` 不是 OpenAI 标准
    字段,必须经 ``extra_body`` 透传。``none``/``off``/空 表示完全不注入 —— 非 OpenRouter
    的上游(V-API / 原生 OpenAI / Gemini 网关)会对这个参数报 400,所以必须可关闭。
    """

    reasoning = request.reasoning
    effort = (default_reasoning_effort or "").strip().lower()
    if reasoning is None and effort and effort not in {"none", "off"}:
        reasoning = {"effort": default_reasoning_effort}
    return {"reasoning": reasoning} if reasoning is not None else None


class Upstream:
    """One configured backend (OpenRouter / V-API / ...) behind the proxy.

    A request is routed to an upstream by the first segment of its model id
    (``openrouter/qwen/...`` -> upstream ``openrouter``); the matched prefix is
    stripped before forwarding so the upstream sees its native model id.
    """

    def __init__(
        self,
        name: str,
        client: AsyncOpenAI | OpenAI,
        reasoning_effort: str | None,
    ) -> None:
        self.name = name
        self.client = client
        self.reasoning_effort = reasoning_effort


def _resolve_api_key(spec: dict) -> str:
    """Resolve a route's API key from ``api_key`` > ``api_key_env`` > ``key_file``."""
    key = spec.get("api_key")
    if key:
        return key
    env_names = spec.get("api_key_env")
    if isinstance(env_names, str):
        env_names = [env_names]
    for env_name in env_names or ():
        env_val = os.getenv(env_name)
        if env_val:
            return env_val
    key_file = spec.get("key_file")
    if key_file and Path(key_file).exists():
        return _load_api_keys(key_file)[0]
    raise ValueError(
        f"route '{spec.get('name', spec)}': no API key "
        f"(set one of api_key / api_key_env / key_file)"
    )


def _build_upstream(
    name: str, spec: dict, async_client: bool, timeout_s: float
) -> Upstream:
    api_key = _resolve_api_key({**spec, "name": name})
    base_url = spec["base_url"]
    default_headers = {
        "HTTP-Referer": "https://github.com/nvidia-gear/CaP-X",
        "X-Title": "CaP-X",
    }
    cls = AsyncOpenAI if async_client else OpenAI
    client = cls(
        api_key=api_key,
        base_url=base_url,
        default_headers=default_headers,
        timeout=timeout_s,
    )
    logger.info("route '%s' -> %s (effort=%s)", name, base_url, spec.get("reasoning_effort"))
    return Upstream(name=name, client=client, reasoning_effort=spec.get("reasoning_effort"))


def _proxy_error(exc: Exception, upstream: Upstream | None, native_model: str | None) -> HTTPException:
    """Convert upstream/proxy exceptions into a diagnosable HTTP error."""
    route = upstream.name if upstream is not None else "unresolved"
    model = native_model or "unresolved"
    response = getattr(exc, "response", None)
    upstream_status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
    body_prefix = ""
    if response is not None:
        try:
            body_prefix = (response.text or "")[:800]
        except Exception:  # noqa: BLE001 - best-effort diagnostics only
            body_prefix = ""
    detail = (
        f"upstream route={route!r} model={model!r} failed: "
        f"{type(exc).__name__}: {exc}"
    )
    if upstream_status:
        detail += f" (upstream_status={upstream_status})"
    if body_prefix:
        detail += f" body_prefix={body_prefix!r}"
    return HTTPException(status_code=502, detail=detail)


def create_app(
    routes: dict[str, Upstream],
    default_name: str,
    async_client: bool = True,
) -> FastAPI:
    if default_name not in routes:
        raise ValueError(f"default route '{default_name}' not in routes {list(routes)}")

    def _route_for(model: str) -> tuple[Upstream, str]:
        """Pick the upstream + native model id for an incoming model string."""
        prefix = model.split("/", 1)[0]
        if "/" in model and prefix in routes:
            return routes[prefix], model[len(prefix) + 1:]
        # Backward compat: a bare "openrouter/..." with no matching route still
        # gets the prefix stripped and goes to the default upstream.
        if model.startswith("openrouter/") and "openrouter" not in routes:
            return routes[default_name], model[len("openrouter/"):]
        return routes[default_name], model

    app = FastAPI(title="LLM Proxy", version="2.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if async_client:

        @app.post("/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            upstream: Upstream | None = None
            native_model: str | None = None
            try:
                client_kwargs = request.model_dump(exclude_none=True)
                # reasoning 经 extra_body 透传(非 OpenAI 标准 kwarg);不传则按路由默认注入。
                client_kwargs.pop("reasoning", None)

                upstream, native_model = _route_for(client_kwargs.get("model", ""))
                client_kwargs["model"] = native_model
                extra_body = _build_extra_body(request, upstream.reasoning_effort)
                client = upstream.client

                if request.stream:
                    client_kwargs["stream"] = True
                    response = await client.chat.completions.create(**client_kwargs, extra_body=extra_body)

                    async def event_stream():
                        async for chunk in response:
                            data = chunk.model_dump_json()
                            yield f"data: {data}\n\n"
                        yield "data: [DONE]\n\n"

                    return StreamingResponse(event_stream(), media_type="text/event-stream")

                client_kwargs["stream"] = False
                response = await client.chat.completions.create(**client_kwargs, extra_body=extra_body)

                choices = [
                    ChatCompletionResponseChoice(
                        index=c.index,
                        message=Message(role=c.message.role, content=c.message.content),
                        finish_reason=c.finish_reason,
                    )
                    for c in response.choices
                ]

                return ChatCompletionResponse(
                    id=response.id, created=response.created, model=response.model, choices=choices
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.exception("LLM proxy request failed")
                raise _proxy_error(e, upstream, native_model) from e

    else:

        @app.post("/chat/completions", response_model=ChatCompletionResponse)
        def chat_completions(request: ChatCompletionRequest):
            upstream: Upstream | None = None
            native_model: str | None = None
            try:
                client_kwargs = request.model_dump(exclude_none=True)
                client_kwargs.pop("reasoning", None)

                upstream, native_model = _route_for(client_kwargs.get("model", ""))
                client_kwargs["model"] = native_model
                extra_body = _build_extra_body(request, upstream.reasoning_effort)

                client_kwargs["stream"] = False
                response = upstream.client.chat.completions.create(**client_kwargs, extra_body=extra_body)

                choices = [
                    ChatCompletionResponseChoice(
                        index=c.index,
                        message=Message(role=c.message.role, content=c.message.content),
                        finish_reason=c.finish_reason,
                    )
                    for c in response.choices
                ]

                return ChatCompletionResponse(
                    id=response.id, created=response.created, model=response.model, choices=choices
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.exception("LLM proxy request failed")
                raise _proxy_error(e, upstream, native_model) from e

    @app.get("/health")
    async def health():
        return {"status": "ok", "routes": list(routes), "default": default_name}

    return app


def _routes_from_file(path: str, async_client: bool, timeout_s: float) -> tuple[dict[str, Upstream], str]:
    """Build the routing table from a JSON config.

    Schema::

        {
          "default": "vapi",
          "routes": {
            "openrouter": {"base_url": "...", "api_key_env": "OPENROUTER_API_KEY",
                            "key_file": ".openrouterkey", "reasoning_effort": "low"},
            "vapi":       {"base_url": "...", "api_key_env": "V_API_KEY",
                            "reasoning_effort": "none"}
          }
        }
    """
    spec = json.loads(Path(path).read_text())
    route_specs = spec.get("routes") or {}
    if not route_specs:
        raise ValueError(f"routes file '{path}' has no 'routes'")
    routes = {
        name: _build_upstream(name, rspec, async_client, timeout_s)
        for name, rspec in route_specs.items()
    }
    default_name = spec.get("default") or next(iter(routes))
    return routes, default_name


def main(
    key_file: str = ".openrouterkey",
    api_key: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8111,
    base_url: str = "https://openrouter.ai/api/v1/",
    async_client: bool = True,
    reasoning_effort: str | None = "low",
    timeout_s: float = 600.0,
    routes_file: str | None = None,
    dotenv_path: str = ".env",
):
    """
    Start the LLM Proxy Server.

    Two ways to configure upstreams:

    * ``--routes-file routes.json``: multi-upstream routing keyed by model prefix
      (e.g. ``openrouter/...`` -> OpenRouter, ``vapi/...`` -> V-API). This lets a
      single proxy serve several backends at once.
    * ``--api-key``/``--base-url``/``--reasoning-effort``: legacy single-upstream
      mode. A model id is forwarded as-is, except a leading ``openrouter/`` is
      stripped for backward compatibility.

    ``reasoning_effort`` 是未显式传 reasoning 的请求的默认 thinking 强度(max/xhigh/high/
    medium/low/minimal/none/off);``none``/``off`` 表示不注入。``timeout_s`` 是到上游的读超时。
    """
    _load_dotenv(dotenv_path)
    if routes_file:
        routes, default_name = _routes_from_file(routes_file, async_client, timeout_s)
        logger.info("loaded %d route(s) from %s, default='%s'", len(routes), routes_file, default_name)
    else:
        if api_key is None:
            api_key = _load_api_keys(key_file)[0]
            logger.info(f"Loaded API key from {key_file}")
        routes = {
            "default": _build_upstream(
                "default",
                {"base_url": base_url, "api_key": api_key, "reasoning_effort": reasoning_effort},
                async_client,
                timeout_s,
            )
        }
        default_name = "default"

    app = create_app(routes=routes, default_name=default_name, async_client=async_client)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
