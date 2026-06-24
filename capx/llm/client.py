"""LLM client utilities for querying language models.

Extracted from capx/utils/launch_utils.py to separate LLM query logic
from launch/config utilities.
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
import os
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

if TYPE_CHECKING:
    from capx.envs.launch import LaunchArgs

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

GPT_MODELS = [
    "openai/gpt-5.4",
    "openai/o4-mini",
]
VLM_MODELS = [
    "google/gemini-3.1-pro-preview",
    "google/gemini-2.5-flash-lite",
    "anthropic/claude-opus-4-5",
    "anthropic/claude-haiku-4-5",
    "openai/gpt-5.4",
    "openai/o1",
    "openai/o4-mini",
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-r1",
    "qwen/qwen3.5-122b-a10b",
    "moonshotai/kimi-k2",
    "openrouter/qwen/qwen3.6-plus",
]
CLAUDE_MODELS = ["anthropic/claude-opus-4-5", "anthropic/claude-haiku-4-5"]
OSS_MODELS = [
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-r1",
    "qwen/qwen3.5-122b-a10b",
    "moonshotai/kimi-k2",
]
OPENROUTER_MODELS = [
    "openrouter/google/gemini-2.5-pro-preview",
    "openrouter/google/gemini-2.5-flash-preview",
    "openrouter/anthropic/claude-sonnet-4",
    "openrouter/anthropic/claude-opus-4",
    "openrouter/deepseek/deepseek-r1",
    "openrouter/deepseek/deepseek-chat-v3-0324",
    "openrouter/openai/gpt-4.1",
    "openrouter/openai/o4-mini",
    "openrouter/meta-llama/llama-4-maverick",
    "openrouter/qwen/qwen3-235b-a22b",
    "openrouter/qwen/qwen3.6-plus",
]
OPENROUTER_SERVER_URL = "http://localhost:8110/chat/completions"

# 单次 HTTP 请求的读超时(秒)与重试上限,均可用环境变量覆盖。默认放大到 600s 以容忍慢
# 模型 / 长 thinking;重试用有上限的指数退避(替代旧的 240±90s 巨长 sleep)。
REQUEST_TIMEOUT = float(os.getenv("CAPX_LLM_TIMEOUT", "600"))
MAX_RETRIES = int(os.getenv("CAPX_LLM_MAX_RETRIES", "5"))
# 这些状态码视为可重试(429 限流、5xx 上游/代理错误、404 偶发路由抖动)。
_RETRYABLE_STATUS = (404, 429, 500, 502, 503, 504)


def _post_with_retries(
    server_url: str, headers: dict[str, str], payload: dict[str, Any]
) -> "requests.Response":
    """POST 到 LLM 端点,对网络异常与可重试状态码做有上限的指数退避重试。

    旧实现只在收到特定状态码时重试,``ReadTimeout`` / ``ConnectionError`` 等异常会直接
    抛出,导致整个 trial 作废;这里把异常也纳入重试,并给重试次数与退避加上限。
    """

    last_exc: Exception | None = None
    response: requests.Response | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                server_url, headers=headers, data=json.dumps(payload), timeout=REQUEST_TIMEOUT
            )
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            sleep_time = min(60.0, 5.0 * (2 ** attempt)) + random.uniform(0, 5)
            print(f"[query_model] 请求异常 {exc!r};{attempt + 1}/{MAX_RETRIES} 次,"
                  f"{sleep_time:.0f}s 后重试…")
            time.sleep(sleep_time)
            continue
        if response.status_code in _RETRYABLE_STATUS:
            sleep_time = min(60.0, 5.0 * (2 ** attempt)) + random.uniform(0, 5)
            print(f"[query_model] 状态码 {response.status_code}: {response.text[:200]};"
                  f"{attempt + 1}/{MAX_RETRIES} 次,{sleep_time:.0f}s 后重试…")
            time.sleep(sleep_time)
            continue
        return response
    if response is not None:
        return response  # 用尽重试仍是坏状态码:交给 raise_for_status 报错
    raise last_exc if last_exc is not None else RuntimeError("query_model: 重试用尽且无响应")

# ---------------------------------------------------------------------------
# Ensemble configuration
# ---------------------------------------------------------------------------

ENSEMBLE_CONFIGS = [
    ("openrouter/qwen/qwen3.6-plus", [0.1, 0.5, 0.9]),
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def is_openrouter_model(model: str) -> bool:
    """Return True if the model should be routed through the OpenRouter proxy."""
    return model.startswith("openrouter/") or model in OPENROUTER_MODELS


@dataclass
class ModelQueryArgs:
    """Arguments for querying a model."""

    model: str
    server_url: str
    api_key: str | None = None
    temperature: float = 0.2
    max_tokens: int = 4096
    reasoning_effort: str = "medium"
    debug: bool = False


def collapse_text_image_inputs(messages: list[dict]) -> list[dict]:
    """
    Collapse a list of messages with sequential text into a single text input, images are still in the same relative position
    """
    new_prompt = []
    current_text_input = ""
    for message in messages:
        if message["type"] == "text":
            current_text_input += message["text"] + "\n"
        else:
            if current_text_input != "":
                new_prompt.append({"type": "text", "text": current_text_input})
                current_text_input = ""
            new_prompt.append(message)
    if current_text_input != "":
        new_prompt.append({"type": "text", "text": current_text_input})
    return new_prompt


def _completions_to_responses_convert_prompt(prompt: list[dict]) -> list[dict]:
    """Convert completions api format to responses api format.

    Args:
        prompt: The prompt in completions api format

    Returns:
        The prompt in responses api format

    Switch prompt structure to api responses api format e.g.:
    From
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image in detail."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    To
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe the image in detail."},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            ]
        }
    ]
    """

    for message in prompt:
        for content in message["content"]:
            if type(content) == str:
                continue
            if content.get("type") == "text":
                content["type"] = "input_text"
                content["text"] = content.pop("text")

            elif content.get("type") == "image_url":
                content["type"] = "input_image"
                content["image_url"] = content["image_url"]["url"]
    return prompt


# ---------------------------------------------------------------------------
# Core query functions
# ---------------------------------------------------------------------------


def _write_raw_llm_log(
    *,
    server_url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    status_code: int | None,
    response_body: Any,
    elapsed_seconds: float,
) -> None:
    """Optionally dump raw LLM request/response pairs for debugging.

    Enable with:
        CAPX_LLM_RAW_LOG_DIR=outputs/llm_raw
    """

    log_dir = os.getenv("CAPX_LLM_RAW_LOG_DIR")
    if not log_dir:
        return

    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)

    safe_headers = dict(headers)
    if "Authorization" in safe_headers:
        safe_headers["Authorization"] = "Bearer <redacted>"

    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "server_url": server_url,
        "headers": safe_headers,
        "payload": payload,
        "status_code": status_code,
        "elapsed_seconds": elapsed_seconds,
        "response": response_body,
    }
    name = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{time.time_ns()}.json"
    (path / name).write_text(json.dumps(record, ensure_ascii=False, indent=2))


def query_model(args: "LaunchArgs | ModelQueryArgs", prompt: list[dict]) -> str:
    """Query vLLM server for code generation.

    Args:
        args: Configuration with server URL and model settings
        prompt: Full prompt containing environment observation and possibly multi-turn decision prompt
    Returns:
        Model response content
    """

    # Route OpenRouter models to the OpenRouter proxy server
    if is_openrouter_model(args.model):
        server_url = OPENROUTER_SERVER_URL
    else:
        server_url = args.server_url

    if args.model in GPT_MODELS:
        if "codex" in args.model:
            prompt = _completions_to_responses_convert_prompt(prompt)
            payload = {
                "model": args.model,
                "input": prompt,
            }
        else:
            payload = {
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "max_completion_tokens": args.max_tokens,  # Total completion tokens = reasoning + output tokens
                "messages": prompt,
            }
    elif is_openrouter_model(args.model):
        payload = {
            "model": args.model,
            "messages": prompt,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
    elif args.model in CLAUDE_MODELS:
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "messages": prompt,
        }
    elif args.model in OSS_MODELS:
        payload = {
            "model": args.model,
            "messages": prompt,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
    else:
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "messages": prompt,
        }
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    elif os.getenv("OPENAI_API_KEY") is not None and args.model in GPT_MODELS:
        headers["Authorization"] = f"Bearer {os.getenv('OPENAI_API_KEY')}"
    start_time = time.time()

    # 网络异常 + 可重试状态码都走有上限的指数退避(见 _post_with_retries)。
    response = _post_with_retries(server_url, headers, payload)

    end_time = time.time()
    print(f"Time taken to query model: {end_time - start_time:.2f} seconds")
    response.raise_for_status()
    body = response.json()
    _write_raw_llm_log(
        server_url=server_url,
        headers=headers,
        payload=payload,
        status_code=response.status_code,
        response_body=body,
        elapsed_seconds=end_time - start_time,
    )
    out = {}
    if args.debug:
        print(json.dumps(body, indent=2))
    try:
        if args.model in GPT_MODELS and "codex" in args.model:
            out["content"] = body["output_text"]
        else:
            out["content"] = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected response format: {body}") from exc
    if body.get("choices") is not None:
        out["reasoning"] = body.get("choices")[0].get("message").get("reasoning", None)
    else:
        out["reasoning"] = None
    return out  # type: ignore[return-value]


def query_model_streaming(
    args: "LaunchArgs | ModelQueryArgs",
    prompt: list[dict],
) -> Iterable[dict]:
    """Query model with streaming enabled, yielding partial responses.

    Yields dictionaries with:
      - {"type": "content_delta", "content": "partial text"}
      - {"type": "reasoning_delta", "content": "partial reasoning"} (if supported)
      - {"type": "done", "content": "full content", "reasoning": "full reasoning or None"}

    Args:
        args: Configuration with server URL and model settings
        prompt: Full prompt containing environment observation

    Yields:
        Partial response chunks as they arrive
    """
    if args.model in GPT_MODELS:
        payload = {
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "max_completion_tokens": args.max_tokens,
            "messages": prompt,
            "stream": True,
        }
    elif args.model in CLAUDE_MODELS:
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "messages": prompt,
            "stream": True,
        }
    else:
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "messages": prompt,
            "stream": True,
        }

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    elif os.getenv("OPENAI_API_KEY") is not None and args.model in GPT_MODELS:
        headers["Authorization"] = f"Bearer {os.getenv('OPENAI_API_KEY')}"

    full_content = ""
    full_reasoning = ""

    start_time = time.time()

    with requests.post(
        args.server_url,
        headers=headers,
        data=json.dumps(payload),
        timeout=200,
        stream=True,
    ) as response:
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        is_sse = "text/event-stream" in content_type
        is_json = "application/json" in content_type

        # If it's a regular JSON response (server doesn't support streaming),
        # fall back to non-streaming behavior
        if is_json and not is_sse:
            print("Warning: Server returned JSON instead of SSE stream, falling back to non-streaming")
            body = response.json()
            try:
                full_content = body["choices"][0]["message"]["content"]
                full_reasoning = body.get("choices", [{}])[0].get("message", {}).get("reasoning")
                if full_reasoning:
                    print(f"Reasoning extracted ({len(full_reasoning)} chars)")
                else:
                    print("No reasoning returned by model")
            except (KeyError, IndexError) as exc:
                raise RuntimeError(f"Unexpected response format: {body}") from exc

            yield {"type": "content_delta", "content": full_content}
            yield {
                "type": "done",
                "content": full_content,
                "reasoning": full_reasoning if full_reasoning else None,
            }
            end_time = time.time()
            print(f"Time taken to query model (streaming fallback): {end_time - start_time:.2f} seconds")
            return

        for line in response.iter_lines():
            if not line:
                continue

            line_str = line.decode("utf-8")

            # SSE format: "data: {...}" or "data: [DONE]"
            if line_str.startswith("data: "):
                data_str = line_str[6:]  # Remove "data: " prefix

                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})

                    # Handle content delta
                    content_delta = delta.get("content", "")
                    if content_delta:
                        full_content += content_delta
                        yield {"type": "content_delta", "content": content_delta}

                    # Handle reasoning delta (some APIs support this)
                    reasoning_delta = delta.get("reasoning", "")
                    if reasoning_delta:
                        full_reasoning += reasoning_delta
                        yield {"type": "reasoning_delta", "content": reasoning_delta}

                except json.JSONDecodeError:
                    continue
            else:
                # Try parsing as raw JSON (non-SSE format)
                try:
                    data = json.loads(line_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content_delta = delta.get("content", "")
                        if content_delta:
                            full_content += content_delta
                            yield {"type": "content_delta", "content": content_delta}
                except json.JSONDecodeError:
                    continue

    end_time = time.time()
    print(f"Time taken to query model (streaming): {end_time - start_time:.2f} seconds")
    if full_reasoning:
        print(f"Reasoning extracted ({len(full_reasoning)} chars)")
    else:
        print("No reasoning returned by model")

    yield {
        "type": "done",
        "content": full_content,
        "reasoning": full_reasoning if full_reasoning else None,
    }


def query_model_ensemble(
    args: "LaunchArgs | ModelQueryArgs",
    prompt: list[dict],
    synthesis_model: str = "openrouter/qwen/qwen3.6-plus",
    is_multiturn = False
) -> dict[str, Any]:
    """Query 9 models (3 models x 3 temperatures) and synthesize final output."""

    def query_single(model: str, temp: float) -> dict:
        query_args = ModelQueryArgs(
            model=model,
            server_url=args.server_url,
            api_key=args.api_key,
            temperature=temp,
            max_tokens=args.max_tokens,
            reasoning_effort=getattr(args, "reasoning_effort", "medium"),
        )
        try:
            result = query_model(query_args, copy.deepcopy(prompt))
            return {"model": model, "temp": temp, "content": result["content"], "ok": True}
        except Exception as e:
            error_msg = str(e)
            print(f"[Multimodel Ensemble] {model} temp={temp} FAILED: {error_msg}")
            return {"model": model, "temp": temp, "content": error_msg, "ok": False}

    # Build all (model, temp) pairs and query in parallel
    tasks = [(m, t) for m, temps in ENSEMBLE_CONFIGS for t in temps]
    responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
        futures = {executor.submit(query_single, m, t): (m, t) for m, t in tasks}
        for future in concurrent.futures.as_completed(futures):
            resp = future.result()
            responses.append(resp)
            if resp['ok']:
                print(f"[Multimodel Ensemble] {resp['model']} temp={resp['temp']} ok={resp['ok']}")

    successful = [r for r in responses if r["ok"]]
    if not successful:
        # Print all errors for debugging
        print("\n=== All ensemble queries failed. Errors: ===")
        for r in responses:
            print(f"  {r['model']} temp={r['temp']}: {r['content']}")
        raise RuntimeError("All ensemble queries failed")

    # Build synthesis prompt
    original_text = ""
    for msg in prompt:
        if msg["role"] == "user":
            c = msg["content"]
            if isinstance(c, list):
                original_text += "".join(x.get("text", "") for x in c if isinstance(x, dict))
            elif isinstance(c, str):
                original_text += c

    candidates = "\n\n".join(
        f"--- Candidate ({r['model']}, temp={r['temp']}) ---\n{r['content']}"
        for r in successful
    )

    # Detect if this is a multiturn decision (candidates contain REGENERATE/FINISH)
    regenerate_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "REGENERATE" in r["content"])
    finish_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "FINISH" in r["content"])

    if is_multiturn:
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate responses for a multi-turn robot control task.

    DECISION ANALYSIS:
    - {regenerate_count} candidates voted REGENERATE
    - {finish_count} candidates voted FINISH

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach
    5. Combine best code ideas from REGENERATE candidates

    OUTPUT FORMAT (strict):
    - You may include brief reasoning first
    - Then output "REGENERATE" on its own line followed by exactly ONE fenced code block, OR output "FINISH" on its own line
    """
    else:
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate Python solutions into one optimal program.

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach

    OUTPUT FORMAT (strict):
    You may include reasoning before the fenced code block.
    Output ONLY ONE fenced code block (```python...```) containing the complete final solution.
    Do NOT include any other code blocks or code snippets outside this single block.
    """

    synthesis_user_prompt = f"""Synthesize the best solution.

    <original_task_description>
    {original_text}
    </original_task_description>

    <candidate_solutions>
    {candidates}
    </candidate_solutions>
    """

    synthesis_prompt = [
        {
            "role": "system",
            "content": synthesis_system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": synthesis_user_prompt,
                }
            ],
        },
    ]

    synth_args = ModelQueryArgs(
        model=synthesis_model,
        server_url=args.server_url,
        api_key=args.api_key,
        temperature=0.2,
        max_tokens=args.max_tokens,
    )
    final = query_model(synth_args, synthesis_prompt)

    # Build text content for saving
    candidates_txt = "\n\n".join(
        f"{'='*60}\nModel: {r['model']}\nTemperature: {r['temp']}\nSuccess: {r['ok']}\n{'='*60}\n{r['content']}"
        for r in responses
    )
    synthesis_txt = f"Model: {synthesis_model}\n\n"
    synthesis_txt += f"{'='*60}\nREASONING\n{'='*60}\n{final.get('reasoning') or '(none)'}\n\n"
    synthesis_txt += f"{'='*60}\nOUTPUT\n{'='*60}\n{final['content']}"

    return {
        "content": final["content"],
        "reasoning": final.get("reasoning"),
        "all_responses": responses,
        "ensemble_candidates_txt": candidates_txt,
        "ensemble_synthesis_txt": synthesis_txt,
    }


def query_single_model_ensemble(
    args: "LaunchArgs | ModelQueryArgs",
    prompt: list[dict],
    model: str,
    is_multiturn = False,
) -> dict[str, Any]:
    """Query the same model 9 times (with temperatures 0.1 to 0.9) and synthesize final output.

    Args:
        args: Configuration with server URL and model settings
        prompt: Full prompt containing environment observation and possibly multi-turn decision prompt
        model: The model to use for both candidate generation and synthesis

    Returns:
        Dictionary containing synthesized content, reasoning, all responses, and text artifacts
    """

    def query_single(temp: float) -> dict:
        query_args = ModelQueryArgs(
            model=model,
            server_url=args.server_url,
            api_key=args.api_key,
            temperature=temp,
            max_tokens=args.max_tokens,
            reasoning_effort=getattr(args, "reasoning_effort", "medium"),
        )
        try:
            result = query_model(query_args, copy.deepcopy(prompt))
            return {"model": model, "temp": temp, "content": result["content"], "ok": True}
        except Exception as e:
            error_msg = str(e)
            print(f"[Single Model Ensemble] {model} temp={temp} FAILED: {error_msg}")
            return {"model": model, "temp": temp, "content": error_msg, "ok": False}

    # Query same model with 9 different temperatures (0.1 to 0.9)
    temperatures = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
        futures = {executor.submit(query_single, t): t for t in temperatures}
        for future in concurrent.futures.as_completed(futures):
            resp = future.result()
            responses.append(resp)
            if resp['ok']:
                print(f"[Single Model Ensemble] {resp['model']} temp={resp['temp']} ok={resp['ok']}")

    successful = [r for r in responses if r["ok"]]
    if not successful:
        # Print all errors for debugging
        print("\n=== All single model ensemble queries failed. Errors: ===")
        for r in responses:
            print(f"  {r['model']} temp={r['temp']}: {r['content']}")
        raise RuntimeError("All single model ensemble queries failed")

    # Build synthesis prompt
    original_text = ""
    for msg in prompt:
        if msg["role"] == "user":
            c = msg["content"]
            if isinstance(c, list):
                original_text += "".join(x.get("text", "") for x in c if isinstance(x, dict))
            elif isinstance(c, str):
                original_text += c

    candidates = "\n\n".join(
        f"--- Candidate (temp={r['temp']}) ---\n{r['content']}"
        for r in successful
    )

    # Detect if this is a multiturn decision (candidates contain REGENERATE/FINISH)
    regenerate_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "REGENERATE" in r["content"])
    finish_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "FINISH" in r["content"])

    if is_multiturn:
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate responses for a multi-turn robot control task.

    DECISION ANALYSIS:
    - {regenerate_count} candidates voted REGENERATE
    - {finish_count} candidates voted FINISH

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach
    5. Combine best code ideas from REGENERATE candidates

    OUTPUT FORMAT (strict):
    - You may include brief reasoning first
    - Then output "REGENERATE" on its own line followed by exactly ONE fenced code block, OR output "FINISH" on its own line
    """
    else: # first generation has no REGEN/FINISH candidates
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate Python solutions into one optimal program.

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach

    OUTPUT FORMAT (strict):
    You may include reasoning before the fenced code block.
    Output ONLY ONE fenced code block (```python...```) containing the complete final solution.
    Do NOT include any other code blocks or code snippets outside this single block.
    """

    synthesis_user_prompt = f"""Synthesize the best solution.

    <original_task_description>
    {original_text}
    </original_task_description>

    <candidate_solutions>
    {candidates}
    </candidate_solutions>
    """

    synthesis_prompt = [
        {
            "role": "system",
            "content": synthesis_system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": synthesis_user_prompt,
                }
            ],
        },
    ]

    # Use the same model for synthesis
    synth_args = ModelQueryArgs(
        model=model,
        server_url=args.server_url,
        api_key=args.api_key,
        temperature=0.2,
        max_tokens=args.max_tokens,
    )
    final = query_model(synth_args, synthesis_prompt)

    # Build text content for saving
    candidates_txt = "\n\n".join(
        f"{'='*60}\nModel: {r['model']}\nTemperature: {r['temp']}\nSuccess: {r['ok']}\n{'='*60}\n{r['content']}"
        for r in responses
    )
    synthesis_txt = f"Model: {model}\n\n"
    synthesis_txt += f"{'='*60}\nREASONING\n{'='*60}\n{final.get('reasoning') or '(none)'}\n\n"
    synthesis_txt += f"{'='*60}\nOUTPUT\n{'='*60}\n{final['content']}"

    return {
        "content": final["content"],
        "reasoning": final.get("reasoning"),
        "all_responses": responses,
        "ensemble_candidates_txt": candidates_txt,
        "ensemble_synthesis_txt": synthesis_txt,
    }


# ---------------------------------------------------------------------------
# Backward-compatible aliases (underscore-prefixed names)
# ---------------------------------------------------------------------------

_query_model = query_model
_query_model_streaming = query_model_streaming
_query_model_ensemble = query_model_ensemble
_query_single_model_ensemble = query_single_model_ensemble
