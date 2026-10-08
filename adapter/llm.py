"""claude -p를 도구 없이, 사용자 설정 없이 부른다. 모든 호출은 calls.jsonl에 비용과 함께 남는다.

같은 모델·같은 프롬프트 틀에서 어댑터만 바꿔 비교해야 하므로, 대화 맥락이 섞이지 않는 단발 호출만 쓴다.
"""

from __future__ import annotations

import json
import os
import subprocess
import time

from . import store

CLAUDE = os.environ.get("PA_CLAUDE", "claude")
MODEL = os.environ.get("PA_MODEL", "sonnet")


class LLMError(RuntimeError):
    pass


# 한도 응답(429)은 잠깐 뒤 풀리는 일이 있어(2026-10-07 시연 중 한 번) 두 번까지 기다렸다 다시 부른다. 그래도 막히면 그대로 던진다.
RETRY_WAITS = (30, 90)


def _run(args: list[str], prompt: str, role: str) -> dict:
    for wait in (*RETRY_WAITS, None):
        run = subprocess.run(args, input=prompt, capture_output=True, text=True, timeout=900)
        data = json.loads(run.stdout) if run.stdout.strip().startswith("{") else {}
        limited = data.get("api_error_status") == 429
        if limited and wait is not None:
            time.sleep(wait)
            continue
        if run.returncode != 0 or data.get("is_error") or not data:
            raise LLMError(
                f"{role}: claude 종료 {run.returncode}: {str(data.get('result') or run.stderr or run.stdout)[-400:]}"
            )
        return data
    raise AssertionError("도달하지 않음")


def ask(prompt: str, *, system: str, role: str, schema: dict | None = None, model: str | None = None) -> str | dict:
    model = model or MODEL
    args = [
        CLAUDE, "-p",
        "--model", model,
        "--system-prompt", system,
        "--tools", "",
        "--setting-sources", "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--output-format", "json",
    ]  # fmt: skip
    if schema:
        args += ["--json-schema", json.dumps(schema, ensure_ascii=False)]
    started = time.monotonic()
    data = _run(args, prompt, role)
    usage = data.get("usage", {})
    store.append_jsonl(
        store.home() / "calls.jsonl",
        {
            "at": store.now_iso(),
            "role": role,
            "model": model,
            "usd": data.get("total_cost_usd"),
            "seconds": round(time.monotonic() - started, 1),
            "input": usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0),
            "cache_read": usage.get("cache_read_input_tokens", 0),
            "output": usage.get("output_tokens", 0),
        },
    )
    if schema:
        out = data.get("structured_output")
        if out is None:
            raise LLMError(f"{role}: 구조화 출력 없음: {str(data.get('result'))[:200]}")
        return out
    return str(data["result"]).strip()
