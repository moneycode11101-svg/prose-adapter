"""실제로 쓰는 과정: 장면 → 초고 → 어댑터로 퇴고 → '사용자가 고칠 곳' 예측 → 저장. 확정 본문으로 작품 상태 갱신."""

from __future__ import annotations

from pathlib import Path

from . import llm, metrics, prompts, retrieve, rules, store


def revise(injected: str, draft: str, examples: str = "", state: str = "") -> str:
    prompt = (
        prompts.section("문체 어댑터", injected)
        + prompts.section("이 사용자가 실제로 고친 예 (초고 → 교정)", examples)
        + prompts.section("작품 상태", state)
        + prompts.section("초고", draft)
    )
    return str(llm.ask(prompt, system=prompts.REVISER, role="revise"))


def predict(injected: str, examples: str, text: str) -> list[dict]:
    prompt = (
        prompts.section("문체 어댑터", injected)
        + prompts.section("이 사용자가 실제로 고친 예", examples)
        + prompts.section("받은 글 (문장 번호)", prompts.numbered(metrics.sentences(text)))
    )
    out = llm.ask(prompt, system=prompts.PREDICTOR, role="predict", schema=prompts.PREDICT_SCHEMA)
    return list(out["items"])  # type: ignore[index]


def write(
    adapter: str,
    brief: str,
    work: str | None = None,
    chars: int = 1500,
    version: str | None = None,
    examples_on: bool = True,
) -> tuple[dict, Path]:
    version, text = store.load_adapter(adapter, version)
    injected = rules.parse(text).injected()
    pool = store.corrections(adapter) if examples_on else []
    examples = retrieve.render_examples(retrieve.similar(brief, pool))
    state = store.load_state(work)
    prompt = (
        prompts.section("문체 어댑터", injected)
        + prompts.section("이 사용자가 실제로 고친 예 (초고 → 교정)", examples)
        + prompts.section("작품 상태", state)
        + prompts.section("이번 장면", brief)
        + f"분량: 약 {chars}자.\n"
    )
    first = str(llm.ask(prompt, system=prompts.WRITER, role="write"))
    draft = revise(injected, first, examples, state)
    meta = {
        "id": store.new_id("d"),
        "at": store.now_iso(),
        "adapter": adapter,
        "version": version,
        "work": work,
        "brief": brief,
        "draft": draft,
        "prediction": predict(injected, examples, draft),
    }
    return meta, store.save_draft(meta)


def record(meta: dict, final: str, source: str = "사용자 교정") -> dict:
    """초고와 사용자가 고친 글 한 쌍 = 판정 한 건. corrections.jsonl에 추가만 한다."""
    draft = meta["draft"]
    actual = metrics.changed(draft, final)
    predicted = {int(p["n"]) for p in meta.get("prediction") or []}
    row = {
        "id": meta["id"],
        "at": store.now_iso(),
        "adapter": meta["adapter"],
        "version": meta.get("version"),
        "work": meta.get("work"),
        "brief": meta.get("brief", ""),
        "draft": draft,
        "final": final.strip(),
        "edit_cost": metrics.edit_cost(draft, final),
        "changed": sorted(actual),
        "prediction": meta.get("prediction") or [],
        "hit": metrics.hit(predicted, actual) if meta.get("prediction") is not None else None,
        "source": source,
    }
    store.append_jsonl(store.corrections_path(), row)
    return row


def update_state(work: str, final: str) -> Path:
    prompt = prompts.section("지금 작품 상태", store.load_state(work) or "(비어 있음)") + prompts.section(
        "확정된 본문", final
    )
    return store.save_state(work, str(llm.ask(prompt, system=prompts.STATE, role="state")))
