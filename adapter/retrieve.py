"""지금 쓰려는 장면과 닮은 과거 교정을 골라 예시로 보여 준다. 임베딩 없이 글자 3-gram으로 충분하다."""

from __future__ import annotations

from collections import Counter
from math import sqrt

from . import metrics


def _grams(text: str) -> Counter[str]:
    flat = "".join(text.split())
    return Counter(flat[i : i + 3] for i in range(max(0, len(flat) - 2)))


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    dot = sum(v * b[k] for k, v in a.items() if k in b)
    na = sqrt(sum(v * v for v in a.values()))
    nb = sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def similar(query: str, pool: list[dict], k: int = 3, exclude: set[str] | None = None) -> list[dict]:
    q = _grams(query)
    scored = [
        (_cosine(q, _grams(c.get("brief", "") + " " + c["draft"])), c)
        for c in pool
        if c["id"] not in (exclude or set()) and c["draft"].strip() != c["final"].strip()
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]


def render_examples(pool: list[dict], max_hunks: int = 3) -> str:
    if not pool:
        return ""
    blocks = []
    for c in pool:
        pairs = metrics.hunks(c["draft"], c["final"], limit=max_hunks)
        lines = [f"- 초고: {a or '(없음)'}\n  교정: {b or '(삭제)'}" for a, b in pairs]
        blocks.append(f"[{c['id']}]\n" + "\n".join(lines))
    return "\n".join(blocks)
