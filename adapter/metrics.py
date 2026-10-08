"""정답 신호. 사용자가 실제로 고친 글이 정답이고, 거기에 얼마나 가까운지를 잰다."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

SENTENCE_END = re.compile(r"(?<=[.!?…。])(?=\s)|(?<=[.!?…][\"'”’」』])(?=\s)")


def norm(text: str) -> str:
    lines = [line.rstrip() for line in text.strip().splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines))


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b), autojunk=False).ratio()


def edit_cost(draft: str, final: str) -> float:
    """사용자가 고친 양: 0이면 그대로 받음, 1에 가까울수록 많이 고침."""
    return round(1 - similarity(draft, final), 4)


def gain(output: str, draft: str, final: str) -> float:
    """어댑터가 초고를 고친 결과가, 손대지 않은 초고보다 사용자의 실제 교정에 얼마나 더 가까운가."""
    return similarity(output, final) - similarity(draft, final)


def location(output: str, draft: str, final: str) -> float | None:
    """위치 일치: 사용자가 고친 문장을 고쳤나(+), 사용자가 둔 문장을 건드렸나(−). 사용자 고침 수로 나눈다.

    감정 → 몸짓처럼 같은 방향으로 고쳐도 단어가 다른 교정은 글자 유사도로는 손해로 잡힌다(2026-10-07 본실험 1차).
    위치는 단어와 상관없이 '어디를 고칠지 아는가'를 잰다. 손대지 않으면 0, 완벽하면 1.
    """
    user = changed(draft, final)
    if not user:
        return None
    mine = changed(draft, output)
    return (len(mine & user) - len(mine - user)) / len(user)


def sentences(text: str) -> list[str]:
    out: list[str] = []
    for line in norm(text).splitlines():
        if line.strip():
            out.extend(s.strip() for s in SENTENCE_END.split(line) if s.strip())
    return out


def changed(draft: str, final: str) -> set[int]:
    """초고 쪽 문장 번호(1부터) 중 사용자가 바꾸거나 지운 것."""
    matcher = SequenceMatcher(None, sentences(draft), sentences(final), autojunk=False)
    return {i + 1 for tag, i1, i2, _, _ in matcher.get_opcodes() if tag != "equal" for i in range(i1, i2)}


def hunks(draft: str, final: str, limit: int = 6) -> list[tuple[str, str]]:
    """바뀐 부분을 (초고, 교정) 문장 묶음으로. 예시와 학습 입력에 쓴다."""
    a, b = sentences(draft), sentences(final)
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    pairs = [
        (" ".join(a[i1:i2]), " ".join(b[j1:j2])) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal"
    ]
    return pairs[:limit]


def hit(predicted: set[int], actual: set[int]) -> dict:
    """dot의 예측 적중과 같은 자리: 어댑터가 '사용자가 고칠 것'을 미리 맞혔나."""
    both = len(predicted & actual)
    return {
        "predicted": len(predicted),
        "changed": len(actual),
        "both": both,
        "precision": round(both / len(predicted), 3) if predicted else None,
        "recall": round(both / len(actual), 3) if actual else None,
    }
