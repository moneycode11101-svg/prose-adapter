"""배우는 과정: 학습용 교정으로 어댑터 편집안을 만들고, 숨겨 둔 검증용 교정에서 기존 버전보다 나을 때만 채택한다.

관문의 점수는 '교정 재현': 검증용 초고를 각 어댑터로 퇴고시켜 사용자의 실제 교정본과 맞춰 본다.
- 위치(loc): 사용자가 고친 문장을 고치고, 둔 문장은 두었나. 채택의 주 기준.
- 글자(gain): 사용자 교정본에 글자로 얼마나 가까워졌나. 엉뚱하게 고치는 것을 막는 하한.
내용이 고정된 같은 초고를 고치므로 LLM 심사위원의 취향이 끼지 않는다.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from statistics import mean

from . import llm, metrics, pipeline, prompts, rules, store

MIN_TRAIN = 3
MIN_VAL = 2
# 위치 점수가 이만큼은 올라야 채택. 흔들림은 반복 평균과 '이긴 항목 > 진 항목'으로 한 번 더 거른다.
LOC_MARGIN = 0.05
# 글자 유사도는 사용자 교정량의 이 비율까지 떨어져도 봐준다. 같은 방향의 바꿔 쓰기는 단어가 달라 유사도가 조금 내려간다.
CHAR_SLACK = 0.5


def split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """작품(없으면 교정 id) 단위 해시 순서로 정확히 3분의 1을 검증용으로 숨긴다. 같은 작품이 양쪽에 섞이지 않게.

    해시를 3으로 나눈 나머지로 고르던 때는 16건 중 3건만 검증으로 가는 쏠림이 있었다(본실험 1차).
    """
    usable = [r for r in rows if r["draft"].strip() != r["final"].strip()]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in usable:
        groups[r.get("work") or r["id"]].append(r)
    order = sorted(groups, key=lambda k: hashlib.sha1(k.encode()).hexdigest())
    target = max(MIN_VAL, round(len(usable) / 3))
    val: list[dict] = []
    for key in order:
        if len(val) >= target:
            break
        val += groups[key]
    train = [r for r in usable if r not in val]
    return train, val


def rule_stats(rows: list[dict]) -> dict[str, dict]:
    """규칙마다: 예측에 쓰인 문장 수, 그중 실제로 고쳐진 수. dot의 '경로 점수' 자리."""
    stats: dict[str, dict] = defaultdict(lambda: {"cited": 0, "hit": 0})
    for r in rows:
        changed = set(r.get("changed") or [])
        for p in r.get("prediction") or []:
            s = stats[p["rule"]]
            s["cited"] += 1
            s["hit"] += int(p["n"]) in changed
    return dict(stats)


def _render_train(rows: list[dict]) -> str:
    blocks = []
    for r in rows:
        predicted = {int(p["n"]) for p in r.get("prediction") or []}
        changed = set(r.get("changed") or [])
        sents = metrics.sentences(r["draft"])
        lines = [f"[{r['id']}] 장면: {r.get('brief', '')[:100]} · 고친 양 {r['edit_cost']}"]
        lines += [
            f"- 초고: {a or '(없음)'}\n  교정: {b or '(삭제)'}" for a, b in metrics.hunks(r["draft"], r["final"], 8)
        ]
        missed = sorted(changed - predicted)
        wrong = sorted(predicted - changed)
        if r.get("prediction") is not None and missed:
            lines.append(f"- 예측 못 한 고침: 문장 {missed[:8]}")
        if wrong:
            lines.append(
                "- 예측했지만 사용자가 그대로 둔 문장: "
                + " | ".join(sents[n - 1] for n in wrong[:4] if n <= len(sents))
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def propose(adapter: rules.Adapter, train: list[dict], k: int = 3) -> dict:
    stats = rule_stats(train)
    stat_lines = "\n".join(
        f"{rid}: 예측에 쓰인 문장 {s['cited']}, 실제로 고쳐짐 {s['hit']}" for rid, s in stats.items()
    )
    prompt = (
        prompts.section("현재 어댑터 (관찰 포함 전체)", adapter.render())
        + prompts.section("규칙별 예측 기록", stat_lines or "(아직 없음)")
        + prompts.section("학습용 교정", _render_train(train))
    )
    out = llm.ask(prompt, system=prompts.PROPOSER.format(k=k), role="propose", schema=prompts.PROPOSE_SCHEMA)
    return dict(out)  # type: ignore[arg-type]


def _arm(injected: str, row: dict) -> tuple[float, float]:
    output = pipeline.revise(injected, row["draft"])
    loc = metrics.location(output, row["draft"], row["final"])
    return metrics.gain(output, row["draft"], row["final"]), loc if loc is not None else 0.0


def replay(
    injected_by_arm: dict[str, str], val: list[dict], repeats: int = 1, workers: int = 4
) -> dict[str, dict[str, list[float]]]:
    """각 어댑터로 검증용 초고를 퇴고시켜 항목별 gain·loc(반복 평균)을 돌려준다.

    예시·작품 상태는 넣지 않는다: 어댑터만의 효과를 잰다.
    """
    jobs = [(arm, i) for arm in injected_by_arm for i in range(len(val)) for _ in range(repeats)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda job: _arm(injected_by_arm[job[0]], val[job[1]]), jobs))
    raw: dict[str, list[list[tuple[float, float]]]] = {arm: [[] for _ in val] for arm in injected_by_arm}
    for (arm, i), pair in zip(jobs, results, strict=True):
        raw[arm][i].append(pair)
    return {
        arm: {
            "gain": [round(mean(p[0] for p in per), 4) for per in items],
            "loc": [round(mean(p[1] for p in per), 4) for per in items],
        }
        for arm, items in raw.items()
    }


def learn_round(name: str, k: int = 3, repeats: int = 1) -> dict:
    train, val = split(store.corrections(name))
    if len(train) < MIN_TRAIN or len(val) < MIN_VAL:
        raise SystemExit(f"교정이 부족함: 학습 {len(train)} / 검증 {len(val)} (최소 {MIN_TRAIN}/{MIN_VAL})")
    version, text = store.load_adapter(name)
    base = rules.parse(text)
    proposal = propose(base, train, k)
    cand, log = rules.apply_edits(base, proposal["edits"])
    scores = replay({"base": base.injected(), "cand": cand.injected()}, val, repeats)
    b, c = scores["base"], scores["cand"]
    wins = sum(x > y for x, y in zip(c["loc"], b["loc"], strict=True))
    losses = sum(x < y for x, y in zip(c["loc"], b["loc"], strict=True))
    slack = CHAR_SLACK * mean(r["edit_cost"] for r in val)
    accepted = (
        mean(c["loc"]) > mean(b["loc"]) + LOC_MARGIN and wins > losses and mean(c["gain"]) >= mean(b["gain"]) - slack
    )
    new_version = None
    if accepted:
        touched = {e.get("rule_id") for e in proposal["edits"]} | {
            r.id for r in cand.rules if r.id not in {x.id for x in base.rules}
        }
        for r in cand.rules:
            if r.id in touched and r.status == "가설":
                r.status = "채택"  # 관문을 통과한 가설
        cand.preamble = base.preamble
        new_version = store.save_version(name, cand.render())
        store.set_current(name, new_version)
    result = {
        "at": store.now_iso(),
        "from": version,
        "to": new_version,
        "accepted": accepted,
        "train": len(train),
        "val": len(val),
        "base_gain": round(mean(b["gain"]), 4),
        "cand_gain": round(mean(c["gain"]), 4),
        "base_loc": round(mean(b["loc"]), 3),
        "cand_loc": round(mean(c["loc"]), 3),
        "slack": round(slack, 4),
        "wins": wins,
        "losses": losses,
        "edits": log,
        "analysis": proposal["analysis"],
    }
    store.add_history(name, _history_entry(result))
    return result


def _history_entry(r: dict) -> str:
    verdict = f"채택 → {r['to']}" if r["accepted"] else "기각"
    edits = "\n".join(f"  - {e}" for e in r["edits"]) or "  - (편집 없음)"
    return (
        f"## {r['at']} · {r['from']} 후보 · {verdict}\n"
        f"- 검증 {r['val']}건(학습 {r['train']}건): 위치 {r['base_loc']} → {r['cand_loc']}, "
        f"글자 gain {r['base_gain']} → {r['cand_gain']} (하한 −{r['slack']}), 이김 {r['wins']} / 짐 {r['losses']}\n"
        f"- 편집:\n{edits}\n"
        f"- 분석: {r['analysis'].strip()}"
    )


def evaluate(
    name: str, version_list: list[str], repeats: int = 1, all_rows: bool = False
) -> dict[str, dict[str, float]]:
    """여러 버전을 같은 검증 세트에서 비교: 기본형(v0)·수동 특화(v1)·학습 특화(그 뒤) 실험."""
    train, val = split(store.corrections(name))
    rows = train + val if all_rows else val
    arms = {v: rules.parse(store.load_adapter(name, v)[1]).injected() for v in version_list}
    scores = replay(arms, rows, repeats)
    return {v: {"loc": round(mean(s["loc"]), 3), "gain": round(mean(s["gain"]), 4)} for v, s in scores.items()}


def cost_summary() -> dict:
    calls = store.read_jsonl(store.home() / "calls.jsonl")
    by_role: dict[str, float] = defaultdict(float)
    for c in calls:
        by_role[c["role"]] += c.get("usd") or 0
    return {
        "calls": len(calls),
        "usd": round(sum(by_role.values()), 3),
        "by_role": {k: round(v, 3) for k, v in by_role.items()},
    }
