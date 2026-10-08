"""본실험: 숨은 취향이 서로 다른(일부는 정반대인) 가상 작가 A·B.

질문 셋:
1. 어댑터는 교정만 보고 각자의 숨은 취향을 찾아내는가 (학습 쪽은 숨은 취향을 한 번도 보지 않는다)
2. 처음 보는 장면에서 가상 작가가 고칠 양이 줄어드는가
3. LoRA처럼 갈아 끼우면: 내 어댑터는 돕고, 남의 어댑터는 해가 되는가

  PA_HOME=data/exp python3 demo/experiment.py
단계 결과를 data/exp/results.json에 쌓고, 끝난 단계는 건너뛴다(끊겨도 이어서 돈다).
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapter import learn, llm, metrics, pipeline, rules, store  # noqa: E402

CHARS = 700
WORKERS = int(os.environ.get("PA_WORKERS", "6"))
ROUNDS = int(os.environ.get("ROUNDS", "4"))

TASTES = {
    "a": """1. 감정 단어(분노·슬픔·두려움·배신감·당황 등)를 직접 쓰지 않는다. 그 문장을 몸의 작은 행동 하나로 바꾼다.
2. 대사 뒤의 발화 지문('~라고 말했다', '~가 외쳤다', '~가 물었다')을 지운다. 누가 말하는지 앞뒤로 분명하면 대사만 남긴다.
3. 장면의 마지막 문장은 열 글자 안팎의 단문 한 줄로 끊는다.
4. '그러나', '하지만', '그리고'로 시작하는 문장에서 그 접속사를 지운다.
5. 숫자는 아라비아 숫자 대신 한글로 쓴다(3명 → 세 명, 10걸음 → 열 걸음).""",
    "b": """1. 대사 뒤에는 발화 지문을 붙인다. 지문이 없는 대사에는 '~가 말했다', '~가 물었다' 같은 짧은 지문을 더한다.
2. 숫자는 한글 대신 아라비아 숫자로 쓴다(세 명 → 3명, 열 걸음 → 10걸음).
3. 의성어·의태어(쾅, 딸랑, 스르륵, 덜덜 등)를 지운다. 문장이 어색해지면 그 부분만 풀어 쓴다.
4. 말줄임표(…, ……)를 마침표로 바꾼다.
5. '그', '그녀' 같은 3인칭 대명사를 인물 이름으로 바꾼다.""",
}

TRAIN = [
    "회귀한 검사가 자신을 죽였던 사형과 수련장에서 처음 다시 마주친다.",
    "헌터 협회 등급 측정에서 F급 판정을 받은 주인공이 측정기 오류를 눈치챈다.",
    "악녀로 빙의한 주인공이 약혼자 황태자에게 파혼을 먼저 꺼낸다.",
    "무림맹 회의장에서 말단 호위무사가 맹주의 독살 시도를 막는다.",
    "던전 3층에서 파티원 4명이 주인공을 미끼로 버리고 도망친다.",
    "망한 식당을 물려받은 주인공 앞에 미슐랭 심사원이 손님으로 앉는다.",
    "아카데미 입학시험 대련에서 평민 주인공이 공작가 장남을 10초 만에 쓰러뜨린다.",
    "시스템 창이 '72시간 안에 첫 살인을 하지 않으면 사망'이라고 띄운다.",
    "전생에 자신을 배신한 재벌 회장의 비서로 면접장에 들어간다.",
    "편의점 야간 알바생이 새벽 3시에 들어온 손님이 어제 죽은 사장이라는 걸 알아챈다.",
    "용병단장이 7년 만에 고향 마을로 돌아오자 마을이 통째로 비어 있다.",
    "황궁 연회에서 독이 든 잔을 마시려는 황제를 시녀가 일부러 넘어지며 막는다.",
    "웹툰 작가가 자기 웹툰 속 악역의 몸으로 눈을 뜬다.",
    "검술 대회 결승, 눈먼 검객이 상대의 숨소리만으로 칼끝을 피한다.",
    "이혼 서류를 내민 아내 앞에서 남편의 손목 시계가 거꾸로 돌기 시작한다.",
    "마왕성 지하 감옥에서 용사 파티의 성녀가 마왕과 거래를 제안한다.",
]

TEST = [
    "멸문한 가문의 막내가 원수 가문의 연회에 하녀로 잠입한다.",
    "S급 게이트가 서울 한복판에 열리고, 은퇴한 헌터가 딸의 학교로 달려간다.",
    "회귀 직후 주인공이 내일 폭락할 코인을 산 친구에게 전화를 건다.",
    "마탑 졸업 심사에서 5년 동안 마법을 못 쓰던 견습생의 손끝에 불꽃이 인다.",
    "계약 결혼 1년 차, 대공이 처음으로 저녁 식탁에 늦지 않고 돌아온다.",
    "조선으로 떨어진 응급의학과 의사가 2시간째 숨을 못 쉬는 세자 앞에 선다.",
    "야구 9회 말 2아웃, 방출 직전의 대타가 타석에 들어선다.",
    "무당의 손녀가 이사 온 첫날 밤 새 집 천장에서 발소리를 듣는다.",
]

JUDGE = (
    "너는 실험 채점자다. '숨은 취향' 목록의 항목마다, '배운 규칙'이 그 취향을 잡았는지 판정한다. "
    "잡음 = 같은 방향의 규칙이 있음, 일부 = 방향은 맞지만 범위가 다르거나 뭉뚱그려짐, 못 잡음 = 없음. "
    "숨은 취향과 반대 방향인 규칙이 있으면 '반대'로 표시한다."
)
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hidden": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["잡음", "일부", "못 잡음", "반대"]},
                    "rule": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["hidden", "verdict", "rule", "why"],
            },
        }
    },
    "required": ["items"],
}


def name(t: str) -> str:
    return f"taste-{t}"


def other(t: str) -> str:
    return "b" if t == "a" else "a"


def edit(t: str, draft: str) -> str:
    system = (
        "너는 아래 '취향'을 가진 웹소설 작가다. 받은 초고를 퇴고한다. 취향에 해당하는 곳만 고치고, "
        "해당하지 않는 문장은 한 글자도 바꾸지 않는다. 고친 본문 전체만 출력한다.\n\n# 취향\n" + TASTES[t]
    )
    return str(llm.ask(draft, system=system, role=f"editor-{t}"))


def sign_test(wins: int, losses: int) -> float:
    """양측 부호 검정 p값. 동률은 뺀다."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def results_path() -> Path:
    return store.home() / "results.json"


def load() -> dict:
    p = results_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save(r: dict) -> None:
    results_path().write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")


def phase_train_data() -> None:
    jobs = []
    for t in TASTES:
        if not store.versions(name(t)):
            store.set_current(name(t), store.save_version(name(t), f"# 문체 어댑터: {name(t)}\n빈 상태에서 시작.\n"))
        seen = {c["brief"] for c in store.corrections(name(t))}
        jobs += [(t, b) for b in TRAIN if b not in seen]

    def run(job: tuple[str, str]) -> None:
        t, brief = job
        meta, _ = pipeline.write(name(t), brief, chars=CHARS, version="v0", examples_on=False)
        row = pipeline.record(meta, edit(t, meta["draft"]), source=f"가상 작가 {t.upper()}")
        print(f"  {t.upper()} 교정 {row['id']}: 고친 양 {row['edit_cost']:.3f}", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(run, jobs))


def phase_learn(r: dict) -> None:
    def run(t: str) -> list[dict]:
        out = []
        for _ in range(ROUNDS):
            res = learn.learn_round(name(t), k=3, repeats=2)
            verdict = f"채택 → {res['to']}" if res["accepted"] else "기각"
            print(
                f"  {t.upper()} {res['from']} 후보 {verdict} · 위치 {res['base_loc']} → {res['cand_loc']}"
                f" · 글자 {res['base_gain']} → {res['cand_gain']} (이김 {res['wins']} 짐 {res['losses']})",
                flush=True,
            )
            out.append(res)
        return out

    with ThreadPoolExecutor(max_workers=2) as pool:
        histories = dict(zip(TASTES, pool.map(run, TASTES), strict=True))
    r["learn"] = histories
    r["final"] = {t: store.current_version(name(t)) for t in TASTES}
    save(r)


def arms_for(t: str, final: dict[str, str]) -> dict[str, tuple[str, str, bool]]:
    """갈래 이름 → (어댑터 이름, 버전, 교정 예시 켬)."""
    o = other(t)
    return {
        "빈 어댑터": (name(t), "v0", False),
        "내 어댑터": (name(t), final[t], False),
        "남의 어댑터": (name(o), final[o], False),
        "교정 예시만": (name(t), "v0", True),
        "내 어댑터+예시": (name(t), final[t], True),
    }


def phase_test(r: dict) -> None:
    jobs = [(t, arm, i) for t in TASTES for arm in arms_for(t, r["final"]) for i in range(len(TEST))]

    def run(job: tuple[str, str, int]) -> dict:
        t, arm, i = job
        adapter, version, ex = arms_for(t, r["final"])[arm]
        meta, _ = pipeline.write(adapter, TEST[i], chars=CHARS, version=version, examples_on=ex)
        final = edit(t, meta["draft"])
        actual = metrics.changed(meta["draft"], final)
        predicted = {int(p["n"]) for p in meta["prediction"]}
        return {
            "user": t,
            "arm": arm,
            "scene": i,
            "edit_cost": metrics.edit_cost(meta["draft"], final),
            "hit": metrics.hit(predicted, actual),
            "draft": meta["draft"],
            "final": final,
        }

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(pool.map(run, jobs))
    r["test"] = rows
    save(r)


def phase_judge(r: dict) -> None:
    judged = {}
    for t in TASTES:
        learned = rules.parse(store.load_adapter(name(t))[1]).render()
        prompt = f"# 숨은 취향\n{TASTES[t]}\n\n# 배운 규칙\n{learned}"
        judged[t] = llm.ask(prompt, system=JUDGE, role="judge", schema=JUDGE_SCHEMA, model="opus")["items"]  # type: ignore[index]
    r["judge"] = judged
    save(r)


def summarize(r: dict) -> dict:
    out: dict = {}
    for t in TASTES:
        rows = [x for x in r["test"] if x["user"] == t]
        by_arm: dict[str, dict[int, float]] = {}
        for x in rows:
            by_arm.setdefault(x["arm"], {})[x["scene"]] = x["edit_cost"]
        base = by_arm["빈 어댑터"]
        arms = {}
        for arm, scores in by_arm.items():
            wins = sum(scores[i] < base[i] for i in scores)
            losses = sum(scores[i] > base[i] for i in scores)
            hits = [x["hit"] for x in rows if x["arm"] == arm]
            prec = [h["precision"] for h in hits if h["precision"] is not None]
            rec = [h["recall"] for h in hits if h["recall"] is not None]
            arms[arm] = {
                "edit_cost": round(mean(scores.values()), 4),
                "vs_base": f"{wins}승 {losses}패",
                "p": round(sign_test(wins, losses), 4),
                "precision": round(mean(prec), 3) if prec else None,
                "recall": round(mean(rec), 3) if rec else None,
            }
        out[t] = arms
    return out


def main() -> None:
    os.environ.setdefault("PA_HOME", str(Path(__file__).resolve().parent.parent / "data" / "exp"))
    r = load()
    print(f"1) 학습용 교정: 작가 A·B × 장면 {len(TRAIN)}개", flush=True)
    phase_train_data()
    if "learn" not in r:
        print(f"2) 학습 {ROUNDS}회씩", flush=True)
        phase_learn(r)
    if "test" not in r:
        print(f"3) 처음 보는 장면 {len(TEST)}개 × 5갈래 × 작가 2명", flush=True)
        phase_test(r)
    if "judge" not in r:
        print("4) 숨은 취향 대조 채점", flush=True)
        phase_judge(r)
    r["summary"] = summarize(r)
    r["cost"] = learn.cost_summary()
    save(r)
    for t, arms in r["summary"].items():
        print(f"\n작가 {t.upper()} (최종 {r['final'][t]})")
        for arm, s in arms.items():
            print(
                f"  {arm:10} 고친 양 {s['edit_cost']:.4f} · 빈 어댑터 대비 {s['vs_base']} (p={s['p']}) · 예측 정밀 {s['precision']} 재현 {s['recall']}"
            )
        print("  숨은 취향: " + " / ".join(f"{j['hidden']}:{j['verdict']}" for j in r["judge"][t]))
    print(f"\n비용: {r['cost']}")


if __name__ == "__main__":
    main()
