"""시연: 숨은 취향을 가진 가상 작가를 두고, 어댑터가 교정만 보고 그 취향을 배우는지 잰다.

학습 쪽(propose·관문·집필)은 숨은 취향을 한 번도 보지 않는다. 보는 것은 가상 작가가 고친 결과뿐이다.
1. v0(빈 어댑터)으로 장면 A를 쓰고 가상 작가가 고친다 → 교정 기록
2. learn 몇 번 → 관문을 통과한 버전만 남는다
3. 처음 보는 장면 B를 v0과 학습 버전으로 각각 쓰고 가상 작가가 고친다 → 고친 양 비교 (같은 장면·같은 예시 풀)

  PA_HOME=data/demo python3 demo/simulate.py
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapter import learn, llm, metrics, pipeline, rules, store  # noqa: E402

NAME = "webnovel"
CHARS = 700

HIDDEN = """1. 감정 단어(분노·슬픔·두려움·배신감·당황 등)를 직접 쓰지 않는다. 그 문장을 몸의 작은 행동 하나로 바꾼다.
2. 대사 뒤의 발화 지문('~라고 말했다', '~가 외쳤다', '~가 물었다')을 지운다. 누가 말하는지 앞뒤로 분명하면 대사만 남긴다.
3. 장면의 마지막 문장은 열 글자 안팎의 단문 한 줄로 끊는다.
4. '그러나', '하지만', '그리고'로 시작하는 문장에서 그 접속사를 지운다.
5. 숫자는 아라비아 숫자 대신 한글로 쓴다(3명 → 세 명, 10걸음 → 열 걸음)."""

EDITOR = (
    "너는 아래 '취향'을 가진 웹소설 작가다. 받은 초고를 퇴고한다. 취향에 해당하는 곳만 고치고, "
    "해당하지 않는 문장은 한 글자도 바꾸지 않는다. 고친 본문 전체만 출력한다.\n\n# 취향\n" + HIDDEN
)

SCENES_A = [
    "회귀한 검사가 자신을 죽였던 사형(師兄)과 수련장에서 처음 다시 마주친다.",
    "헌터 협회 등급 측정에서 F급 판정을 받은 주인공이 측정기 오류를 눈치챈다.",
    "악녀로 빙의한 주인공이 약혼자 황태자에게 파혼을 먼저 꺼낸다.",
    "무림맹 회의장에서 말단 호위무사가 맹주의 독살 시도를 막는다.",
    "던전 3층에서 파티원 4명이 주인공을 미끼로 버리고 도망친다.",
    "망한 식당을 물려받은 주인공 앞에 미슐랭 심사원이 손님으로 앉는다.",
    "아카데미 입학시험 대련에서 평민 주인공이 공작가 장남을 10초 만에 쓰러뜨린다.",
    "시스템 창이 '72시간 안에 첫 살인을 하지 않으면 사망'이라고 띄운다.",
    "전생에 자신을 배신한 재벌 회장의 비서로 면접장에 들어간다.",
]

SCENES_B = [
    "멸문한 가문의 막내가 원수 가문의 연회에 하녀로 잠입한다.",
    "S급 게이트가 서울 한복판에 열리고, 은퇴한 헌터가 딸의 학교로 달려간다.",
    "회귀 직후 주인공이 내일 폭락할 코인을 산 친구에게 전화를 건다.",
    "마탑 졸업 심사에서 5년 동안 마법을 못 쓰던 견습생의 손끝에 불꽃이 인다.",
    "계약 결혼 1년 차, 대공이 처음으로 저녁 식탁에 늦지 않고 돌아온다.",
    "조선으로 떨어진 응급의학과 의사가 2시간째 숨을 못 쉬는 세자 앞에 선다.",
]


def edit(draft: str) -> str:
    return str(llm.ask(draft, system=EDITOR, role="sim-editor"))


def write_and_edit(brief: str, version: str | None = None, examples_on: bool = True) -> tuple[dict, str]:
    meta, _ = pipeline.write(NAME, brief, chars=CHARS, version=version, examples_on=examples_on)
    return meta, edit(meta["draft"])


def phase_a() -> None:
    with ThreadPoolExecutor(max_workers=4) as pool:
        done = list(pool.map(write_and_edit, SCENES_A))
    for meta, final in done:
        row = pipeline.record(meta, final, source="가상 작가")
        h = row["hit"]
        print(
            f"  A {row['id']}: 고친 양 {row['edit_cost']:.3f} · 예측 {h['both']}/{h['predicted']} (바뀐 {h['changed']})"
        )


def phase_b(versions: list[str]) -> dict[str, list[float]]:
    """같은 처음 보는 장면을 버전 × 교정 예시(끔·켬)로 쓰고 가상 작가가 고친다. 기록하지 않는다(시험지)."""
    arms = [(v, ex) for v in versions for ex in (False, True)]
    jobs = [(arm, b) for arm in arms for b in SCENES_B]
    with ThreadPoolExecutor(max_workers=4) as pool:
        done = list(pool.map(lambda j: write_and_edit(j[1], j[0][0], j[0][1]), jobs))
    costs: dict[str, list[float]] = {f"{v} 예시{'켬' if ex else '끔'}": [] for v, ex in arms}
    for ((v, ex), _), (meta, final) in zip(jobs, done, strict=True):
        costs[f"{v} 예시{'켬' if ex else '끔'}"].append(metrics.edit_cost(meta["draft"], final))
    return costs


def main() -> None:
    os.environ.setdefault("PA_HOME", str(Path(__file__).resolve().parent.parent / "data" / "demo"))
    if not store.versions(NAME):
        store.set_current(NAME, store.save_version(NAME, f"# 문체 어댑터: {NAME}\n빈 상태에서 시작.\n"))
    if not store.corrections(NAME):
        print("1) v0으로 장면 A 9개 → 가상 작가 교정")
        phase_a()
    print("2) 학습")
    for i in range(int(os.environ.get("ROUNDS", "3"))):
        r = learn.learn_round(NAME, k=3, repeats=2)
        verdict = f"채택 → {r['to']}" if r["accepted"] else "기각"
        print(
            f"  [{i + 1}] {r['from']} 후보 {verdict} · gain {r['base_gain']} → {r['cand_gain']} (이김 {r['wins']} 짐 {r['losses']})"
        )
        for e in r["edits"]:
            print(f"      {e}")
    final_version = store.current_version(NAME)
    print(f"3) 처음 보는 장면 B {len(SCENES_B)}개: v0 vs {final_version}")
    costs = phase_b(["v0", final_version] if final_version != "v0" else ["v0"])
    for v, cs in costs.items():
        print(f"  {v}: 가상 작가가 고친 양 평균 {mean(cs):.3f}  {[round(c, 3) for c in cs]}")
    print("4) 배운 규칙 (숨은 취향과 대조)")
    for r in rules.parse(store.load_adapter(NAME)[1]).rules:
        print(f"  {r.id} [{r.status}] {r.title} — {r.do}")
    print(f"비용: {learn.cost_summary()}")


if __name__ == "__main__":
    main()
