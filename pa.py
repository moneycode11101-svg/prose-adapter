#!/usr/bin/env python3
"""pa — 교정으로 배우는 문체 어댑터.

pa init webnovel [--from 규칙.md]      어댑터 만들기 (파일을 주면 새 버전으로 올리고 지금 버전으로)
pa write webnovel "장면" [--work 제목]  초고 → 퇴고 → 고칠 곳 예측, data/drafts/<id>.md 로 저장
(drafts/<id>.md 를 직접 고친다)
pa collect [--accept ID] [--no-state]  고친 초고를 교정으로 기록, 작품 상태 갱신
pa add webnovel 초고.md 교정.md         바깥에서 가져온 고치기 전/후 한 쌍 기록
pa learn webnovel [--rounds N]         편집안 → 검증 관문 → 나아질 때만 새 버전
pa eval webnovel v0 v1 v3              같은 검증 세트에서 버전 비교
pa stats webnovel                      버전별 고친 양·예측 적중·비용
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

from adapter import learn, metrics, pipeline, rules, store

EMPTY = (
    "# 문체 어댑터: {name}\n모델은 그대로 두고 이 문서만 바뀐다. 규칙은 교정에서 배우고, 검증을 통과한 것만 남는다.\n"
)


def cmd_init(a: argparse.Namespace) -> None:
    if a.source:
        text = Path(a.source).read_text(encoding="utf-8")
        rules.parse(text)  # 형식 확인
    elif store.versions(a.name):
        sys.exit(f"이미 있음: {store.adapter_dir(a.name)} (새 버전은 --from)")
    else:
        text = EMPTY.format(name=a.name)
    version = store.save_version(a.name, text)
    store.set_current(a.name, version)
    print(f"{a.name} {version} 저장 · 지금 버전 {version} · {store.adapter_dir(a.name)}")


def cmd_write(a: argparse.Namespace) -> None:
    meta, path = pipeline.write(a.name, a.brief, a.work, a.chars)
    print(meta["draft"])
    print(f"\n── 저장: {path}  (이 파일을 직접 고친 뒤 `pa collect`)")
    if meta["prediction"]:
        sents = metrics.sentences(meta["draft"])
        print("── 사용자가 고칠 것 같은 곳 (어댑터 예측):")
        for p in meta["prediction"]:
            n = int(p["n"])
            quote = sents[n - 1][:40] if 0 < n <= len(sents) else ""
            print(f"  {n}. [{p['rule']}] {quote} — {p['why']}")


def cmd_collect(a: argparse.Namespace) -> None:
    accept = set(a.accept or [])
    count = 0
    for meta in store.pending_drafts():
        face = (store.drafts_dir() / f"{meta['id']}.md").read_text(encoding="utf-8")
        same = metrics.norm(face) == metrics.norm(meta["draft"])
        if same and meta["id"] not in accept:
            continue
        row = pipeline.record(meta, face)
        count += 1
        h = row["hit"] or {}
        print(
            f"{row['id']}: 고친 양 {row['edit_cost']} · 바뀐 문장 {h.get('changed')} · 예측 맞힘 {h.get('both')}/{h.get('predicted')}"
        )
        if meta.get("work") and not a.no_state:
            print(f"  작품 상태 갱신: {pipeline.update_state(meta['work'], face)}")
    print(f"기록 {count}건" if count else "고친 초고 없음 (그대로 받을 초고는 --accept ID)")


def cmd_add(a: argparse.Namespace) -> None:
    meta = {
        "id": store.new_id("x"),
        "adapter": a.name,
        "version": None,
        "work": a.work,
        "brief": a.brief or "",
        "draft": Path(a.draft).read_text(encoding="utf-8").strip(),
        "prediction": None,
    }
    row = pipeline.record(meta, Path(a.final).read_text(encoding="utf-8"), source=a.source)
    print(f"{row['id']}: 고친 양 {row['edit_cost']} · 바뀐 문장 {len(row['changed'])}")


def cmd_learn(a: argparse.Namespace) -> None:
    for i in range(a.rounds):
        r = learn.learn_round(a.name, a.k, a.repeats)
        verdict = f"채택 → {r['to']}" if r["accepted"] else "기각"
        print(
            f"[{i + 1}] {r['from']} 후보 {verdict} · 검증 {r['val']}건 위치 {r['base_loc']} → {r['cand_loc']}"
            f" · 글자 {r['base_gain']} → {r['cand_gain']} (이김 {r['wins']} 짐 {r['losses']})"
        )
        for e in r["edits"]:
            print(f"    {e}")


def cmd_eval(a: argparse.Namespace) -> None:
    scores = learn.evaluate(a.name, a.versions, a.repeats, a.all)
    for v, s in scores.items():
        print(f"{v}: 위치 {s['loc']:+.3f} · 글자 gain {s['gain']:+.4f}")


def cmd_stats(a: argparse.Namespace) -> None:
    rows = store.corrections(a.name)
    current = store.current_version(a.name)
    print(f"{a.name} · 지금 {current} · 버전 {', '.join(store.versions(a.name))} · 교정 {len(rows)}건")
    for v in store.versions(a.name):
        mine = [r for r in rows if r.get("version") == v]
        if not mine:
            continue
        hits = [r["hit"] for r in mine if r.get("hit")]
        prec = [h["precision"] for h in hits if h["precision"] is not None]
        rec = [h["recall"] for h in hits if h["recall"] is not None]
        print(
            f"  {v}: 초고 {len(mine)}건 · 고친 양 평균 {mean(r['edit_cost'] for r in mine):.3f}"
            + (f" · 예측 정밀 {mean(prec):.2f} 재현 {mean(rec):.2f}" if prec and rec else "")
        )
    adapter = rules.parse(store.load_adapter(a.name)[1])
    for r in adapter.rules:
        print(f"  {r.id} [{r.status}] {r.title} · 근거 {len(r.evidence)}")
    c = learn.cost_summary()
    print(f"호출 {c['calls']}회 · ${c['usd']} · {c['by_role']}")


def main() -> None:
    p = argparse.ArgumentParser(prog="pa", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init")
    s.add_argument("name")
    s.add_argument("--from", dest="source")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("write")
    s.add_argument("name")
    s.add_argument("brief")
    s.add_argument("--work")
    s.add_argument("--chars", type=int, default=1500)
    s.set_defaults(fn=cmd_write)

    s = sub.add_parser("collect")
    s.add_argument("--accept", nargs="*")
    s.add_argument("--no-state", action="store_true")
    s.set_defaults(fn=cmd_collect)

    s = sub.add_parser("add")
    s.add_argument("name")
    s.add_argument("draft")
    s.add_argument("final")
    s.add_argument("--brief")
    s.add_argument("--work")
    s.add_argument("--source", default="가져온 교정")
    s.set_defaults(fn=cmd_add)

    s = sub.add_parser("learn")
    s.add_argument("name")
    s.add_argument("--rounds", type=int, default=1)
    s.add_argument("-k", type=int, default=3, help="한 번에 고칠 규칙 수 상한")
    s.add_argument("--repeats", type=int, default=1, help="검증 항목당 반복 (흔들림 줄이기)")
    s.set_defaults(fn=cmd_learn)

    s = sub.add_parser("eval")
    s.add_argument("name")
    s.add_argument("versions", nargs="+")
    s.add_argument("--repeats", type=int, default=1)
    s.add_argument("--all", action="store_true", help="검증용만이 아니라 전체 교정으로")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("stats")
    s.add_argument("name")
    s.set_defaults(fn=cmd_stats)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
