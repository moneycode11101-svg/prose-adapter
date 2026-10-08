"""data/exp/results.json → 보고서 HTML 한 장. 그림은 페이지 안의 스크립트가 그린다.

PA_HOME=data/exp python3 demo/report.py
"""

from __future__ import annotations

import json
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiment as exp  # noqa: E402

from adapter import metrics, rules, store  # noqa: E402

ARMS = ["빈 어댑터", "교정 예시만", "내 어댑터", "내 어댑터+예시", "남의 어댑터"]


def marks(draft: str, final: str) -> list[list[str]]:
    """교정 부호용 조각: [종류, 글자]. 종류는 eq·del·ins."""
    a, b = metrics.norm(draft), metrics.norm(final)
    out: list[list[str]] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            out.append(["eq", a[i1:i2]])
            continue
        if i2 > i1:
            out.append(["del", a[i1:i2]])
        if j2 > j1:
            out.append(["ins", b[j1:j2]])
    return out


def build(r: dict) -> dict:
    per_scene: dict[str, dict[str, list[float]]] = {}
    for t in exp.TASTES:
        rows = [x for x in r["test"] if x["user"] == t]
        per_scene[t] = {
            arm: [x["edit_cost"] for x in sorted(rows, key=lambda x: x["scene"]) if x["arm"] == arm] for arm in ARMS
        }
    examples = {}
    for t in exp.TASTES:
        base, own = per_scene[t]["빈 어댑터"], per_scene[t]["내 어댑터"]
        i = max(range(len(base)), key=lambda k: base[k] - own[k])
        pick = {x["arm"]: x for x in r["test"] if x["user"] == t and x["scene"] == i}
        examples[t] = {
            "scene": exp.TEST[i],
            "base": {
                "cost": pick["빈 어댑터"]["edit_cost"],
                "marks": marks(pick["빈 어댑터"]["draft"], pick["빈 어댑터"]["final"]),
            },
            "own": {
                "cost": pick["내 어댑터"]["edit_cost"],
                "marks": marks(pick["내 어댑터"]["draft"], pick["내 어댑터"]["final"]),
            },
        }
    learned = {}
    for t in exp.TASTES:
        adapter = rules.parse(store.load_adapter(exp.name(t))[1])
        learned[t] = [
            {
                "id": x.id,
                "title": x.title,
                "status": x.status,
                "do": x.do,
                "when": x.when,
                "unless": x.unless,
                "n": len(x.evidence),
            }
            for x in adapter.rules
        ]
    train = {t: round(mean(c["edit_cost"] for c in store.corrections(exp.name(t))), 4) for t in exp.TASTES}
    return {
        "tastes": {t: [line.split(". ", 1)[1] for line in s.splitlines()] for t, s in exp.TASTES.items()},
        "final": r["final"],
        "summary": r["summary"],
        "per_scene": per_scene,
        "learn": {
            t: [
                {
                    k: h[k]
                    for k in (
                        "from",
                        "to",
                        "accepted",
                        "base_gain",
                        "cand_gain",
                        "base_loc",
                        "cand_loc",
                        "wins",
                        "losses",
                        "edits",
                        "val",
                        "train",
                    )
                    if k in h
                }
                for h in hs
            ]
            for t, hs in r["learn"].items()
        },
        "judge": r["judge"],
        "learned": learned,
        "examples": examples,
        "train_cost": train,
        "n_train": len(exp.TRAIN),
        "n_test": len(exp.TEST),
        "cost": r["cost"],
        "arms": ARMS,
    }


def run1(path: str | None) -> dict | None:
    """1차 실행 결과(관문을 고치기 전). 같은 학습용 교정·같은 시험 장면이라 나란히 놓을 수 있다."""
    if not path:
        return None
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "final": r["final"],
        "summary": r["summary"],
        "learn": {t: [{"accepted": h["accepted"], "val": h["val"]} for h in hs] for t, hs in r["learn"].items()},
        "judge": {t: sum(j["verdict"] == "잡음" for j in js) for t, js in r["judge"].items()},
    }


def main() -> None:
    os.environ.setdefault("PA_HOME", str(Path(__file__).resolve().parent.parent / "data" / "exp"))
    r = json.loads((store.home() / "results.json").read_text(encoding="utf-8"))
    built = build(r)
    built["run1"] = run1(os.environ.get("PA_RUN1"))
    data = json.dumps(built, ensure_ascii=False).replace("</", "<\\/")
    template = (Path(__file__).resolve().parent / "report.html").read_text(encoding="utf-8")
    out = store.home() / "report.html"
    page = template.replace("/*__DATA__*/null", data)
    out.write_text(page, encoding="utf-8")
    # 게시할 때는 Artifact가 문서 뼈대를 씌운다. 로컬 미리보기에만 뼈대를 붙인다.
    preview = store.home() / "report-preview.html"
    preview.write_text(
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">' + page,
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
