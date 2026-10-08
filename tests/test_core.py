import pytest

from adapter import learn, metrics, pipeline, rules, store

ADAPTER = """# 문체 어댑터: t
설명

## R1 감정은 행동으로 [가설]
- 언제: 갈등 장면
- 하라: 감정 단어 대신 절제된 행동 하나
- 근거: c1, c2
- 예: 몸을 떨었다 → 계약서를 접었다

## R2 짧은 대사 [관찰]
- 하라: 대사는 한 줄
- 근거: c3
"""


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("PA_HOME", str(tmp_path))
    return tmp_path


def test_rules_roundtrip_and_injection():
    a = rules.parse(ADAPTER)
    assert [r.id for r in a.rules] == ["R1", "R2"]
    assert a.rules[0].evidence == ["c1", "c2"]
    assert rules.parse(a.render()).render() == a.render()
    assert "R1" in a.injected() and "R2" not in a.injected()  # 관찰은 집필에 안 들어간다


def test_one_correction_never_makes_an_injected_rule():
    a = rules.parse(ADAPTER)
    spec = {"title": "새 규칙", "status": "가설", "when": "x", "do": "y", "evidence": ["c9"]}
    b, log = rules.apply_edits(a, [{"op": "add", "rule": spec, "why": ""}])
    assert b.find("R3").status == "관찰"
    assert "R3" not in b.injected()
    assert len(a.rules) == 2  # 원본은 그대로
    c, _ = rules.apply_edits(
        b,
        [{"op": "delete", "rule_id": "R2", "why": ""}, {"op": "status", "rule_id": "R1", "status": "관찰", "why": ""}],
    )
    assert [r.id for r in c.rules] == ["R1", "R3"] and c.find("R1").status == "관찰"


def test_metrics_changed_hunks_hit_gain():
    draft = "민재는 분노했다. 그는 문을 열었다. 밖은 비가 왔다."
    final = "민재는 계약서를 접었다. 그는 문을 열었다. 밖은 비가 왔다."
    assert metrics.sentences(draft) == ["민재는 분노했다.", "그는 문을 열었다.", "밖은 비가 왔다."]
    assert metrics.changed(draft, final) == {1}
    assert metrics.hunks(draft, final) == [("민재는 분노했다.", "민재는 계약서를 접었다.")]
    assert metrics.hit({1, 3}, {1}) == {"predicted": 2, "changed": 1, "both": 1, "precision": 0.5, "recall": 1.0}
    assert metrics.edit_cost(draft, draft) == 0
    assert metrics.gain(final, draft, final) > 0 > metrics.gain("전혀 다른 글", draft, final)


def test_versions_are_never_overwritten():
    assert store.save_version("t", "a") == "v0"
    assert store.save_version("t", "b") == "v1"
    store.set_current("t", "v1")
    assert store.load_adapter("t") == ("v1", "b")
    assert store.load_adapter("t", "v0")[1] == "a"


def test_record_and_split_keep_work_on_one_side():
    for i in range(9):
        meta = {
            "id": f"d{i}",
            "adapter": "t",
            "version": "v0",
            "work": f"w{i % 3}",
            "brief": "",
            "draft": f"초고 {i}.",
            "prediction": [],
        }
        pipeline.record(meta, f"교정 {i}.")
    rows = store.corrections("t")
    assert len(rows) == 9 and rows[0]["hit"]["changed"] == 1
    train, val = learn.split(rows)
    assert {r["work"] for r in train}.isdisjoint({r["work"] for r in val}) or len(val) >= learn.MIN_VAL


def test_rule_stats_counts_hits():
    rows = [
        {"changed": [1, 2], "prediction": [{"n": 1, "rule": "R1"}, {"n": 3, "rule": "R1"}, {"n": 2, "rule": "새 습관"}]}
    ]
    assert learn.rule_stats(rows) == {"R1": {"cited": 2, "hit": 1}, "새 습관": {"cited": 1, "hit": 1}}


def test_bad_edits_are_skipped_not_fatal():
    a = rules.parse(ADAPTER)
    b, log = rules.apply_edits(a, [{"op": "delete", "rule_id": "R9", "why": ""}, {"op": "add", "why": ""}])
    assert b.render() == a.render() and len(log) == 2 and log[0].startswith("무시")


def test_location_rewards_right_sentences_even_with_other_words():
    draft = "민재는 분노했다. 그는 문을 열었다. 밖은 비가 왔다."
    final = "민재는 계약서를 접었다. 그는 문을 열었다. 밖은 비가 왔다."
    paraphrase = "민재는 어금니를 깨물었다. 그는 문을 열었다. 밖은 비가 왔다."
    assert metrics.location(paraphrase, draft, final) == 1.0
    assert metrics.location(draft, draft, final) == 0.0
    over = "민재는 어금니를 깨물었다. 그가 문을 밀었다. 밖엔 비."
    assert metrics.location(over, draft, final) == -1.0
    assert metrics.location(final, draft, draft) is None


def test_split_holds_out_exactly_a_third():
    rows = [{"id": f"d{i}", "work": None, "draft": f"초고 {i}.", "final": f"교정 {i}."} for i in range(16)]
    train, val = learn.split(rows)
    assert len(val) == 5 and len(train) == 11
    assert learn.split(rows) == (train, val)  # 결정적
