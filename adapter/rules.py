"""어댑터 문서 = 규칙 목록. 규칙마다 근거(교정 id)와 확인 수준을 단다.

확인 수준(dot의 criteria와 같은 사다리):
- 관찰: 교정 하나뿐. 저장만 하고 집필에는 넣지 않는다.
- 가설: 같은 쪽 교정이 둘 이상, 또는 사용자가 이유를 말함. 집필에 넣는다.
- 채택: 검증 관문을 통과했거나 사용자가 직접 정함. 집필에 넣는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

STATUSES = ("채택", "가설", "관찰")
INJECTED = ("채택", "가설")
FIELDS = {"언제": "when", "하라": "do", "예외": "unless", "근거": "evidence", "예": "example"}
HEADER = re.compile(r"^## (R\d+) (.+?) \[(채택|가설|관찰)\]\s*$")


@dataclass
class Rule:
    id: str
    title: str
    status: str
    when: str = ""
    do: str = ""
    unless: str = ""
    evidence: list[str] = field(default_factory=list)
    example: str = ""

    def render(self) -> str:
        lines = [f"## {self.id} {self.title} [{self.status}]"]
        for label, attr in FIELDS.items():
            value = ", ".join(self.evidence) if attr == "evidence" else getattr(self, attr)
            if value:
                lines.append(f"- {label}: {value}")
        return "\n".join(lines)


@dataclass
class Adapter:
    preamble: str
    rules: list[Rule]

    def render(self) -> str:
        blocks = [self.preamble.rstrip()] + [r.render() for r in self.rules]
        return "\n\n".join(b for b in blocks if b) + "\n"

    def injected(self) -> str:
        """집필 프롬프트에 넣는 부분: 관찰 수준은 뺀다."""
        kept = [r.render() for r in self.rules if r.status in INJECTED]
        return "\n\n".join(kept) if kept else "(규칙 없음)"

    def find(self, rule_id: str) -> Rule:
        for r in self.rules:
            if r.id == rule_id:
                return r
        raise KeyError(f"규칙 {rule_id} 없음")

    def next_id(self) -> str:
        top = max((int(r.id[1:]) for r in self.rules), default=0)
        return f"R{top + 1}"


def parse(text: str) -> Adapter:
    preamble: list[str] = []
    rules: list[Rule] = []
    for line in text.splitlines():
        head = HEADER.match(line)
        if head:
            rules.append(Rule(id=head[1], title=head[2], status=head[3]))
            continue
        if not rules:
            preamble.append(line)
            continue
        item = re.match(r"^- (언제|하라|예외|근거|예): (.*)$", line)
        if not item:
            continue
        attr = FIELDS[item[1]]
        if attr == "evidence":
            rules[-1].evidence = [e.strip() for e in item[2].split(",") if e.strip()]
        else:
            setattr(rules[-1], attr, item[2].strip())
    return Adapter(preamble="\n".join(preamble).strip(), rules=rules)


def status_for(evidence: list[str], requested: str) -> str:
    """근거 수가 허락하는 만큼만 올린다. 한 번의 교정으로 집필 규칙을 만들지 않는다."""
    if requested == "채택":
        return "가설" if len(evidence) >= 2 else "관찰"
    if requested == "가설" and len(evidence) < 2:
        return "관찰"
    return requested if requested in STATUSES else "관찰"


def apply_edits(adapter: Adapter, edits: list[dict]) -> tuple[Adapter, list[str]]:
    """제안된 편집(add·replace·delete·status)을 적용한 새 어댑터와 적용 기록을 돌려준다."""
    rules = [Rule(**vars(r)) for r in adapter.rules]
    out = Adapter(preamble=adapter.preamble, rules=rules)
    log: list[str] = []
    for e in edits:
        op = e["op"]
        if op in ("replace", "delete", "status") and e.get("rule_id") not in {r.id for r in out.rules}:
            log.append(f"무시 {op} {e.get('rule_id')}: 없는 규칙")
            continue
        if op in ("add", "replace") and not e.get("rule"):
            log.append(f"무시 {op}: 규칙 내용 없음")
            continue
        if op == "add":
            r = _rule_from(e["rule"], out.next_id())
            out.rules.append(r)
            log.append(f"추가 {r.id} {r.title} [{r.status}] 근거 {', '.join(r.evidence)}")
        elif op == "replace":
            old = out.find(e["rule_id"])
            r = _rule_from(e["rule"], old.id)
            out.rules[out.rules.index(old)] = r
            log.append(f"교체 {r.id} {r.title} [{r.status}]")
        elif op == "delete":
            old = out.find(e["rule_id"])
            out.rules.remove(old)
            log.append(f"삭제 {old.id} {old.title}")
        elif op == "status":
            old = out.find(e["rule_id"])
            old.status = status_for(old.evidence, e["status"])
            log.append(f"수준 {old.id} → {old.status}")
        else:
            raise ValueError(f"모르는 편집: {op}")
    return out, log


def _rule_from(spec: dict, rule_id: str) -> Rule:
    evidence = [str(x) for x in spec.get("evidence", [])]
    return Rule(
        id=rule_id,
        title=spec["title"].strip(),
        status=status_for(evidence, spec.get("status", "관찰")),
        when=spec.get("when", "").strip(),
        do=spec.get("do", "").strip(),
        unless=spec.get("unless", "").strip(),
        evidence=evidence,
        example=spec.get("example", "").strip().replace("\n", " / "),
    )
