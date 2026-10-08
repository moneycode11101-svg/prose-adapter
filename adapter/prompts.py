"""프롬프트 틀. 비교가 공정하려면 어댑터 칸만 바뀌고 나머지는 같아야 하므로 한 곳에 둔다."""

WRITER = (
    "너는 한국 웹소설 작가다. '문체 어댑터'의 규칙을 지키고, '작품 상태'와 어긋나는 내용을 쓰지 않는다. "
    "인물은 자기가 아는 것만 말하고 행동한다. 본문만 출력한다. 제목·설명·머리말을 붙이지 않는다."
)

# 사용자는 대개 글의 몇 %만 고친다. 규칙 밖의 문장을 '더 좋게' 다듬으면 사용자 교정본에서 멀어진다(첫 시연에서 관문이 이걸 잡았다).
REVISER = (
    "너는 이 사용자의 퇴고를 대신하는 편집자다. 문체 어댑터의 규칙이나 교정 예에 해당하는 문장만 고친다. "
    "해당하지 않는 문장은 한 글자도 바꾸지 않는다. 더 좋아 보여도 바꾸지 않는다. 규칙이 없거나 해당하는 곳이 없으면 초고를 그대로 출력한다. "
    "사건·정보·순서는 바꾸지 않는다(작품 상태와 충돌하는 부분만 예외). 고친 본문 전체만 출력한다."
)

PREDICTOR = (
    "너는 이 사용자가 글을 받고 어디를 고칠지 예측한다. 규칙과 교정 예에 근거해 손댈 가능성이 높은 문장만 고른다. "
    "확신이 없으면 고르지 않는다."
)

PROPOSER = """너는 문체 어댑터를 고치는 편집장이다. 사용자의 실제 교정(초고→교정)과 예측 기록을 보고 어댑터를 작게 고친다.
원칙:
- 편집은 최대 {k}개. 교정에서 반복되는 방향을 먼저, 한 번뿐인 것은 '관찰'로만 남긴다.
- 근거(evidence)는 교정 id로 단다. 교정 하나뿐이면 status '관찰', 같은 방향 교정이 둘 이상이면 '가설'. '채택'은 쓰지 않는다.
- 사용자의 구체적인 고침 방식을 일반론으로 뭉개지 않는다. 언제(적용 장면)와 예외를 같이 쓴다.
- 특정 작품의 인물·설정·사건은 넣지 않는다. 그건 작품 상태의 몫이다.
- 기존 규칙과 겹치면 add 대신 replace로 근거를 합친다. 예측에 쓰였지만 실제로는 안 고쳐진 규칙은 delete나 수준 낮추기 후보다.
- example에는 실제 교정을 짧게 인용한다(초고 → 교정).
- 초고와 교정이 같은 문장은 사용자가 받아들인 것이다. 그걸 고치게 만드는 규칙은 손해다."""

STATE = """너는 연재 작품의 설정 관리자다. 확정된 본문을 읽고 작품 상태 문서를 갱신한다.
섹션은 이 다섯 개로 고정한다:
## 세계 사실 — 독자에게 확정된 것
## 인물별 아는 것·믿는 것 — 사실과 믿음을 구분하고, 틀린 믿음은 '(오해)'로 표시
## 작가 계획 — 아직 본문에 안 나온 것. 본문에서 실현되면 세계 사실로 옮긴다
## 미회수 복선
## 연표 — 사건 한 줄씩
본문에 근거가 없는 것은 추가하지 않는다. 기존 내용은 본문과 충돌할 때만 고친다. 문서 전체만 출력한다."""

PREDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "문장 번호"},
                    "rule": {"type": "string", "description": "근거 규칙 id, 규칙에 없으면 '새 습관'"},
                    "why": {"type": "string"},
                },
                "required": ["n", "rule", "why"],
            },
        }
    },
    "required": ["items"],
}

PROPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string", "description": "교정에서 본 반복 패턴, 3~6줄"},
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["add", "replace", "delete", "status"]},
                    "rule_id": {"type": "string", "description": "replace·delete·status일 때"},
                    "status": {"type": "string", "enum": ["가설", "관찰"], "description": "status일 때"},
                    "rule": {
                        "type": "object",
                        "description": "add·replace일 때",
                        "properties": {
                            "title": {"type": "string"},
                            "status": {"type": "string", "enum": ["가설", "관찰"]},
                            "when": {"type": "string"},
                            "do": {"type": "string"},
                            "unless": {"type": "string"},
                            "evidence": {"type": "array", "items": {"type": "string"}},
                            "example": {"type": "string"},
                        },
                        "required": ["title", "status", "when", "do", "evidence"],
                    },
                    "why": {"type": "string"},
                },
                "required": ["op", "why"],
            },
        },
    },
    "required": ["analysis", "edits"],
}


def section(title: str, body: str) -> str:
    return f"# {title}\n{body.strip()}\n\n" if body.strip() else ""


def numbered(sentences: list[str]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))
