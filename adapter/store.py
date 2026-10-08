"""데이터 폴더. 원본(corrections·calls·HISTORY·state 이력)은 추가만 하고, 입구(CURRENT·state.md)만 덮어쓴다."""

from __future__ import annotations

import json
import os
import re
import secrets
from datetime import datetime
from pathlib import Path


def home() -> Path:
    return Path(os.environ.get("PA_HOME", Path(__file__).resolve().parent.parent / "data"))


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def corrections_path() -> Path:
    return home() / "corrections.jsonl"


def corrections(adapter: str | None = None) -> list[dict]:
    rows = read_jsonl(corrections_path())
    return [r for r in rows if adapter is None or r["adapter"] == adapter]


# --- 어댑터 버전: vN.md는 한 번 쓰면 고치지 않는다. CURRENT가 지금 쓰는 버전을 가리킨다.


def adapter_dir(name: str) -> Path:
    return home() / "adapters" / name


def versions(name: str) -> list[str]:
    found = [p.stem for p in adapter_dir(name).glob("v*.md") if re.fullmatch(r"v\d+", p.stem)]
    return sorted(found, key=lambda v: int(v[1:]))


def current_version(name: str) -> str:
    path = adapter_dir(name) / "CURRENT"
    if not path.exists():
        raise FileNotFoundError(f"어댑터 '{name}'이 없음. 먼저 `pa init {name}`")
    return path.read_text(encoding="utf-8").strip()


def load_adapter(name: str, version: str | None = None) -> tuple[str, str]:
    version = version or current_version(name)
    return version, (adapter_dir(name) / f"{version}.md").read_text(encoding="utf-8")


def save_version(name: str, text: str) -> str:
    existing = versions(name)
    version = f"v{int(existing[-1][1:]) + 1}" if existing else "v0"
    path = adapter_dir(name) / f"{version}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(text)
    return version


def set_current(name: str, version: str) -> None:
    (adapter_dir(name) / "CURRENT").write_text(version + "\n", encoding="utf-8")


def add_history(name: str, entry: str) -> None:
    with (adapter_dir(name) / "HISTORY.md").open("a", encoding="utf-8") as f:
        f.write(entry.rstrip() + "\n\n")


# --- 초고: drafts/<id>.md는 사용자가 직접 고치는 면, drafts/.orig/<id>.json은 손대지 않는 원본.


def drafts_dir() -> Path:
    return home() / "drafts"


def save_draft(meta: dict) -> Path:
    orig = drafts_dir() / ".orig" / f"{meta['id']}.json"
    orig.parent.mkdir(parents=True, exist_ok=True)
    orig.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    face = drafts_dir() / f"{meta['id']}.md"
    face.write_text(meta["draft"] + "\n", encoding="utf-8")
    return face


def pending_drafts() -> list[dict]:
    done = {r["id"] for r in read_jsonl(corrections_path())}
    metas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((drafts_dir() / ".orig").glob("*.json"))]
    return [m for m in metas if m["id"] not in done]


# --- 작품 상태: 작품마다 state.md 하나. 고칠 때마다 이전 판을 history/에 남긴다.


def work_dir(title: str) -> Path:
    return home() / "works" / re.sub(r"[\s/\\:]+", "-", title.strip())


def load_state(title: str | None) -> str:
    if not title:
        return ""
    path = work_dir(title) / "state.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def save_state(title: str, text: str) -> Path:
    folder = work_dir(title)
    path = folder / "state.md"
    if path.exists():
        snap = folder / "history" / f"{datetime.now():%Y%m%d-%H%M%S}.md"
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path
