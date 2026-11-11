"""Загрузка базы знаний, структурная нарезка, извлечение блоков правил."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import config

TOPIC_BY_STEM = {
    "00-product-map": "map", "01-numbers-and-indicators": "numbers",
    "02-product-scope-and-limits": "scope", "03-calculation-and-requests": "requests",
    "10-market": "market", "11-price-categories": "price", "12-power-thresholds": "thresholds",
    "13-metering": "metering", "14-tariffs": "tariffs", "15-microgeneration": "micro",
    "16-regulation": "regulation", "17-faq": "faq", "20-portfolio": "portfolio",
    "21-orem": "orem", "22-askue": "askue", "23-solar": "solar", "24-losses": "losses",
    "25-meter-reading": "photo", "26-general": "general",
}

ALLOW_HEAD = "факты, которые можно утверждать"
DENY_HEAD = "чего утверждать нельзя"


def norm(s: str) -> str:
    return s.lower().replace("ё", "е")


def is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 90:
        return False
    if s.startswith(("-", "•", "*", "|", ">", "[")):
        return False
    if re.search(r"\s{3,}", s):          # строка таблицы
        return False
    if s.endswith((".", ",", ";", ":", "!", "?", "»")):
        return False
    if ". " in s or re.match(r"^\d+[.)]\s", s):
        return False
    if not s[0].isupper():
        return False
    return len(s.split()) <= 12


def split_sections(text: str) -> list[dict]:
    lines = text.replace("\r\n", "\n").split("\n")
    title = ""
    head = ""
    buf: list[str] = []
    sections: list[dict] = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            h = norm(head)
            kind = "allow" if ALLOW_HEAD in h else "deny" if DENY_HEAD in h else "text"
            sections.append({"head": head or title, "text": body, "kind": kind})

    for line in lines:
        raw = line.strip()
        md = raw.startswith("#")
        if md or is_heading(raw):
            flush()
            buf = []
            head = raw.lstrip("#").strip()
            if not title:
                title = head
                head = ""
        else:
            buf.append(line)
    flush()
    if not sections and text.strip():
        sections = [{"head": title or "Документ", "text": text.strip(), "kind": "text"}]
    return sections


def chunk_section(sec: dict, limit: int) -> list[str]:
    text = sec["text"]
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    cur = ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(cur) + len(para) + 1 <= limit:
            cur = (cur + "\n" + para).strip()
        else:
            if cur:
                parts.append(cur)
            if len(para) <= limit:
                cur = para
            else:
                for i in range(0, len(para), limit - 120):
                    parts.append(para[i:i + limit])
                cur = ""
    if cur:
        parts.append(cur)
    return parts or [text[:limit]]


class KB:
    def __init__(self):
        self.docs: list[dict] = []
        self.chunks: list[dict] = []

    def by_topic(self, topic: str) -> dict | None:
        for d in self.docs:
            if d["topic"] == topic:
                return d
        return None

    def sections_of(self, doc: dict, kinds=("text",)) -> list[dict]:
        return [s for s in doc["sections"] if s["kind"] in kinds]

    def find_section(self, doc: dict | None, needle: str) -> dict | None:
        if not doc:
            return None
        n = norm(needle)
        for s in doc["sections"]:
            if n in norm(s["head"]):
                return s
        return None


def _read_sources() -> list[tuple[str, str]]:
    path = config.find_file(config.KB_FILE, "knowledge-base.zip", "knowledge_base.zip", "kb.zip")
    items: list[tuple[str, str]] = []
    if path and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                if name.startswith("__MACOSX/") or Path(name).suffix.lower() not in (".md", ".txt"):
                    continue
                text = zf.read(name).decode("utf-8-sig", errors="replace")
                if text.strip():
                    items.append((name, text))
        return items
    folder = config.find_dir("knowledge-base", "knowledge_base", "kb", "00-reference")
    if folder:
        root = folder.parent if folder.name == "00-reference" else folder
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in (".md", ".txt") and p.is_file():
                text = p.read_text(encoding="utf-8-sig", errors="replace")
                if text.strip():
                    items.append((str(p.relative_to(root)), text))
    return items


def load() -> KB:
    kb = KB()
    for name, text in _read_sources():
        stem = Path(name).stem.lower()
        topic = TOPIC_BY_STEM.get(stem, "")
        sections = split_sections(text)
        title = sections[0]["head"] if sections else Path(name).name
        if not topic:
            low = norm(title)
            for key, val in (("солнеч", "solar"), ("оптов", "orem"), ("платформа учета", "askue"),
                             ("потерь", "losses"), ("фотограф", "photo"), ("ценовые категории", "price")):
                if key in low:
                    topic = val
                    break
            topic = topic or "general"
        doc = {"name": Path(name).name, "path": name, "title": title,
               "topic": topic, "sections": sections, "text": text}
        di = len(kb.docs)
        kb.docs.append(doc)
        for si, sec in enumerate(sections):
            header = f"{doc['name']} — {sec['head']}"
            for part in chunk_section(sec, config.CHUNK_CHARS):
                kb.chunks.append({"doc": di, "sec": si, "kind": sec["kind"],
                                  "header": header, "raw": part, "text": header + "\n" + part})
    return kb