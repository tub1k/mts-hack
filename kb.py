"""Load the supplied knowledge-base.zip and split its Markdown into small evidence chunks.

No knowledge base, cached index, or project data is bundled with the submission.
"""
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
    "21-orem": "orem", "22-askue": "askue", "23-solar": "solar",
    "24-losses": "losses", "25-meter-reading": "photo", "26-general": "general",
}
TOPIC_TERMS = {
    "solar": ("солн", "фотоэлектр", "панел", "инвертор"),
    "orem": ("оптов", "орэм", "орем"),
    "askue": ("аскуэ", "удал", "облачн", "учет", "счетчик", "счётчик"),
    "photo": ("фото", "распознаван", "штрих"),
    "losses": ("потер", "хищен", "небаланс"),
    "price": ("ценов", "категор"),
    "tariffs": ("тариф", "счёт"),
    "requests": ("заявк", "расчёт", "расчет", "оценка"),
    "thresholds": ("порог", "мощност"),
    "regulation": ("закон", "постановлен"),
}
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
BOLD_HEADING_RE = re.compile(r"^\s*\*\*([^*]{4,110})\*\*\s*$")
DENY_RE = re.compile(r"(чего\s+утверждать\s+нельзя|не\s+утверждать|что\s+нельзя\s+утверждать|запрещен\w*\s+утвержден|недопустим\w*\s+утвержден)", re.I)
ALLOW_RE = re.compile(r"(факт\w*\s*,?\s*котор\w*\s+можно\s+утверждать|что\s+можно\s+утверждать)", re.I)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.;!?])\s+(?=[А-ЯЁA-Z0-9])")


def norm(text: str) -> str:
    return (text or "").lower().replace("ё", "е")


def _plain_heading(s: str) -> bool:
    if not s or len(s) > 90 or len(s.split()) > 10:
        return False
    if s.startswith(("-", "*", "•", ">", "|", "[", "`", "#", "+")):
        return False
    if s.endswith((".", ",", ";", ":", "?", "!")):
        return False
    if re.match(r"^\d+[.)]\s", s) or "  " in s:
        return False
    if not s[0].isupper() and not s[0].isdigit():
        return False
    return True


def split_sections(text: str) -> list[dict]:
    title = "Документ"
    headings = {}
    has_md_headings = bool(re.search(r"(?m)^\s{0,3}#{1,6}\s+", text))
    buf = []
    sections = []
    current_head = ""

    def flush():
        body = "\n".join(buf).strip()
        if not body:
            return
        head = current_head or title
        nh = norm(head)
        kind = "deny" if DENY_RE.search(nh) else "allow" if ALLOW_RE.search(nh) else "text"
        sections.append({"head": head, "text": body, "kind": kind})

    for source_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = source_line.rstrip()
        s = line.strip()
        match = HEADING_RE.match(line)
        bold_match = BOLD_HEADING_RE.match(s)
        level = 3
        head = None
        if match:
            level = len(match.group(1))
            head = match.group(2).strip(" *")
        elif bold_match and not has_md_headings:
            head = bold_match.group(1).strip(" :")
        elif not has_md_headings and _plain_heading(s) and (not buf or not buf[-1].strip()) and s:
            head = s.strip(" :")
        if head:
            flush()
            buf = []
            if level == 1 and title == "Документ":
                title = head
            headings = {k: v for k, v in headings.items() if k < level}
            headings[level] = head
            path = [headings[k] for k in sorted(headings)]
            current_head = " / ".join(path[-3:])
        else:
            buf.append(line)
    flush()
    if not sections and text.strip():
        sections = [{"head": title, "text": text.strip(), "kind": "text"}]
    return sections


def _long_segments(text: str, limit: int):
    """Keep sentences intact where possible; cut very long table/text runs safely."""
    if len(text) <= limit:
        return [text]
    pieces = []
    phrases = SENTENCE_BOUNDARY.split(text)
    if len(phrases) == 1:
        phrases = re.split(r"(?<=,)\s+", text)
    acc = ""
    for phrase in phrases:
        if len(acc) + len(phrase) + 1 <= limit:
            acc = (acc + " " + phrase).strip()
            continue
        if acc:
            pieces.append(acc)
            acc = ""
        if len(phrase) > limit:
            stride = max(50, limit - 100)
            for i in range(0, len(phrase), stride):
                pieces.append(phrase[i:i + limit].strip())
        else:
            acc = phrase
    if acc:
        pieces.append(acc)
    return pieces


def chunk_section(sec: dict, limit: int) -> list[str]:
    body = sec["text"].strip()
    if len(body) <= limit:
        return [body]
    paras = re.split(r"\n\s*\n", body)
    if len(paras) == 1:
        # Lists in the same paragraph are usually separated by newlines.
        paras = re.split(r"(?<=\n)(?=[-•*\d])", body)
    spans = []
    for para in paras:
        if not para.strip():
            continue
        spans.extend(_long_segments(para.strip(), limit))
    chunks = []
    cur = ""
    for piece in spans:
        if len(cur) + len(piece) + 2 <= limit:
            cur = (cur + "\n\n" + piece).strip()
        else:
            if cur:
                chunks.append(cur)
            # Small overlap keeps definition+qualification close at chunk boundaries.
            tail = cur[-100:].strip() if cur else ""
            cur = (tail + "\n" + piece).strip() if len(tail) + len(piece) + 1 <= limit else piece
    if cur:
        chunks.append(cur)
    return chunks or [body[:limit]]


class KB:
    def __init__(self):
        self.docs: list[dict] = []
        self.chunks: list[dict] = []

    def by_topic(self, topic: str) -> dict | None:
        return next((d for d in self.docs if d["topic"] == topic), None)

    def docs_for_topic(self, topic: str) -> list[dict]:
        return [d for d in self.docs if d["topic"] == topic]

    def find_section(self, doc: dict | None, needle: str) -> dict | None:
        if not doc:
            return None
        key = norm(needle)
        return next((s for s in doc["sections"] if key in norm(s["head"])), None)


def _read_sources() -> list[tuple[str, str]]:
    path = config.find_file(config.KB_FILE, "knowledge-base.zip", "knowledge_base.zip", "kb.zip")
    items = []

    def decode(payload: bytes) -> str:
        text = payload.decode("utf-8-sig", errors="replace")
        if text.count("\ufffd") > max(3, len(text) // 300):
            try:
                text = payload.decode("cp1251")
            except UnicodeDecodeError:
                pass
        return text

    valid = {".md", ".markdown", ".txt"}
    if path and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if name.startswith(("__MACOSX/", ".")) or Path(name).suffix.lower() not in valid:
                    continue
                # protect against huge/accidental binary files in supplied archive
                meta = archive.getinfo(name)
                if meta.file_size > 8 * 1024 * 1024:
                    continue
                text = decode(archive.read(meta))
                if text.strip():
                    items.append((name, text))
        return items
    folder = config.find_dir("knowledge-base", "knowledge_base", "kb", "00-reference")
    if folder:
        root = folder.parent if folder.name == "00-reference" else folder
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in valid and p.stat().st_size < 8 * 1024 * 1024:
                text = decode(p.read_bytes())
                if text.strip():
                    items.append((str(p.relative_to(root)), text))
    return items


def load() -> KB:
    kb = KB()
    for name, text in _read_sources():
        stem = Path(name).stem.lower()
        sections = split_sections(text)
        title = sections[0]["head"] if sections else Path(name).name
        topic = TOPIC_BY_STEM.get(stem, "")
        if not topic:
            header = norm(name + " " + title)
            for key, terms in TOPIC_TERMS.items():
                if any(term in header for term in terms):
                    topic = key
                    break
            if not topic:
                topic = "general"
        doc = {"name": Path(name).name, "path": name, "title": title,
               "topic": topic, "sections": sections, "text": text}
        di = len(kb.docs)
        kb.docs.append(doc)
        for si, sec in enumerate(sections):
            for part in chunk_section(sec, config.CHUNK_CHARS):
                header = f"{doc['name']} — {sec['head']}"
                kb.chunks.append({"doc": di, "sec": si, "kind": sec["kind"],
                                  "header": header, "raw": part,
                                  "text": f"{header}\n{part}"})
    return kb
