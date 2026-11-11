"""Гибридный поиск (BM25 + эмбеддинги + RRF + реранкер) и сборка контекста."""
from __future__ import annotations

import math
import re
import sys
import time

import config
import dialog
import kb as kbmod
import llm

try:
    import numpy as np
except Exception:
    np = None

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

STOP = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы
по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже
или ни быть был него до вас нибудь опять уж вам сказал ведь там потом себя ничего ей может
они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под
будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем
чтобы нее сейчас были куда зачем всех никогда можно при наконец два об другой хоть после
над больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо
свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между это
""".split())

TOKEN_RE = re.compile(r"[а-яеa-z0-9]+")


def tokens(text: str) -> list[str]:
    out = []
    for w in TOKEN_RE.findall(text.lower().replace("ё", "е")):
        if w in STOP or len(w) < 2:
            continue
        out.append(w if w.isdigit() or len(w) <= 5 else w[:5])
    return out


SCOPE_MAP = {"orem": "Выход на оптовый рынок", "askue": "Облачная платформа учета",
             "solar": "Солнечные электростанции", "losses": "Поиск коммерческих потерь",
             "photo": "Распознавание показаний по фотографии"}
NUM_MAP = {"orem": "Выход на оптовый рынок", "askue": "Платформа учета",
           "solar": "Солнечные станции", "photo": "Распознавание показаний",
           "thresholds": "Пороги, связанные с учетом", "metering": "Пороги, связанные с учетом",
           "price": "Пороги, связанные с учетом"}


class Index:
    def __init__(self, kb: kbmod.KB):
        self.kb = kb
        self.bm25 = None
        self.matrix = None
        self.doc_tokens: list[list[str]] = []

    def build(self) -> None:
        texts = [c["text"] for c in self.kb.chunks]
        if not texts:
            return
        self.doc_tokens = [tokens(t) for t in texts]
        if BM25Okapi is not None:
            try:
                self.bm25 = BM25Okapi(self.doc_tokens)
            except Exception:
                self.bm25 = None
        if np is not None and llm.embeddings_available():
            started = time.monotonic()
            try:
                vecs = []
                for i in range(0, len(texts), 32):
                    if time.monotonic() - started > config.INDEX_BUDGET_SEC:
                        raise TimeoutError("Бюджет индексации исчерпан")
                    vecs.extend(llm.embeddings(texts[i:i + 32]))
                mat = np.asarray(vecs, dtype="float32")
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self.matrix = mat / norms
            except Exception as exc:
                print(f"Индекс эмбеддингов не построен: {exc}", file=sys.stderr, flush=True)
                self.matrix = None

    def _bm25_rank(self, q: str, n: int) -> list[int]:
        qt = tokens(q)
        if not qt:
            return []
        if self.bm25 is not None:
            scores = self.bm25.get_scores(qt)
            return sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
        # запасной вариант без внешних библиотек
        df: dict[str, int] = {}
        for toks in self.doc_tokens:
            for w in set(toks):
                df[w] = df.get(w, 0) + 1
        N = max(1, len(self.doc_tokens))
        scores = []
        for toks in self.doc_tokens:
            s = 0.0
            for w in qt:
                f = toks.count(w)
                if f:
                    s += math.log(1 + (N - df.get(w, 0) + 0.5) / (df.get(w, 0) + 0.5)) * f / (f + 1.5)
            scores.append(s)
        return sorted(range(len(scores)), key=lambda i: -scores[i])[:n]

    def _dense_rank(self, q: str, n: int) -> list[int]:
        if self.matrix is None or np is None:
            return []
        try:
            qv = np.asarray(llm.embeddings([q])[0], dtype="float32")
            qv = qv / (np.linalg.norm(qv) or 1.0)
            sims = self.matrix @ qv
            return sorted(range(len(sims)), key=lambda i: -float(sims[i]))[:n]
        except Exception:
            return []

    def search(self, question: str, topics: list[str]) -> list[tuple[int, float]]:
        q = dialog.expand_query(question)
        n = min(50, len(self.kb.chunks))
        bm = self._bm25_rank(q, n)
        dn = self._dense_rank(q, n)
        fused: dict[int, float] = {}
        for rank, idx in enumerate(bm):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (60 + rank)
        for rank, idx in enumerate(dn):
            fused[idx] = fused.get(idx, 0.0) + 1.2 / (60 + rank)
        if not fused:
            return []
        for idx in fused:
            topic = self.kb.docs[self.kb.chunks[idx]["doc"]]["topic"]
            if topic in topics[:2]:
                fused[idx] *= 1.15
        ranked = sorted(fused.items(), key=lambda x: -x[1])

        if llm.reranker_available() and time.monotonic() < config.DEADLINE - 60:
            cand = ranked[:config.RERANK_CANDIDATES]
            try:
                order = llm.rerank(question, [self.kb.chunks[i]["text"][:900] for i, _ in cand],
                                   min(12, len(cand)))
                boosted = [(cand[r][0], cand[r][1] + 0.05 * (len(order) - p)) for p, r in enumerate(order)]
                rest = [c for k, c in enumerate(cand) if k not in set(order)]
                ranked = boosted + rest + ranked[len(cand):]
            except Exception:
                pass
        return ranked


def _doc_scores(index: Index, ranked: list[tuple[int, float]]) -> list[tuple[int, float]]:
    per: dict[int, list[float]] = {}
    for ci, sc in ranked[:40]:
        di = index.kb.chunks[ci]["doc"]
        per.setdefault(di, []).append(sc)
    out = []
    for di, scores in per.items():
        scores.sort(reverse=True)
        out.append((di, scores[0] + 0.35 * sum(scores[1:3])))
    return sorted(out, key=lambda x: -x[1])


def _sec_scores(index: Index, ranked: list[tuple[int, float]]) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for ci, sc in ranked[:60]:
        c = index.kb.chunks[ci]
        key = (c["doc"], c["sec"])
        out[key] = max(out.get(key, 0.0), sc)
    return out


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    pos = max(cut.rfind("\n"), cut.rfind(". "))
    return cut[:pos] if pos > limit * 0.5 else cut


def assemble(index: Index, question: str, topics: list[str],
             ranked: list[tuple[int, float]]) -> str:
    kb = index.kb
    budget = config.CONTEXT_CHARS
    blocks: list[str] = []
    used = 0

    def add(title: str, body: str, limit: int) -> None:
        nonlocal used
        body = _clip(body.strip(), max(0, min(limit, budget - used)))
        if len(body) < 40:
            return
        blocks.append(f"### {title}\n{body}")
        used += len(body) + len(title) + 8

    # 1. Границы консультации — всегда
    scope = kb.by_topic("scope")
    if scope:
        parts = []
        for head in ["Общие границы"] + [SCOPE_MAP[t] for t in topics if t in SCOPE_MAP] + \
                    ["Когда требуется специалист"]:
            sec = kb.find_section(scope, head)
            if sec:
                parts.append(sec["text"])
        seen, uniq = set(), []
        for p in parts:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        add("Границы консультации и ограничения продуктов", "\n".join(uniq), 5200)

    # 2. Числа и обязательные оговорки — по маршруту
    numbers = kb.by_topic("numbers")
    if numbers:
        parts = []
        for t in topics:
            if t in NUM_MAP:
                sec = kb.find_section(numbers, NUM_MAP[t])
                if sec:
                    parts.append(sec["text"])
        low = question.lower()
        if ("тариф" in low and ("рост" in low or "раст" in low)) or not parts:
            sec = kb.find_section(numbers, "Неуниверсальные показатели")
            if sec:
                parts.append(sec["text"])
        seen, uniq = set(), []
        for p in parts:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        if uniq:
            add("Числовые показатели и обязательные оговорки", "\n".join(uniq), 4200)

    # 3. Основные материалы
    doc_rank = [d for d in _doc_scores(index, ranked)
                if kb.docs[d[0]]["topic"] not in ("scope", "numbers")]
    sec_rank = _sec_scores(index, ranked)
    shares = [0.50, 0.30, 0.14, 0.06, 0.04, 0.02]
    for pos, (di, _) in enumerate(doc_rank[:config.TOP_DOCS]):
        doc = kb.docs[di]
        share = shares[pos] if pos < len(shares) else 0.03
        limit = int((budget - used) * share / max(0.05, sum(shares[pos:config.TOP_DOCS])))
        limit = max(1200, min(limit, budget - used))
        secs = sorted(
            [s for s in enumerate(doc["sections"]) if s[1]["kind"] == "text"],
            key=lambda pair: -sec_rank.get((di, pair[0]), 0.0),
        )
        body, local = [], 0
        for si, sec in secs:
            if sec_rank.get((di, si), 0.0) <= 0 and local > limit * 0.6:
                continue
            piece = _clip(sec["text"], max(400, limit - local))
            if len(piece) < 40:
                continue
            body.append(f"[{sec['head']}]\n{piece}")
            local += len(piece)
            if local >= limit:
                break
        rules = []
        for kind, label in (("allow", "Что можно утверждать"), ("deny", "Чего утверждать НЕЛЬЗЯ")):
            for sec in doc["sections"]:
                if sec["kind"] == kind:
                    rules.append(f"{label}:\n{_clip(sec['text'], 1800)}")
        text = "\n\n".join(body + rules)
        if text:
            add(f"Материалы: {doc['name']} — {doc['title']}", text, limit + 3600)
        if used >= budget - 500:
            break

    return "\n\n".join(blocks) if blocks else "Релевантные материалы базы знаний не найдены."