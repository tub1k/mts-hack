"""Hybrid RAG: indexed Russian BM25 + optional BGE embeddings + optional reranker.

Context is assembled from the most relevant *passages*, not whole long documents.
"""
from __future__ import annotations

from array import array
from collections import Counter, defaultdict
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
except ImportError:
    np = None

STOPWORDS = set("""а без более бы был была были быть в вам вас весь все всего всех вы где да для до его ее
если есть еще же за и из или им их как когда кто к ли либо мне мы на над нам нас не нет но о об от по под
после при про с со так такой там те тем то того тоже только тут ты у уже хотя чем что чтобы эта эти этого
это этот я у нас меня свой свою свои какой какая каких какие котором которые можно нужно надо хотим хочу
будет ли стоит вообще сейчас сначала сделать расскажите подскажите знаю кажется пожалуйста хотели
самой самой большинстве пользователю пользователи система услуга сервис ваши нашего вам вам
""".split())
TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.I)
SUFFIXES = tuple(sorted({
    "иями", "ями", "ами", "ение", "ения", "ений", "иях", "ость", "ости", "ового", "овыми",
    "ого", "ему", "ому", "ыми", "ими", "ией", "иях", "иях", "овать", "ются", "ется",
    "ая", "яя", "ые", "ие", "ое", "ее", "ый", "ий", "ой", "ую", "юю", "ых", "их",
    "ям", "ам", "ях", "ах", "ов", "ев", "ей", "ия", "ии", "ию", "ий", "ые",
    "ого", "ом", "ем", "ет", "ут", "ют", "ать", "ять", "ить", "тся", "ся", "сь",
    "а", "я", "ы", "и", "е", "у", "ю", "о",
}, key=len, reverse=True))


def stem(word: str) -> str:
    w = word.lower().replace("ё", "е")
    if len(w) < 5 or not re.search(r"[а-я]", w):
        return w
    for suf in SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 5:
            return w[:-len(suf)]
    return w


def tokens(text: str) -> list[str]:
    words = TOKEN_RE.findall(text.lower().replace("ё", "е"))
    return [stem(w) for w in words if len(w) > 1 and w not in STOPWORDS]


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    p = max(text.rfind("\n", 0, limit), text.rfind(". ", 0, limit),
            text.rfind("; ", 0, limit))
    return text[:p + 1].strip() if p > limit * 0.58 else text[:limit].rstrip() + "…"


class Index:
    def __init__(self, kb: kbmod.KB):
        self.kb = kb
        self.postings = defaultdict(list)
        self.idf = {}
        self.lengths = []
        self.avgdl = 1.0
        self.embedded_ids = []
        self.matrix = None
        self.vectors: list[array] = []

    def build(self) -> None:
        chunks = self.kb.chunks
        if not chunks:
            return
        for i, chunk in enumerate(chunks):
            terms = Counter(tokens(chunk["text"]))
            self.lengths.append(sum(terms.values()))
            for term, count in terms.items():
                self.postings[term].append((i, count))
        n = len(chunks)
        self.avgdl = max(1.0, sum(self.lengths) / n)
        self.idf = {word: math.log(1.0 + (n - len(hits) + 0.5) / (len(hits) + 0.5))
                    for word, hits in self.postings.items()}

        if not llm.embeddings_available() or config.remaining() < 120:
            return
        started = time.monotonic()
        if len(chunks) <= config.EMBED_MAX_CHUNKS:
            chosen = list(range(len(chunks)))
        else:
            # Even coverage avoids systematically excluding the later topic files.
            n_take = config.EMBED_MAX_CHUNKS
            chosen = sorted(set(round(j * (len(chunks)-1) / (n_take-1)) for j in range(n_take)))
        vectors = []
        chosen_ok = []
        for start in range(0, len(chosen), 24):
            if time.monotonic() - started > config.INDEX_BUDGET_SEC or config.remaining() < 95:
                break
            part_ids = chosen[start:start + 24]
            part = [chunks[i]["text"][:1550] for i in part_ids]
            try:
                new = llm.embeddings(part)
                if not new or (vectors and len(new[0]) != len(vectors[0])):
                    break
                vectors.extend(new)
                chosen_ok.extend(part_ids)
            except Exception as exc:
                print("Embedding index partially disabled:", type(exc).__name__, file=sys.stderr)
                break
        if not vectors:
            return
        self.embedded_ids = chosen_ok
        if np is not None:
            mat = np.asarray(vectors, dtype="float32")
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.matrix = mat / norms
        else:
            for v in vectors:
                mag = math.sqrt(sum(x*x for x in v)) or 1.0
                self.vectors.append(array("f", [x / mag for x in v]))

    def _bm25(self, query: str, n: int = 75) -> list[tuple[int, float]]:
        qterms = Counter(tokens(query))
        scores = defaultdict(float)
        for term, qf in qterms.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            qw = 1.0 + 0.20 * min(qf - 1, 3)
            for idx, tf in self.postings[term]:
                norm = 1.4 * (0.28 + 0.72 * self.lengths[idx] / self.avgdl)
                scores[idx] += qw * idf * tf * 2.4 / (tf + norm)
        if not scores:
            return []
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]

    def _dense(self, query: str, n: int = 60) -> list[tuple[int, float]]:
        if not self.embedded_ids or not llm.embeddings_available() or config.remaining() < 50:
            return []
        try:
            vec = llm.embeddings([query[:1800]])[0]
            if self.matrix is not None:
                qv = np.asarray(vec, dtype="float32")
                qv /= (np.linalg.norm(qv) or 1.0)
                sims = self.matrix @ qv
                order = np.argsort(-sims)[:n]
                return [(self.embedded_ids[int(i)], float(sims[i])) for i in order]
            norm = math.sqrt(sum(v*v for v in vec)) or 1.0
            qnorm = [v / norm for v in vec]
            sims = [(idx, sum(a*b for a, b in zip(row, qnorm)))
                    for idx, row in zip(self.embedded_ids, self.vectors)]
            return sorted(sims, key=lambda x: x[1], reverse=True)[:n]
        except Exception:
            return []

    def search(self, query: str, topics: list[str]) -> list[tuple[int, float]]:
        q = dialog.expand_query(query)
        bm = self._bm25(q)
        dense = self._dense(q)
        if not bm and not dense:
            return []
        ranks = defaultdict(float)
        for pos, (idx, _) in enumerate(bm):
            ranks[idx] += 1.0 / (45 + pos)
        for pos, (idx, sim) in enumerate(dense):
            if sim >= 0.12:
                ranks[idx] += 1.15 / (45 + pos)
        if not ranks:
            return []
        for idx in ranks:
            topic = self.kb.docs[self.kb.chunks[idx]["doc"]]["topic"]
            if topic in topics[:2]:
                ranks[idx] *= 1.25
            elif topic in topics[:4]:
                ranks[idx] *= 1.11
            elif topic in ("map", "scope", "numbers"):
                # General guidance should not replace detailed product evidence.
                ranks[idx] *= 0.85
        ranked = sorted(ranks.items(), key=lambda x: -x[1])

        # Fast reranker: use at most one call, and keep lexical/dense fallback.
        if (llm.reranker_available() and len(ranked) >= 5
                and config.remaining() > max(100, 2 * config.RERANK_TIMEOUT)):
            candidates = ranked[:config.RERANK_CANDIDATES]
            docs = [self.kb.chunks[i]["text"][:1100] for i, _ in candidates]
            try:
                indices = llm.rerank(query, docs, min(config.RERANKER_TOP_N, len(docs)))
                seen = set()
                reranked = []
                for pos in indices:
                    if 0 <= pos < len(candidates) and pos not in seen:
                        # Reranker decides the leading passages, BM25/dense breaks ties.
                        reranked.append((candidates[pos][0], candidates[pos][1] + 0.06))
                        seen.add(pos)
                ranked = reranked + [v for j, v in enumerate(candidates) if j not in seen] + ranked[len(candidates):]
            except Exception:
                pass
        return ranked


# Evidence shortcuts are used ONLY to retrieve passages from the supplied KB.
# They do not inject canned answers or invented product facts.
ANCHOR_RULES = [
    (r"симк|\bsim\b|другого оператор", "askue", r"приборы каких производителей", r"сим|оператор"),
    (r"симк|\bsim\b|другого оператор", "askue", r"совместимость", r"сим|оператор"),
    (r"цех|производственн.*учет", "askue", r"кому подходит|что это", r"цех|промышленн"),
    (r"магазин|ретейл|разн.*точк", "askue", r"что дает клиенту|кому подходит", r"ретейл|сравнен"),
    (r"арендатор|бизнес.центр", "askue", r"кому подходит|что это", r"арендатор|торгов"),
    (r"придется.*менят.*схем|менят.*сим|подключ.*существующ", "askue", r"придется ли менять всю схему|совместимость", r"менят|существующ"),
    (r"предварительн.*оценк|сколько.*сто|стоимост.*учет|расчет.*склад", "askue", r"сколько это стоит", r"количеств|конфигурац"),
    (r"предварительн.*оценк|сколько.*сто|стоимост.*учет|расчет.*склад", "askue", r"две конфигурации", r"конфигурац"),
    (r"реактив", "faq", r"реактивн", r"измер|компенсац"),
    (r"сч[её]т.*вырос|сумм.*заметн|сравн.*сч[её]т|рост.*сч[её]т", "tariffs", r"что именно растет", r"сравнит|мощност|средн"),
    (r"ценов.*категор|четверт|пят.*категор|двухставочн", "price", r"шесть категорий", r"четверт|двухставочн"),
    (r"провер.*учет|аудит.*учет|менят.*систем.*учет|сохранит.*счетчик", "orem", r"что делает мтс|нужно ли менять существующ", r"аудит|модернизац"),
    (r"оптов.*рын|порог.*750|орэм", "orem", r"порог доступа", r"750|20 мвт"),
    (r"невыгод|не окуп|экономическ.*эффект", "orem", r"что если расчет покажет|экономическая целесообразность", r"невыгод|расчет"),
    (r"потер|хищен|архив|почасов.*данн|суточн.*данн", "losses", r"какой результат дает продукт|что нужно от клиента|что мы не делаем", r"глубина архива|суточн|дневное|ночное"),
    (r"фотограф|по фото|распозна|двухтариф|год выпуск", "photo", r"возможност|распозна|какие данные|что это|частые вопросы", r"показан|серийн|год производ|год выпуск|тариф"),
    (r"скат|фасад|шифер|кровл|расшир|солнеч|сетевая станц", "solar", r"кровл|размещен|станц|частые вопросы|кому подходит", r"фасад|шифер|скат|восток|запад|расшир|накопит"),
]


def _anchors(index: Index, query: str) -> list[int]:
    """Find high-information chunks in exact FAQ sections and requirement paragraphs."""
    q = kbmod.norm(query)
    chosen = []
    for qpattern, topic, heading_pattern, body_pattern in ANCHOR_RULES:
        if not re.search(qpattern, q):
            continue
        scored = []
        for di, doc in enumerate(index.kb.docs):
            if doc["topic"] != topic:
                continue
            for ci, c in enumerate(index.kb.chunks):
                if c["doc"] != di or c["kind"] == "deny":
                    continue
                # Match the leaf heading, not an ancestor (e.g. an unrelated
                # subsection of "Что делает МТС" must not beat its audit text).
                leaf = kbmod.norm(doc["sections"][c["sec"]]["head"].split(" / ")[-1])
                if re.search(heading_pattern, leaf):
                    score = (2.0 if re.search(body_pattern, kbmod.norm(c["raw"])) else 0.0)
                    score += len(set(tokens(q)) & set(tokens(c["raw"]))) * 0.15
                    scored.append((score, ci))
        scored.sort(reverse=True)
        if scored:
            for _, ci in scored[:1]:
                if ci not in chosen:
                    chosen.append(ci)
        if len(chosen) >= 5:
            break
    return chosen[:5]

SCOPE_MAP = {
    "orem": "Выход на оптовый рынок", "askue": "Облачная платформа учета",
    "solar": "Солнечные электростанции", "losses": "Поиск коммерческих потерь",
    "photo": "Распознавание показаний по фотографии",
}
NUM_MAP = {
    "orem": "Выход на оптовый рынок", "askue": "Платформа учета",
    "solar": "Солнечные станции", "photo": "Распознавание показаний",
    "price": "Пороги, связанные с учетом", "thresholds": "Пороги, связанные с учетом",
}


def assemble(index: Index, question: str, topics: list[str], ranked: list[tuple[int, float]],
             full_query: str | None = None) -> str:
    kb = index.kb
    if not ranked:
        return "ПОДТВЕРЖДАЮЩИЕ ФРАГМЕНТЫ НЕ НАЙДЕНЫ. Не угадывать, уточнить вопрос или направить к специалисту."
    budget = config.CONTEXT_CHARS
    blocks = []
    used_chars = 0

    def add(label: str, body: str, max_chars: int) -> bool:
        nonlocal used_chars
        available = budget - used_chars - len(label) - 12
        if available < 130:
            return False
        text = _clip(body, min(available, max_chars))
        if len(text) < 30:
            return False
        blocks.append("### " + label + "\n" + text)
        used_chars += len(label) + len(text) + 7
        return True

    # Choose diverse, scored passages. Guarantee coverage of the relevant product
    # topics, rather than letting the most verbose single file occupy the context.
    chosen = []
    seen_chunks = set()
    per_doc = Counter()
    per_sec = Counter()

    def choose(ci: int, force: bool = False):
        c = kb.chunks[ci]
        di, si = c["doc"], c["sec"]
        if ci in seen_chunks:
            return False
        if not force and (per_doc[di] >= 4 or per_sec[(di, si)] >= 2):
            return False
        chosen.append(ci)
        seen_chunks.add(ci)
        per_doc[di] += 1
        per_sec[(di, si)] += 1
        return True

    for ci in _anchors(index, full_query or question):
        choose(ci, force=True)

    for topic in topics[:3]:
        matching = [ci for ci, _ in ranked[:55]
                    if kb.docs[kb.chunks[ci]["doc"]]["topic"] == topic
                    and kb.chunks[ci]["kind"] in ("text", "allow")]
        for ci in matching[:(2 if topic == topics[0] else 1)]:
            choose(ci)
    for ci, score in ranked:
        if len(chosen) >= config.TOP_K:
            break
        if kb.chunks[ci]["kind"] in ("text", "allow"):
            choose(ci)
    # If only special sections were retrieved, allow them as evidence.
    if not chosen:
        for ci, _ in ranked[:min(5, config.TOP_K)]:
            choose(ci, force=True)

    used_doc_ids = set()
    for i, ci in enumerate(chosen):
        c = kb.chunks[ci]
        doc = kb.docs[c["doc"]]
        used_doc_ids.add(c["doc"])
        remaining_blocks = max(1, len(chosen)-i)
        reserved = 1700
        available = budget - used_chars - reserved
        if available < 270:
            break
        max_chars = max(400, min(1500, available // remaining_blocks + 200))
        add(f"Источник {i+1}: {doc['name']} | {c['header'].split(' — ', 1)[-1]}",
            c["raw"], max_chars)

    # Append only relevant prohibitions/qualification notes from product docs.
    for di in list(used_doc_ids)[:4]:
        doc = kb.docs[di]
        limitations = [s["text"] for s in doc["sections"] if s["kind"] == "deny"]
        if limitations:
            add("Ограничения по " + doc["name"], "\n".join(limitations), 800)

    # Scope and numeric limitations are supplemental rather than the entire corpus.
    scope = kb.by_topic("scope")
    if scope and used_chars < budget - 700:
        parts = []
        for name in ([SCOPE_MAP[t] for t in topics[:3] if t in SCOPE_MAP]):
            sec = kb.find_section(scope, name)
            if sec and sec["text"] not in parts:
                parts.append(sec["text"])
        if parts:
            add("Границы продуктов", "\n".join(parts), 1250)

    if (re.search(r"\d|порог|сколько|точност|окупаем|гаранти|процент|квт|мвт", question.lower())
            and used_chars < budget - 600):
        num = kb.by_topic("numbers")
        if num:
            for topic in topics[:3]:
                if topic in NUM_MAP:
                    sec = kb.find_section(num, NUM_MAP[topic])
                    if sec:
                        add("Показатели и оговорки: " + topic, sec["text"], 950)
                        break

    return "\n\n".join(blocks) if blocks else (
        "ПОДТВЕРЖДАЮЩИЕ ФРАГМЕНТЫ НЕ НАЙДЕНЫ. Нужны данные базы или консультация специалиста."
    )
