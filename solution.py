"""MTS Energytool challenge entry point: questions.csv -> submission.csv.

Run as `python solution.py`. The judge supplies questions.csv and
knowledge-base.zip under DATA_DIR, and the LLM proxy via env vars.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import dialog
import guard
import kb as kbmod
import llm
import policy
import retrieval

_WRITE_LOCK = threading.Lock()


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def read_questions() -> list[tuple[str, str]]:
    path = config.find_file(config.QUESTIONS_FILE, "questions.csv", "train.csv")
    if path is None:
        raise FileNotFoundError("В DATA_DIR не найден questions.csv")
    out = []
    ids = set()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "id" not in reader.fieldnames or "question" not in reader.fieldnames:
            raise ValueError("questions.csv: требуются столбцы id и question")
        for row in reader:
            qid = str(row.get("id") or "").strip()
            if not qid or qid in ids:
                # CSV specification requires unique ID; skip a malformed duplicate.
                continue
            ids.add(qid)
            out.append((qid, str(row.get("question") or "")))
    if not out:
        raise ValueError("В questions.csv нет вопросов")
    return out


def write_csv(path: str, questions: list[tuple[str, str]], answers: list[str]) -> None:
    """Save an always-valid CSV snapshot, atomically and without duplicate IDs."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with _WRITE_LOCK:
        with temporary.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
            writer.writerow(["id", "answer"])
            for (qid, question), answer in zip(questions, answers):
                text = (answer or "").strip()[:10000]
                if not text:
                    text = policy.fallback(question)
                writer.writerow([qid, text])
        os.replace(temporary, destination)


def answer_one(index: retrieval.Index, raw: str) -> str:
    parsed = dialog.parse(raw)
    question = parsed["question"]
    facts = dialog.extract_facts(parsed["user_text"])
    topics = dialog.route_topics(question, parsed["routing_text"])
    if not index.kb.chunks:
        return policy.fallback(question, topics)

    ranked = index.search(parsed["search_query"], topics)
    if not ranked:
        return policy.fallback(question, topics)
    context = retrieval.assemble(index, question, topics, ranked, parsed["search_query"])
    messages = policy.build_messages(context, question, parsed["history"], facts, topics)
    answer = policy.postprocess(llm.chat(messages))
    if not answer:
        return policy.fallback(question, topics)

    if config.USE_REPAIR and llm.time_left() > 75:
        issues = guard.check(answer, context, question)
        if issues:
            try:
                fixed = policy.postprocess(llm.chat(guard.repair_messages(messages, answer, issues),
                                                   max_tokens=min(config.MAX_TOKENS, 1400)))
                if fixed and len(guard.check(fixed, context, question)) < len(issues):
                    return fixed
            except Exception as exc:
                log("Repair skipped: " + type(exc).__name__)
            cleaned = policy.postprocess(guard.strip_bad_sentences(answer))
            if cleaned and len(cleaned) >= 30 and len(guard.check(cleaned, context, question)) < len(issues):
                return cleaned
    return answer


def main() -> None:
    started = time.monotonic()
    config.DEADLINE = started + config.TOTAL_BUDGET_SEC
    questions = read_questions()
    log(f"Questions: {len(questions)}, model present: {bool(config.LLM_MODEL)}, "
        f"proxy present: {bool(config.BASE_URL)}")

    # Produce a valid submission *before* any indexing/HTTP work. Later snapshots
    # preserve successfully generated answers even if the container times out.
    answers = []
    for _, raw in questions:
        p = dialog.parse(raw)
        topics = dialog.route_topics(p["question"], p["routing_text"])
        answers.append(policy.fallback(p["question"], topics))
    write_csv(config.OUT_FILE, questions, answers)

    try:
        kb = kbmod.load()
        index = retrieval.Index(kb)
        log(f"KB: {len(kb.docs)} documents, {len(kb.chunks)} chunks")
        if kb.chunks:
            index.build()
        log(f"Index: postings={len(index.postings)}, embedded={len(index.embedded_ids)}, "
            f"elapsed={time.monotonic()-started:.1f}s")
    except Exception as exc:
        log("KB/index load failed: " + type(exc).__name__ + ": " + str(exc))
        kb = kbmod.KB()
        index = retrieval.Index(kb)

    # Questions are independent; only exact duplicates are reused, never any
    # cross-ID dialogue state.
    grouped = {}
    for i, (_, raw) in enumerate(questions):
        grouped.setdefault(raw.strip(), []).append(i)
    completed = 0
    errors = 0

    def work(raw: str) -> tuple[str, str, bool]:
        if llm.time_left() < 25:
            return raw, "", True
        try:
            answer = answer_one(index, raw)
            return raw, answer, False
        except Exception as exc:
            log("Question generation failed: " + type(exc).__name__ + ": " + str(exc)[:120])
            return raw, "", True

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = [pool.submit(work, raw) for raw in grouped]
        for f in as_completed(futures):
            try:
                raw, answer, failed = f.result()
            except Exception:
                errors += 1
                log(traceback.format_exc(limit=2))
                continue
            if failed:
                errors += 1
            for i in grouped[raw]:
                if answer:
                    answers[i] = answer[:10000]
                completed += 1
            if completed % 4 == 0 or completed >= len(questions):
                write_csv(config.OUT_FILE, questions, answers)
                log(f"Answered {completed}/{len(questions)}, errors={errors}, "
                    f"time_left={llm.time_left():.0f}s")

    write_csv(config.OUT_FILE, questions, answers)
    log(f"Finished: {len(questions)} rows, errors={errors}, "
        f"total_time={time.monotonic()-started:.1f}s; file={config.OUT_FILE}")


if __name__ == "__main__":
    main()
