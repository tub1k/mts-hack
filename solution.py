"""Точка входа этапа 1: questions.csv -> submission.csv."""
from __future__ import annotations

import csv
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

_write_lock = threading.Lock()


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def read_questions() -> list[tuple[str, str]]:
    path = config.find_file(config.QUESTIONS_FILE, "questions.csv", "train.csv")
    if not path:
        raise FileNotFoundError("Не найден questions.csv / train.csv")
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        qid = str(row.get("id", "")).strip()
        question = str(row.get("question", "") or "")
        if qid:
            out.append((qid, question))
    if not out:
        raise ValueError("questions.csv пуст")
    return out


def write_csv(path: str, questions: list[tuple[str, str]], answers: list[str]) -> None:
    with _write_lock:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
            w.writerow(["id", "answer"])
            for (qid, _), ans in zip(questions, answers):
                w.writerow([qid, ans or policy.FALLBACK])
        with open(tmp, "rb") as src, open(path, "wb") as dst:
            dst.write(src.read())


def answer_one(index: retrieval.Index, raw: str) -> str:
    parsed = dialog.parse(raw)
    question = parsed["question"]
    facts = dialog.extract_facts(parsed["user_text"])
    topics = dialog.route_topics(question, parsed["user_text"])

    ranked = index.search(question, topics)
    context = retrieval.assemble(index, question, topics, ranked)
    messages = policy.build_messages(context, question, parsed["history"], facts)

    answer = policy.postprocess(llm.chat(messages))
    if not answer:
        return policy.FALLBACK

    if config.USE_REPAIR and llm.time_left() > 75:
        issues = guard.check(answer, context, question)
        if issues:
            try:
                fixed = policy.postprocess(llm.chat(guard.repair_messages(messages, answer, issues)))
                if fixed and not guard.check(fixed, context, question):
                    return fixed
                if fixed and len(guard.check(fixed, context, question)) < len(issues):
                    return fixed
            except Exception as exc:
                log(f"repair: {type(exc).__name__}")
            stripped = policy.postprocess(guard.strip_bad_sentences(answer))
            if stripped and len(stripped) > 40:
                return stripped
    return answer


def main() -> None:
    started = time.monotonic()
    config.DEADLINE = started + config.TOTAL_BUDGET_SEC

    questions = read_questions()
    log(f"Вопросов: {len(questions)}; модель: {config.LLM_MODEL or 'НЕ ЗАДАНА'}; API: {config.BASE_URL or 'НЕ ЗАДАН'}")

    kb = kbmod.load()
    log(f"Документов: {len(kb.docs)}; фрагментов: {len(kb.chunks)}")
    index = retrieval.Index(kb)
    index.build()
    log(f"Индекс готов за {time.monotonic() - started:.1f} c; "
        f"embeddings={index.matrix is not None}; bm25={index.bm25 is not None}")

    answers = [policy.FALLBACK] * len(questions)
    write_csv(config.OUT_FILE, questions, answers)

    cache: dict[str, str] = {}
    cache_lock = threading.Lock()
    done = 0

    def work(i: int, raw: str) -> tuple[int, str]:
        if llm.time_left() < 20:
            return i, policy.FALLBACK
        key = raw.strip()
        with cache_lock:
            if key in cache:
                return i, cache[key]
        try:
            result = answer_one(index, raw)
        except Exception as exc:
            log(f"Вопрос {i + 1}: {type(exc).__name__}: {exc}")
            result = policy.FALLBACK
        result = result or policy.FALLBACK
        with cache_lock:
            cache[key] = result
        return i, result

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = [pool.submit(work, i, q) for i, (_, q) in enumerate(questions)]
        for fut in as_completed(futures):
            try:
                i, ans = fut.result()
            except Exception:
                log(traceback.format_exc(limit=2))
                continue
            answers[i] = ans
            done += 1
            if done % 5 == 0 or done == len(questions):
                write_csv(config.OUT_FILE, questions, answers)
                log(f"Готово {done}/{len(questions)}; осталось {llm.time_left():.0f} c")

    write_csv(config.OUT_FILE, questions, answers)
    empty = sum(1 for a in answers if not a.strip())
    log(f"submission.csv записан: {len(answers)} строк, пустых {empty}, "
        f"время {time.monotonic() - started:.1f} c")


if __name__ == "__main__":
    main()