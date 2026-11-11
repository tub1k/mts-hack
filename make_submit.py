"""Сборка плоского ZIP-сабмита и проверка формата submission.csv."""
from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from pathlib import Path

FILES = ["solution.py", "config.py", "llm.py", "kb.py", "dialog.py",
         "retrieval.py", "policy.py", "guard.py", "requirements.txt"]
FORBIDDEN = {".env", "questions.csv", "train.csv", "knowledge-base.zip", "submission.csv"}
MAX_ZIP = 128 * 1024 * 1024


def build(name: str = "submit.zip") -> None:
    root = Path(__file__).resolve().parent
    missing = [f for f in FILES if not (root / f).is_file()]
    if missing:
        sys.exit(f"Нет файлов: {missing}")
    for f in FILES:
        if f.endswith(".py"):
            compile((root / f).read_text(encoding="utf-8"), f, "exec")
    with zipfile.ZipFile(root / name, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in FILES:
            zf.write(root / f, arcname=f)
    size = (root / name).stat().st_size
    with zipfile.ZipFile(root / name) as zf:
        names = zf.namelist()
    assert all("/" not in n for n in names), "В архиве есть вложенные папки"
    assert not (set(names) & FORBIDDEN), "В архив попали запрещённые файлы"
    assert size <= MAX_ZIP, "Архив больше 128 МБ"
    print(f"{name}: {size / 1024:.1f} КБ, {len(names)} файлов в корне")
    print("\n".join("  " + n for n in names))


def check(path: str = "submission.csv") -> None:
    p = Path(path)
    if not p.is_file():
        sys.exit("submission.csv не найден")
    with p.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "answer"]:
            sys.exit(f"Неверный заголовок: {reader.fieldnames}")
        rows = list(reader)
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        sys.exit("Есть повторяющиеся ID")
    empty = [r["id"] for r in rows if not (r["answer"] or "").strip()]
    if empty:
        sys.exit(f"Пустые ответы: {empty}")
    long = [r["id"] for r in rows if len(r["answer"]) > 10000]
    if long:
        sys.exit(f"Ответы длиннее 10000 символов: {long}")
    words = [len(r["answer"].split()) for r in rows]
    print(f"OK: {len(rows)} строк; слов мин/сред/макс = "
          f"{min(words)}/{sum(words) // len(words)}/{max(words)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", metavar="CSV", nargs="?", const="submission.csv")
    args = ap.parse_args()
    if args.check:
        check(args.check)
    else:
        build()