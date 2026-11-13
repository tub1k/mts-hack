"""Build the flat, compliant ZIP and validate produced submission.csv."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import zipfile

FILES = ["solution.py", "config.py", "llm.py", "kb.py", "dialog.py",
         "retrieval.py", "policy.py", "guard.py", "requirements.txt"]
FORBIDDEN = {".env", "questions.csv", "demo_questions.csv", "demo_output.csv",
             "knowledge-base.zip", "submission.csv", "train.csv"}


def build(name: str = "mts_energytool_submission.zip") -> Path:
    root = Path(__file__).resolve().parent
    missing = [f for f in FILES if not (root / f).is_file()]
    if missing:
        raise FileNotFoundError("Missing files: " + ", ".join(missing))
    for name_py in FILES:
        if name_py.endswith(".py"):
            compile((root / name_py).read_text(encoding="utf-8"), name_py, "exec")
    destination = root / name
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as out:
        for file in FILES:
            out.write(root / file, arcname=file)
    with zipfile.ZipFile(destination) as zipped:
        names = zipped.namelist()
        assert names == FILES, "Unexpected archive contents"
        assert all("/" not in name for name in names)
        assert not (set(names) & FORBIDDEN)
        assert zipped.testzip() is None
    assert destination.stat().st_size <= 128 * 1024 * 1024
    print(f"Created: {destination} ({destination.stat().st_size} bytes)")
    return destination


def check(path: str = "submission.csv", questions_path: str | None = None) -> int:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["id", "answer"], reader.fieldnames
        rows = list(reader)
    assert rows, "No answers"
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "Duplicate IDs"
    assert all(r["answer"].strip() and len(r["answer"]) <= 10000 for r in rows)
    if questions_path:
        with open(questions_path, encoding="utf-8-sig", newline="") as f:
            source_ids = [r["id"] for r in csv.DictReader(f)]
        assert ids == source_ids, "IDs or their order differ from questions.csv"
    print(f"CSV is valid: {len(rows)} IDs, no empty answers, exactly two columns")
    return len(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", help="Validate an existing submission.csv")
    p.add_argument("--questions", help="Compare IDs with questions.csv")
    p.add_argument("--output", default="submit.zip")
    args = p.parse_args()
    if args.check:
        check(args.check, args.questions)
    else:
        build(args.output)
