"""Runtime settings for the MTS Energytool challenge.

Only Python modules and requirements.txt are shipped. All data and credentials
come from the platform environment at runtime.
"""
from __future__ import annotations

import os
from pathlib import Path
import time

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_local_dotenv() -> None:
    """Read a local .env without extra dependencies; never override judge env vars."""
    for filepath in (SCRIPT_DIR / ".env", Path.cwd() / ".env"):
        try:
            lines = filepath.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeError):
            continue
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            name, separator, value = line.partition("=")
            name = name.strip()
            if not separator or not name.isidentifier():
                continue
            value = value.strip()
            if value.startswith(("\"", "'")):
                quote = value[0]
                closing = value.find(quote, 1)
                value = value[1:closing] if closing >= 0 else value[1:]
            else:
                value = value.split(" #", 1)[0].strip()
            os.environ.setdefault(name, value)


_load_local_dotenv()


def env(name: str, default: str = "") -> str:
    value = os.environ.get(name, "")
    value = value.strip() if isinstance(value, str) else ""
    return default if not value or value == name else value


def number(name: str, default, low, high, cast=int):
    try:
        result = cast(float(env(name, str(default))))
        return max(low, min(high, result))
    except (ValueError, OverflowError, TypeError):
        return default


BASE_URL = (env("LLM_BASE_URL") or env("OPENAI_BASE_URL") or env("BASE_URL")).rstrip("/")
API_KEY = env("LLM_API_KEY") or env("OPENAI_API_KEY") or "local-key"
LLM_MODEL = env("LLM_MODEL") or env("OPENAI_MODEL") or env("GEN_MODEL")
EMB_MODEL = env("EMB_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = env("RERANKER_MODEL", "Qwen3-Reranker-8B")

DATA_DIR = Path(env("DATA_DIR", "data"))
QUESTIONS_FILE = env("QUESTIONS_FILE")
KB_FILE = env("KB_FILE")
OUT_FILE = env("OUT_FILE", "submission.csv")

# The platform kills a run after 20 minutes; reserve at least 60s for flushing.
TOTAL_BUDGET_SEC = number("TOTAL_BUDGET_SEC", 1080, 120, 1140)
CHAT_TIMEOUT = number("CHAT_TIMEOUT", 85, 15, 240)
EMB_TIMEOUT = number("EMB_TIMEOUT", 55, 10, 150)
RERANK_TIMEOUT = number("RERANK_TIMEOUT", 35, 10, 90)
INDEX_BUDGET_SEC = number("INDEX_BUDGET_SEC", 130, 10, 300)
MAX_WORKERS = number("MAX_WORKERS", 4, 1, 8)
MAX_TOKENS = number("MAX_TOKENS", 1300, 512, 4096)
TEMPERATURE = number("TEMPERATURE", 0.0, 0.0, 0.7, float)

# Retrieval budgets are in characters, not tokens.
CHUNK_CHARS = number("CHUNK_CHARS", 1150, 550, 2200)
CONTEXT_CHARS = number("CONTEXT_CHARS", 13700, 6000, 25000)
TOP_K = number("TOP_K", 9, 5, 24)
RERANK_CANDIDATES = number("RERANK_CANDIDATES", 18, 8, 32)
RERANKER_TOP_N = number("RERANKER_TOP_N", 12, 4, 24)
EMBED_MAX_CHUNKS = number("EMBED_MAX_CHUNKS", 850, 150, 2000)
USE_EMBEDDINGS = env("USE_EMBEDDINGS", "1").lower() not in ("0", "false", "no")
USE_RERANKER = (env("USE_RERANKER") or env("USE_RERAINKER", "1")).lower() not in ("0", "false", "no")
USE_REPAIR = env("USE_REPAIR", "1").lower() not in ("0", "false", "no")

# Assigned in solution.main(); module-level default also supports local tests.
DEADLINE = float("inf")


def remaining() -> float:
    return DEADLINE - time.monotonic()


def candidate_paths(*names: str):
    roots = [DATA_DIR, SCRIPT_DIR / DATA_DIR, Path.cwd(), SCRIPT_DIR]
    for name in names:
        if not name:
            continue
        p = Path(name).expanduser()
        if p.is_absolute():
            yield p
        else:
            for root in roots:
                yield root / p


def find_file(*names: str) -> Path | None:
    seen = set()
    for path in candidate_paths(*names):
        if str(path) in seen:
            continue
        seen.add(str(path))
        if path.is_file():
            return path
    return None


def find_dir(*names: str) -> Path | None:
    for path in candidate_paths(*names):
        if path.is_dir():
            return path
    return None
