"""Конфигурация. На платформе значения приходят из окружения, локально — из .env."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)  # окружение платформы всегда важнее .env
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent


def env(name: str, default: str = "") -> str:
    value = os.getenv(name, "")
    value = value.strip() if isinstance(value, str) else ""
    if not value or value == name:  # платформа иногда передаёт литерал-заглушку
        return default
    return value


def env_num(name, default, lo, hi, cast=int):
    try:
        return max(lo, min(hi, cast(float(env(name, str(default))))))
    except (TypeError, ValueError):
        return default


# --- доступ к API ---
BASE_URL = (env("LLM_BASE_URL") or env("BASE_URL") or env("OPENAI_BASE_URL")).rstrip("/")
API_KEY = env("LLM_API_KEY") or env("OPENAI_API_KEY") or "local-key"
EMB_KEY = env("EMBEDDER_API_KEY") or API_KEY
RERANK_KEY = env("RERANKER_API_KEY") or API_KEY

LLM_MODEL = env("LLM_MODEL") or env("GEN_MODEL") or env("OPENAI_MODEL")
EMB_MODEL = env("EMB_MODEL", "baai/bge-m3")
RERANKER_MODEL = env("RERANKER_MODEL", "qwen/qwen3-reranker-8b")

# --- данные ---
DATA_DIR = Path(env("DATA_DIR", "data"))
QUESTIONS_FILE = env("QUESTIONS_FILE")
KB_FILE = env("KB_FILE")
OUT_FILE = env("OUT_FILE", "submission.csv")

# --- бюджеты ---
TOTAL_BUDGET_SEC = env_num("TOTAL_BUDGET_SEC", 1020, 120, 1140)
CHAT_TIMEOUT = env_num("CHAT_TIMEOUT", 150, 20, 400)
EMB_TIMEOUT = env_num("EMB_TIMEOUT", 90, 15, 300)
INDEX_BUDGET_SEC = env_num("INDEX_BUDGET_SEC", 240, 30, 600)

MAX_WORKERS = env_num("MAX_WORKERS", 4, 1, 8)
MAX_TOKENS = env_num("MAX_TOKENS", 2048, 512, 8192)
TEMPERATURE = env_num("TEMPERATURE", 0.0, 0.0, 0.4, float)

# --- поиск ---
CONTEXT_CHARS = env_num("CONTEXT_CHARS", 22000, 4000, 60000)
CHUNK_CHARS = env_num("CHUNK_CHARS", 1000, 300, 2500)
TOP_DOCS = env_num("TOP_DOCS", 3, 1, 6)
RERANK_CANDIDATES = env_num("RERANK_CANDIDATES", 24, 4, 60)
USE_EMBEDDINGS = env("USE_EMBEDDINGS", "1") != "0"
USE_RERANKER = env("USE_RERANKER", "1") != "0"
USE_REPAIR = env("USE_REPAIR", "1") != "0"

# --- стиль ---
TARGET_WORDS = env_num("TARGET_WORDS", 80, 30, 140)
HARD_WORDS = env_num("HARD_WORDS", 110, 50, 150)

DEADLINE = float("inf")  # выставляется в solution.main()


def find_file(*names: str) -> Path | None:
    roots = [DATA_DIR, SCRIPT_DIR / DATA_DIR, Path.cwd(), SCRIPT_DIR]
    for root in roots:
        for name in names:
            if not name:
                continue
            p = Path(name)
            cand = p if p.is_absolute() else root / p
            if cand.is_file():
                return cand
    return None


def find_dir(*names: str) -> Path | None:
    roots = [DATA_DIR, SCRIPT_DIR / DATA_DIR, Path.cwd(), SCRIPT_DIR]
    for root in roots:
        for name in names:
            cand = root / name
            if cand.is_dir() and any(cand.rglob("*.md")):
                return cand
    return None