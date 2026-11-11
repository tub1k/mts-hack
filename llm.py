"""Тонкий HTTP-клиент к OpenAI-совместимому прокси: ретраи, таймауты, предохранители."""
from __future__ import annotations

import random
import re
import sys
import threading
import time

import requests

import config

_local = threading.local()
_lock = threading.Lock()
_state = {"emb_fail": 0, "rerank_fail": 0, "emb_ok": True, "rerank_ok": True}

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
THINK_RE = re.compile(r"", re.S | re.I)


class ApiError(RuntimeError):
    pass


def _session() -> requests.Session:
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=0)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        _local.session = s
    return s


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def time_left() -> float:
    return config.DEADLINE - time.monotonic()


def post(endpoint: str, key: str, payload: dict, timeout: int, attempts: int = 3) -> dict:
    if not config.BASE_URL:
        raise ApiError("Не задан LLM_BASE_URL")
    url = config.BASE_URL + endpoint
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last: Exception | None = None
    for attempt in range(attempts):
        if time_left() <= 5:
            raise ApiError("Исчерпан бюджет времени")
        try:
            resp = _session().post(url, headers=headers, json=payload, timeout=(10, timeout))
            if resp.status_code in RETRY_STATUS and attempt < attempts - 1:
                wait = min(12.0, 1.5 * (2 ** attempt)) + random.random()
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = min(20.0, max(wait, float(retry_after)))
                    except ValueError:
                        pass
                time.sleep(min(wait, max(0.0, time_left() - 5)))
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last = exc
            if attempt < attempts - 1:
                time.sleep(min(1.0 * (2 ** attempt) + random.random(), max(0.0, time_left() - 5)))
    raise ApiError(f"{endpoint}: {type(last).__name__}: {last}")


def _clean(text: str) -> str:
    text = THINK_RE.sub("", text or "")
    return text.strip()


def chat(messages: list[dict], max_tokens: int | None = None) -> str:
    tokens = max_tokens or config.MAX_TOKENS
    last_err: Exception | None = None
    for attempt in range(3):
        payload = {
            "model": config.LLM_MODEL,
            "messages": messages,
            "temperature": config.TEMPERATURE,
            "max_tokens": tokens,
        }
        try:
            data = post("/chat/completions", config.API_KEY, payload, config.CHAT_TIMEOUT)
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content = _clean(msg.get("content") or "")
            if content:
                return content
            if not msg.get("content") and choice.get("finish_reason") == "length":
                tokens = min(8192, tokens * 2)
                continue
            last_err = ApiError("Пустой ответ модели")
        except (ApiError, KeyError, TypeError, IndexError) as exc:
            last_err = exc
        if attempt == 1 and time_left() > 90:
            # последний шанс: некоторые прокси понимают управление режимом размышления
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            try:
                data = post("/chat/completions", config.API_KEY, payload, config.CHAT_TIMEOUT, attempts=1)
                content = _clean(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
                if content:
                    return content
            except Exception:
                pass
        tokens = min(8192, tokens * 2)
    raise ApiError(str(last_err or "LLM недоступна"))


def embeddings(texts: list[str]) -> list[list[float]]:
    if not config.USE_EMBEDDINGS or not _state["emb_ok"]:
        raise ApiError("Эмбеддинги отключены")
    out: list[list[float]] = []
    batch = 16
    try:
        for i in range(0, len(texts), batch):
            data = post("/embeddings", config.EMB_KEY,
                        {"model": config.EMB_MODEL, "input": texts[i:i + batch]},
                        config.EMB_TIMEOUT, attempts=2)
            items = data.get("data") or []
            if len(items) != len(texts[i:i + batch]):
                raise ApiError("Не все эмбеддинги получены")
            ordered = sorted(items, key=lambda x: int(x.get("index", 0)))
            out.extend([[float(v) for v in it["embedding"]] for it in ordered])
        with _lock:
            _state["emb_fail"] = 0
        return out
    except Exception as exc:
        with _lock:
            _state["emb_fail"] += 1
            if _state["emb_fail"] >= 2:
                _state["emb_ok"] = False
                _log("Embeddings отключены до конца прогона")
        raise ApiError(str(exc))


def rerank(query: str, documents: list[str], top_n: int) -> list[int]:
    if not config.USE_RERANKER or not _state["rerank_ok"] or not documents:
        raise ApiError("Реранкер отключён")
    try:
        data = post("/rerank", config.RERANK_KEY,
                    {"model": config.RERANKER_MODEL, "query": query,
                     "documents": documents, "top_n": min(top_n, len(documents))},
                    config.CHAT_TIMEOUT, attempts=1)
        entries = data.get("results") or data.get("data") or []
        idx = [int(e["index"]) for e in entries if 0 <= int(e.get("index", -1)) < len(documents)]
        if not idx:
            raise ApiError("Пустой ответ реранкера")
        with _lock:
            _state["rerank_fail"] = 0
        return idx
    except Exception as exc:
        with _lock:
            _state["rerank_fail"] += 1
            if _state["rerank_fail"] >= 2:
                _state["rerank_ok"] = False
                _log("Reranker отключён до конца прогона")
        raise ApiError(str(exc))


def embeddings_available() -> bool:
    return config.USE_EMBEDDINGS and _state["emb_ok"]


def reranker_available() -> bool:
    return config.USE_RERANKER and _state["rerank_ok"]