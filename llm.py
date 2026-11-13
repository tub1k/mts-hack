"""Dependency-free, deadline-aware client for the platform's OpenAI-compatible API."""
from __future__ import annotations

import json
import random
import re
import threading
import time
from urllib import error, request
from urllib.parse import urlsplit

import config

THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.I | re.S)
THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.I)
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_state_lock = threading.Lock()
_state = {"emb_fail": 0, "rerank_fail": 0, "emb_ok": True,
          "rerank_ok": True, "thinking_flag": True}


class ApiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        self.status = status
        super().__init__(message)


def time_left() -> float:
    return config.remaining()


def _time_budget(requested: int) -> float:
    remaining = time_left()
    if remaining < 7:
        raise ApiError("Общий лимит времени исчерпан")
    return max(1.5, min(float(requested), remaining - 5))


def post(endpoint: str, payload: dict, timeout: int, attempts: int = 2) -> dict:
    if not config.BASE_URL:
        raise ApiError("LLM_BASE_URL / OPENAI_BASE_URL не задан")
    url = endpoint if endpoint.startswith(("http://", "https://")) else config.BASE_URL + endpoint
    headers = {
        "Authorization": "Bearer " + config.API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last = None
    for attempt in range(attempts):
        delay = None
        try:
            req = request.Request(url, data=body, headers=headers, method="POST")
            with request.urlopen(req, timeout=_time_budget(timeout)) as response:
                raw = response.read()
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise ApiError(endpoint + " вернул не объект JSON")
            return data
        except error.HTTPError as exc:
            # Never log payloads or authorization headers.
            last = ApiError(f"{endpoint}: HTTP {exc.code}", exc.code)
            if exc.code not in RETRY_STATUS or attempt + 1 >= attempts:
                raise last from None
            retry_after = exc.headers.get("Retry-After", "") if exc.headers else ""
            try:
                delay = max(0.0, min(10.0, float(retry_after)))
            except (ValueError, TypeError):
                delay = None
        except (error.URLError, TimeoutError, OSError, ValueError, ApiError) as exc:
            last = exc if isinstance(exc, ApiError) else ApiError(f"{endpoint}: {type(exc).__name__}")
            if attempt + 1 >= attempts:
                break
        if delay is None:
            delay = min(7.0, 0.8 * (2 ** attempt) + random.random())
        if time_left() < delay + 9:
            break
        time.sleep(delay)
    raise last or ApiError("API недоступно")


def _is_local_ollama() -> bool:
    """Detect Ollama localhost. The platform's OpenAI proxy is unaffected."""
    try:
        address = urlsplit(config.BASE_URL)
        return address.hostname in ("localhost", "127.0.0.1", "::1") and address.port == 11434
    except ValueError:
        return False


def _ollama_chat(messages: list[dict], tokens: int) -> str:
    """Use native Ollama chat to reliably suppress costly thinking on Qwen3.5."""
    base = config.BASE_URL.removesuffix("/v1").rstrip("/")
    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": config.TEMPERATURE, "num_predict": min(tokens, 1024)},
    }
    # Local CPU/GPU can take longer to load and process a context than the judge.
    # One sufficiently long request prevents multiple costly timeout retries.
    result = post(base + "/api/chat", payload, max(240, config.CHAT_TIMEOUT), attempts=1)
    message = result.get("message") or {}
    answer = _clean(_content(message.get("content"))) if isinstance(message, dict) else ""
    if not answer:
        raise ApiError("Ollama /api/chat вернула пустой ответ")
    return answer


def _content(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join((v.get("text") or "") if isinstance(v, dict) else ""
                         for v in value)
    return ""


def _clean(text: str) -> str:
    text = THINK_BLOCK_RE.sub("", text or "")
    # If the model returned only an unfinished thinking block, do not show it.
    if THINK_OPEN_RE.search(text):
        text = text[:THINK_OPEN_RE.search(text).start()]
    text = re.sub(r"</?\|?im_(?:start|end)\|?>", "", text, flags=re.I)
    text = re.sub(r"^\s*(?:assistant|final)\s*:\s*", "", text, flags=re.I)
    return text.strip()


def chat(messages: list[dict], max_tokens: int | None = None) -> str:
    if not config.LLM_MODEL:
        raise ApiError("Не задан LLM_MODEL / OPENAI_MODEL / GEN_MODEL")
    tokens = max_tokens or config.MAX_TOKENS
    if _is_local_ollama():
        return _ollama_chat(messages, tokens)
    with _state_lock:
        use_thinking_flag = _state["thinking_flag"]
    last = None
    for attempt in range(3):
        payload = {
            "model": config.LLM_MODEL,
            "messages": messages,
            "temperature": config.TEMPERATURE,
            "max_tokens": tokens,
        }
        if use_thinking_flag:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            data = post("/chat/completions", payload, config.CHAT_TIMEOUT, attempts=2)
        except ApiError as exc:
            last = exc
            if use_thinking_flag and exc.status in (400, 422):
                # Some OpenAI-compatible proxies reject vLLM-specific fields.
                with _state_lock:
                    _state["thinking_flag"] = False
                use_thinking_flag = False
                continue
            if exc.status in (401, 403, 404):
                break
            if time_left() < 35:
                break
            continue
        try:
            choice = (data.get("choices") or [])[0]
            message = choice.get("message") or {}
            text = _clean(_content(message.get("content")))
            if text:
                return text
            # Empty content often means the reasoning budget consumed the output.
            last = ApiError("Модель вернула пустой финальный ответ")
        except (TypeError, KeyError, IndexError, AttributeError):
            last = ApiError("Некорректный формат chat/completions")
        if time_left() < 40:
            break
        tokens = min(4096, max(tokens + 500, int(tokens * 1.6)))
        if use_thinking_flag:
            with _state_lock:
                _state["thinking_flag"] = False
            use_thinking_flag = False
    raise last or ApiError("Модель не вернула ответ")


def embeddings_available() -> bool:
    with _state_lock:
        return config.USE_EMBEDDINGS and _state["emb_ok"]


def reranker_available() -> bool:
    with _state_lock:
        return config.USE_RERANKER and _state["rerank_ok"]


def embeddings(texts: list[str]) -> list[list[float]]:
    if not embeddings_available():
        raise ApiError("Эмбеддинги отключены")
    if not texts:
        return []
    out: list[list[float]] = []
    try:
        for start in range(0, len(texts), 24):
            batch = texts[start:start + 24]
            payload = {"model": config.EMB_MODEL, "input": batch}
            data = post("/embeddings", payload, config.EMB_TIMEOUT, attempts=2)
            result = data.get("data") or []
            if len(result) != len(batch):
                raise ApiError("Не все эмбеддинги вернулись")
            order = sorted(result, key=lambda v: int(v.get("index", 0)))
            vectors = [[float(x) for x in item["embedding"]] for item in order]
            if not vectors or not vectors[0]:
                raise ApiError("Пустые эмбеддинги")
            out.extend(vectors)
        with _state_lock:
            _state["emb_fail"] = 0
        return out
    except (ApiError, TypeError, KeyError, ValueError) as exc:
        with _state_lock:
            _state["emb_fail"] += 1
            if isinstance(exc, ApiError) and exc.status in (400, 401, 403, 404, 422):
                _state["emb_ok"] = False
            elif _state["emb_fail"] >= 2:
                _state["emb_ok"] = False
        raise ApiError(str(exc)) from None


def rerank(query: str, docs: list[str], top_n: int) -> list[int]:
    if not reranker_available() or not docs:
        raise ApiError("Reranker недоступен")
    payload = {"model": config.RERANKER_MODEL, "query": query,
               "documents": docs, "top_n": max(1, min(top_n, len(docs)))}
    try:
        data = post("/rerank", payload, config.RERANK_TIMEOUT, attempts=1)
        result = data.get("results") or data.get("data") or data.get("rankings") or []
        if isinstance(result, dict):
            result = result.get("results") or []
        if not isinstance(result, list):
            raise ApiError("Неизвестный формат rerank")
        pairs = []
        for i, entry in enumerate(result):
            index = int(entry.get("index", entry.get("document_index", -1)))
            if 0 <= index < len(docs):
                pairs.append((index, float(entry.get("relevance_score", entry.get("score", -i)))))
        if not pairs:
            raise ApiError("Пустой результат rerank")
        with _state_lock:
            _state["rerank_fail"] = 0
        return [i for i, _ in pairs]
    except (ApiError, KeyError, TypeError, ValueError) as exc:
        with _state_lock:
            _state["rerank_fail"] += 1
            if isinstance(exc, ApiError) and exc.status in (400, 401, 403, 404, 422):
                _state["rerank_ok"] = False
            elif _state["rerank_fail"] >= 2:
                _state["rerank_ok"] = False
        raise ApiError(str(exc)) from None
