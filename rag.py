import os
import re
import time
import zipfile
import httpx
import numpy as np
import faiss
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import config


def tokenize_ru(text: str) -> list[str]:
    return re.findall(r"[а-яёa-z0-9]+", text.lower())


class KnowledgeBaseRAG:
    def __init__(self):
        self.client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
        self.chunks: list[dict] = []
        self.index = None
        self.bm25 = None

    def _read_file_text(self, full_path: str, filename: str) -> str:
        ext = os.path.splitext(filename)[1].lower()
        try:
            if ext in (".txt", ".md", ".csv", ".json", ".yaml", ".yml"):
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read().strip()
            elif ext == ".html":
                from bs4 import BeautifulSoup
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    soup = BeautifulSoup(f.read(), "lxml")
                    return soup.get_text(separator="\n", strip=True)
            elif ext == ".docx":
                import docx
                doc = docx.Document(full_path)
                return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            elif ext == ".pdf":
                from pypdf import PdfReader
                reader = PdfReader(full_path)
                return "\n".join([page.extract_text() or "" for page in reader.pages])
        except Exception:
            pass
        return ""

    def build_from_zip(self, zip_path: str = config.KB_ZIP_PATH, extract_dir: str = "/tmp/mts_kb"):
        if os.path.exists(zip_path):
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            search_root = extract_dir
        elif os.path.exists(config.DATA_DIR):
            search_root = config.DATA_DIR
        else:
            return

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
        )

        for root, _, files in os.walk(search_root):
            for file in sorted(files):
                if file.startswith(".") or file.endswith(".zip"):
                    continue
                full_path = os.path.join(root, file)
                text = self._read_file_text(full_path, file)
                if not text:
                    continue

                doc_title = os.path.splitext(file)[0].replace("_", " ").replace("-", " ")
                split_texts = splitter.split_text(text)
                for chunk in split_texts:
                    # Вшиваем название документа в текст чанка, чтобы не терялся контекст
                    enriched_text = f"Документ: {doc_title}\n{chunk.strip()}"
                    self.chunks.append({"source": file, "text": enriched_text})

        if not self.chunks:
            return

        # 1. Строим плотный векторный индекс FAISS (BAAI/bge-m3)
        vectors = self._embed_texts([c["text"] for c in self.chunks])
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)

        # 2. Строим лексический индекс BM25 для точного поиска цифр и терминов
        tokenized_corpus = [tokenize_ru(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for attempt in range(3):
                try:
                    resp = self.client.embeddings.create(model=config.EMB_MODEL, input=batch)
                    all_embs.extend([item.embedding for item in resp.data])
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
        return np.array(all_embs, dtype="float32")

    def _rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        if not candidates or len(candidates) <= top_n:
            return candidates

        base_url = config.LLM_BASE_URL.rstrip("/")
        headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"}
        payload = {
            "model": config.RERANKER_MODEL,
            "query": query,
            "documents": [c["text"] for c in candidates],
            "top_n": top_n,
        }

        # Пробуем стандартные пути OpenAI-совместимого прокси для реранкера
        for endpoint in (f"{base_url}/rerank", f"{base_url.replace('/v1', '')}/v1/rerank"):
            try:
                resp = httpx.post(endpoint, json=payload, headers=headers, timeout=15.0)
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", [])
                    if results:
                        reranked = []
                        for item in results[:top_n]:
                            idx = item["index"]
                            cand = dict(candidates[idx])
                            cand["score"] = float(item.get("relevance_score", cand["score"]))
                            reranked.append(cand)
                        return reranked
            except Exception:
                continue

        # Фоллбэк: если локально в Ollama нет реранкера, возвращаем топ по гибридному скору RRF
        return candidates[:top_n]

    def search(self, query: str, top_k: int = config.TOP_K, top_n: int = config.RERANKER_TOP_N) -> list[dict]:
        if not self.index or not self.chunks:
            return []

        k = min(top_k, len(self.chunks))

        # 1. Векторный поиск (FAISS)
        q_vec = self._embed_texts([query])
        faiss.normalize_L2(q_vec)
        vec_scores, vec_indices = self.index.search(q_vec, k)

        # 2. Полнотекстовый поиск (BM25)
        bm25_scores = self.bm25.get_scores(tokenize_ru(query))
        bm25_indices = np.argsort(bm25_scores)[::-1][:k]

        # 3. Гибридное слияние Reciprocal Rank Fusion (RRF)
        rrf_scores: dict[int, float] = {}
        for rank, idx in enumerate(vec_indices[0]):
            if idx != -1:
                rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0.0) + 1.0 / (60 + rank + 1)

        for rank, idx in enumerate(bm25_indices):
            if bm25_scores[idx] > 0:
                rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0.0) + 1.0 / (60 + rank + 1)

        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:k]
        candidates = [
            {
                "source": self.chunks[i]["source"],
                "text": self.chunks[i]["text"],
                "score": round(rrf_scores[i], 5),
            }
            for i in sorted_indices
        ]

        # 4. Дожимаем точность через Qwen3 Reranker 8B
        return self._rerank(query, candidates, top_n)