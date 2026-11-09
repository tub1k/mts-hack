import os
import zipfile
from pathlib import Path
import numpy as np
import faiss
from openai import OpenAI
import config

class KnowledgeBaseRAG:
    def __init__(self):
        self.client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
        self.chunks: list[dict] = []
        self.index = None

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

        # Сбор текстовых файлов, .md, .docx, .pdf (команда допишет парсеры под формат базы)
        for root, _, files in os.walk(search_root):
            for file in files:
                if file.endswith((".txt", ".md", ".html")):
                    full_path = os.path.join(root, file)
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read().strip()
                    if text:
                        # Базовая нарезка по 1000 символов (замените на RecursiveCharacterTextSplitter)
                        for i in range(0, len(text), 800):
                            chunk_text = text[i:i + 1000]
                            self.chunks.append({"source": file, "text": chunk_text})

        if not self.chunks:
            return

        # Получение эмбеддингов батчами через прокси BAAI/bge-m3
        vectors = self._embed_texts([c["text"] for c in self.chunks])
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)

    def _embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self.client.embeddings.create(model=config.EMB_MODEL, input=batch)
            all_embs.extend([item.embedding for item in resp.data])
        return np.array(all_embs, dtype="float32")

    def search(self, query: str, top_k: int = config.RERANKER_TOP_N) -> list[dict]:
        if not self.index or not self.chunks:
            return []
        q_vec = self._embed_texts([query])
        faiss.normalize_L2(q_vec)
        scores, indices = self.index.search(q_vec, min(top_k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append({
                    "source": self.chunks[idx]["source"],
                    "text": self.chunks[idx]["text"],
                    "score": float(score)
                })
        return results