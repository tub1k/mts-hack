import time
from openai import OpenAI
import config
from prompts import SYSTEM_PROMPT
from rag import KnowledgeBaseRAG

class ConsultantAgent:
    def __init__(self, rag: KnowledgeBaseRAG):
        self.rag = rag
        self.client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)

    def answer_question(self, question_or_history: str, chat_history: list[dict] | None = None) -> dict:
        start_time = time.perf_counter()

        # Поиск релевантных фрагментов в базе знаний
        docs = self.rag.search(question_or_history)
        context_str = "\n\n---\n\n".join(
            [f"[{d['source']}]: {d['text']}" for d in docs]
        ) if docs else "Релевантных документов в базе знаний не найдено."

        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context_str)}]
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": question_or_history})

        # Ограниченное число повторов с паузой на случай 429 / 5xx / таймаута
        answer_text = "Извините, сервис временно перегружен. Пожалуйста, оставьте контакты для связи со специалистом."
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=messages,
                    temperature=config.TEMPERATURE,
                    max_tokens=config.MAX_TOKENS,
                    timeout=60.0
                )
                content = (resp.choices[0].message.content or "").strip()
                if content:
                    answer_text = content[:10000]  # Жесткий лимит 10 000 символов по п. 3
                    break
            except Exception:
                time.sleep(2 ** attempt)

        elapsed = round(time.perf_counter() - start_time, 3)
        return {
            "answer": answer_text,
            "sources": docs,
            "elapsed_sec": elapsed
        }