import re
import time
from openai import OpenAI
import config
from prompts import SYSTEM_PROMPT
from rag import KnowledgeBaseRAG


def parse_question_field(raw_text: str, external_history: list[dict] | None = None) -> tuple[str, list[dict], str]:
    if external_history:
        user_msgs = [m["content"] for m in external_history if m["role"] == "user"] + [raw_text]
        return " ".join(user_msgs).strip(), list(external_history), raw_text.strip()

    if "Пользователь:" in raw_text and "Ассистент:" in raw_text:
        parts = re.split(r"(?=Пользователь:|Ассистент:)", raw_text.strip())
        messages = []
        user_utterances = []
        for part in parts:
            part = part.strip()
            if part.startswith("Пользователь:"):
                content = part.replace("Пользователь:", "", 1).strip()
                messages.append({"role": "user", "content": content})
                user_utterances.append(content)
            elif part.startswith("Ассистент:"):
                content = part.replace("Ассистент:", "", 1).strip()
                messages.append({"role": "assistant", "content": content})

        if messages and messages[-1]["role"] == "user":
            return " ".join(user_utterances), messages[:-1], messages[-1]["content"]

    clean = raw_text.strip()
    return clean, [], clean


def clean_model_output(text: str) -> str:
    # 1. Вырезаем блоки размышлений <think>...</think> (если модель их сгенерировала)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # На случай незакрытого <think> при упоре в лимит токенов
    if "<think>" in text:
        text = text.split("</think>")[-1].replace("<think>", "").strip()

    # 2. Убираем лишнее форматирование Markdown (жирный шрифт и заголовки)
    text = text.replace("**", "").replace("__", "")
    # 3. Схлопываем множественные переносы строк в один абзац, как в эталонах
    lines = [line.lstrip("-•* ").strip() for line in text.splitlines() if line.strip()]
    single_paragraph = " ".join(lines)
    return single_paragraph.strip(' "«»')


class ConsultantAgent:
    def __init__(self, rag: KnowledgeBaseRAG):
        self.rag = rag
        self.client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)

    def answer_question(self, question_or_history: str, chat_history: list[dict] | None = None) -> dict:
        start_time = time.perf_counter()

        search_query, parsed_history, current_user_msg = parse_question_field(
            question_or_history, chat_history
        )

        docs = self.rag.search(search_query)
        context_str = "\n\n---\n\n".join(
            [f"[{d['source']}]:\n{d['text']}" for d in docs]
        ) if docs else "Релевантных документов в базе знаний не найдено."

        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context_str)}]
        messages.extend(parsed_history)
        messages.append({"role": "user", "content": current_user_msg})

        answer_text = "Уточните, пожалуйста, параметры вашего объекта и текущую задачу, чтобы мы подобрали подходящее решение МТС Energytool."
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=messages,
                    temperature=config.TEMPERATURE,
                    max_tokens=config.MAX_TOKENS,
                    timeout=90.0,
                )
                raw_content = (resp.choices[0].message.content or "").strip()
                cleaned = clean_model_output(raw_content)
                if cleaned:
                    answer_text = cleaned[:10000]
                    break
            except Exception:
                time.sleep(2 ** attempt)

        elapsed = round(time.perf_counter() - start_time, 3)
        return {
            "answer": answer_text,
            "sources": docs,
            "elapsed_sec": elapsed,
        }