import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import config
from legacy.rag import KnowledgeBaseRAG
from legacy.agent import ConsultantAgent

def main():
    # 1. Инициализируем и строим индекс из knowledge-base.zip
    rag = KnowledgeBaseRAG()
    rag.build_from_zip(config.KB_ZIP_PATH)
    agent = ConsultantAgent(rag)

    # 2. Читаем входной questions.csv
    df = pd.read_csv(config.QUESTIONS_CSV_PATH, encoding="utf-8")
    questions = df[["id", "question"]].to_dict("records")

    def process_row(row: dict) -> tuple:
        q_id = row["id"]
        q_text = str(row["question"]).strip()
        res = agent.answer_question(q_text)
        ans = res["answer"].strip() or "Уточните, пожалуйста, вашу задачу для подбора решения."
        return q_id, ans

    # 3. Параллельная генерация с сохранением исходного порядка ID
    results_map = {}
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        future_to_id = {executor.submit(process_row, r): r["id"] for r in questions}
        for future in as_completed(future_to_id):
            q_id = future_to_id[future]
            try:
                _, ans = future.result()
                results_map[q_id] = ans
            except Exception:
                results_map[q_id] = "Для ответа на этот вопрос рекомендуем связаться со специалистом МТС Energytool."

    # 4. Сохраняем строго две колонки id,answer в UTF-8 со стандартным экранированием
    out_rows = [{"id": r["id"], "answer": results_map[r["id"]]} for r in questions]
    out_df = pd.DataFrame(out_rows, columns=["id", "answer"])
    out_df.to_csv(
        config.SUBMISSION_CSV_PATH,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL
    )

if __name__ == "__main__":
    main()