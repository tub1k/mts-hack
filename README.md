### Структура папок

```
mts-energytool/
│
├── solution.py          → ZIP   точка входа: questions.csv → submission.csv
├── config.py            → ZIP   env/.env, пути, бюджеты
├── llm.py               → ZIP   HTTP-клиент: chat / embeddings / rerank
├── kb.py                → ZIP   чтение базы, нарезка по секциям, блоки правил
├── dialog.py            → ZIP   история диалога, факты о клиенте, маршрут темы
├── retrieval.py         → ZIP   BM25 + эмбеддинги + RRF + реранкер, сборка контекста
├── policy.py            → ZIP   промпт, эталонные примеры, постобработка
├── guard.py             → ZIP   проверка фактов и формы, починка
├── requirements.txt     → ZIP   4 зависимости
│
├── make_submit.py       локально: сборка архива + валидация CSV
├── .env                 локально: Ollama (в ZIP НЕ кладём!)
├── .gitignore           локально
│
├── data/                локально; платформа подкладывает эти файлы сама
│   ├── questions.csv        твой урезанный на 1 вопрос
│   ├── knowledge-base.zip   исходный архив с 3 папками, распаковывать не нужно
│   └── demo_output.csv      эталон для сверки, кодом не читается
│
├── legacy/              старые файлы, чтобы не мешали импортам
│   ├── rag.py
│   ├── agent.py
│   ├── prompts.py
│   ├── app.py               понадобится на этапе 2
│   └── solution_old.py
│
├── submission.csv       создаётся при прогоне
├── submit.zip           создаётся make_submit.py
├── __pycache__/
└── venv/
```

### Среда разработки

```
Python 3.12.10
```