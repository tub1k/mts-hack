### Структура папок

```
mts-energytool-bot/
├── config.py            # Чтение всех переменных окружения с фоллбэками по регламенту
├── prompts.py           # Системные промпты консультанта (до 150 слов, сбор заявки)
├── rag.py               # Распаковка knowledge-base.zip, чанкинг, FAISS + BM25 + Reranker
├── agent.py             # Клиент к локальному прокси (OpenAI API) с ретраями от 429/5xx
├── solution.py          # Точка входа Этапа 1: читает questions.csv -> пишет submission.csv
├── app.py               # Точка входа Этапа 2: FastAPI-сервер + SQLite + выдача фронтенда
├── make_submit.py       # Скрипт упаковки плоского ZIP-сабмита для лидерборда
├── static/
│   └── index.html       # Веб-чат с кнопкой «Новый диалог» и панелью режима отладки
├── data/                # В .gitignore! Локальная папка для questions.csv и knowledge-base.zip
├── requirements.txt     # Фиксированный список пакетов из регламента
├── .env.example         # Шаблон переменных окружения без секретов
├── .gitignore
├── .dockerignore
├── Dockerfile
├── compose.yaml
└── README.mdmts-energytool-bot/
```

### Среда разработки

```
Python 3.12.10
```