import os
from dotenv import load_dotenv

load_dotenv()

# Настройки прокси и моделей (с фоллбэками по п. 6 регламента)
LLM_BASE_URL = (
    os.getenv("LLM_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or "http://localhost:8000/v1"
)
LLM_API_KEY = (
    os.getenv("LLM_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or "dummy-key"
)
LLM_MODEL = (
    os.getenv("LLM_MODEL")
    or os.getenv("OPENAI_MODEL")
    or os.getenv("GEN_MODEL")
    or "Qwen3.8-27B"
)

EMB_MODEL = os.getenv("EMB_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Qwen3 Reranker 8B")

# Пути к данным
DATA_DIR = os.getenv("DATA_DIR", "data")
KB_ZIP_PATH = os.path.join(DATA_DIR, "knowledge-base.zip")
QUESTIONS_CSV_PATH = os.path.join(DATA_DIR, "questions.csv")
SUBMISSION_CSV_PATH = "submission.csv"

# Параметры генерации и поиска
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
TOP_K = int(os.getenv("TOP_K", "10"))
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "4"))

CHUNK_SIZE = 900
CHUNK_OVERLAP = 200
TOP_K = 16            
RERANKER_TOP_N = 5    
TEMPERATURE = 0.05    
MAX_TOKENS = 4096