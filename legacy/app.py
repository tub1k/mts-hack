import uuid
from fastapi import FastAPI
from pydantic import BaseModel
import config
from legacy.rag import KnowledgeBaseRAG
from legacy.agent import ConsultantAgent

app = FastAPI(title="MTS Energytool AI Consultant")

rag = KnowledgeBaseRAG()
rag.build_from_zip(config.KB_ZIP_PATH)
agent = ConsultantAgent(rag)

# Хранилище всех сессий (для финала переведите на SQLAlchemy + SQLite)
SESSIONS: dict[str, list[dict]] = {}

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    history = SESSIONS.setdefault(session_id, [])

    result = agent.answer_question(req.message, chat_history=history)

    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": result["answer"]})

    return {
        "session_id": session_id,
        "answer": result["answer"],
        "debug": {
            "elapsed_sec": result["elapsed_sec"],
            "sources": result["sources"]
        }
    }

@app.post("/api/reset")
def reset_session():
    new_id = str(uuid.uuid4())
    SESSIONS[new_id] = []
    return {"session_id": new_id}

@app.get("/api/sessions")
def list_sessions():
    return {"sessions": SESSIONS}