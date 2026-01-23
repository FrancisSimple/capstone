from fastapi import APIRouter, HTTPException, status
from .schema import IngestRequest, QueryRequest
from . import store,chat

aiApp = APIRouter(prefix='/ai',tags=["AI ENGINE"])

@aiApp.post("/ingest")
async def ingest_knowledge(request: IngestRequest):
    try:
        store.add_text_to_base(request.texts)
        return {"status": "success", "message": "Knowledge added to ai brain"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@aiApp.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        answer = chat.ask_ai(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))