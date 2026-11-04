from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from datetime import date
from app.database import get_db
from app.models.conversation import Conversation
from app.models.message import Message
from app.utils.rag import ask_with_rag
from app.utils.security import get_current_user

router = APIRouter(prefix="/chat", tags=["Chat RAG"])


class ChatRequest(BaseModel):
    conversation_id: int
    question: str
    use_graph: bool = True
    top_k: int = 10


class ChatResponse(BaseModel):
    reponse: str
    conversation_id: int
    sources: list = []


@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest, db: Session = Depends(get_db), payload: dict = Depends(get_current_user)):
    conv = db.query(Conversation).filter(
        Conversation.idConversation == body.conversation_id
    ).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation non trouvee")
    if int(payload["sub"]) != conv.utilisateur_id and payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Acces a votre conversation uniquement")

    msg_user = Message(
        contenu=body.question,
        dateEnvoi=date.today(),
        role="user",
        conversation_id=body.conversation_id,
    )
    db.add(msg_user)
    db.commit()

    reponse, sources = await ask_with_rag(
        question=body.question,
        top_k=body.top_k,
        use_graph=body.use_graph,
    )

    msg_assistant = Message(
        contenu=reponse,
        dateEnvoi=date.today(),
        role="assistant",
        conversation_id=body.conversation_id,
    )
    db.add(msg_assistant)
    db.commit()

    return {"reponse": reponse, "conversation_id": body.conversation_id, "sources": sources}


from fastapi.responses import StreamingResponse
import httpx
import json
from app.database import SessionLocal

@router.post("/stream")
async def chat_stream(body: ChatRequest, db: Session = Depends(get_db), payload: dict = Depends(get_current_user)):
    conv = db.query(Conversation).filter(
        Conversation.idConversation == body.conversation_id
    ).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation non trouvee")
    if int(payload["sub"]) != conv.utilisateur_id and payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Acces a votre conversation uniquement")

    msg_user = Message(
        contenu=body.question,
        dateEnvoi=date.today(),
        role="user",
        conversation_id=body.conversation_id,
    )
    db.add(msg_user)
    db.commit()

    async def event_generator():
        from app.utils.rag import RAG_API_URL, _rewrite_file_url
        full_response = ""
        sources_list = []
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST",
                    f"{RAG_API_URL}/query/stream",
                    json={"question": body.question, "top_k": body.top_k, "use_graph": body.use_graph}
                ) as r:
                    async for line in r.aiter_lines():
                        if not line:
                            continue
                        
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if data.get("type") == "chunk":
                                    full_response += data.get("text", "")
                                    yield f"{line}\n\n"
                                elif data.get("type") == "sources":
                                    sources_list = data.get("sources", [])
                                    for s in sources_list:
                                        s["url"] = _rewrite_file_url(s.get("url", ""))
                                    # Forward the updated line to frontend
                                    modified_line = f"data: {json.dumps(data)}"
                                    yield f"{modified_line}\n\n"
                                else:
                                    yield f"{line}\n\n"
                            except Exception:
                                yield f"{line}\n\n"
                        else:
                            yield f"{line}\n\n"
        finally:
            with SessionLocal() as db_session:
                msg_assistant = Message(
                    contenu=full_response,
                    dateEnvoi=date.today(),
                    role="assistant",
                    conversation_id=body.conversation_id,
                )
                db_session.add(msg_assistant)
                db_session.commit()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


class NewConversationRequest(BaseModel):
    utilisateur_id: int
    titre: Optional[str] = "Nouvelle conversation"


@router.post("/nouvelle-conversation")
def nouvelle_conversation(body: NewConversationRequest, db: Session = Depends(get_db), payload: dict = Depends(get_current_user)):
    if int(payload["sub"]) != body.utilisateur_id and payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Impossible de creer pour un autre utilisateur")
    conv = Conversation(
        titre=body.titre,
        estFavori=False,
        dateDernierModification=date.today(),
        utilisateur_id=body.utilisateur_id,
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return {"idConversation": conv.idConversation, "titre": conv.titre}
