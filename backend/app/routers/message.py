from datetime import date
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.models.conversation import Conversation
from app.models.message import Message
from app.schemas.message import MessageCreate, MessageOut
from app.utils.security import get_current_user

router = APIRouter(prefix="/messages", tags=["Messages"])


@router.post("", response_model=MessageOut, status_code=201)
def create(body: MessageCreate, db: Session = Depends(get_db), payload: dict = Depends(get_current_user)):
    conv = db.query(Conversation).filter(Conversation.idConversation == body.conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation non trouvee")
    if int(payload["sub"]) != conv.utilisateur_id and payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Acces a votre conversation uniquement")
    data = body.model_dump()
    if data.get("dateEnvoi") is None:
        data["dateEnvoi"] = date.today()
    msg = Message(**data)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


@router.get("/conversation/{conv_id}", response_model=List[MessageOut])
def get_by_conversation(conv_id: int, db: Session = Depends(get_db), payload: dict = Depends(get_current_user)):
    conv = db.query(Conversation).filter(Conversation.idConversation == conv_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation non trouvee")
    if int(payload["sub"]) != conv.utilisateur_id and payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Acces a votre conversation uniquement")
    return db.query(Message).filter(Message.conversation_id == conv_id).all()


@router.delete("/{msg_id}")
def delete(msg_id: int, db: Session = Depends(get_db), payload: dict = Depends(get_current_user)):
    msg = db.query(Message).filter(Message.idMessage == msg_id).first()
    if not msg:
        raise HTTPException(status_code=404, detail="Message non trouve")
    conv = db.query(Conversation).filter(Conversation.idConversation == msg.conversation_id).first()
    if conv and int(payload["sub"]) != conv.utilisateur_id and payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Acces a votre conversation uniquement")
    db.delete(msg)
    db.commit()
    return {"message": "Message supprime"}
