from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import date
from app.database import get_db
from app.models.conversation import Conversation
from app.schemas.conversation import ConversationCreate, ConversationOut
from app.utils.security import get_current_user

router = APIRouter(prefix="/conversations", tags=["Conversations"])


@router.post("", response_model=ConversationOut, status_code=201)
def create(body: ConversationCreate, db: Session = Depends(get_db), payload: dict = Depends(get_current_user)):
    if int(payload["sub"]) != body.utilisateur_id and payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Impossible de creer pour un autre utilisateur")
    conv = Conversation(**body.model_dump())
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


@router.get("/utilisateur/{user_id}", response_model=List[ConversationOut])
def get_by_user(user_id: int, db: Session = Depends(get_db), payload: dict = Depends(get_current_user)):
    if int(payload["sub"]) != user_id and payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Acces a vos conversations uniquement")
    return db.query(Conversation).filter(Conversation.utilisateur_id == user_id).order_by(Conversation.idConversation.desc()).all()


@router.get("/{conv_id}", response_model=ConversationOut)
def get_one(conv_id: int, db: Session = Depends(get_db), payload: dict = Depends(get_current_user)):
    conv = db.query(Conversation).filter(Conversation.idConversation == conv_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation non trouvee")
    if int(payload["sub"]) != conv.utilisateur_id and payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Acces a vos conversations uniquement")
    return conv


@router.patch("/{conv_id}/favori", response_model=ConversationOut)
def toggle_favori(conv_id: int, db: Session = Depends(get_db), payload: dict = Depends(get_current_user)):
    conv = db.query(Conversation).filter(Conversation.idConversation == conv_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation non trouvee")
    if int(payload["sub"]) != conv.utilisateur_id:
        raise HTTPException(status_code=403, detail="Acces a vos conversations uniquement")
    conv.estFavori = not conv.estFavori
    db.commit()
    db.refresh(conv)
    return conv


@router.delete("/{conv_id}")
def delete(conv_id: int, db: Session = Depends(get_db), payload: dict = Depends(get_current_user)):
    conv = db.query(Conversation).filter(Conversation.idConversation == conv_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation non trouvee")
    if int(payload["sub"]) != conv.utilisateur_id and payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Acces a vos conversations uniquement")
    db.delete(conv)
    db.commit()
    return {"message": "Conversation supprimee"}
