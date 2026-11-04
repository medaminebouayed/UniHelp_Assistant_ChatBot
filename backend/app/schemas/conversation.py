from pydantic import BaseModel
from typing import Optional, List
from datetime import date
from app.schemas.message import MessageOut

class ConversationBase(BaseModel):
    titre: Optional[str] = None
    estFavori: bool = False
    dateDernierModification: Optional[date] = None

class ConversationCreate(ConversationBase):
    utilisateur_id: int

class ConversationOut(ConversationBase):
    idConversation: int
    messages: List[MessageOut] = []
    class Config:
        from_attributes = True
