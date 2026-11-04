from pydantic import BaseModel
from datetime import date

class MessageBase(BaseModel):
    contenu: str
    dateEnvoi: date = None
    role: str = "user"

class MessageCreate(MessageBase):
    conversation_id: int

class MessageOut(MessageBase):
    idMessage: int
    conversation_id: int
    class Config:
        from_attributes = True
