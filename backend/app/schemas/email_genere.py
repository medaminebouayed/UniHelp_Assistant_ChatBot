from pydantic import BaseModel
from typing import Optional

class EmailGenereBase(BaseModel):
    object: Optional[str] = None
    destinataire: Optional[str] = None
    contenu: Optional[str] = None

class EmailGenereCreate(EmailGenereBase):
    etudiant_id: int

class EmailGenereOut(EmailGenereBase):
    idEmail: int
    class Config:
        from_attributes = True
