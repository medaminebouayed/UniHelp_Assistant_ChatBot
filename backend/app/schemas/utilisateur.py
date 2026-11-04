from pydantic import BaseModel, EmailStr
from typing import Optional

class UtilisateurBase(BaseModel):
    nom: str
    prenom: str
    email: EmailStr
    photo: Optional[str] = None
    estActif: Optional[bool] = True

class UtilisateurCreate(UtilisateurBase):
    motDePasse: str

class UtilisateurOut(UtilisateurBase):
    id: int
    class Config:
        from_attributes = True
