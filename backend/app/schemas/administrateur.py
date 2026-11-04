from pydantic import BaseModel
from typing import Optional, List
from app.models.enums import DepartementEnum
from app.schemas.utilisateur import UtilisateurCreate, UtilisateurOut
from app.schemas.document import DocumentOut

class AdministrateurCreate(UtilisateurCreate):
    departement: Optional[DepartementEnum] = None
    telephone: Optional[str] = None

class AdministrateurUpdate(BaseModel):
    nom: Optional[str] = None
    prenom: Optional[str] = None
    email: Optional[str] = None
    motDePasse: Optional[str] = None
    photo: Optional[str] = None
    departement: Optional[DepartementEnum] = None
    telephone: Optional[str] = None

class AdministrateurOut(UtilisateurOut):
    departement: Optional[DepartementEnum] = None
    telephone: Optional[str] = None
    documents: List[DocumentOut] = []
    class Config:
        from_attributes = True
