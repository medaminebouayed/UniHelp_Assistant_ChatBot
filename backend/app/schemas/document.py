from pydantic import BaseModel
from typing import Optional
from datetime import date


class DocumentBase(BaseModel):
    titre: str
    site: Optional[str] = ""
    source: Optional[str] = "uploaded"
    categorie: Optional[str] = ""
    fichier: Optional[str] = None
    nom_fichier: Optional[str] = None
    taille: Optional[float] = None
    dateAjout: Optional[date] = None


class DocumentCreate(DocumentBase):
    administrateur_id: Optional[int] = None


class DocumentOut(DocumentBase):
    idDocument: int

    class Config:
        from_attributes = True
