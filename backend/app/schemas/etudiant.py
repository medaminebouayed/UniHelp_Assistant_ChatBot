from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import date
from app.models.enums import (
    SexeEnum, DiplomeEnum, UniversiteEnum,
    DiplomeSouhaiteEnum, NiveauEnum, SourceEnum
)
from app.schemas.utilisateur import UtilisateurCreate, UtilisateurOut
from app.schemas.email_genere import EmailGenereOut

class EtudiantCreate(UtilisateurCreate):
    CIN: Optional[str] = None
    passport: Optional[str] = None
    pays: Optional[str] = None
    dateNaissance: Optional[date] = None
    sexe: Optional[SexeEnum] = None
    telephone: Optional[str] = None
    adresse: Optional[str] = None
    dernierDiplomeObtenu: Optional[DiplomeEnum] = None
    anneeObtention: Optional[str] = None
    universite: Optional[UniversiteEnum] = None
    diplomeSouhaite: Optional[DiplomeSouhaiteEnum] = None
    niveau: Optional[NiveauEnum] = None
    cours: Optional[str] = None
    sourceConnaissanceUniversite: Optional[SourceEnum] = None

class EtudiantUpdate(BaseModel):
    nom: Optional[str] = None
    prenom: Optional[str] = None
    email: Optional[EmailStr] = None
    motDePasse: Optional[str] = None
    photo: Optional[str] = None
    CIN: Optional[str] = None
    passport: Optional[str] = None
    pays: Optional[str] = None
    dateNaissance: Optional[date] = None
    sexe: Optional[SexeEnum] = None
    telephone: Optional[str] = None
    adresse: Optional[str] = None
    dernierDiplomeObtenu: Optional[DiplomeEnum] = None
    anneeObtention: Optional[str] = None
    universite: Optional[UniversiteEnum] = None
    diplomeSouhaite: Optional[DiplomeSouhaiteEnum] = None
    niveau: Optional[NiveauEnum] = None
    cours: Optional[str] = None
    sourceConnaissanceUniversite: Optional[SourceEnum] = None

class EtudiantOut(UtilisateurOut):
    CIN: Optional[str] = None
    passport: Optional[str] = None
    pays: Optional[str] = None
    dateNaissance: Optional[date] = None
    sexe: Optional[SexeEnum] = None
    telephone: Optional[str] = None
    adresse: Optional[str] = None
    dernierDiplomeObtenu: Optional[DiplomeEnum] = None
    anneeObtention: Optional[str] = None
    universite: Optional[UniversiteEnum] = None
    diplomeSouhaite: Optional[DiplomeSouhaiteEnum] = None
    niveau: Optional[NiveauEnum] = None
    cours: Optional[str] = None
    sourceConnaissanceUniversite: Optional[SourceEnum] = None
    emails_generes: List[EmailGenereOut] = []
    class Config:
        from_attributes = True
