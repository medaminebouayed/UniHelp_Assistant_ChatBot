from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from typing import Optional
from app.database import get_db
from app.models.utilisateur import Utilisateur
from app.models.etudiant import Etudiant
from app.models.administrateur import Administrateur
from app.schemas.auth import LoginSchema, TokenOut
from app.utils.security import verify_password, create_token, hash_password, get_current_user

router = APIRouter(prefix="/auth", tags=["Auth"])


def _get_role(user_id: int, db: Session) -> str:
    if db.query(Administrateur).filter(Administrateur.id == user_id).first():
        return "admin"
    if db.query(Etudiant).filter(Etudiant.id == user_id).first():
        return "etudiant"
    return "utilisateur"


@router.post("/login", response_model=TokenOut)
def login(body: LoginSchema, db: Session = Depends(get_db)):
    user = db.query(Utilisateur).filter(Utilisateur.email == body.email).first()
    if not user or not verify_password(body.motDePasse, user.motDePasse):
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")
    if not user.estActif:
        raise HTTPException(status_code=403, detail="Compte inactif")
    role = _get_role(user.id, db)
    token = create_token({"sub": str(user.id), "email": user.email, "role": role})
    return {"access_token": token}


class RegisterSchema(BaseModel):
    nom: str
    prenom: str
    email: EmailStr
    motDePasse: str
    CIN: Optional[str] = None
    passport: Optional[str] = None
    pays: Optional[str] = None
    dateNaissance: Optional[str] = None
    sexe: Optional[str] = None
    telephone: Optional[str] = None
    adresse: Optional[str] = None
    dernierDiplomeObtenu: Optional[str] = None
    anneeObtention: Optional[str] = None
    universite: Optional[str] = None
    diplomeSouhaite: Optional[str] = None
    niveau: Optional[str] = None
    cours: Optional[str] = None
    sourceConnaissanceUniversite: Optional[str] = None


@router.post("/register", status_code=201)
def register(body: RegisterSchema, db: Session = Depends(get_db)):
    if db.query(Utilisateur).filter(Utilisateur.email == body.email).first():
        raise HTTPException(status_code=400, detail="Email deja utilise")

    from app.models.enums import SexeEnum, DiplomeEnum, UniversiteEnum, DiplomeSouhaiteEnum, NiveauEnum, SourceEnum

    user = Utilisateur(
        nom=body.nom, prenom=body.prenom,
        email=body.email, motDePasse=hash_password(body.motDePasse),
    )
    db.add(user)
    db.flush()

    sexe_val = None
    if body.sexe:
        try:
            sexe_val = SexeEnum(body.sexe)
        except ValueError:
            pass
    diplome_val = None
    if body.dernierDiplomeObtenu:
        try:
            diplome_val = DiplomeEnum(body.dernierDiplomeObtenu)
        except ValueError:
            pass
    uni_val = None
    if body.universite:
        try:
            uni_val = UniversiteEnum(body.universite)
        except ValueError:
            pass
    diplome_souhaite_val = None
    if body.diplomeSouhaite:
        try:
            diplome_souhaite_val = DiplomeSouhaiteEnum(body.diplomeSouhaite)
        except ValueError:
            pass
    niveau_val = None
    if body.niveau:
        try:
            niveau_val = NiveauEnum(body.niveau)
        except ValueError:
            pass
    source_val = None
    if body.sourceConnaissanceUniversite:
        try:
            source_val = SourceEnum(body.sourceConnaissanceUniversite)
        except ValueError:
            pass

    naissance = None
    if body.dateNaissance:
        from datetime import date
        try:
            parts = body.dateNaissance.split("-")
            naissance = date(int(parts[0]), int(parts[1]), int(parts[2]))
        except (ValueError, IndexError):
            pass

    etudiant = Etudiant(
        id=user.id, CIN=body.CIN, passport=body.passport, pays=body.pays,
        dateNaissance=naissance, sexe=sexe_val, telephone=body.telephone,
        adresse=body.adresse, dernierDiplomeObtenu=diplome_val,
        anneeObtention=body.anneeObtention, universite=uni_val,
        diplomeSouhaite=diplome_souhaite_val, niveau=niveau_val,
        cours=body.cours, sourceConnaissanceUniversite=source_val,
    )
    db.add(etudiant)
    db.commit()
    db.refresh(etudiant)
    return {"id": user.id, "email": user.email, "role": "etudiant"}


@router.get("/me")
def me(payload: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    user = db.query(Utilisateur).filter(Utilisateur.id == int(payload["sub"])).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouve")
    if not user.estActif:
        raise HTTPException(status_code=403, detail="Compte inactif")
    role = _get_role(user.id, db)
    result = {
        "id": user.id, "nom": user.nom, "prenom": user.prenom,
        "email": user.email, "role": role, "photo": user.photo,
        "universite": None, "diplomeSouhaite": None,
        "telephone": None, "departement": None,
    }
    if role == "etudiant":
        etudiant = db.query(Etudiant).filter(Etudiant.id == user.id).first()
        if etudiant:
            result["universite"] = etudiant.universite.value if etudiant.universite else None
            result["diplomeSouhaite"] = etudiant.diplomeSouhaite.value if etudiant.diplomeSouhaite else None
            result["telephone"] = etudiant.telephone
    elif role == "admin":
        admin = db.query(Administrateur).filter(Administrateur.id == user.id).first()
        if admin:
            result["departement"] = admin.departement.value if admin.departement else None
            result["telephone"] = admin.telephone
    return result
