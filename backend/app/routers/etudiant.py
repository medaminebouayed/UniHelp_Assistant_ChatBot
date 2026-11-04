from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.models.utilisateur import Utilisateur
from app.models.etudiant import Etudiant
from app.schemas.etudiant import EtudiantCreate, EtudiantUpdate, EtudiantOut
from app.utils.security import hash_password, get_current_user, require_role, optional_user

router = APIRouter(prefix="/etudiants", tags=["Etudiants"])


def _build_out(user: Utilisateur, etudiant: Etudiant) -> EtudiantOut:
    return EtudiantOut(
        id=user.id, nom=user.nom, prenom=user.prenom, email=user.email, photo=user.photo, estActif=user.estActif,
        CIN=etudiant.CIN, passport=etudiant.passport, pays=etudiant.pays,
        dateNaissance=etudiant.dateNaissance, sexe=etudiant.sexe,
        telephone=etudiant.telephone, adresse=etudiant.adresse,
        dernierDiplomeObtenu=etudiant.dernierDiplomeObtenu,
        anneeObtention=etudiant.anneeObtention, universite=etudiant.universite,
        diplomeSouhaite=etudiant.diplomeSouhaite, niveau=etudiant.niveau,
        cours=etudiant.cours,
        sourceConnaissanceUniversite=etudiant.sourceConnaissanceUniversite,
        emails_generes=etudiant.emails_generes or [],
    )


@router.post("", response_model=EtudiantOut, status_code=201)
def create(body: EtudiantCreate, db: Session = Depends(get_db)):
    if db.query(Utilisateur).filter(Utilisateur.email == body.email).first():
        raise HTTPException(status_code=400, detail="Email deja utilise")
    user = Utilisateur(nom=body.nom, prenom=body.prenom,
                       email=body.email, motDePasse=hash_password(body.motDePasse))
    db.add(user)
    db.flush()
    etudiant = Etudiant(
        id=user.id, CIN=body.CIN, passport=body.passport, pays=body.pays,
        dateNaissance=body.dateNaissance, sexe=body.sexe, telephone=body.telephone,
        adresse=body.adresse, dernierDiplomeObtenu=body.dernierDiplomeObtenu,
        anneeObtention=body.anneeObtention, universite=body.universite,
        diplomeSouhaite=body.diplomeSouhaite, niveau=body.niveau, cours=body.cours,
        sourceConnaissanceUniversite=body.sourceConnaissanceUniversite,
    )
    db.add(etudiant)
    db.commit()
    db.refresh(etudiant)
    return _build_out(user, etudiant)


@router.get("", response_model=List[EtudiantOut])
def get_all(db: Session = Depends(get_db), _=Depends(require_role("admin"))):
    return [_build_out(e.utilisateur, e) for e in db.query(Etudiant).all()]


@router.get("/{etudiant_id}", response_model=EtudiantOut)
def get_one(etudiant_id: int, db: Session = Depends(get_db), payload: dict = Depends(get_current_user)):
    e = db.query(Etudiant).filter(Etudiant.id == etudiant_id).first()
    if not e:
        raise HTTPException(status_code=404, detail="Etudiant non trouve")
    if payload.get("role") != "admin" and int(payload["sub"]) != etudiant_id:
        raise HTTPException(status_code=403, detail="Acces a vos donnees uniquement")
    return _build_out(e.utilisateur, e)


@router.put("/{etudiant_id}", response_model=EtudiantOut)
def update(etudiant_id: int, body: EtudiantCreate, db: Session = Depends(get_db), payload: dict = Depends(get_current_user)):
    if payload.get("role") != "admin" and int(payload["sub"]) != etudiant_id:
        raise HTTPException(status_code=403, detail="Acces a vos donnees uniquement")
    e = db.query(Etudiant).filter(Etudiant.id == etudiant_id).first()
    if not e:
        raise HTTPException(status_code=404, detail="Etudiant non trouve")
    for field in ["CIN", "passport", "pays", "dateNaissance", "sexe", "telephone",
                  "adresse", "dernierDiplomeObtenu", "anneeObtention", "universite",
                  "diplomeSouhaite", "niveau", "cours", "sourceConnaissanceUniversite"]:
        setattr(e, field, getattr(body, field))
    db.commit()
    db.refresh(e)
    return _build_out(e.utilisateur, e)


@router.patch("/{etudiant_id}", response_model=EtudiantOut)
def patch(etudiant_id: int, body: EtudiantUpdate, db: Session = Depends(get_db), payload: dict = Depends(get_current_user)):
    if payload.get("role") != "admin" and int(payload["sub"]) != etudiant_id:
        raise HTTPException(status_code=403, detail="Acces a vos donnees uniquement")
    e = db.query(Etudiant).filter(Etudiant.id == etudiant_id).first()
    if not e:
        raise HTTPException(status_code=404, detail="Etudiant non trouve")
    user = e.utilisateur
    if body.nom is not None:
        user.nom = body.nom
    if body.prenom is not None:
        user.prenom = body.prenom
    if body.email is not None:
        user.email = body.email
    if body.motDePasse is not None:
        user.motDePasse = hash_password(body.motDePasse)
    if body.photo is not None:
        user.photo = body.photo
    for field in ["CIN", "passport", "pays", "dateNaissance", "sexe", "telephone",
                  "adresse", "dernierDiplomeObtenu", "anneeObtention", "universite",
                  "diplomeSouhaite", "niveau", "cours", "sourceConnaissanceUniversite"]:
        val = getattr(body, field)
        if val is not None:
            setattr(e, field, val)
    db.commit()
    db.refresh(e)
    return _build_out(user, e)


@router.delete("/{etudiant_id}")
def delete(etudiant_id: int, db: Session = Depends(get_db), _=Depends(require_role("admin"))):
    e = db.query(Etudiant).filter(Etudiant.id == etudiant_id).first()
    if not e:
        raise HTTPException(status_code=404, detail="Etudiant non trouve")
    user = e.utilisateur
    db.delete(e)
    db.delete(user)
    db.commit()
    return {"message": "Etudiant supprime"}


@router.patch("/{etudiant_id}/toggle-active")
def toggle_active(etudiant_id: int, db: Session = Depends(get_db), _=Depends(require_role("admin"))):
    e = db.query(Etudiant).filter(Etudiant.id == etudiant_id).first()
    if not e:
        raise HTTPException(status_code=404, detail="Etudiant non trouve")
    user = e.utilisateur
    user.estActif = not user.estActif
    db.commit()
    db.refresh(user)
    return {"id": user.id, "estActif": user.estActif}
