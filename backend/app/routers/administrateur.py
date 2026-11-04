from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.models.utilisateur import Utilisateur
from app.models.administrateur import Administrateur
from app.schemas.administrateur import AdministrateurCreate, AdministrateurOut, AdministrateurUpdate
from app.utils.security import hash_password, require_role, get_current_user

router = APIRouter(prefix="/administrateurs", tags=["Administrateurs"])


@router.post("", response_model=AdministrateurOut, status_code=201)
def create(body: AdministrateurCreate, db: Session = Depends(get_db), _=Depends(require_role("admin"))):
    if db.query(Utilisateur).filter(Utilisateur.email == body.email).first():
        raise HTTPException(status_code=400, detail="Email déjà utilisé")
    user = Utilisateur(nom=body.nom, prenom=body.prenom,
                       email=body.email, motDePasse=hash_password(body.motDePasse))
    db.add(user)
    db.flush()
    admin = Administrateur(id=user.id, departement=body.departement, telephone=body.telephone)
    db.add(admin)
    db.commit()
    db.refresh(admin)
    return AdministrateurOut(id=user.id, nom=user.nom, prenom=user.prenom,
                             email=user.email, photo=user.photo, estActif=user.estActif, departement=admin.departement,
                             telephone=admin.telephone, documents=[])


@router.get("", response_model=List[AdministrateurOut])
def get_all(db: Session = Depends(get_db), _=Depends(require_role("admin"))):
    admins = db.query(Administrateur).all()
    return [AdministrateurOut(id=a.utilisateur.id, nom=a.utilisateur.nom,
            prenom=a.utilisateur.prenom, email=a.utilisateur.email,
            photo=a.utilisateur.photo, estActif=a.utilisateur.estActif,
            departement=a.departement, telephone=a.telephone,
            documents=a.documents or []) for a in admins]


@router.get("/{admin_id}", response_model=AdministrateurOut)
def get_one(admin_id: int, db: Session = Depends(get_db), _=Depends(require_role("admin"))):
    a = db.query(Administrateur).filter(Administrateur.id == admin_id).first()
    if not a:
        raise HTTPException(status_code=404, detail="Administrateur non trouve")
    return AdministrateurOut(id=a.utilisateur.id, nom=a.utilisateur.nom,
                             prenom=a.utilisateur.prenom, email=a.utilisateur.email,
                             photo=a.utilisateur.photo, estActif=a.utilisateur.estActif,
                             departement=a.departement, telephone=a.telephone,
                             documents=a.documents or [])


@router.delete("/{admin_id}")
def delete(admin_id: int, db: Session = Depends(get_db), _=Depends(require_role("admin"))):
    a = db.query(Administrateur).filter(Administrateur.id == admin_id).first()
    if not a:
        raise HTTPException(status_code=404, detail="Administrateur non trouve")
    user = a.utilisateur
    db.delete(a)
    db.delete(user)
    db.commit()
    return {"message": "Administrateur supprime"}


@router.patch("/{admin_id}", response_model=AdministrateurOut)
def patch(admin_id: int, body: AdministrateurUpdate, db: Session = Depends(get_db), payload: dict = Depends(get_current_user)):
    if payload.get("role") != "admin" and int(payload["sub"]) != admin_id:
        raise HTTPException(status_code=403, detail="Acces a vos donnees uniquement")
    a = db.query(Administrateur).filter(Administrateur.id == admin_id).first()
    if not a:
        raise HTTPException(status_code=404, detail="Administrateur non trouve")
    user = a.utilisateur
    if body.nom is not None:
        user.nom = body.nom
    if body.prenom is not None:
        user.prenom = body.prenom
    if body.email is not None:
        user.email = body.email
    if body.motDePasse is not None:
        user.motDePasse = hash_password(body.motDePasse)
    if hasattr(body, 'photo') and body.photo is not None:
        user.photo = body.photo
    if body.departement is not None:
        a.departement = body.departement
    if body.telephone is not None:
        a.telephone = body.telephone
    db.commit()
    db.refresh(a)
    return AdministrateurOut(id=user.id, nom=user.nom, prenom=user.prenom,
                             email=user.email, photo=user.photo, estActif=user.estActif, departement=a.departement,
                             telephone=a.telephone, documents=a.documents or [])
