from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from app.database import get_db
from app.models.email_genere import EmailGenere
from app.models.etudiant import Etudiant
from app.schemas.email_genere import EmailGenereOut
from app.utils.rag import generate_email
from app.utils.mail import send_email
from app.utils.security import get_current_user

router = APIRouter(prefix="/emails-generes", tags=["Emails Generes"])


class GenerateEmailRequest(BaseModel):
    etudiant_id: int
    type_email: str
    destinataire: str = ""  # real recipient email (e.g., stage@iit.tn)
    details: str = ""


class EmailGenereWithStatus(EmailGenereOut):
    send_success: bool = False
    send_error: Optional[str] = None


@router.post("/generer", response_model=EmailGenereWithStatus)
async def generer_email(body: GenerateEmailRequest, db: Session = Depends(get_db), payload: dict = Depends(get_current_user)):
    etudiant = db.query(Etudiant).filter(Etudiant.id == body.etudiant_id).first()
    if not etudiant:
        raise HTTPException(status_code=404, detail="Etudiant non trouve")
    if int(payload["sub"]) != body.etudiant_id and payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Acces a vos donnees uniquement")

    user = etudiant.utilisateur
    uni = etudiant.universite.value if etudiant.universite else "Universite"

    result = await generate_email(
        type_email=body.type_email,
        nom=user.nom,
        prenom=user.prenom,
        universite=uni,
        details=body.details,
    )

    objet = result.get("objet", "")
    contenu = result.get("contenu", "")
    destinataire = body.destinataire or result.get("destinataire", "")

    email = EmailGenere(
        object=objet,
        destinataire=destinataire or "administration@universite.tn",
        contenu=contenu,
        etudiant_id=body.etudiant_id,
    )
    db.add(email)
    db.commit()
    db.refresh(email)

    send_result = send_email(to=destinataire, subject=objet, body=contenu)

    return EmailGenereWithStatus(
        idEmail=email.idEmail,
        object=email.object,
        destinataire=email.destinataire,
        contenu=email.contenu,
        send_success=send_result["success"],
        send_error=send_result["error"],
    )


@router.get("/etudiant/{etudiant_id}", response_model=List[EmailGenereOut])
def get_emails(etudiant_id: int, db: Session = Depends(get_db), payload: dict = Depends(get_current_user)):
    if int(payload["sub"]) != etudiant_id and payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Acces a vos emails uniquement")
    return db.query(EmailGenere).filter(EmailGenere.etudiant_id == etudiant_id).all()


@router.delete("/{email_id}")
def delete_email(email_id: int, db: Session = Depends(get_db), payload: dict = Depends(get_current_user)):
    email = db.query(EmailGenere).filter(EmailGenere.idEmail == email_id).first()
    if not email:
        raise HTTPException(status_code=404, detail="Email non trouve")
    if int(payload["sub"]) != email.etudiant_id and payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Acces a vos emails uniquement")
    db.delete(email)
    db.commit()
    return {"message": "Email supprime"}
