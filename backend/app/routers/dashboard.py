from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import date, timedelta
from app.database import get_db
from app.models.utilisateur import Utilisateur
from app.models.etudiant import Etudiant
from app.models.administrateur import Administrateur
from app.models.document import Document
from app.models.email_genere import EmailGenere
from app.models.conversation import Conversation
from app.utils.security import get_current_user

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/stats")
def get_stats(payload: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    user_id = int(payload["sub"])
    role = payload.get("role")

    base = {
        "user_id": user_id,
        "role": role,
    }

    if role == "etudiant":
        total_emails = db.query(func.count(EmailGenere.idEmail)).filter(
            EmailGenere.etudiant_id == user_id
        ).scalar() or 0
        total_convs = db.query(func.count(Conversation.idConversation)).filter(
            Conversation.utilisateur_id == user_id
        ).scalar() or 0
        total_fav_convs = db.query(func.count(Conversation.idConversation)).filter(
            Conversation.utilisateur_id == user_id,
            Conversation.estFavori == True
        ).scalar() or 0
        total_docs = db.query(func.count(Document.idDocument)).scalar() or 0
        today = date.today()

        recent_convs = db.query(Conversation).filter(
            Conversation.utilisateur_id == user_id
        ).order_by(Conversation.dateDernierModification.desc().nullslast()).limit(5).all()

        fav_convs = db.query(Conversation).filter(
            Conversation.utilisateur_id == user_id,
            Conversation.estFavori == True
        ).order_by(Conversation.dateDernierModification.desc().nullslast()).limit(5).all()

        week_ago = today - timedelta(days=7)
        week_conv_count = db.query(func.count(Conversation.idConversation)).filter(
            Conversation.utilisateur_id == user_id,
            Conversation.dateDernierModification >= week_ago
        ).scalar() or 0
        month_ago = today - timedelta(days=30)
        month_conv_count = db.query(func.count(Conversation.idConversation)).filter(
            Conversation.utilisateur_id == user_id,
            Conversation.dateDernierModification >= month_ago
        ).scalar() or 0

        recent_activity = []
        for c in recent_convs:
            recent_activity.append({
                "id": c.idConversation,
                "action": f"A posé une question au Chatbot IA — {c.titre or 'Sans titre'}",
                "time": c.dateDernierModification.isoformat() if c.dateDernierModification else "",
                "type": "chat",
            })

        fav_messages = []
        for c in fav_convs:
            title = c.titre or "Conversation"
            fav_messages.append({
                "id": c.idConversation,
                "text": title,
                "source": "Chatbot IA",
                "time": c.dateDernierModification.isoformat() if c.dateDernierModification else "",
            })

        return {
            **base,
            "total_conversations": total_convs,
            "total_emails": total_emails,
            "total_documents": total_docs,
            "total_favorites": total_fav_convs,
            "conversations_this_week": week_conv_count,
            "conversations_this_month": month_conv_count,
            "recent_activity": recent_activity,
            "favorite_conversations": fav_messages,
        }

    elif role == "admin":
        total_etudiants = db.query(func.count(Etudiant.id)).scalar() or 0
        total_documents = db.query(func.count(Document.idDocument)).scalar() or 0
        total_emails = db.query(func.count(EmailGenere.idEmail)).scalar() or 0
        total_admins = db.query(func.count(Administrateur.id)).scalar() or 0
        total_conversations = db.query(func.count(Conversation.idConversation)).scalar() or 0

        recent_students = db.query(Etudiant).order_by(Etudiant.id.desc()).limit(5).all()
        students_out = []
        for e in recent_students:
            u = e.utilisateur
            students_out.append({
                "id": e.id,
                "name": f"{u.prenom or ''} {u.nom or ''}".strip(),
                "field": e.diplomeSouhaite.value if e.diplomeSouhaite else "",
                "status": "Actif" if u.estActif else "Inactif",
            })
            
        weekly_data = [
            {"name": "Lun", "inscrits": db.query(func.count(Utilisateur.id)).filter(func.extract('isodow', Utilisateur.dateCreation) == 1).scalar() or 0},
            {"name": "Mar", "inscrits": db.query(func.count(Utilisateur.id)).filter(func.extract('isodow', Utilisateur.dateCreation) == 2).scalar() or 0},
            {"name": "Mer", "inscrits": db.query(func.count(Utilisateur.id)).filter(func.extract('isodow', Utilisateur.dateCreation) == 3).scalar() or 0},
            {"name": "Jeu", "inscrits": db.query(func.count(Utilisateur.id)).filter(func.extract('isodow', Utilisateur.dateCreation) == 4).scalar() or 0},
            {"name": "Ven", "inscrits": db.query(func.count(Utilisateur.id)).filter(func.extract('isodow', Utilisateur.dateCreation) == 5).scalar() or 0},
            {"name": "Sam", "inscrits": db.query(func.count(Utilisateur.id)).filter(func.extract('isodow', Utilisateur.dateCreation) == 6).scalar() or 0},
            {"name": "Dim", "inscrits": db.query(func.count(Utilisateur.id)).filter(func.extract('isodow', Utilisateur.dateCreation) == 7).scalar() or 0},
        ]
        
        monthly_data = [
            {"name": "Jan", "inscrits": db.query(func.count(Utilisateur.id)).filter(func.extract('month', Utilisateur.dateCreation) == 1).scalar() or 0},
            {"name": "Fév", "inscrits": db.query(func.count(Utilisateur.id)).filter(func.extract('month', Utilisateur.dateCreation) == 2).scalar() or 0},
            {"name": "Mar", "inscrits": db.query(func.count(Utilisateur.id)).filter(func.extract('month', Utilisateur.dateCreation) == 3).scalar() or 0},
            {"name": "Avr", "inscrits": db.query(func.count(Utilisateur.id)).filter(func.extract('month', Utilisateur.dateCreation) == 4).scalar() or 0},
            {"name": "Mai", "inscrits": db.query(func.count(Utilisateur.id)).filter(func.extract('month', Utilisateur.dateCreation) == 5).scalar() or 0},
            {"name": "Juin", "inscrits": db.query(func.count(Utilisateur.id)).filter(func.extract('month', Utilisateur.dateCreation) == 6).scalar() or 0},
            {"name": "Juil", "inscrits": db.query(func.count(Utilisateur.id)).filter(func.extract('month', Utilisateur.dateCreation) == 7).scalar() or 0},
            {"name": "Août", "inscrits": db.query(func.count(Utilisateur.id)).filter(func.extract('month', Utilisateur.dateCreation) == 8).scalar() or 0},
            {"name": "Sep", "inscrits": db.query(func.count(Utilisateur.id)).filter(func.extract('month', Utilisateur.dateCreation) == 9).scalar() or 0},
            {"name": "Oct", "inscrits": db.query(func.count(Utilisateur.id)).filter(func.extract('month', Utilisateur.dateCreation) == 10).scalar() or 0},
            {"name": "Nov", "inscrits": db.query(func.count(Utilisateur.id)).filter(func.extract('month', Utilisateur.dateCreation) == 11).scalar() or 0},
            {"name": "Déc", "inscrits": db.query(func.count(Utilisateur.id)).filter(func.extract('month', Utilisateur.dateCreation) == 12).scalar() or 0},
        ]
        
        top_questions = [
            {"text": "Certificat de scolarité", "count": 31, "icon": "file", "color": "text-blue-500", "bg": "bg-blue-50"},
            {"text": "Calendrier des examens", "count": 28, "icon": "calendar", "color": "text-indigo-500", "bg": "bg-indigo-50"},
            {"text": "Bourse & aides", "count": 24, "icon": "coins", "color": "text-amber-500", "bg": "bg-amber-50"},
            {"text": "Justification d'absence", "count": 19, "icon": "x-circle", "color": "text-red-500", "bg": "bg-red-50"}
        ]
        
        categories = [
            {"name": "Scolarité", "percent": 38, "color": "bg-slate-800"},
            {"name": "Examens", "percent": 25, "color": "bg-amber-500"},
            {"name": "Bourses", "percent": 20, "color": "bg-blue-500"},
            {"name": "Absences", "percent": 10, "color": "bg-emerald-500"}
        ]

        return {
            **base,
            "total_etudiants": total_etudiants,
            "total_documents": total_documents,
            "total_emails": total_emails,
            "total_admins": total_admins,
            "total_conversations": total_conversations,
            "recent_students": students_out,
            "weekly_data": weekly_data,
            "monthly_data": monthly_data,
            "top_questions": top_questions,
            "categories": categories,
        }

    return {"error": "Role non reconnu"}
