import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import httpx

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_DIR))

from app.database import engine, SessionLocal
from app.models import *
from app.database import Base

Base.metadata.create_all(bind=engine)

from app.routers import auth, etudiant, administrateur, conversation, message, document, email_genere, chat, dashboard

app = FastAPI(
    title="UniHelp API",
    description="Backend FastAPI - Assistant IA pour services universitaires (IIT & ISB)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200", "http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:8002", "http://127.0.0.1:8002", "http://localhost", "http://127.0.0.1"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(etudiant.router)
app.include_router(administrateur.router)
app.include_router(conversation.router)
app.include_router(message.router)
app.include_router(document.router)
app.include_router(email_genere.router)
app.include_router(chat.router)
app.include_router(dashboard.router)


@app.on_event("startup")
def init_admin_and_migrate():
    from app.models.utilisateur import Utilisateur
    from app.models.administrateur import Administrateur
    from app.utils.security import hash_password

    # ── Migration: add columns to document table ──────────────────────
    try:
        from sqlalchemy import text
        conn = SessionLocal().connection()
        conn.execute(text("ALTER TABLE document ADD COLUMN IF NOT EXISTS site VARCHAR(10) DEFAULT ''"))
        conn.execute(text("ALTER TABLE document ADD COLUMN IF NOT EXISTS source VARCHAR(50) DEFAULT 'uploaded'"))
        conn.execute(text("ALTER TABLE document ADD COLUMN IF NOT EXISTS contenu BYTEA"))
        conn.execute(text("ALTER TABLE document ADD COLUMN IF NOT EXISTS nom_fichier VARCHAR(255)"))
        try:
            conn.execute(text("ALTER TABLE document ALTER COLUMN categorie TYPE VARCHAR(100) USING categorie::text"))
        except Exception:
            pass
        conn.connection.commit()
        conn.close()
    except Exception as e:
        print(f"[startup] Migration notice: {e}")

    # ── Auto-create admin if none exists ──────────────────────────────
    admin_email = os.getenv("ADMIN_EMAIL", "admin@unihelp.tn")
    admin_password = os.getenv("ADMIN_PASSWORD", "admin123")

    db = SessionLocal()
    try:
        existing = db.query(Utilisateur).filter(Utilisateur.email == admin_email).first()
        if existing:
            admin = db.query(Administrateur).filter(Administrateur.id == existing.id).first()
            if not admin:
                admin = Administrateur(id=existing.id, departement=None, telephone="")
                db.add(admin)
                db.commit()
            print(f"[startup] Admin existe deja: {admin_email}")
        else:
            user = Utilisateur(
                nom="Admin",
                prenom="System",
                email=admin_email,
                motDePasse=hash_password(admin_password),
            )
            db.add(user)
            db.flush()
            admin = Administrateur(id=user.id, departement=None, telephone="+21600000000")
            db.add(admin)
            db.commit()
            print(f"[startup] Admin cree: {admin_email} / {admin_password}")
    except Exception as e:
        print(f"[startup] Erreur admin: {e}")
        db.rollback()
    finally:
        db.close()


@app.get("/files/{doc_key:path}", tags=["RAG"])
async def proxy_rag_file(doc_key: str):
    """Proxy file downloads to RAG API."""
    rag_url = os.getenv("RAG_API_URL", "http://rag-api:8002")
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{rag_url}/files/{doc_key}")
        return Response(
            content=resp.content,
            media_type=resp.headers.get("content-type", "application/octet-stream"),
            headers={
                "Content-Disposition": resp.headers.get("content-disposition", f'attachment; filename="{doc_key}"'),
            },
        )


@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "projet": "UniHelp",
        "version": "2.0.0",
        "docs": "/docs",
    }
