from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base

class Utilisateur(Base):
    __tablename__ = "utilisateur"

    id = Column(Integer, primary_key=True, index=True)
    nom = Column(String(100), nullable=False)
    prenom = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, nullable=False, index=True)
    motDePasse = Column(String(255), nullable=False)
    photo = Column(String, nullable=True)
    estActif = Column(Boolean, default=True)
    dateCreation = Column(DateTime, server_default=func.now())

    etudiant = relationship("Etudiant", back_populates="utilisateur", uselist=False)
    administrateur = relationship("Administrateur", back_populates="utilisateur", uselist=False)
    conversations = relationship("Conversation", back_populates="utilisateur")
