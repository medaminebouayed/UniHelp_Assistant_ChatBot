from sqlalchemy import Column, Integer, String, Date, Enum, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base
from app.models.enums import (
    SexeEnum, DiplomeEnum, UniversiteEnum,
    DiplomeSouhaiteEnum, NiveauEnum, SourceEnum
)

class Etudiant(Base):
    __tablename__ = "etudiant"

    id = Column(Integer, ForeignKey("utilisateur.id"), primary_key=True)
    CIN = Column(String(20), unique=True)
    passport = Column(String(20))
    pays = Column(String(100))
    dateNaissance = Column(Date)
    sexe = Column(Enum(SexeEnum))
    telephone = Column(String(20))
    adresse = Column(String(255))
    dernierDiplomeObtenu = Column(Enum(DiplomeEnum))
    anneeObtention = Column(String(10))
    universite = Column(Enum(UniversiteEnum))
    diplomeSouhaite = Column(Enum(DiplomeSouhaiteEnum))
    niveau = Column(Enum(NiveauEnum))
    cours = Column(String(255))
    sourceConnaissanceUniversite = Column(Enum(SourceEnum))

    utilisateur = relationship("Utilisateur", back_populates="etudiant")
    emails_generes = relationship("EmailGenere", back_populates="etudiant")
