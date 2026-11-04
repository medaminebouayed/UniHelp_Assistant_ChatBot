from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey, LargeBinary
from sqlalchemy.orm import relationship
from app.database import Base


class Document(Base):
    __tablename__ = "document"

    idDocument = Column(Integer, primary_key=True, index=True)
    titre = Column(String(255), nullable=False)
    site = Column(String(10), default="")
    source = Column(String(50), default="uploaded")
    categorie = Column(String(100), default="")
    fichier = Column(String(500))
    contenu = Column(LargeBinary, nullable=True)
    nom_fichier = Column(String(255), nullable=True)
    taille = Column(Float)
    dateAjout = Column(Date)
    administrateur_id = Column(Integer, ForeignKey("administrateur.id"), nullable=False)

    administrateur = relationship("Administrateur", back_populates="documents")
