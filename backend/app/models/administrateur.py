from sqlalchemy import Column, Integer, String, Enum, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base
from app.models.enums import DepartementEnum

class Administrateur(Base):
    __tablename__ = "administrateur"

    id = Column(Integer, ForeignKey("utilisateur.id"), primary_key=True)
    departement = Column(Enum(DepartementEnum))
    telephone = Column(String(20))

    utilisateur = relationship("Utilisateur", back_populates="administrateur")
    documents = relationship("Document", back_populates="administrateur")
