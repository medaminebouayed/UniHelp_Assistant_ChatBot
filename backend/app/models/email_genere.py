from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base

class EmailGenere(Base):
    __tablename__ = "email_genere"

    idEmail = Column(Integer, primary_key=True, index=True)
    object = Column(String(255))
    destinataire = Column(String(150))
    contenu = Column(Text)
    etudiant_id = Column(Integer, ForeignKey("etudiant.id"), nullable=False)

    etudiant = relationship("Etudiant", back_populates="emails_generes")
