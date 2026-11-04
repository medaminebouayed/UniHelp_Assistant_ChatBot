from sqlalchemy import Column, Integer, String, Boolean, Date, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base

class Conversation(Base):
    __tablename__ = "conversation"

    idConversation = Column(Integer, primary_key=True, index=True)
    titre = Column(String(255))
    estFavori = Column(Boolean, default=False)
    dateDernierModification = Column(Date)
    utilisateur_id = Column(Integer, ForeignKey("utilisateur.id"), nullable=False)

    utilisateur = relationship("Utilisateur", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
