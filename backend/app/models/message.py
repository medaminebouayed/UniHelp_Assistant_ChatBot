from sqlalchemy import Column, Integer, Text, Date, ForeignKey, String
from sqlalchemy.orm import relationship
from app.database import Base

class Message(Base):
    __tablename__ = "message"

    idMessage = Column(Integer, primary_key=True, index=True)
    contenu = Column(Text, nullable=False)
    dateEnvoi = Column(Date, nullable=False)
    role = Column(String(10), default="user")  # "user" ou "assistant"
    conversation_id = Column(Integer, ForeignKey("conversation.idConversation", ondelete="CASCADE"), nullable=False)

    conversation = relationship("Conversation", back_populates="messages")
