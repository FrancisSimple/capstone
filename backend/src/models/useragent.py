from src.models.base import Base
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy import ForeignKey
from sqlalchemy.dialects.postgresql import UUID as pgID
from uuid import UUID


class UserAgentLink(Base):
    __tablename__ = "user_agent_link"
    user_id: Mapped[UUID] = mapped_column(pgID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),primary_key=True)
    agent_id: Mapped[UUID] = mapped_column(pgID(as_uuid=True), ForeignKey("agents.id", ondelete="CASCADE"),primary_key=True)
    user = relationship("User",back_populates="user_link")
    agent = relationship("Agent",back_populates="agent_link")