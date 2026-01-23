from src.models.base import TimestampedModel, UUIDModel
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey
from sqlalchemy.dialects.postgresql import UUID as pgID
from uuid import UUID
from typing import Optional
from src.objects.agents.model import Agent
from src.objects.user.model import User


class Message(UUIDModel,TimestampedModel):
    imageUrl: Mapped[Optional[str]] = mapped_column(nullable=True)
    videoUrl: Mapped[Optional[str]] = mapped_column(nullable=True)
    audioUrl: Mapped[Optional[str]] = mapped_column(nullable=True)
    content: Mapped[Optional[str]] = mapped_column(nullable=True)
    sender: Mapped[str]
    # foreign keys columns
    user_id: Mapped[UUID] = mapped_column(pgID(as_uuid=True), ForeignKey("users.id",ondelete='CASCADE'),nullable=False)
    agent_id: Mapped[UUID] = mapped_column(pgID(as_uuid=True), ForeignKey("agents.id",ondelete='CASCADE'),nullable=False)
    # relationship
    user: Mapped["User"] = relationship(back_populates="messages",lazy='selectin')
    agent: Mapped["Agent"] = relationship(back_populates="messages",lazy='selectin')