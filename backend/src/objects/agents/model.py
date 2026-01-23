from src.models.base import TimestampedModel, UUIDModel
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import List, TYPE_CHECKING
from src.models.useragent import UserAgentLink

if TYPE_CHECKING:
    from src.objects.messages.model import Message
    from src.objects.user.model import User

class Agent(UUIDModel,TimestampedModel):
    url: Mapped[str]
    name: Mapped[str]
    description: Mapped[str]
    contact: Mapped[str]
    email: Mapped[str] = mapped_column(unique=True, nullable=False)
    password: Mapped[str]
    # ================
    # Relationships
    # ================
    # to link table
    agent_link: Mapped[List["UserAgentLink"]] = relationship(back_populates="agent",cascade="all, delete-orphan")
    # to agents
    users: Mapped[List["User"]] = relationship(back_populates="agents",viewonly=True,secondary="user_agent_link")
    # to messages
    messages: Mapped[List["Message"]] = relationship(back_populates="agent",lazy="selectin")