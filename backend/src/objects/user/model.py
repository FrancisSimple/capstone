from src.models.base import UUIDModel, TimestampedModel
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import Optional, List, TYPE_CHECKING
from src.models.useragent import UserAgentLink

if TYPE_CHECKING:
    from src.objects.agents.model import Agent
    from src.objects.messages.model import Message

class User(UUIDModel,TimestampedModel):
    firstName: Mapped[str]
    middleName: Mapped[Optional[str]] = mapped_column(nullable=True)
    lastName: Mapped[str]
    email: Mapped[str] = mapped_column(unique=True)
    isVerified: Mapped[bool] = mapped_column(default=False, nullable=False)
    isActive: Mapped[bool] = mapped_column(default=False, nullable=False)
    password: Mapped[str]
    # =============
    # relationship
    # =============
    # to link table
    user_link: Mapped[List["UserAgentLink"]] = relationship(back_populates="user",cascade="all, delete-orphan")
    # to agents
    agents: Mapped[List["Agent"]] = relationship(back_populates="users",viewonly=True,secondary="user_agent_link")
    # to messages
    messages: Mapped[List["Message"]] = relationship(back_populates="user",lazy="selectin")
    