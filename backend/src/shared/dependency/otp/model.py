from uuid import UUID, uuid4
from datetime import datetime, timezone

from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID as pgUUID
from sqlalchemy import Boolean, String, TIMESTAMP

from src.models.base import Base



class OTP(Base):
    __tablename__ = "otps"

    id: Mapped[UUID] = mapped_column(
        pgUUID(as_uuid=True),
        primary_key=True,
        default=uuid4
    )
    email:Mapped[str] = mapped_column(index=True, nullable=False)
    otp :Mapped[str]= mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.now(timezone.utc),
    )

    expires_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False
    )

class ResetToken(Base):
    __tablename__ = "reset_tokens"

    id: Mapped[uuid4] = mapped_column(
        pgUUID(as_uuid=True),
        primary_key=True,
        default=uuid4
    )
    email: Mapped[str] = mapped_column(String, nullable=False, index=True)
    token: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    expires_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    used: Mapped[bool] = mapped_column(Boolean, default=False)
