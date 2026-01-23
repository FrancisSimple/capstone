from uuid import UUID, uuid4
from datetime import datetime, timezone

from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID as pgUUID
from sqlalchemy import Boolean, String, TIMESTAMP

from src.models.base import Base

class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[UUID] = mapped_column(
        pgUUID(as_uuid=True),
        primary_key=True,
        default=uuid4
    )

    token: Mapped[str] = mapped_column(String, nullable=False)

    user_id: Mapped[UUID] = mapped_column(
        pgUUID(as_uuid=True),
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=datetime.now(timezone.utc),
    )

    expires_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False
    )

    revoked: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )
