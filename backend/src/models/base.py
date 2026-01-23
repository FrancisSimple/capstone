from sqlalchemy.orm import DeclarativeBase, declared_attr, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID as pgUUID
from sqlalchemy import func, TIMESTAMP
from uuid import UUID, uuid4
from datetime import datetime

# Shared Declarative Base
class Base(DeclarativeBase):
    pass

# Automatically add __tablename__
class BaseModel(Base):
    __abstract__ = True

    @declared_attr # type: ignore
    def __tablename__(cls) -> str:
        return cls.__name__.lower() + "s"

# ✅ UUID mixin MUST inherit from BaseModel
class UUIDModel(BaseModel):
    __abstract__ = True
    id: Mapped[UUID] = mapped_column(pgUUID(as_uuid=True), primary_key=True, default=uuid4)

# Timestamp mixin (optional)
class TimestampedModel:
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
    )
    last_updated: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
