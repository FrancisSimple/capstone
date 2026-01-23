from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from typing import AsyncGenerator

from src.models.base import Base  # <-- your new DeclarativeBase
from src.config import Config
import src.models # import needed to classes available before handling db
from src.shared.services.db_service import DatabaseService  # this ensures all models are registered

# Create async engine
if Config.PRODUCTION == 1:
    DB = Config.DATABASE_URL
else:
    DB = Config.DEV_DATABASE_URL

async_engine = create_async_engine(
    url=DB,
    echo=False,
)

# Session factory
AsyncSessionLocal = sessionmaker(
    bind=async_engine, # type: ignore
    class_=AsyncSession,
    autocommit=False,
) # type: ignore


async def init_db():
    async with async_engine.begin() as conn:
        print("Creating tables...")
        print("Tables to create:", Base.metadata.tables.keys())
        # await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)



# Dependency for FastAPI
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session: # type: ignore
        yield session


async def get_db_service(
    session: AsyncSession = Depends(get_session),
) -> DatabaseService:
    return DatabaseService(session=session)
