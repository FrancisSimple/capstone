from typing import Any, Dict, List, Optional, Type
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, delete
from sqlalchemy.orm import DeclarativeMeta

from src.shared.exceptions import CustomException

class DatabaseService:
    """Generic async database service for basic CRUD operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_field(
        self,
        model: Type[DeclarativeMeta],
        field_name: str,
        value: Any,
    ) -> Optional[Any]:
        """Get a single item by field."""
        try:
            field = getattr(model, field_name)
            stmt = select(model).where(field == value)
            result = await self.session.scalars(stmt)
            return result.one_or_none()
        except Exception as e:
            raise CustomException(dev_message=f"Error getting by field: {e}", user_message="Something unexpected happened")
    
    async def return_all(self, stmt):
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def get_by_fields(
        self,
        model: Type[DeclarativeMeta],
        filters: Dict[str, Any]
    ):
        """
        Fetch a single row matching multiple column filters.
        Returns the first matching row or None.
        """
        stmt = select(model)
        for field_name, value in filters.items():
            stmt = stmt.where(getattr(model, field_name) == value)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def exists_by_field(
        self,
        model: Type[DeclarativeMeta],
        field_name: str,
        value: Any,
    ) -> bool:
        """Check if an item exists by field."""
        return bool(await self.get_by_field(model, field_name, value))

    async def create(
        self,
        obj: Any,
    ) -> Any:
        """Add and commit a new object."""
        self.session.add(obj)
        await self.session.commit()
        await self.session.refresh(obj)
        return obj

    async def update_by_field(
        self,
        model: Type[DeclarativeMeta],
        lookup_field: str,
        lookup_value: Any,
        update_data: dict,
    ) -> int:
        """Update item(s) by a field."""
        field = getattr(model, lookup_field)
        stmt = (
            update(model)
            .where(field == lookup_value)
            .values(**update_data)
            .execution_options(synchronize_session="fetch")
        )
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount
    
    async def update_single_field(
    self,
    model: Type[DeclarativeMeta],
    lookup_field: str,
    lookup_value: Any,
    update_field: str,
    update_data: Any,
) -> int:
    
        where_column = getattr(model, lookup_field)
        target_column = getattr(model, update_field)

        stmt = (
            update(model)
            .where(where_column == lookup_value)
            .values({target_column: update_data})
            .execution_options(synchronize_session="fetch")
    )

        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount

    async def delete_by_field(
        self,
        model: Type[DeclarativeMeta],
        field_name: str,
        value: Any,
    ) -> int:
        """Delete item(s) by field."""
        field = getattr(model, field_name)
        stmt = delete(model).where(field == value)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount
    
