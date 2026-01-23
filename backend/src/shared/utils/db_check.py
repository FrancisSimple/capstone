from typing import Type, Any, Optional, Union
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.db.session import get_session
from src.shared.exceptions import CustomException


class UniqueChecker:
    def __init__(self, session: AsyncSession = Depends(get_session)):
        self.session = session

    async def exists(
        self,
        model: Type[Any],
        field_name: str,
        value: Union[str, float, int],
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Check if a value exists in a given SQLAlchemy model field.
        Optionally raises an AlreadyExistsException with a custom user message.
        """
        try:
            column_attr = getattr(model, field_name)
        except AttributeError:
            raise ValueError(f"'{field_name}' is not a valid field for model '{model.__name__}'")

        stmt = select(model).where(column_attr == value)
        result = await self.session.execute(stmt)
        instance = result.scalar_one_or_none()

        if instance and error_message:
            raise CustomException(
                dev_message=f"{model.__name__}.{field_name} already exists: {value}",
                user_message=error_message
            )

        return instance is not None
