from fastapi import HTTPException

from psycopg2 import DataError, IntegrityError, OperationalError


# MY CUSTOM EXCEPTIONS CLASSES:
from typing import Optional


class CustomException(Exception):
    def __init__(
        self,
        dev_message: str,
        user_message: Optional[str] = "Item already exists",
        status_code: int = 409
    ):
        self.dev_message = dev_message
        self.user_message = user_message
        self.status_code = status_code
        super().__init__(dev_message)





def handle_exception(e: Exception) -> HTTPException:
    if isinstance(e, HTTPException):
        status_code = e.status_code
        detail = e.detail
    elif isinstance(e, ValueError):
        status_code = 400
        detail = "Invalid input provided. Please check your data."
    elif isinstance(e, KeyError):
        status_code = 404
        detail = "Requested resource could not be found."
    elif isinstance(e, IntegrityError):
        status_code = 409  # Conflict
        detail = "A conflict occurred. Please ensure the data is unique."
    elif isinstance(e, DataError):
        status_code = 400
        detail = "Invalid data provided. Please verify your input."
    elif isinstance(e, OperationalError):
        status_code = 503
        detail = "Service temporarily unavailable. Please try again later."
    elif isinstance(e, AttributeError):
        status_code = 400
        detail = "Invalid request. Missing required attributes."
    elif isinstance(e, TypeError):
        status_code = 400
        detail = "Invalid data type in request. Please correct it."
    elif isinstance(e, PermissionError):
        status_code = 403
        detail = "You do not have permission to access this resource."
    elif isinstance(e, FileNotFoundError):
        status_code = 404
        detail = "The requested file could not be found."
    elif isinstance(e, TimeoutError):
        status_code = 504
        detail = "The request took too long. Please try again later."
    elif isinstance(e, NotImplementedError):
        status_code = 501
        detail = "This feature is not available yet."
    else:
        status_code = 500
        detail = "Something went wrong. Please try again later."

    raise HTTPException(
        status_code=status_code,
        detail=detail,
    )
