from typing import Dict
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.shared.exceptions import CustomException
from src.shared.response_structure import model_response
from fastapi import FastAPI


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=model_response(
            status=False,
            message=exc.detail,
            data={},
            error={
                "dev_message": str(exc),
                "user_message": exc.detail,
                "extra": {
                    "method": request.method,
                    "path": request.url.path,
                },
            },
        ),
    )


async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=model_response(
            status=False,
            message="Internal server error",
            data={},
            error={
                "dev_message": str(exc),
                "user_message": "An unexpected error occurred",
                "extra": {
                    "method": request.method,
                    "path": request.url.path,
                    "exception": (
                        str(exc) if hasattr(exc, "args") else "No additional info"
                    ),
                },
            },
        ),
    )

# custom exceptions functions:
async def custom_exception_handler(request: Request, exc: CustomException):
    return JSONResponse(
        status_code=exc.status_code,
        content=model_response(
            status=False,
            message=exc.user_message,
            data={},
            error={
                "dev_message": exc.dev_message,
                "user_message": exc.user_message,
                "extra": {
                    "method": request.method,
                    "path": request.url.path,
                },
            },
        )
    )
    

# ==============================
# Data validator function
# ==============================
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    dev_messages: Dict[str, str] = {}
    user_message = []

    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"] if loc != "body")
        dev_messages[field] = error["msg"]  # raw dev message
        user_message.append(field)
    
    return JSONResponse(
        status_code=422,
        content=model_response(
            status=False,
            message="Invalid data input. Kindly check your data.",
            data={},
            error={
                "dev_message": dev_messages,
                "user_message": f"Invalid input. Check: {user_message}",
                "extra": {
                    "method": request.method,
                    "path": request.url.path,
                },
            },
        )
    )
# end of validator function:
# =========================================================



def register_exception_handlers(app: FastAPI):
    """Middleware to wrap failed/exception JSON responses in a standard format.
    This middleware intercepts responses with status codes 4xx and 5xx,
    ensuring they are wrapped in a consistent JSON structure.
    It is particularly useful for APIs to maintain a uniform response format,
    even for successful requests.
    """
    app.add_exception_handler(
        exc_class_or_status_code=StarletteHTTPException, handler=http_exception_handler  # type: ignore
    )
    app.add_exception_handler(
        RequestValidationError, handler=validation_exception_handler  # type: ignore
    )
    app.add_exception_handler(
        exc_class_or_status_code=CustomException, handler=custom_exception_handler # type: ignore
    )
    app.add_exception_handler(
        exc_class_or_status_code=Exception, handler=general_exception_handler
    )
