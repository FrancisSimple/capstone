import re
import logging
from src.config import Config
from fastapi import Request
from fastapi.responses import JSONResponse, Response
from typing import Callable, Awaitable
from datetime import datetime, timezone

from src.shared.response_structure import model_response

# Use uvicorn.error logger so logs appear in standard output
logger = logging.getLogger("uvicorn.error")

# ✅ ALLOWED_ORIGINS now comes from environment variables for security flexibility
#    so you don't have to hardcode in production
if len(Config.ALLOWED_ORIGINS) > 0:
    ALLOWED_ORIGINS = Config.ALLOWED_ORIGINS
else:
    ALLOWED_ORIGINS= [
        r"^http://localhost:\d+$",      # Dev machine ports
        r"^http://127\.0\.0\.1:8000$",  # Localhost FastAPI default
        r"https://keeyoapi.binbyte.dev",
    ]

# ✅ Extra: Precompile regex for performance (avoid re-compiling every request)
ALLOWED_ORIGIN_PATTERNS = [re.compile(pattern) for pattern in ALLOWED_ORIGINS]

# ✅ Extra: Explicit allowed HTTP methods & headers for clarity
ALLOWED_METHODS = Config.ALLOWED_METHODS
ALLOWED_HEADERS = Config.ALLOWED_HEADERS

# ✅ Function to check origin
def is_valid_origin(origin: str) -> bool:
    return any(pattern.match(origin) for pattern in ALLOWED_ORIGIN_PATTERNS)

# ✅ Middleware function
async def cors_validation_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]]
):
    origin = request.headers.get("origin")

    # ✅ SECURITY: Log every incoming origin with timestamp & IP
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"[CORS CHECK] Time={datetime.now(timezone.utc).isoformat()} "
                f"Origin={origin} ClientIP={client_ip} Path={request.url.path}")

    # ✅ Block if origin exists but not in allowed list
    if origin and not is_valid_origin(origin):
        logger.warning(f"❌ Unauthorized CORS origin attempt: {origin} from IP {client_ip}")
        return JSONResponse(
            status_code=403,
            content=model_response(
                status=False,
                message="Unauthorized CORS origin",
                error={
                    "dev_message": f"The given origin is not allowed: {origin}",
                    "user_message": "Request not allowed from your location.",
                    "extra": {
                        "method": request.method,
                        "path": request.url.path,
                        "client_ip": client_ip
                    },
                },
                data={}
            ),
        )

    # ✅ Handle OPTIONS preflight requests early (performance & correctness)
    if request.method == "OPTIONS" and origin and is_valid_origin(origin):
        logger.info(f"[CORS] Preflight request approved for origin {origin}")
        preflight_response = Response()
        preflight_response.headers["Access-Control-Allow-Origin"] = origin
        preflight_response.headers["Access-Control-Allow-Methods"] = ", ".join(ALLOWED_METHODS)
        preflight_response.headers["Access-Control-Allow-Headers"] = ", ".join(ALLOWED_HEADERS)
        preflight_response.headers["Access-Control-Allow-Credentials"] = "true"
        preflight_response.headers["Access-Control-Max-Age"] = "3600"  # Cache for 1 hour
        return preflight_response

    # ✅ Continue request if origin is valid
    response = await call_next(request)

    # ✅ Add actual CORS headers to the response
    if origin and is_valid_origin(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = ", ".join(ALLOWED_METHODS)
        response.headers["Access-Control-Allow-Headers"] = ", ".join(ALLOWED_HEADERS)
        response.headers["Access-Control-Allow-Credentials"] = "true"

    return response
