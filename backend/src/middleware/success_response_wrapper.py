from fastapi.responses import JSONResponse
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import json

from src.shared.response_structure import model_response
from typing import Callable, Awaitable
from starlette.responses import Response

class SuccessResponseWrapperMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]):
        # Skip OpenAPI and excluded paths
        EXCLUDE_PATHS = {"/openapi.json", "/health", "/metrics", "/docs"}
        if request.url.path in EXCLUDE_PATHS:
            return await call_next(request)

        response = await call_next(request)
        content_type = response.headers.get("content-type", "").lower()

        if response.status_code < 400 and content_type.startswith("application/json"):
            body = [section async for section in response.body_iterator]
            raw_body = b"".join(body).decode("utf-8")

            try:
                parsed = json.loads(raw_body)
            except json.JSONDecodeError:
                parsed = raw_body

            # Prevent double wrapping
            if isinstance(parsed, dict) and "status" in parsed and "message" in parsed and "data" in parsed:
                return Response(
                    content=json.dumps(parsed),
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type="application/json"
                )

            wrapped = model_response(
                status=True,
                message="Request was successful",
                error={
                    "dev_message": "Everything went well",
                    "user_message": "Request was successful",
                    "extra": {"method": request.method, "path": request.url.path},
                },
                data=parsed,
            )

            return JSONResponse(content=wrapped, status_code=response.status_code)

        return response
