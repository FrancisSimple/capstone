from starlette.responses import Response
import json
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from typing import Dict, Any, List
from typing import Callable, Awaitable

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class LoggingMiddleware(BaseHTTPMiddleware):

    @staticmethod
    def filter_sensitive_data(
        data: Dict[str, Any], sensitive_fields: List[str]
    ) -> Dict[str, Any]:
        """
        Redacts sensitive fields from a dictionary.
        :param data: The original data dictionary.
        :param sensitive_fields: List of fields to redact.
        :return: Data dictionary with sensitive fields redacted.
        """
        filtered_data = data.copy()
        for field in sensitive_fields:
            if field in filtered_data:
                filtered_data[field] = "REDACTED"
        return filtered_data

    async def log_request(self, request: Request):
        """
        Logs request details, filtering sensitive data if applicable.
        """
        logging.info(f"Incoming Request: {request.method} {request.url}")

        # Redact sensitive headers
        headers = dict(request.headers)
        sensitive_headers = ["authorization", "api-key", "token_header"]
        filtered_headers = self.filter_sensitive_data(headers, sensitive_headers)
        logging.info(f"Headers: {filtered_headers}")

        # Only log body for specific request methods
        if request.method in ["POST", "PUT", "PATCH", "DELETE"]:
            content_type = request.headers.get("content-type", "").lower()

            # Handle different content types
            if "multipart/form-data" in content_type:
                logging.info("Request Body: [Multipart Form Data]")
                return

            if "application/json" in content_type:
                try:
                    body = await request.body()
                    if body:
                        body_dict = json.loads(body.decode("utf-8"))
                        filtered_body = self.filter_sensitive_data(
                            body_dict, ["password", "token", "token_header"]
                        )
                        logging.info(f"Request Body: {json.dumps(filtered_body)}")
                except Exception as e:
                    logging.info(f"Request Body: [Could not parse JSON: {str(e)}]")
            else:
                logging.info(f"Request Body: [Content-Type: {content_type}]")

    def log_response(self, response: Response, duration: float):
        """
        Logs response details.
        """
        logging.info(
            f"Response Status: {response.status_code}, Duration: {duration:.2f}s"
        )

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ):
        """
        Middleware execution: Logs request and response details.
        """
        start_time = time.time()

        try:
            await self.log_request(request)
            response = await call_next(request)
            duration = time.time() - start_time
            self.log_response(response, duration)
            return response
        except Exception as e:
            duration = time.time() - start_time
            logging.error(f"Error in request processing: {str(e)}")
            raise
