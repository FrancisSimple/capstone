from typing import Any, Dict, Optional


def model_response(
    status: Optional[bool] = None,
    message: Optional[str] = None,
    data: Any = None,
    error: Optional[Dict[str, Any]] = None,
    pagination: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Utility function to create a consistent API response structure.

    Args:
        status (bool): Whether the request was successful or not.
        message (str): A user-friendly message (success or error).
        data (Any, optional): The data to be returned. Defaults to None.
        error (Dict[str, any], optional): Contains error messages (for dev and user). Defaults to None.
        pagination (Dict[str, int], optional): Pagination information (total, page, per_page). Defaults to None.

    Returns:
        Dict[str, Any]: The formatted response dictionary.
    """

    # Ensure status is a boolean
    status = status if status is not None else False

    # Ensure data is not None, default to empty dict if it is
    if data is None:
        data = {}

    # default message if not provided
    if message is None:
        message = "Operation completed successfully" if status else "An error occurred"

    # Default error structure if not provided
    if error is None:
        error = {
            "dev_message": "Developer validation error here for debugging",
            "user_message": "Error that can be displayed to the user when there is an error",
            "extra": {
                "method": "METHOD",
                "path": "api/path/here",
            },
        }

    # Default pagination structure if not provided
    if pagination is None:
        pagination = {}

    response = {
        "status": status or False,
        "message": message or "No message provided",
        "error": error or {},
        "data": data if data is not None else {},
    }

    # Include pagination metadata if provided
    if pagination:
        response["meta"] = {"pagination": pagination}

    return response
