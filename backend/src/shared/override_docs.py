from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi import APIRouter

router = APIRouter()


@router.get("/docs", include_in_schema=False)
def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="KEEYO API",
    )


@router.get("/redoc", include_in_schema=False)
def custom_redoc():
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="KEEYO REDOC"
    )
