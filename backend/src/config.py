from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuration settings for the application.
    """

    DATABASE_URL: str
    GOOGLE_API_KEY:SecretStr
    GROQ_API_KEY:SecretStr
    DEV_DATABASE_URL :str
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    JTI_EXPIRY: int
    TOKEN_EXPIRY_MINUTES: int
    JWT_SECRET_KEY:str
    JWT_REFRESH_SECRET_KEY:str
    ALGORITHM:str
    TOKEN_EXPIRY_DAYS: int
    EMAIL_API_KEY: str
    EMAIL_APP_PASSWORD:str
    EMAIL_ADDRESS:str
    REDIS_URL:str
    ALLOWED_ORIGINS: list[str]
    ALLOWED_METHODS: list[str]
    ALLOWED_HEADERS: list[str]
    PRODUCTION: int
    SMTP_SERVER: str
    SMTP_PORT:int
    OTP_EXPIRY_MINUTES:int
    CLOUD_NAME:str
    CLOUD_API_KEY:str
    CLOUD_API_SECRET:str


    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


Config = Settings()  # type: ignore
