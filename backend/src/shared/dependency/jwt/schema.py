# src/schemas/token.py
from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    user_id: str  # user_id
    user_email: str
    user_role:int
    user_contact:str
    exp: int
