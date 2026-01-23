from pydantic import BaseModel

from src.objects.authentication.schema import StudentResponse, ProviderResponse
from src.shared.dependency.jwt.schema import Token

class StudentToken(BaseModel):
  user:StudentResponse
  token:Token

class ProviderToken(BaseModel):
  user:ProviderResponse
  token:Token
