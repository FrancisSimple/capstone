from src.objects.user.schema import UserResponse
from src.objects.agents.schema import AgentResponse
from src.shared.dependency.jwt.schema import Token
from pydantic import BaseModel, EmailStr

class UserToken(BaseModel):
    user: UserResponse
    token: Token
    
class AgentToken(BaseModel):
    user: AgentResponse
    token: Token
    
    

# ===============
# CONSTANTS
# ===============
class EmailRequest(BaseModel):
    email: EmailStr

class VerifyRequest(BaseModel):
    email: EmailStr
    code: str