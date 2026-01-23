from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID
from typing import Optional

class User(BaseModel):
    firstName: str = Field(examples=["Francis"])
    middleName: Optional[str] = Field(examples=["Kwame"], default=None)
    lastName: str = Field(examples=["Sewor"])

class RegisterUser(User):
    email: str = Field(examples=["francis@gmail.com","samuel@gmail.com"])
    password: str = Field(examples=["mysecurity@1234-"])
    
class UserResponse(User):
    id: UUID
    isVerified: bool
    isActive: bool
    email: str
    model_config = ConfigDict(from_attributes=True)

class UserLogin(BaseModel):
    email: str = Field(examples=["francis@gmail.com","samuel@gmail.com"])
    password: str = Field(examples=["mysecurity@1234-"])
    
    