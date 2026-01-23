from pydantic import BaseModel, Field,HttpUrl
from uuid import UUID
from typing import Optional

class Agent(BaseModel):
    name: str = Field(examples=["Fr3Doctor"],description="Name of the agent")
    description: Optional[str] = Field(examples=["Connect Clients to verified lawyers instantly"], default=None)
    url: HttpUrl = Field(examples=["https://www.youtube.com/"])
    contact: str = Field(examples=["0256591970"])
    email: Optional[str] = Field(examples=["fr3doctor@gmail.com"])
    
class RegisterAgent(Agent):
    password: str = Field(examples=["mysecurity@1234-"])
    
class AgentResponse(Agent):
    id: UUID
    # isVerified: bool
    # isActive: bool

class SaveAgent(BaseModel):
    agent_url: HttpUrl = Field(examples=["https://www.youtube.com"])
    
    