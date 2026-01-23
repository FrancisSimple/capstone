from pydantic import BaseModel, Field,HttpUrl
from uuid import UUID
from typing import Optional

class Message(BaseModel):
    videoUrl: HttpUrl = Field(examples=["https://www.youtube.com/"])
    imageUrl:HttpUrl = Field(examples=["https://www.youtube.com/"])
    audioUrl: HttpUrl = Field(examples=["https://www.youtube.com/"])
    content: str = Field(examples=["This may be a tag on a media file"])

    
class MessageCreate(Message):
    agent_id: UUID = Field(examples=["d70dc3fa-1011-4e76-b157-0dff7913a19f"])
    user_id: UUID = Field(examples=["51166604-052d-403c-bb5c-e7c1b457a794"])
    sender: str = Field(default="user", examples=["user", "agent"])
    
class MessageResponse(MessageCreate):
    id: UUID
    # isVerified: bool
    # isActive: bool
    
    