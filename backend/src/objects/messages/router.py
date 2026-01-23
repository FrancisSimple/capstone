from fastapi import APIRouter, Depends, status
from src.db.session import get_db_service
from src.shared.services.db_service import DatabaseService
from src.objects.messages.operations import save_message
from src.objects.messages.schema import MessageCreate, MessageResponse
from src.shared.dependency.jwt.operations import tokenOperations

messageApp = APIRouter(tags=["Messages"],prefix="/message")

# send new message
@messageApp.post("/send", status_code=status.HTTP_201_CREATED,response_model=MessageResponse)
async def send_message(message:MessageCreate, db:DatabaseService=Depends(get_db_service),user: dict = Depends(tokenOperations.get_token_decoded)):
    response = await save_message(message,db)
    return response