from fastapi import APIRouter, Depends, Query, status
from src.db.session import get_db_service
from src.objects.agents.model import Agent
from src.shared.exceptions import CustomException
from src.shared.services.db_service import DatabaseService
from src.objects.messages.operations import save_message
from src.objects.messages.schema import MessageCreate, MessageResponse
from src.shared.dependency.jwt.operations import tokenOperations
from src.objects.agents.schema import SaveAgent, AgentResponse

agentApp = APIRouter(tags=["Agents"],prefix="/agent")

# send new message
@agentApp.get("/save", status_code=status.HTTP_201_CREATED,response_model=AgentResponse)
async def send_message(urlData: str = Query(title="Agent url",default = "https://youtube.com"), db:DatabaseService=Depends(get_db_service),user: dict = Depends(tokenOperations.get_token_decoded)):
    # try fetching the url
    agent = await db.get_by_field(model=Agent,field_name="url",value=urlData) # type: ignore
    if not agent:
        raise CustomException(dev_message="Agent Not found",user_message="Agent Not available")
    return agent