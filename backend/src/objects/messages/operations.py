from src.objects.messages.schema import MessageCreate
from src.shared.services.db_service import DatabaseService
from src.objects.messages.model import Message


async def save_message(message: MessageCreate,db: DatabaseService):
    data = message.model_dump()
    if message.videoUrl:
        url = str(data["videoUrl"])
        data["videoUrl"] = url
    if message.audioUrl:
        url = str(data["audioUrl"])
        data["audioUrl"] = url
    if message.imageUrl:
        url = str(data["imageUrl"])
        data["imageUrl"] = url
    new_account = Message(**data)
    new_account = await db.create(new_account)
    # new_tokens = tokenOperations.generate_tokens(new_account)
    # await tokenOperations.store_refresh_token(
    #     refresh_token=new_tokens['refresh_token'],
    #     user_id=str(new_account.id),
    #     session=db.session # type: ignore
    # )
    return new_account