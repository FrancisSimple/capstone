from src.objects.user.schema import RegisterUser, UserResponse, UserLogin
from src.shared.dependency.passwords import verify_password
from src.shared.exceptions import CustomException
from src.shared.services.db_service import DatabaseService
from src.objects.user.model import User
from src.objects.agents.model import Agent
from src.shared.dependency.jwt.operations import tokenOperations
from src.objects.agents.schema import RegisterAgent
from src.objects.authentication.schema import UserToken
from src.shared.dependency.jwt.schema import Token
import secrets
import string







async def create_user(user: RegisterUser,db: DatabaseService):
    data = user.model_dump()
    new_account = User(**data)
    new_account = await db.create(new_account)
    user_schema = UserResponse.model_validate(new_account)
    new_tokens = tokenOperations.generate_tokens(new_account)
    await tokenOperations.store_refresh_token(
        refresh_token=new_tokens['refresh_token'],
        user_id=str(new_account.id),
        session=db.session # type: ignore
    )
    
    return UserToken(user=user_schema, token=Token(**new_tokens))


async def create_agent(agent: RegisterAgent,db: DatabaseService):
    data = agent.model_dump()
    url = str(agent.url)
    data["url"] = url
    new_account = Agent(**data)
    new_account = await db.create(new_account)
    # new_tokens = tokenOperations.generate_tokens(new_account)
    # await tokenOperations.store_refresh_token(
    #     refresh_token=new_tokens['refresh_token'],
    #     user_id=str(new_account.id),
    #     session=db.session # type: ignore
    # )
    return new_account


async def validate_login(data: UserLogin, db_service: DatabaseService):
    # Try Student
    db_user = await db_service.get_by_field(
        model=User, # type: ignore
        field_name="email",
        value=data.email
    )

    # If not found, try Provider
    if not db_user:
        db_user = await db_service.get_by_field(
            model=Agent, # type: ignore
            field_name="email",
            value=data.email
        )

    if not db_user:
        raise CustomException(
            dev_message="Email not found",
            user_message="Entered email not found.",
            status_code=404
        )

    # Verify password
    if not verify_password(data.password, db_user.password):
        raise CustomException(
            dev_message="Password is incorrect.",
            user_message="Invalid password."
        )

    # Generate tokens
    tokens = await tokenOperations.get_access_from_existing_refresh_token(
        session=db_service.session,  # type: ignore
        user_id=str(db_user.id)
    )

    return {"user": db_user, "token": tokens}




# ===========================
# CONSTANT OPERATIONS
# ===========================
def generate_otp(length: int = 6) -> str:
    """Generates a secure random numeric string."""
    return ''.join(secrets.choice(string.digits) for _ in range(length))