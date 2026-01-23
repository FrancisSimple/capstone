from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from src.objects.user.schema import RegisterUser, UserLogin
from src.objects.authentication.schema import EmailRequest, UserToken, AgentToken, VerifyRequest
from src.db.session import get_db_service
from src.shared.services.email_service import email_service
from src.shared.services.redis_service import redis_service
from src.shared.services.db_service import DatabaseService
from src.shared.dependency.passwords import generate_password_hash
from src.objects.authentication.operations import create_user, create_agent, generate_otp, validate_login
from src.objects.agents.schema import AgentResponse, RegisterAgent
from typing import Union


authApp = APIRouter(tags=["Authentication"],prefix="/auth")


# register new user:
@authApp.post("/user",response_model=UserToken)
async def register_user(user:RegisterUser, db: DatabaseService = Depends(get_db_service)):
    hpw = generate_password_hash(user.password)
    user.password = hpw
    response = await create_user(user,db)
    return response

# register new agent
@authApp.post("/agent", status_code=status.HTTP_201_CREATED,response_model=AgentResponse)
async def register_agent(agent:RegisterAgent, db:DatabaseService=Depends(get_db_service)):
    hpw = generate_password_hash(agent.password)
    agent.password = hpw
    response = await create_agent(agent,db)
    return response

# # send new message
@authApp.post("/login", status_code=status.HTTP_201_CREATED,response_model=Union[UserToken,AgentToken])
async def login(data: UserLogin, db:DatabaseService=Depends(get_db_service)):
    response = await validate_login(data,db)
    return response

# Refresh access token endpoint
@authApp.post("/refresh")
async def refresh_access_token(token: dict,):
    return {'new_access_token': ""}



# =================
# CONSTANT ENDPOINTS
# ================
@authApp.post("/send-otp")
async def send_otp(request: EmailRequest, background_tasks: BackgroundTasks):
    """
    Generates an OTP, stores it in Redis for 5 minutes, 
    and queues an email to be sent.
    """
    email = request.email
    
    # 1. Generate the Code
    otp_code = generate_otp()
    
    # 2. Create a unique Redis key, e.g., "otp:user@email.com"
    # It is good practice to prefix keys to avoid collisions
    redis_key = f"otp:{email}"
    
    # 3. Save to Redis with Expiry
    # We await this because redis_service is async
    await redis_service.set_key(redis_key, otp_code)
    
    # 4. Send Email via Background Task
    # We use background_tasks so the API returns instantly without waiting for Gmail
    background_tasks.add_task(email_service.send_verification_email, email, otp_code)
    
    return {"message": "OTP sent successfully. Please check your email."}


@authApp.post("/verify-otp")
async def verify_otp(request: VerifyRequest):
    """
    Validates the OTP provided by the user.
    """
    redis_key = f"otp:{request.email}"
    
    # 1. Retrieve the code from Redis
    stored_code = await redis_service.get_key(redis_key)
    
    # 2. Check if it exists (if None, it expired or never existed)
    if not stored_code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="OTP has expired or is invalid."
        )
    
    # 3. Compare the codes
    if stored_code != request.code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Incorrect OTP code."
        )
    
    # 4. Success! - Clean up
    # Delete the key so the OTP cannot be used a second time
    await redis_service.delete_key(redis_key)
    
    # Logic to login user or return JWT token goes here...
    return {"message": "Verification successful", "verified": True}