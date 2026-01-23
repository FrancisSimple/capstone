from redis.asyncio import Redis
from src.config import Config

token_blocklist = Redis(
    host=Config.REDIS_HOST,
    port=Config.REDIS_PORT,
    db=0,
    decode_responses=True,  # so you get strings instead of bytes
)


async def add_jti_to_blocklist(jti: str) -> None:
    """
    Add a JWT ID (jti) to the Redis blocklist.
    :param jti: The JWT ID to add to the blocklist.
    """
    await token_blocklist.set(name=jti, value="", ex=Config.JTI_EXPIRY)


async def token_in_blocklist(jti: str) -> bool:
    """
    Check if a JWT ID (jti) is in the Redis blocklist.
    :param jti: The JWT ID to check.
    :return: True if the jti is in the blocklist, False otherwise.
    """
    return await token_blocklist.exists(jti) > 0
