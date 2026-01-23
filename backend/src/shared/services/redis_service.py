# redis_service.py
from redis import asyncio as aioredis

from src.config import Config

class RedisService:
    def __init__(self):
        # create redis connection
        self.redis = aioredis.from_url(Config.REDIS_URL, decode_responses=True)

    async def set_key(self, key: str, value: str, expiry_seconds: int = 300):
        """
        Stores a value under a key in Redis with expiration time
        """
        await self.redis.set(key, value, ex=expiry_seconds)

    async def get_key(self, key: str):
        """
        Retrieves the value of a given key
        """
        return await self.redis.get(key)

    async def delete_key(self, key: str):
        """
        Deletes a key-value pair
        """
        await self.redis.delete(key)

    async def exists(self, key: str) -> bool:
        """
        Checks if a key exists
        """
        return await self.redis.exists(key) == 1

redis_service = RedisService()