# src/shared/services/token_service.py

from datetime import datetime, timedelta
from typing import Any

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlmodel import select, Session
from sqlmodel.ext.asyncio.session import AsyncSession

from src.config import Config
from src.shared.dependency.jwt.model import RefreshToken
from src.shared.exceptions import CustomException

security = HTTPBearer(auto_error=False)


class TokenService:
    def __init__(self):
        self.access_secret = Config.JWT_SECRET_KEY
        self.refresh_secret = Config.JWT_REFRESH_SECRET_KEY
        self.access_expiry = Config.TOKEN_EXPIRY_MINUTES
        self.refresh_expiry = Config.TOKEN_EXPIRY_DAYS
        self.algorithm = Config.ALGORITHM

    def create_access_token(self, data: dict) -> str:
        """Generates a short-lived access token."""
        to_encode = data.copy()
        to_encode["exp"] = datetime.now() + timedelta(minutes=self.access_expiry)
        return jwt.encode(to_encode, self.access_secret, algorithm=self.algorithm)

    def create_refresh_token(self, data: dict) -> str:
        """Generates a long-lived refresh token."""
        to_encode = data.copy()
        to_encode["exp"] = datetime.now() + timedelta(days=self.refresh_expiry)
        return jwt.encode(to_encode, self.refresh_secret, algorithm=self.algorithm)

    def decode_token(self, token: str, is_refresh: bool = False) -> dict:
        """Decodes a token using the appropriate secret."""
        secret = self.refresh_secret if is_refresh else self.access_secret
        return jwt.decode(token, secret, algorithms=[self.algorithm])

    def generate_tokens(self, user: Any) -> dict:
        """Generates both access and refresh tokens for a beneficiary."""
        base_payload = {
            "user_id": str(user.id),
            "email": user.email,
            #"role": user.role
        }

        access_token = self.create_access_token(base_payload)
        refresh_token = self.create_refresh_token(base_payload)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }

    async def store_refresh_token(self, session: Session, refresh_token: str, user_id: str) -> RefreshToken:
        """Stores a refresh token in the database."""
        token = RefreshToken(
            user_id=user_id,
            token=refresh_token,            
            expires_at=datetime.now() + timedelta(days=self.refresh_expiry)
        )
        session.add(token)
        await session.commit() # type: ignore
        return token

    async def rotate_refresh_token(self, session: Session, old_token: str,user_type:str) -> dict:
        """Handles refresh token rotation and revocation of old token."""
        try:
            payload = self.decode_token(old_token, is_refresh=True)
            user_id = payload.get("user_id")
        except JWTError:
            raise CustomException(
                dev_message="Failed to decode the refresh token",
                user_message="Something went wrong. Kindly contact the team.",
                status_code=401
            )

        stmt =select(RefreshToken).where(
                RefreshToken.token == old_token,
                RefreshToken.revoked == False
            )
        
        result = await session.scalars(stmt) # type: ignore
        token_in_db = result.one_or_none()

        if not token_in_db or token_in_db.expires_at < datetime.now():
            raise CustomException(
                dev_message="Token is revoked or expired",
                user_message="An issue showed up. Kindly contact the team",
                status_code=401
            )

        # Revoke old token
        token_in_db.revoked = True
        session.add(token_in_db)

        # Generate and store new token
        new_tokens = self.generate_tokens(payload)
        await self.store_refresh_token(session, new_tokens["refresh_token"], user_id) # type: ignore
        return new_tokens

    def get_token_decoded(self, credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
        """Retrieves and validates the current user from access token."""
        if not credentials or not credentials.scheme == "Bearer":
            raise CustomException(
                dev_message="Token is missing",
                user_message="There is credential issues. Kindly contact the team.",
                status_code=401
            )
        token = credentials.credentials
        try:
            payload = self.decode_token(token)
            return payload
        except JWTError:
            raise CustomException(
                dev_message="Token failed to be authenticated. It might be expired or invalid.",
                user_message="There might be issues with your credentials. Try signing in again.",
                status_code=401
            )

    async def get_access_from_existing_refresh_token(
            self, session: AsyncSession, user_id: str
        ) -> dict:
            """
            Checks if the user has an existing valid (non-revoked, unexpired) refresh token.
            If valid, generates a new access token and returns both.
            Otherwise, raises an exception.
            """
            # Step 1: Query for the most recent valid refresh token for the user
            stmt = (
                select(RefreshToken)
                .where(
                    RefreshToken.user_id == user_id,
                    RefreshToken.revoked == False,
                    RefreshToken.expires_at > datetime.utcnow()
                )
                .order_by(RefreshToken.created_at.desc())
            )

            result = await session.scalars(stmt)
            token = result.one_or_none()

            # Step 2: Handle case where no valid refresh token is found
            if not token:
                raise CustomException(
                    dev_message=f"No valid refresh token found for user {user_id}",
                    user_message="Session expired. Please sign in again.",
                    status_code=401
                )

            # Step 3: Decode refresh token to extract user payload
            try:
                payload = self.decode_token(token.token, is_refresh=True)
            except JWTError:
                raise CustomException(
                    dev_message="Stored refresh token is invalid or corrupted.",
                    user_message="There was a token issue. Please sign in again.",
                    status_code=401
                )

            # Step 4: Generate a new access token
            access_token = self.create_access_token({
                "user_id": payload.get("user_id"),
                "email": payload.get("email"),
                "role": payload.get("role")
            })

            return {
                "access_token": access_token,
                "refresh_token": token.token,
                "token_type": "bearer"
            }


# class this throughout the application
tokenOperations = TokenService()