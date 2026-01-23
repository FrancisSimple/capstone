import re

from fastapi import HTTPException, status


class DataValidator:
    @staticmethod
    def normalize_phone_number(phone: str) -> str:
        """Normalize phone number by removing all formatting except + and digits."""
        return re.sub(r"[^\d+]", "", phone)

    @staticmethod
    def normalize_email(email: str) -> str:
        """Normalize email by converting to lowercase and stripping whitespace."""
        return email.strip().lower()

    @staticmethod
    def validate_phone_number(phone: str) -> bool:
        """
        Validate phone number format.
        Accepts formats like: +1234567890, +1-234-567-8900, +1 (234) 567-8900
        """
        # Remove all non-digit characters except +
        cleaned_phone = re.sub(r"[^\d+]", "", phone)

        # Check if it starts with + and has 10-15 digits after country code
        phone_pattern = r"^\+\d{10,15}$"
        return bool(re.match(phone_pattern, cleaned_phone))

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format using regex."""
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(email_pattern, email.strip().lower()))

    @staticmethod
    async def validate_identifier(identifier: str) -> None:
        """Validate email or phone number format."""
        is_email = DataValidator.isEmail(identifier=identifier)

        if is_email:
            if not DataValidator.validate_email(email=identifier):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid email format",
                )
        else:
            if not DataValidator.validate_phone_number(phone=identifier):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid phone number format",
                )

    @staticmethod
    def isEmail(identifier: str) -> bool:
        """Check if the identifier is an email."""
        return "@" in identifier and DataValidator.validate_email(email=identifier)
