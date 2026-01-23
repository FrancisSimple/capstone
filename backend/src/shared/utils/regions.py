from enum import Enum
from fastapi import HTTPException, status

from src.shared.exceptions import CustomException

class RegionEnum(str, Enum):
    AHAFO = "Ahafo"
    ASHANTI = "Ashanti"
    BONO = "Bono"
    BONO_EAST = "Bono East"
    CENTRAL = "Central"
    EASTERN = "Eastern"
    GREATER_ACCRA = "Greater Accra"
    NORTH_EAST = "North East"
    NORTHERN = "Northern"
    OTI = "Oti"
    SAVANNAH = "Savannah"
    UPPER_EAST = "Upper East"
    UPPER_WEST = "Upper West"
    VOLTA = "Volta"
    WESTERN = "Western"
    WESTERN_NORTH = "Western North"









async def validate_region(region: str) -> RegionEnum:
    """Normalizes input and validates it against RegionEnum.

    Steps:
    - Remove the word 'region' (case-insensitive)
    - Strip extra spaces
    - Capitalize each word (e.g., 'greater accra' → 'Greater Accra')
    - Match exactly with RegionEnum values
    """
    # Remove 'region' and clean up
    normalized = region.lower().replace("region", "").strip()
    # Capitalize each word correctly (for multi-word regions like 'Greater Accra')
    normalized = normalized.title()

    try:
        print(f"Normalised region: {normalized}")
        print(RegionEnum(normalized))
        return RegionEnum(normalized)
    except ValueError:
        raise CustomException(
            dev_message=f"The region given is not captured in the system: {region}",
            user_message=f"The region is not captured. Contact the team for clarification.",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
