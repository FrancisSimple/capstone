import cloudinary
from src.config import Config

cloudinary.config(
    cloud_name= Config.CLOUD_NAME,
    api_key= Config.CLOUD_API_KEY,
    api_secret = Config.CLOUD_API_SECRET,
    secure=True
)
