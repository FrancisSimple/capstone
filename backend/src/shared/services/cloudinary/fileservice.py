import cloudinary
import cloudinary.uploader
import cloudinary.api
from urllib.parse import urlparse
from src.shared.services.cloudinary import config


class FileService:
    """
    A service class to handle file operations with Cloudinary.
    This does not interact with any database — only Cloudinary.
    """

    @staticmethod
    def upload(file, folder: str = "fastapi_uploads", resource_type: str = "auto") -> dict:
        """
        Upload a file to Cloudinary.
        
        Args:
            file: A file-like object (e.g., UploadFile.file from FastAPI).
            folder (str): Folder name in Cloudinary.
        
        Returns:
            dict: { "url": <secure_url>, "public_id": <public_id> }
        """
        
        result = cloudinary.uploader.upload(
            file,
            folder=folder,
            resource_type=resource_type  # force only images
        )
        
        return {
            "url": result["secure_url"],
            "public_id": result["public_id"]
        }

    @staticmethod
    def delete(identifier: str) -> dict:
        """
        Delete a file from Cloudinary.
        
        Args:
            identifier: Can be a public_id or a Cloudinary URL.
        
        Returns:
            dict: Cloudinary's deletion result.
        """
        public_id = identifier

        # If a URL is provided, extract public_id
        if identifier.startswith("http"):
            public_id = FileService._extract_public_id(identifier)

        result = cloudinary.uploader.destroy(public_id)
        return result

    @staticmethod
    def retrieve(url: str) -> dict:
        """
        Retrieve basic info about a file from Cloudinary.
        
        Args:
            url: Cloudinary URL of the file.
        
        Returns:
            dict: File details (from Cloudinary API).
        """
        public_id = FileService._extract_public_id(url)
        result = cloudinary.api.resource(public_id)
        return result

    @staticmethod
    def _extract_public_id(url: str) -> str:
        """
        Helper method to extract Cloudinary public_id from a URL.
        """
        path = urlparse(url).path  # /<folder>/<public_id>.<ext>
        public_id = path.split("/")[-1].split(".")[0]  # remove folder + extension
        folder = "/".join(path.split("/")[1:-1])  # ignore leading '/'
        return f"{folder}/{public_id}" if folder else public_id
