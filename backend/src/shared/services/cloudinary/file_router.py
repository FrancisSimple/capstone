from fastapi import File, UploadFile, APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.db.session import get_db_service
from src.shared.exceptions import CustomException
from src.shared.services.cloudinary.fileservice import FileService
from src.shared.services.db_service import DatabaseService

# Import your models
from src.shared.services.cloudinary.image_model import Image
# from src.shared.services.cloudinary.video_model import Video  # Uncomment when you create this model
# from src.shared.services.cloudinary.audio_model import Audio  # Uncomment when you create this model
# from src.shared.services.cloudinary.document_model import Document # Uncomment when you create this model

mediaApp = APIRouter(prefix='/media', tags=["Media Uploads"])

# ==========================================
# 1. IMAGE UPLOAD (JPG, PNG, GIF, WEBP)
# ==========================================
@mediaApp.post("/image/upload")
async def upload_image(
    file: UploadFile = File(...), 
    db: DatabaseService = Depends(get_db_service)
):
    """
    Uploads an image file.
    Cloudinary Resource Type: 'image'
    """
    try:
        if not file.content_type.startswith("image/"): # type: ignore
            raise HTTPException(status_code=400, detail="File provided is not an image.")

        # Upload to Cloudinary (Stream file directly)
        result = FileService.upload(file.file, resource_type="image")
        
        if result:
            # Save to Database
            image = Image(
                url=result["url"], 
                public_id=result["public_id"]
            )
            await db.create(image)
            
            return {
                "id": image.id,
                "url": image.url, 
                "public_id": image.public_id,
                "type": "image"
            }
            
    except Exception as e:
        raise CustomException(
            dev_message=f"Image upload failed: {str(e)}",
            user_message="Failed to upload image",
            status_code=500,
        )


# ==========================================
# 2. VIDEO UPLOAD (MP4, MOV, AVI)
# ==========================================
@mediaApp.post("/video/upload")
async def upload_video(
    file: UploadFile = File(...), 
    db: DatabaseService = Depends(get_db_service)
):
    """
    Uploads a video file.
    Cloudinary Resource Type: 'video'
    """
    try:
        if not file.content_type.startswith("video/"): # type: ignore
            raise HTTPException(status_code=400, detail="File provided is not a video.")

        # Upload to Cloudinary
        result = FileService.upload(file.file, resource_type="video")
        
        if result:
            # TODO: Create a Video model and uncomment below
            # video = Video(url=result["url"], public_id=result["public_id"])
            # await db.create(video)
            
            return {
                # "id": video.id,
                "url": result["url"], 
                "public_id": result["public_id"],
                "type": "video"
            }
            
    except Exception as e:
        raise CustomException(
            dev_message=f"Video upload failed: {str(e)}",
            user_message="Failed to upload video",
            status_code=500,
        )


# ==========================================
# 3. AUDIO UPLOAD (MP3, WAV, AAC)
# ==========================================
@mediaApp.post("/audio/upload")
async def upload_audio(
    file: UploadFile = File(...), 
    db: DatabaseService = Depends(get_db_service)
):
    """
    Uploads an audio file.
    Cloudinary Resource Type: 'video' (Standard for Audio in Cloudinary)
    """
    try:
        if not file.content_type.startswith("audio/"): # type: ignore
            raise HTTPException(status_code=400, detail="File provided is not audio.")

        # Upload to Cloudinary using "video" resource type
        result = FileService.upload(file.file, resource_type="video")
        
        if result:
            # TODO: Create an Audio model and uncomment below
            # audio = Audio(url=result["url"], public_id=result["public_id"])
            # await db.create(audio)
            
            return {
                # "id": audio.id,
                "url": result["url"], 
                "public_id": result["public_id"],
                "type": "audio"
            }
            
    except Exception as e:
        raise CustomException(
            dev_message=f"Audio upload failed: {str(e)}",
            user_message="Failed to upload audio",
            status_code=500,
        )


# ==========================================
# 4. DOCUMENT UPLOAD (PDF, DOCX, TXT)
# ==========================================
# ==========================================
# 4. DOCUMENT UPLOAD (STRICT)
# ==========================================
@mediaApp.post("/document/upload")
async def upload_document(
    file: UploadFile = File(...), 
    db: DatabaseService = Depends(get_db_service)
):
    """
    Uploads raw documents. REJECTS images/videos.
    """
    try:
        # 1. DEFINE ALLOWED TYPES
        allowed_mime_types = [
            "application/pdf",                                                          # .pdf
            "text/plain",                                                               # .txt
            "application/msword",                                                       # .doc
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
            "application/vnd.ms-excel",                                                 # .xls
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",        # .xlsx
            "application/zip",                                                          # .zip
            "application/x-zip-compressed"                                              # .zip (windows sometimes sends this)
        ]

        # 2. CHECK IF IT IS AN IMAGE OR VIDEO (Fail fast)
        if file.content_type.startswith(("image/", "video/", "audio/")): # type: ignore
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type: {file.content_type}. Please use the Image or Video upload endpoints."
            )

        # 3. CHECK STRICT ALLOW LIST (Optional but recommended)
        if file.content_type not in allowed_mime_types:
             raise HTTPException(
                status_code=400, 
                detail=f"File type {file.content_type} is not supported as a document."
            )

        # 4. Upload
        result = FileService.upload(file.file, resource_type="raw")
        
        if result:
            # Document DB logic here...
            return {
                "url": result["url"], 
                "public_id": result["public_id"],
                "original_name": file.filename,
                "type": "document"
            }
            
    except Exception as e:
        # Re-raise HTTP exceptions so the frontend gets the 400 error
        if isinstance(e, HTTPException):
            raise e
            
        raise CustomException(
            dev_message=f"Document upload failed: {str(e)}",
            user_message="Failed to upload document",
            status_code=500,
        )