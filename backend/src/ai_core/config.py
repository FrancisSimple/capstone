import os
from dotenv import load_dotenv


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

if not GROQ_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing from .env file")