import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from . import config

# Global variable to hold the database instance
_vector_db = None

def _initialize_db():
    """
    Internal function to create the database connection if it doesn't exist.
    """
    print("Initializing Vector Database...")
    
    # 1. Setup Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2. Setup ChromaDB
    # We explicitly verify the path exists to avoid silent errors
    if not os.path.exists(config.VECTOR_DB_PATH):
        os.makedirs(config.VECTOR_DB_PATH)

    db = Chroma(
        persist_directory=config.VECTOR_DB_PATH,
        embedding_function=embeddings,
        collection_name="my_knowledge_base"
    )
    print(f"Vector Database initialized at: {config.VECTOR_DB_PATH}")
    return db

def get_vector_store():
    """
    Singleton Accessor: Ensures we always return a valid database object.
    """
    global _vector_db
    if _vector_db is None:
        _vector_db = _initialize_db()
    return _vector_db

def add_text_to_base(texts: list[str]):
    """
    Adds text to the database.
    """
    db = get_vector_store() # Use the getter to ensure it's initialized
    print("Adding documents to vector store...")
    db.add_texts(texts)
    print("Done!")