from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Literal
from dotenv import load_dotenv

# Explicitly load the .env file
load_dotenv()


class Settings(BaseSettings):
    # Database
    database_url: str
    
    # OpenRouter
    openrouter_api_key: str
    embedding_model: str = "sentence-transformers/all-minilm-l6-v2"
    llm_model: str = "nousresearch/hermes-3-llama-3.1-405b"
    
    # Vector DB
    vector_db_type: Literal["chroma", "pinecone"] = "chroma"
    chroma_path: str = "./chroma_db"
    pinecone_api_key: str = ""
    pinecone_environment: str = ""
    pinecone_index_name: str = "documents"
    
    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_upload_size_bytes: int = 5 * 1024 * 1024  # 5 MB
    
    # RAG
    top_k_results: int = 5
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()
