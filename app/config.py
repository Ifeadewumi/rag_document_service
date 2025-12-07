from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Literal


class Settings(BaseSettings):
    # Database
    database_url: str
    
    # OpenRouter
    openrouter_api_key: str
    embedding_model: str = "openai/text-embedding-3-small"
    llm_model: str = "anthropic/claude-3.5-sonnet"
    
    # Vector DB
    vector_db_type: Literal["chroma", "pinecone"] = "chroma"
    chroma_path: str = "./chroma_db"
    pinecone_api_key: str = ""
    pinecone_environment: str = ""
    pinecone_index_name: str = "documents"
    
    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # RAG
    top_k_results: int = 5
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()