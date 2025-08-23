import os
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    upstage_api_key: str
    solar_pro_2_endpoint: str = "https://api.upstage.ai/v1/solar/chat/completions"
    solar_embedding_endpoint: str = "https://api.upstage.ai/v1/solar/embeddings"
    log_level: str = "INFO"
    
    qdrant_host: str
    qdrant_port: int = 6333
    qdrant_api_key: str
    qdrant_collection_name: str = "research"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings():
    return Settings()