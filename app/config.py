from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    collection_prefix: str = "pg18_docs"
    docs_version: int = 18

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
