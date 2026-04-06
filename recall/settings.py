import pathlib

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    db_path: pathlib.Path = pathlib.Path.home() / ".local/share/recall/memory.db"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    debug: bool = True
    max_emails: int | None = None


settings = Settings()
