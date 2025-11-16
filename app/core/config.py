from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    cors_origins: str = "*"
    database_url: str = "postgresql+psycopg://gymy:gymy_pass@localhost:5432/gymy_smart"
    login_timeout_seconds: int = 120
    cosine_login_threshold: float = 0.60
    class Config: env_file = ".env"

@lru_cache
def get_settings() -> "Settings":
    return Settings()
