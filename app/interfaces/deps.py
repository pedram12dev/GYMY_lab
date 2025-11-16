from typing import Generator
from sqlalchemy.orm import Session
from app.core.config import get_settings
from app.modules.db.test_database import SessionLocal
from app.modules.db.create_db import init_db

_settings = get_settings()
init_db()

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try: yield db
    finally: db.close()

def get_settings_dep(): return _settings
