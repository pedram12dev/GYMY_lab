# lab/db/models.py
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String

from lab.db.test_database import Base


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    profile_id = Column(Integer, nullable=False)
    embedding = Column(String, nullable=False)
    confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
