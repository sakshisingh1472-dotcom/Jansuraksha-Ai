from sqlalchemy import Column, String, Boolean, DateTime
from datetime import datetime
from app.core.database import Base
class Camera(Base):
    __tablename__ = "cameras"
    camera_id = Column(String, primary_key=True); name = Column(String); location = Column(String, nullable=True); is_active = Column(Boolean, default=True); last_seen = Column(DateTime, default=datetime.utcnow)
