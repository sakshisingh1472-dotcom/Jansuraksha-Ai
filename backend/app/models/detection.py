from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from app.core.database import Base
class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True); camera_id = Column(String); zone_id = Column(String, nullable=True); object_type = Column(String); track_id = Column(Integer, nullable=True); confidence = Column(Float, nullable=True); timestamp = Column(DateTime, default=datetime.utcnow)
