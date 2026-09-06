from sqlalchemy import Column, Integer, String, Boolean, DateTime
from datetime import datetime
from app.core.database import Base
class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True); alert_type = Column(String); zone_id = Column(String); risk_level = Column(String); message = Column(String); acknowledged = Column(Boolean, default=False); created_at = Column(DateTime, default=datetime.utcnow)
