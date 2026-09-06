from sqlalchemy import Column, String, Integer, Float
from app.core.database import Base
class Zone(Base):
    __tablename__ = "zones"
    zone_id = Column(String, primary_key=True); name = Column(String); capacity = Column(Integer, default=100); current_people = Column(Integer, default=0); current_vehicles = Column(Integer, default=0); density_percent = Column(Float, default=0.0); risk_level = Column(String, default="LOW")
