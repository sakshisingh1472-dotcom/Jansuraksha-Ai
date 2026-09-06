from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.zone import Zone
from pydantic import BaseModel
class ZoneCreate(BaseModel):
    zone_id: str; name: str; capacity: int=100
router = APIRouter(prefix="/zones", tags=["Zones"])
@router.post("/")
def create_zone(zone: ZoneCreate, db: Session = Depends(get_db)):
    z=Zone(zone_id=zone.zone_id, name=zone.name, capacity=zone.capacity); db.add(z); db.commit(); db.refresh(z); return z
@router.get("/")
def get_zones(db: Session = Depends(get_db)): return db.query(Zone).all()
