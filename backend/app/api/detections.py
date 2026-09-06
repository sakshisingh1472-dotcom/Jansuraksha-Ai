from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.detection import Detection
from app.models.zone import Zone
from pydantic import BaseModel
from typing import List, Optional
class BatchItem(BaseModel):
    object_type: str; zone_id: Optional[str]=None; track_id: Optional[int]=None; confidence: Optional[float]=None
class BatchInput(BaseModel):
    camera_id: str; detections: List[BatchItem]
class DemoInput(BaseModel):
    camera_id: str; zone_id: str; people_count: int=0; vehicle_count: int=0
router = APIRouter(prefix="/detections", tags=["Detections"])
@router.post("/demo")
def demo(p: DemoInput, db: Session = Depends(get_db)):
    from app.services.risk_engine import calculate_risk
    from app.services.alert_service import check_and_generate_alerts
    zone=db.query(Zone).filter(Zone.zone_id==p.zone_id).first()
    if not zone:
        zone=Zone(zone_id=p.zone_id, name=p.zone_id, capacity=100); db.add(zone); db.commit(); db.refresh(zone)
    density=(p.people_count/zone.capacity*100) if zone.capacity else 0
    risk=calculate_risk(density)
    zone.current_people=p.people_count; zone.current_vehicles=p.vehicle_count; zone.density_percent=density; zone.risk_level=risk
    check_and_generate_alerts(db, p.zone_id, p.people_count, p.vehicle_count, density, risk)
    db.commit()
    return {"zone_id": p.zone_id, "density_percent": density, "risk_level": risk}
@router.post("/batch")
def batch(b: BatchInput, db: Session = Depends(get_db)):
    for item in b.detections:
        db.add(Detection(camera_id=b.camera_id, zone_id=item.zone_id, object_type=item.object_type, track_id=item.track_id, confidence=item.confidence))
    db.commit(); return {"status":"accepted","count":len(b.detections)}
@router.get("/")
def list_det(db: Session = Depends(get_db)): return db.query(Detection).order_by(Detection.id.desc()).limit(100).all()
