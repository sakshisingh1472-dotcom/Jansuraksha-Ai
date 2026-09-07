from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.detection import Detection
from app.models.zone import Zone
from pydantic import BaseModel
from typing import List, Optional

class BatchItem(BaseModel):
    object_type: str
    zone_id: Optional[str] = None
    track_id: Optional[int] = None
    confidence: Optional[float] = None

class BatchInput(BaseModel):
    camera_id: str
    detections: List[BatchItem]

class DemoInput(BaseModel):
    camera_id: str
    zone_id: str
    people_count: int = 0
    vehicle_count: int = 0

router = APIRouter(prefix="/detections", tags=["Detections"])

@router.post("/demo")
def demo(p: DemoInput, db: Session = Depends(get_db)):
    from app.services.risk_engine import calculate_risk
    from app.services.alert_service import check_and_generate_alerts

    zone = db.query(Zone).filter(Zone.zone_id == p.zone_id).first()
    if not zone:
        zone = Zone(zone_id=p.zone_id, name=p.zone_id, capacity=100)
        db.add(zone)
        db.commit()
        db.refresh(zone)

    density = (p.people_count / zone.capacity * 100) if zone.capacity else 0
    risk = calculate_risk(density)

    zone.current_people = p.people_count
    zone.current_vehicles = p.vehicle_count
    zone.density_percent = density
    zone.risk_level = risk
    db.commit()

    check_and_generate_alerts(db, p.zone_id, p.people_count, p.vehicle_count, density, risk)

    return {"status": "ok", "zone_id": p.zone_id, "people": p.people_count, "density": density, "risk": risk}

@router.post("/batch")
def batch_input(p: BatchInput, db: Session = Depends(get_db)):
    from app.services.risk_engine import calculate_risk
    from app.services.alert_service import check_and_generate_alerts

    # Count people from batch
    people_count = sum(1 for d in p.detections if d.object_type == "person")
    vehicle_count = sum(1 for d in p.detections if d.object_type == "vehicle")

    # Group by zone if provided
    zone_id = p.detections[0].zone_id if p.detections and p.detections[0].zone_id else "zone_1"

    zone = db.query(Zone).filter(Zone.zone_id == zone_id).first()
    if not zone:
        zone = Zone(zone_id=zone_id, name=zone_id, capacity=100)
        db.add(zone)
        db.commit()
        db.refresh(zone)

    density = (people_count / zone.capacity * 100) if zone.capacity else 0
    risk = calculate_risk(density)

    zone.current_people = people_count
    zone.current_vehicles = vehicle_count
    zone.density_percent = density
    zone.risk_level = risk
    db.commit()

    check_and_generate_alerts(db, zone_id, people_count, vehicle_count, density, risk)

    return {"zone_id": zone_id, "people": people_count, "vehicles": vehicle_count, "risk": risk}

@router.post("/analyze")
def analyze_image(camera_id: str, zone_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    from app.ai_engine.crowd_counter import count_crowd_from_frame
    import cv2
    import numpy as np
    from app.services.risk_engine import calculate_risk
    from app.services.alert_service import check_and_generate_alerts

    data = file.file.read()
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    result = count_crowd_from_frame(frame)

    people = result.get("count", 0)

    zone = db.query(Zone).filter(Zone.zone_id == zone_id).first()
    if not zone:
        zone = Zone(zone_id=zone_id, name=zone_id, capacity=100)
        db.add(zone)
        db.commit()
        db.refresh(zone)

    density = (people / zone.capacity * 100) if zone.capacity else 0
    risk = calculate_risk(density)

    zone.current_people = people
    zone.density_percent = density
    zone.risk_level = risk
    db.commit()

    check_and_generate_alerts(db, zone_id, people, 0, density, risk)

    return {"zone_id": zone_id, "people_count": people, "density": density, "risk": risk, "details": result}