from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.core.database import get_db
from app.models.zone import Zone
from app.models.alert import Alert
router = APIRouter(prefix="/dashboard", tags=["Dashboard"])
@router.get("/")
def get_dashboard(db: Session = Depends(get_db)):
    tp=db.query(Zone).with_entities(func.sum(Zone.current_people)).scalar() or 0
    tv=db.query(Zone).with_entities(func.sum(Zone.current_vehicles)).scalar() or 0
    aa=db.query(Alert).filter(Alert.acknowledged==False).count()
    return {"system_status":"ONLINE","total_people":tp,"total_vehicles":tv,"active_alerts":aa,"zones":db.query(Zone).all(),"alerts":db.query(Alert).filter(Alert.acknowledged==False).order_by(Alert.created_at.desc()).limit(10).all()}
