from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.alert import Alert
router = APIRouter(prefix="/alerts", tags=["Alerts"])
@router.get("/")
def get_alerts(db: Session = Depends(get_db)): return db.query(Alert).order_by(Alert.created_at.desc()).limit(50).all()
