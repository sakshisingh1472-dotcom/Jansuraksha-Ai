from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.camera import Camera
from pydantic import BaseModel
from typing import Optional
class CameraCreate(BaseModel):
    camera_id: str; name: str; location: Optional[str]=None; is_active: bool=True
router = APIRouter(prefix="/cameras", tags=["Cameras"])
@router.post("/")
def create_camera(camera: CameraCreate, db: Session = Depends(get_db)):
    if db.query(Camera).filter(Camera.camera_id==camera.camera_id).first(): raise HTTPException(400,"exists")
    c=Camera(**camera.model_dump()); db.add(c); db.commit(); db.refresh(c); return c
@router.get("/")
def get_cameras(db: Session = Depends(get_db)): return db.query(Camera).all()
