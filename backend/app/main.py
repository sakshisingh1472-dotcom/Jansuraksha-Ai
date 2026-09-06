from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.database import Base, engine
from app.api import health, cameras, zones, detections, alerts, dashboard
Base.metadata.create_all(bind=engine)
app = FastAPI(title="Jansuraksha AI - Abhinav Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(health.router)
app.include_router(cameras.router)
app.include_router(zones.router)
app.include_router(detections.router)
app.include_router(alerts.router)
app.include_router(dashboard.router)
@app.get("/")
def root(): return {"project": "Jansuraksha AI", "owner": "Abhinav", "status": "online"}
