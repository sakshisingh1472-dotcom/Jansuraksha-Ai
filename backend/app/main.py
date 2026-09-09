from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import detections

app = FastAPI(title="Jansuraksha AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detections.router, prefix="/api")

@app.get("/")
def root():
    return {"status":"connected","backend":"http://127.0.0.1:8000"}