from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64

app = FastAPI(
    title="AI Voice Detection API",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

API_KEY = "vd_api_key_98231"


class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


@app.get("/")
def root():
    return {
        "message": "FASTAPI IS RUNNING",
        "service": "voice-detection-api"
    }


@app.get("/__self_test")
def self_test():
    return {
        "status": "ok",
        "message": "API reachable and responding"
    }


@app.post("/api/voice-detection")
def detect_voice(
    request: VoiceRequest,
    x_api_key: str = Header(None)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return {
        "status": "success",
        "classification": "HUMAN",
        "confidenceScore": 0.80,
        "explanation": "Baseline test response"
    }
