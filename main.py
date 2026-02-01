from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import io
import numpy as np
import librosa

# -------------------------------------------------
# APP CONFIG
# -------------------------------------------------
app = FastAPI(
    title="AI Voice Detection API",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

API_KEY = "vd_api_key_98231"

SUPPORTED_LANGUAGES = [
    "Tamil",
    "English",
    "Hindi",
    "Malayalam",
    "Telugu"
]

# -------------------------------------------------
# REQUEST MODEL
# -------------------------------------------------
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# -------------------------------------------------
# ROOT ENDPOINTS
# -------------------------------------------------
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

# -------------------------------------------------
# FEATURE EXTRACTION
# -------------------------------------------------
def extract_features(audio_bytes: bytes):
    try:
        audio_stream = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_stream, sr=None)

        if y is None or len(y) == 0:
            return None

        # Feature 1: MFCC variance
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = float(np.var(mfcc))

        # Feature 2: Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = float(np.mean(zcr))

        # Feature 3: Spectral Centroid variance
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_var = float(np.var(centroid))

        return {
            "mfcc_var": mfcc_var,
            "zcr_mean": zcr_mean,
            "centroid_var": centroid_var
        }

    except Exception:
        return None

# -------------------------------------------------
# CLASSIFICATION LOGIC
# -------------------------------------------------
def classify_voice(features):
    if features is None:
        return "HUMAN", 0.50, "Audio could not be confidently analyzed"

    ai_score = 0.0
    reasons = []

    # MFCC variance check
    if features["mfcc_var"] < 20:
        ai_score += 0.4
        reasons.append("Low MFCC variance")

    # Zero Crossing Rate check
    if features["zcr_mean"] < 0.05:
        ai_score += 0.3
        reasons.append("Low zero-crossing rate")

    # Spectral Centroid variance check
    if features["centroid_var"] < 150:
        ai_score += 0.3
        reasons.append("Stable spectral centroid")

    # Final classification
    if ai_score >= 0.6:
        classification = "AI_GENERATED"
    else:
        classification = "HUMAN"

    confidence = round(min(0.95, max(0.55, ai_score)), 2)

    explanation = (
        ", ".join(reasons)
        if reasons
        else "Natural speech variations detected"
    )

    return classification, confidence, explanation

# -------------------------------------------------
# MAIN API ENDPOINT
# -------------------------------------------------
@app.post("/api/voice-detection")
def detect_voice(
    request: VoiceRequest,
    x_api_key: str = Header(None)
):
    # API KEY VALIDATION
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # LANGUAGE VALIDATION
    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    # FORMAT VALIDATION
    if request.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only mp3 format supported")

    # BASE64 DECODE
    try:
        audio_bytes = base64.b64decode(request.audioBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    # FEATURE EXTRACTION
    features = extract_features(audio_bytes)

    # CLASSIFICATION
    classification, confidence, explanation = classify_voice(features)

    # RESPONSE
    return {
        "status": "success",
        "language": request.language,
        "classification": classification,
        "confidenceScore": confidence,
        "explanation": explanation
    }
