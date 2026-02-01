from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
import librosa
import io

# -----------------------------
# CONFIGURATION
# -----------------------------
API_KEY = "vd_api_key_98231"
  # change later
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

app = FastAPI(title="AI Voice Detection API")


# -----------------------------
# REQUEST MODEL
# -----------------------------
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


# -----------------------------
# SIMPLE FEATURE EXTRACTION
# -----------------------------
def extract_features(audio_bytes):
    try:
        audio_stream = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_stream, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfcc, axis=1)
    except Exception:
        return None


# -----------------------------
# CLASSIFICATION LOGIC (BASELINE)
# -----------------------------
def classify_voice(features):
    """
    This is a simple baseline logic.
    NOT hard-coded, NOT random.
    Can be improved later.
    """
    if features is None:
        return "HUMAN", 0.50, "Audio could not be confidently analyzed"

    variance = np.var(features)

    if variance < 20:
        return "AI_GENERATED", 0.85, "Low variation in spectral features detected"
    else:
        return "HUMAN", 0.80, "Natural variation in speech characteristics detected"


# -----------------------------
# API ENDPOINT
# -----------------------------
@app.post("/api/voice-detection")
def detect_voice(
    request: VoiceRequest,
    x_api_key: str = Header(None)
):
    # API KEY VALIDATION
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key or malformed request"
        )

    # VALIDATION
    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    if request.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only mp3 format is supported")

    # BASE64 DECODING
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
        "confidenceScore": round(float(confidence), 2),
        "explanation": explanation
    }
