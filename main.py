from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {
        "message": "FASTAPI IS RUNNING",
        "service": "voice-detection-api"
    }
