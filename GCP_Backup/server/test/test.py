from fastapi import FastAPI
import requests

app = FastAPI()

# Ngrok endpoint of the Raspberry Pi
RPI_ENDPOINT = "https://humane-marmot-entirely.ngrok-free.app/get_audio_transcription"

@app.get("/")
def root():
    return {"message": "GCP Server is running"}

@app.get("/test_rpi")
def test_rpi():
    try:
        response = requests.get(RPI_ENDPOINT, timeout=10)
        response.raise_for_status()
        return {
            "status": "success",
            "data": response.json()
        }
    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "error": str(e)
        }

