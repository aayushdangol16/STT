from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import whisper
import logging
import time
import os
import shutil

# Set up logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "transcriptions.log"),
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Initialize FastAPI
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/transcribe", response_class=HTMLResponse)
async def transcribe(
    request: Request,
    model_size: str = Form(...),
    file: UploadFile = File(...)
):
    # Save uploaded file to disk
    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load model and transcribe
    model = whisper.load_model(model_size)
    start_time = time.time()
    result = model.transcribe(temp_path)
    duration = time.time() - start_time

    # Clean up uploaded file
    os.remove(temp_path)

    # Log transcription with timing
    logging.info(f"MODEL: {model_size} - TIME: {duration:.2f}s - TRANSCRIPT: {result['text']}")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "transcription": result["text"]
    })
