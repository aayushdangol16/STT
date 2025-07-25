from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import os
import time
import torchaudio
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

device = torch.device("cpu")

models_dir = "models"
models = {}
processors = {}

for model_name in os.listdir(models_dir):
    model_path = os.path.join(models_dir, model_name)
    if os.path.isdir(model_path):
        try:
            processors[model_name] = Wav2Vec2Processor.from_pretrained(model_path)
            models[model_name] = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
            print(f"Loaded model: {model_name}")
        except Exception as e:
            print(f"Skipping {model_name}, error loading: {e}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_names": list(models.keys())
    })

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...), model_name: str = Form(...)):
    if model_name not in models:
        return JSONResponse(content={"error": "Model not found"}, status_code=400)

    file_path = "temp.wav"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    processor = processors[model_name]
    model = models[model_name]

    input_values = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values.to(device)

    start_time = time.time()
    with torch.no_grad():
        logits = model(input_values).logits
    elapsed_time = time.time() - start_time

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    os.remove(file_path)

    return {
        "transcription": transcription.strip(),
        "time_taken": round(elapsed_time, 3)
    }
