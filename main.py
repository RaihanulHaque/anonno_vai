from fastapi import FastAPI, UploadFile, File
from my_models import BaseModel, AudioCNN
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np
from io import BytesIO
from typing import Dict

app = FastAPI()

# Model registry for adding new models
model_registry: Dict[str, BaseModel] = {}

def register_model(name: str, model: BaseModel, model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    model_registry[name] = model

# Load initial models
audio_model = "audio_fake_classifier"
register_model(audio_model, AudioCNN(), "models/audio_deep_fake_cnn_states.pth")

# Function to preprocess audio
def preprocess_audio(file: bytes, sample_rate=22050, duration=2.0):
    waveform, sr = torchaudio.load(BytesIO(file))
    if sr != sample_rate:
        resampler = T.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    target_length = int(sample_rate * duration)
    if waveform.size(1) < target_length:
        waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.size(1)))
    else:
        waveform = waveform[:, :target_length]
    mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=128, n_fft=1024, hop_length=512)
    db_transform = T.AmplitudeToDB()
    spectrogram = mel_transform(waveform)
    spectrogram = db_transform(spectrogram)
    spectrogram = spectrogram.unsqueeze(0)
    return spectrogram

# Generic Prediction Route
@app.post("/classify/audio")
async def predict(file: UploadFile = File(...)):
    
    model = model_registry[audio_model]
    device = next(model.parameters()).device
    
    audio_bytes = await file.read()
    spectrogram = preprocess_audio(audio_bytes)
    spectrogram = spectrogram.to(device)
    
    with torch.no_grad():
        output = model(spectrogram)
        probability = output.item()
        prediction = 1 if probability > 0.5 else 0
    
    label = "Real" if prediction == 1 else "Fake"
    return {"prediction": label, "confidence": probability}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
# To run the server, use the command:
# uvicorn main:app --reload
# You can test the API using a tool like Postman or curl.
# Make sure to have the models saved in the correct path.
# The model loading and preprocessing functions are designed to be reusable for future models.