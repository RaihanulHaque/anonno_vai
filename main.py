# --- Imports ---
from fastapi import FastAPI, UploadFile, File
from my_models import BaseModel, AudioCNN, FaceClassifier, VideoClassifier
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from io import BytesIO
from typing import Dict
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import tempfile
import cv2
import os

# --- Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
FRAME_COUNT = 10
FRAME_SIZE = (224, 224)  # Matches ViT input size

# --- FastAPI App ---
app = FastAPI()

# --- Model Registry ---
model_registry: Dict[str, BaseModel] = {}

def register_model(name: str, model: BaseModel):
    """Register a pre-loaded model in the registry."""
    model_registry[name] = model

# --- Load and Register Models ---
face_model = FaceClassifier()
face_model.load_state_dict(torch.load("models/real_vs_fake_face_cnn_state.pth", map_location=DEVICE))
face_model.to(DEVICE)
face_model.eval()
register_model("face_classifier", face_model)

audio_model = AudioCNN()
audio_model.load_state_dict(torch.load("models/audio_deep_fake_cnn_states.pth", map_location=DEVICE))
audio_model.to(DEVICE)
audio_model.eval()
register_model("audio_classifier", audio_model)

vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=2, ignore_mismatched_sizes=True)
video_model = VideoClassifier(vit)
video_model.load_state_dict(torch.load("models/real_fake_video_classifier_vit.pth", map_location=DEVICE))
video_model.to(DEVICE)
video_model.eval()
register_model("video_classifier", video_model)

# --- Preprocessing Functions ---
def preprocess_image(file_bytes: bytes):
    """Preprocess image bytes for face classification."""
    image = Image.open(BytesIO(file_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def preprocess_audio(file_bytes: bytes, sample_rate=22050, duration=2.0):
    """Preprocess audio bytes for audio classification."""
    waveform, sr = torchaudio.load(BytesIO(file_bytes))
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
    return spectrogram.unsqueeze(0)

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

def extract_frames_from_bytes(video_bytes, frame_count=FRAME_COUNT):
    """Extract frames from video bytes without resizing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(total_frames // frame_count, 1)
        frames = []
        for i in range(frame_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    finally:
        os.remove(tmp_path)  # Clean up temporary file
    return frames if len(frames) == frame_count else None

def preprocess_video_frames(frames):
    """Preprocess video frames using ViTImageProcessor."""
    images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    inputs = processor(images=images, return_tensors="pt")
    return inputs['pixel_values'].unsqueeze(0)  # Shape: (1, seq_len, 3, 224, 224)

# --- API Endpoints ---
@app.post("/classify/image")
async def classify_face(file: UploadFile = File(...)):
    """Classify an uploaded face image as Real or Fake."""
    model = model_registry["face_classifier"]
    file_bytes = await file.read()
    input_tensor = preprocess_image(file_bytes).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        probability = output.item()
        prediction = 1 if probability > 0.5 else 0
    label = "Real" if prediction == 1 else "Fake"
    return {"prediction": label, "probability": probability}

@app.post("/classify/audio")
async def classify_audio(file: UploadFile = File(...)):
    """Classify an uploaded audio file as Real or Fake."""
    model = model_registry["audio_classifier"]
    file_bytes = await file.read()
    spectrogram = preprocess_audio(file_bytes).to(DEVICE)
    with torch.no_grad():
        output = model(spectrogram)
        probability = output.item()
        prediction = 1 if probability > 0.5 else 0
    label = "Real" if prediction == 1 else "Fake"
    return {"prediction": label, "confidence": probability}

@app.post("/classify/video")
async def classify_video(file: UploadFile = File(...)):
    """Classify an uploaded video as real or fake."""
    model = model_registry["video_classifier"]
    video_bytes = await file.read()
    frames = extract_frames_from_bytes(video_bytes)
    if frames is None:
        return {"error": "Could not extract enough frames from the video"}
    tensor = preprocess_video_frames(frames).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1).item()
        probability = torch.softmax(logits, dim=1).max().item()
    label = "real" if pred == 0 else "fake"
    return {"prediction": label, "confidence": probability}

# --- Run the Server ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
    # Access the API at http://<your-ip>:8000 from any device on the network