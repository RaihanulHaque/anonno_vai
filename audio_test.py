import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np

# Define the model class (must match the one used during training)
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 16 * 10, 256)  # Corrected size from training
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# Function to preprocess MP3 audio
def preprocess_audio(mp3_path, sample_rate=22050, duration=2.0):
    # Load the MP3 file
    waveform, sr = torchaudio.load(mp3_path)
    
    # Resample if necessary
    if sr != sample_rate:
        resampler = T.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Ensure mono (single channel)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Trim or pad to target duration (2 seconds)
    target_length = int(sample_rate * duration)
    if waveform.size(1) < target_length:
        waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.size(1)))
    else:
        waveform = waveform[:, :target_length]
    
    # Define the transformations used during training
    mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=128, n_fft=1024, hop_length=512)
    db_transform = T.AmplitudeToDB()
    
    # Apply transformations sequentially
    spectrogram = mel_transform(waveform)  # Shape: [1, 128, 87]
    spectrogram = db_transform(spectrogram)  # Convert to dB scale
    spectrogram = spectrogram.unsqueeze(0)  # Add batch dimension: [1, 1, 128, 87]
    
    return spectrogram

# Function to load model and run inference
def run_inference(mp3_path, model_path='best_model.pth', device='cpu'):
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Instantiate the model
    model = AudioCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Preprocess the audio
    spectrogram = preprocess_audio(mp3_path)
    spectrogram = spectrogram.to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(spectrogram)
        probability = output.item()  # Scalar value between 0 and 1
        prediction = 1 if probability > 0.5 else 0  # Binary classification
    
    return prediction, probability

# Main execution
if __name__ == "__main__":
    # Path to your MP3 file
    mp3_file = "test_data/tts.mp3"  # Replace with actual path
    
    # Path to the saved model
    model_file = "models/audio_deep_fake_cnn_states.pth"
    
    # Run inference
    prediction, probability = run_inference(mp3_file, model_file, device='cuda')
    
    # Interpret results
    label = "Real" if prediction == 1 else "Fake"
    print(f"Prediction: {label}")
    print(f"Probability of being Real: {probability:.4f}")