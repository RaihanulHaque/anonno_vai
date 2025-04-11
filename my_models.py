import torch
import torch.nn as nn
import torchaudio.transforms as T
from torchvision import models
from transformers import ViTForImageClassification

# Base Model class for consistency across all models
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
    
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method")

class FaceClassifier(BaseModel):
    def __init__(self):
        super(FaceClassifier, self).__init__()
        self.backbone = models.densenet121(pretrained=True)
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.backbone(x)

class AudioCNN(BaseModel):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 16 * 10, 256)  # Adjust based on input size
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

class VideoClassifier(BaseModel):
    def __init__(self, vit_model):
        super(VideoClassifier, self).__init__()
        self.vit = vit_model.vit  # Use ViT backbone only
        self.lstm = nn.LSTM(input_size=768, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        x = x.view(batch_size * seq_len, C, H, W)
        x = self.vit(x).last_hidden_state[:, 0, :]
        x = x.view(batch_size, seq_len, 768)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x