import torch
import torch.nn as nn
import torchaudio.transforms as T
import numpy as np
from torchvision import models

# Base Model class to allow for future extensibility
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
    
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method")

class FaceClassifier(nn.Module):
    def __init__(self):
        super(FaceClassifier, self).__init__()
        # Load pretrained DenseNet121
        self.backbone = models.densenet121(pretrained=True)
        
        # Get the number of features from the last layer
        num_features = self.backbone.classifier.in_features
        
        # Replace the classifier
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
        self.fc1 = nn.Linear(128 * 16 * 10, 256)  # Adjust based on your input size
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