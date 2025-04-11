# Anonno Vai

Welcome to the Anonno Vai repository, a comprehensive collection of deep learning models for detecting deepfakes across multiple modalities (audio, images, and videos).

## Overview

This repository provides ready-to-use notebooks and scripts for training, evaluating, and deploying deepfake detection models. The implementation focuses on Convolutional Neural Networks (CNN) and Vision Transformer (ViT) architectures to distinguish between real and AI-generated content.

## Features

- **Audio Deepfake Detection**: CNN-based models to identify synthetic audio
- **Image Deepfake Detection**: Models for detecting manipulated faces and images
- **Video Deepfake Detection**: Both CNN and ViT implementations for video deepfake analysis
- **Pre-trained Models**: Ready-to-use models trained on standard deepfake datasets

## Repository Structure

- **Notebooks**:
  - [`audio_deep_fake_cnn.ipynb`](audio_deep_fake_cnn.ipynb): Audio deepfake detection using CNN
  - [`real_fake_face_cnn.ipynb`](real_fake_face_cnn.ipynb): Face deepfake detection
  - [`real_fake_video_vit.ipynb`](real_fake_video_vit.ipynb): Video deepfake detection with Vision Transformer
  - [`real_fake_video_cnn.ipynb`](real_fake_video_cnn.ipynb): Video deepfake detection with CNN


- **Python Scripts**:
  - [`audio_test.py`](audio_test.py): Testing script for audio models
  - [`vit.py`](vit.py): Vision Transformer implementation

- **Pre-trained Models**:
  - Audio deepfake detection
  - Image classification
  - Video deepfake detection

## Download Pre-trained Models

You can download our pre-trained models from the following link:

[Download Models](https://drive.google.com/drive/folders/1nY-ylZbvnTxqJKAtBrbAeNPWSXivJn3G?usp=sharing)

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- MPS (Apple Silicon) support for MacOS
- Basic knowledge of deep learning and PyTorch
- Familiarity with Jupyter notebooks
- Basic understanding of audio and video processing
- Basic understanding of image processing
- Basic understanding of deepfake detection techniques

### Setup

1. Clone this repository to your local machine:

```bash
git clone https://github.com/RaihanulHaque/anonno_vai.git
cd anonno_vai
```

2. Create a virtual environment and install dependencies:

```bash
# Create and activate virtual environment
python -m venv env
source env/bin/activate  # On Windows, use: env\Scripts\activate

# Install required packages
pip install torch torchvision torchaudio
pip install pandas numpy matplotlib seaborn scikit-learn
pip install opencv-python fastapi uvicorn
pip install transformers
pip install librosa soundfile tqdm Pillow
```

3. Download the pre-trained models and place them in the `models` directory.

## Usage

### Running Notebooks

1. Start the Jupyter notebook server:

```bash
jupyter notebook
```

2. Open any of the provided notebooks to:
    - Train a new model
    - Evaluate the model
    - Perform inference on new data


### Testing Audio Models

To test the audio models, run the following command:

```bash
python audio_test.py
```