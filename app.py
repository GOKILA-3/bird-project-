import streamlit as st
import torch
import numpy as np
from PIL import Image
import librosa
import joblib
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(page_title="Bird Classifier", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🕊️ Bird Species Classifier</h1>
    <p style='text-align: center;'>Multimodal Deep Learning (Image + Audio)</p>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL
# ==============================

model = joblib.load("model.pkl")

class_names = ["crow","sparrow","parrot","pigeon","peacock",
               "eagle","owl","kingfisher","woodpecker","duck"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# IMAGE TRANSFORM
# ==============================

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ==============================
# MODELS
# ==============================

class VisionTransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        self.vit.heads = nn.Linear(self.vit.heads.head.in_features, 128)

    def forward(self, x):
        return self.vit(x)

class AudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        return self.net(x)

class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )

    def forward(self, v, a):
        return self.fc(torch.cat((v, a), dim=1))

vit = VisionTransformerModel().to(device)
audio_model = AudioModel().to(device)
fusion = FusionModel().to(device)

vit.eval()
audio_model.eval()
fusion.eval()

# ==============================
# AUDIO FEATURE
# ==============================

def extract_audio_features(file):
    signal, sr = librosa.load(file, sr=22050)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    if len(mfcc) < 256:
        mfcc = np.pad(mfcc, (0, 256 - len(mfcc)))
    else:
        mfcc = mfcc[:256]

    return torch.tensor(mfcc).float().to(device)

# ==============================
# UI LAYOUT
# ==============================

col1, col2 = st.columns(2)

with col1:
    img_file = st.file_uploader("📸 Upload Bird Image", type=["jpg", "png"])

with col2:
    audio_file = st.file_uploader("🎵 Upload Bird Audio", type=["wav", "mp3"])

# ==============================
# PREDICTION
# ==============================

if st.button("🔍 Predict", use_container_width=True):

    if img_file and audio_file:

        colA, colB = st.columns(2)

        with colA:
            img = Image.open(img_file).convert("RGB")
            st.image(img, caption="Uploaded Image")

        with colB:
            st.audio(audio_file)

        img = image_transform(img).unsqueeze(0).to(device)
        audio_features = extract_audio_features(audio_file)

        with torch.no_grad():
            v = vit(img)
            a = audio_model(audio_features).unsqueeze(0)
            f = fusion(v, a).cpu().numpy()

        probs = model.predict_proba(f)[0]
        top3_idx = np.argsort(probs)[-3:][::-1]

        st.markdown("## 🧠 Prediction Results")

        for i in top3_idx:
            st.write(f"**{class_names[i].upper()}**")
            st.progress(float(probs[i]))

    else:
        st.warning("⚠️ Upload both image and audio!")