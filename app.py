import streamlit as st
import os
import zipfile
import numpy as np
import librosa
import joblib
import torch
from torchvision import models, transforms
from PIL import Image

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="🐦 Bird AI", layout="wide")

# ===============================
# LOAD AUDIO MODEL
# ===============================
audio_model = joblib.load("bird_model.pkl")
le = joblib.load("label_encoder.pkl")

# ===============================
# REBUILD IMAGE MODEL FROM PARTS
# ===============================
def rebuild_model():
    if not os.path.exists("bird_image_model.pth"):

        parts = sorted([f for f in os.listdir() if f.startswith("model_part_")])

        if len(parts) > 0:
            with open("bird_image_model.pth", "wb") as outfile:
                for part in parts:
                    with open(part, "rb") as infile:
                        outfile.write(infile.read())

# run rebuild
rebuild_model()

# ===============================
# LOAD IMAGE MODEL
# ===============================
num_classes = len(le.classes_)

image_model = models.mobilenet_v2(pretrained=False)
image_model.classifier[1] = torch.nn.Linear(image_model.last_channel, num_classes)

image_model.load_state_dict(torch.load("bird_image_model.pth", map_location="cpu"))
image_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ===============================
# AUDIO FEATURE EXTRACTION
# ===============================
def extract_audio_features(file):
    y, sr = librosa.load(file, sr=22050)
    y = librosa.util.normalize(y)
    y, _ = librosa.effects.trim(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)

    return np.hstack([
        np.mean(mfcc.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(mel.T, axis=0)
    ])

# ===============================
# AUDIO PREDICTION
# ===============================
def predict_audio(file):
    features = extract_audio_features(file)
    probs = audio_model.predict_proba([features])[0]

    top3 = probs.argsort()[-3:][::-1]

    return [(le.inverse_transform([i])[0], probs[i]) for i in top3]

# ===============================
# IMAGE PREDICTION
# ===============================
def predict_image(img):
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = image_model(img)

    probs = torch.nn.functional.softmax(output[0], dim=0)

    top3 = torch.argsort(probs)[-3:].flip(0)

    return [(le.inverse_transform([i.item()])[0], probs[i].item()) for i in top3]

# ===============================
# UI
# ===============================
st.sidebar.title("🐦 Bird AI System")
mode = st.sidebar.radio("Select Mode", ["Home", "Audio", "Image"])

# ===============================
# HOME
# ===============================
if mode == "Home":
    st.title("🐦 Bird Species Classifier")
    st.write("Audio + Image AI Model")

# ===============================
# AUDIO
# ===============================
elif mode == "Audio":
    st.title("🎧 Audio Prediction")

    file = st.file_uploader("Upload audio", type=["wav", "mp3"])

    if file and st.button("Predict"):
        results = predict_audio(file)

        for label, conf in results:
            st.write(f"**{label}**")
            st.progress(float(conf))
            st.write(f"{conf*100:.2f}%")

# ===============================
# IMAGE
# ===============================
elif mode == "Image":
    st.title("🖼 Image Prediction")

    file = st.file_uploader("Upload image", type=["jpg", "png"])

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, use_column_width=True)

        if st.button("Predict"):
            results = predict_image(img)

            for label, conf in results:
                st.write(f"**{label}**")
                st.progress(float(conf))
                st.write(f"{conf*100:.2f}%")
