import streamlit as st
import os
import numpy as np
import librosa
import joblib
import onnxruntime as ort
from PIL import Image
import gdown

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="🐦 Bird AI System", layout="wide")

# ===============================
# DOWNLOAD ONNX FROM GOOGLE DRIVE
# ===============================
def download_model():
    file_id = "19HuHFO-avUg2K9kPL_CEkzPIL7A25rYr"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "bird_image_model.onnx"

    if not os.path.exists(output):
        st.info("⬇️ Downloading image model from Google Drive...")
        gdown.download(url, output, quiet=False, fuzzy=True)

    # safety check
    if not os.path.exists(output):
        raise Exception("❌ ONNX file not found after download")

    if os.path.getsize(output) < 1_000_000:
        raise Exception("❌ Corrupted ONNX file (too small)")

download_model()

# ===============================
# LOAD AUDIO MODEL
# ===============================
audio_model = joblib.load("bird_model.pkl")
le = joblib.load("label_encoder.pkl")

# ===============================
# LOAD ONNX IMAGE MODEL (SAFE)
# ===============================
def load_onnx():
    model_path = "bird_image_model.onnx"

    session = ort.InferenceSession(model_path)

    return session

session = load_onnx()

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

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

    return [(le.inverse_transform([i])[0], float(probs[i])) for i in top3]

# ===============================
# IMAGE PREPROCESS (ONNX)
# ===============================
def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0

    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img

# ===============================
# IMAGE PREDICTION
# ===============================
def predict_image(img):
    input_data = preprocess(img)

    outputs = session.run([output_name], {input_name: input_data})
    probs = outputs[0][0]

    top3 = np.argsort(probs)[-3:][::-1]

    return [(le.inverse_transform([i])[0], float(probs[i])) for i in top3]

# ===============================
# UI
# ===============================
st.sidebar.title("🐦 Bird AI System")
mode = st.sidebar.radio("Select Mode", ["Home", "Audio", "Image"])

# ===============================
# HOME
# ===============================
if mode == "Home":
    st.title("🐦 Bird Species Prediction System")
    st.write("Audio + Image AI Model (Fully Fixed + Cloud Ready)")

# ===============================
# AUDIO
# ===============================
elif mode == "Audio":
    st.title("🎧 Audio Prediction")

    file = st.file_uploader("Upload audio", type=["wav", "mp3"])

    if file and st.button("Predict Audio"):
        results = predict_audio(file)

        for label, conf in results:
            st.write(f"**{label}**")
            st.progress(conf)
            st.write(f"{conf*100:.2f}%")

# ===============================
# IMAGE
# ===============================
elif mode == "Image":
    st.title("🖼 Image Prediction")

    file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict Image"):
            results = predict_image(img)

            for label, conf in results:
                st.write(f"**{label}**")
                st.progress(conf)
                st.write(f"{conf*100:.2f}%")
