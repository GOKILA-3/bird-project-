import streamlit as st
import numpy as np
import librosa
import joblib

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="🐦 Bird Audio AI", layout="wide")

# ===============================
# LOAD MODEL
# ===============================
audio_model = joblib.load("bird_model.pkl")
le = joblib.load("label_encoder.pkl")

# ===============================
# FEATURE EXTRACTION
# ===============================
def extract_features(file):
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
# PREDICTION
# ===============================
def predict_audio(file):
    features = extract_features(file)
    probs = audio_model.predict_proba([features])[0]

    top3 = probs.argsort()[-3:][::-1]

    return [(le.inverse_transform([i])[0], float(probs[i])) for i in top3]

# ===============================
# UI
# ===============================
st.title("🐦 Bird Audio Species Prediction")

file = st.file_uploader("Upload bird sound (wav/mp3)", type=["wav", "mp3"])

if file and st.button("Predict"):
    results = predict_audio(file)

    for label, conf in results:
        st.write(f"**{label}**")
        st.progress(conf)
        st.write(f"{conf*100:.2f}%")
