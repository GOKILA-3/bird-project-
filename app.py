import streamlit as st
import numpy as np
from PIL import Image
import librosa
import joblib
from io import BytesIO
import requests

# ==============================
# PAGE
# ==============================
st.set_page_config(page_title="Bird Classifier", layout="wide")

st.markdown("""
<h1 style='text-align: center; color: #4CAF50;'>🕊️ Bird Species Classifier</h1>
<p style='text-align: center;'>Audio-based ML Prediction (Smart UI)</p>
""", unsafe_allow_html=True)

# ==============================
# MODEL
# ==============================
model = joblib.load("model_compressed.pkl")

class_names = ["crow","sparrow","parrot","pigeon","peacock",
               "eagle","owl","kingfisher","woodpecker","duck"]

# ==============================
# AUDIO FEATURES
# ==============================
def extract_audio_features(file):
    signal, sr = librosa.load(file, sr=22050)

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    return mfcc

# ==============================
# UI INPUT
# ==============================
col1, col2 = st.columns(2)

with col1:
    img_file = st.file_uploader("📸 Upload Bird Image", type=["jpg", "png"])

with col2:
    audio_file = st.file_uploader("🎵 Upload Bird Audio", type=["wav", "mp3"])

# ==============================
# PREDICT
# ==============================
if st.button("🔍 Predict", use_container_width=True):

    if img_file and audio_file:

        st.markdown("## 📥 Inputs")

        colA, colB = st.columns([1,2])  # 👈 smaller image

        with colA:
            img = Image.open(img_file)
            st.image(img, caption="Uploaded Image", width=150)  # 👈 SMALL

        with colB:
            st.audio(audio_file)

        # ==============================
        # PROCESS AUDIO
        # ==============================
        audio_features = extract_audio_features(audio_file)

        expected = model.n_features_in_

        if len(audio_features) < expected:
            audio_features = np.pad(audio_features, (0, expected - len(audio_features)))
        else:
            audio_features = audio_features[:expected]

        final_features = audio_features.reshape(1, -1)

        # ==============================
        # PREDICTION
        # ==============================
        probs = model.predict_proba(final_features)[0]
        top3_idx = np.argsort(probs)[-3:][::-1]

        st.markdown("## 🧠 Prediction Results")

        for i in top3_idx:
            st.write(f"### {class_names[i].upper()}")
            st.progress(float(probs[i]))
            st.caption(f"Confidence: {probs[i]*100:.2f}%")

        # ==============================
        # FINAL RESULT
        # ==============================
        best_idx = top3_idx[0]
        best = class_names[best_idx]
        best_conf = probs[best_idx]

        st.markdown("## 🎯 Final Prediction")

        if best_conf > 0.75:
            st.success(f"✅ Detected: {best.upper()} (High Confidence)")
        elif best_conf > 0.5:
            st.info(f"⚡ Possibly: {best.upper()} (Medium Confidence)")
        else:
            st.warning(f"⚠️ Uncertain Result: {best.upper()}")

        # ==============================
        # SMART MESSAGE
        # ==============================
        st.markdown("## 💡 Insight")

        st.info(
            "Prediction is mainly based on AUDIO. "
            "If image and audio are different, audio result will dominate."
        )

    else:
        st.warning("⚠️ Upload both image and audio!")
