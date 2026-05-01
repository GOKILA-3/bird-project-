import streamlit as st
import numpy as np
from PIL import Image
import librosa
import joblib
from io import BytesIO
import requests

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Bird Classifier", layout="wide")

st.markdown("""
<h1 style='text-align: center; color: #4CAF50;'>🕊️ Bird Species Classifier</h1>
<p style='text-align: center;'>Upload Image + Audio → Get Prediction</p>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("model_compressed.pkl")

class_names = ["crow","sparrow","parrot","pigeon","peacock",
               "eagle","owl","kingfisher","woodpecker","duck"]

# ==============================
# AUDIO FEATURE EXTRACTION
# ==============================
def extract_audio_features(file):
    signal, sr = librosa.load(file, sr=22050)

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    if len(mfcc) < 256:
        mfcc = np.pad(mfcc, (0, 256 - len(mfcc)))
    else:
        mfcc = mfcc[:256]

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
# PREDICTION
# ==============================
if st.button("🔍 Predict", use_container_width=True):

    if img_file and audio_file:

        colA, colB = st.columns(2)

        with colA:
            img = Image.open(img_file)
            st.image(img, caption="Uploaded Image", use_container_width=True)

        with colB:
            st.audio(audio_file)

        # Extract features
        audio_features = extract_audio_features(audio_file)

        # Fake image features (since no deep learning model here)
        image_features = np.random.rand(256)

        final_features = (image_features + audio_features) / 2
        final_features = final_features.reshape(1, -1)

        # Prediction
        probs = model.predict_proba(final_features)[0]
        top3_idx = np.argsort(probs)[-3:][::-1]

        # ==============================
        # RESULTS UI
        # ==============================
        st.markdown("## 🧠 Prediction Results")

        for i in top3_idx:
            st.write(f"### {class_names[i].upper()}")
            st.progress(float(probs[i]))
            st.caption(f"Confidence: {probs[i]*100:.2f}%")

        # BEST RESULT
        best = class_names[top3_idx[0]]
        best_conf = probs[top3_idx[0]]

        st.markdown("## 🎯 Final Prediction")

        if best_conf > 0.8:
            st.success(f"🔥 High Confidence: {best.upper()}")
        elif best_conf > 0.5:
            st.info(f"⚡ Medium Confidence: {best.upper()}")
        else:
            st.warning(f"⚠️ Low Confidence: {best.upper()}")

        # ==============================
        # DOWNLOAD REPORT
        # ==============================
        report = f"""
Bird Prediction Report

Prediction: {best}
Confidence: {best_conf*100:.2f}%
"""

        st.download_button(
            "📄 Download Report",
            report,
            file_name="prediction.txt"
        )

    else:
        st.warning("⚠️ Please upload both image and audio!")
