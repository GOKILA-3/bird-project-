import streamlit as st
import numpy as np
from PIL import Image
import librosa
import joblib

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Bird Classifier", layout="wide")

st.markdown(
    "<h1 style='text-align:center;color:#4CAF50;'>🕊️ Bird Species Classifier</h1>",
    unsafe_allow_html=True
)

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("model_compressed.pkl")

class_names = ["crow","sparrow","parrot","pigeon","peacock",
               "eagle","owl","woodpecker","duck"]

# ==============================
# AUDIO FEATURE
# ==============================
def extract_audio_features(file):
    signal, sr = librosa.load(file, sr=22050)

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    # match training shape
    if len(mfcc) < 256:
        mfcc = np.pad(mfcc, (0, 256 - len(mfcc)))
    else:
        mfcc = mfcc[:256]

    return mfcc

# ==============================
# SIMPLE IMAGE FEATURE (COLOR)
# ==============================
def extract_image_features(img):
    img = img.resize((64, 64))
    arr = np.array(img)

    # simple RGB mean features
    features = np.mean(arr, axis=(0,1))  # 3 values

    # expand to match 256 size
    features = np.pad(features, (0, 256 - len(features)))

    return features

# ==============================
# UI
# ==============================
col1, col2 = st.columns(2)

with col1:
    img_file = st.file_uploader("📸 Upload Bird Image", type=["jpg","png"])

with col2:
    audio_file = st.file_uploader("🎵 Upload Bird Audio", type=["wav","mp3"])

# ==============================
# PREDICT
# ==============================
if st.button("🔍 Predict", use_container_width=True):

    if img_file or audio_file:

        features_list = []

        colA, colB = st.columns(2)

        # IMAGE
        if img_file:
            img = Image.open(img_file).convert("RGB")
            colA.image(img, caption="Uploaded Image", width=200)

            img_features = extract_image_features(img)
            features_list.append(img_features)

        # AUDIO
        if audio_file:
            colB.audio(audio_file)

            audio_features = extract_audio_features(audio_file)
            features_list.append(audio_features)

        # ==============================
        # FINAL FEATURE (SMART FUSION)
        # ==============================
        final_features = np.mean(features_list, axis=0)
        final_features = final_features.reshape(1, -1)

        # ==============================
        # PREDICTION
        # ==============================
        probs = model.predict_proba(final_features)[0]
        best_idx = np.argmax(probs)

        st.markdown("## 🎯 Prediction Result")

        st.success(f"**{class_names[best_idx].upper()}**")

        st.progress(float(probs[best_idx]))
        st.write(f"Confidence: {probs[best_idx]*100:.2f}%")

    else:
        st.warning("⚠️ Upload at least image or audio")
