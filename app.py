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
<p style='text-align: center;'>Audio-based Machine Learning Prediction</p>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("model_compressed.pkl")

class_names = ["crow","sparrow","parrot","pigeon","peacock",
               "eagle","owl","kingfisher","woodpecker","duck"]

# ==============================
# REFERENCE IMAGES
# ==============================
reference_images = {
    "crow": "https://upload.wikimedia.org/wikipedia/commons/1/11/Crow_in_flight.jpg",
    "sparrow": "https://upload.wikimedia.org/wikipedia/commons/5/5e/House_sparrow04.jpg",
    "parrot": "https://upload.wikimedia.org/wikipedia/commons/0/05/Scarlet_Macaw_and_Blue-and-yellow_Macaw.jpg",
    "pigeon": "https://upload.wikimedia.org/wikipedia/commons/9/9b/Rock_Pigeon_01.jpg",
    "peacock": "https://upload.wikimedia.org/wikipedia/commons/e/e3/Peacock_Plumage.jpg",
    "eagle": "https://upload.wikimedia.org/wikipedia/commons/1/1a/Bald_Eagle_Portrait.jpg",
    "owl": "https://upload.wikimedia.org/wikipedia/commons/1/1c/Barn_Owl.jpg",
    "kingfisher": "https://upload.wikimedia.org/wikipedia/commons/5/5a/Common_Kingfisher.jpg",
    "woodpecker": "https://upload.wikimedia.org/wikipedia/commons/6/6e/Great_Spotted_Woodpecker.jpg",
    "duck": "https://upload.wikimedia.org/wikipedia/commons/7/74/Mallard2.jpg"
}

# ==============================
# IMAGE FROM URL
# ==============================
def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

# ==============================
# AUDIO FEATURE EXTRACTION
# ==============================
def extract_audio_features(file):
    signal, sr = librosa.load(file, sr=22050)

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    # Convert to 1D vector
    return mfcc

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
            img = Image.open(img_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)

        with colB:
            st.audio(audio_file)

        # ==============================
        # FEATURE PROCESSING (FIXED)
        # ==============================
        audio_features = extract_audio_features(audio_file)

        expected = model.n_features_in_

        # match feature size
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
        # BEST RESULT
        # ==============================
        best_idx = top3_idx[0]
        best = class_names[best_idx]
        best_conf = probs[best_idx]

        st.markdown("## 🎯 Final Prediction")

        if best_conf > 0.8:
            st.success(f"🔥 High Confidence: {best.upper()}")
        elif best_conf > 0.5:
            st.info(f"⚡ Medium Confidence: {best.upper()}")
        else:
            st.warning(f"⚠️ Low Confidence: {best.upper()}")

        # ==============================
        # REFERENCE IMAGE
        # ==
