import streamlit as st
import numpy as np
from PIL import Image
import joblib

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Bird Classifier", layout="centered")

st.title("🕊️ Bird Species Classifier")

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("model_compressed.pkl")

class_names = [
    "crow", "sparrow", "parrot", "pigeon", "peacock",
    "eagle", "owl", "woodpecker", "duck"
]

# ==============================
# AUTO FEATURE MATCH (IMPORTANT)
# ==============================
EXPECTED_FEATURES = model.n_features_in_

def extract_features(img):
    img = img.resize((64, 64))
    arr = np.array(img) / 255.0
    flat = arr.flatten()

    # 🔥 MATCH MODEL INPUT SIZE
    if len(flat) > EXPECTED_FEATURES:
        flat = flat[:EXPECTED_FEATURES]
    else:
        flat = np.pad(flat, (0, EXPECTED_FEATURES - len(flat)))

    return flat

# ==============================
# UI
# ==============================
uploaded_file = st.file_uploader("📸 Upload Bird Image", type=["jpg", "png"])

if st.button("🔍 Predict", key="predict_btn"):

    if uploaded_file:

        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, width=220)

        features = extract_features(img).reshape(1, -1)

        probs = model.predict_proba(features)[0]
        best = np.argmax(probs)

        st.success(f"Prediction: {class_names[best]}")
        st.info(f"Confidence: {probs[best]*100:.2f}%")

    else:
        st.warning("Upload image")
