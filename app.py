import streamlit as st
import numpy as np
from PIL import Image
import joblib

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Bird Species Classifier",
    layout="centered"
)

# ==============================
# TITLE
# ==============================
st.markdown("<h1 style='text-align: center;'>🕊️ Bird Species Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a bird image to predict the species</p>", unsafe_allow_html=True)

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("model_compressed.pkl")

class_names = [
    "crow", "sparrow", "parrot", "pigeon", "peacock",
    "eagle", "owl", "woodpecker", "duck"
]

# ==============================
# IMAGE FEATURE FUNCTION (IMPORTANT)
# ==============================
def extract_image_features(img):
    img = img.resize((64, 64))          # must match training
    img = np.array(img) / 255.0         # normalize
    return img.flatten()                # flatten

# ==============================
# FILE UPLOAD
# ==============================
uploaded_file = st.file_uploader(
    "📸 Upload Bird Image",
    type=["jpg", "jpeg", "png"]
)

# ==============================
# PREDICT BUTTON
# ==============================
if st.button("🔍 Predict", key="predict_btn"):

    if uploaded_file is not None:

        # Load image
        image = Image.open(uploaded_file).convert("RGB")

        # Show image small
        st.image(image, caption="Uploaded Image", width=220)

        # Extract features
        features = extract_image_features(image)
        features = features.reshape(1, -1)

        # Prediction
        probs = model.predict_proba(features)[0]
        best_index = np.argmax(probs)

        predicted_class = class_names[best_index]
        confidence = probs[best_index] * 100

        # ==============================
        # OUTPUT
        # ==============================
        st.markdown("### 🎯 Prediction Result")

        st.success(f"**{predicted_class.upper()}**")
        st.info(f"Confidence: {confidence:.2f}%")

        # Optional: show top 3
        st.markdown("### 📊 Top Predictions")
        top3 = np.argsort(probs)[-3:][::-1]

        for i in top3:
            st.write(f"{class_names[i]} → {probs[i]*100:.2f}%")

    else:
        st.warning("⚠️ Please upload an image first")
