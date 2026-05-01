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
               "eagle","owl","kingfisher","woodpecker","duck"]

# auto-detect feature size
EXPECTED_FEATURES = model.n_features_in_

# ==============================
# IMAGE FEATURE (IMPORTANT)
# ==============================
def extract_image_features(img):
    img = img.resize((64, 64))
    arr = np.array(img)

    # RGB mean (3 values)
    features = np.mean(arr, axis=(0, 1))

    # match model input size
    if len(features) < EXPECTED_FEATURES:
        features = np.pad(features, (0, EXPECTED_FEATURES - len(features)))
    else:
        features = features[:EXPECTED_FEATURES]

    return features

# ==============================
# AUDIO FEATURE (OPTIONAL)
# ==============================
def extract_audio_features(file):
    signal, sr = librosa.load(file, sr=22050)

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    # match model input size
    if len(mfcc) < EXPECTED_FEATURES:
        mfcc = np.pad(mfcc, (0, EXPECTED_FEATURES - len(mfcc)))
    else:
        mfcc = mfcc[:EXPECTED_FEATURES]

    return mfcc

# ==============================
# UI
# ==============================
col1, col2 = st.columns(2)

with col1:
    img_file = st.file_uploader("📸 Upload Bird Image", type=["jpg", "png"])

with col2:
    audio_file = st.file_uploader("🎵 Upload Bird Audio (optional)", type=["wav", "mp3"])

# ==============================
# SINGLE BUTTON (NO DUPLICATE)
# ==============================
predict_clicked = st.button(
    "🔍 Predict",
    use_container_width=True,
    key="predict_btn"
)

# ==============================
# PREDICTION (IMAGE PRIORITY)
# ==============================
if predict_clicked:

    if img_file:

        colA, colB = st.columns(2)

        # show image
        img = Image.open(img_file).convert("RGB")
        colA.image(img, caption="Uploaded Image", width=200)

        # image features
        img_features = extract_image_features(img)

        # OPTIONAL: include audio slightly
        if audio_file:
            colB.audio(audio_file)
            audio_features = extract_audio_features(audio_file)

            # weighted fusion (image priority)
            final_features = (0.8 * img_features + 0.2 * audio_features)
        else:
            final_features = img_features

        final_features = final_features.reshape(1, -1)

        # prediction
        probs = model.predict_proba(final_features)[0]
        best_idx = np.argmax(probs)

        # output
        st.markdown("## 🎯 Prediction Result")

        st.success(f"**{class_names[best_idx].upper()}**")

        st.progress(float(probs[best_idx]))
        st.write(f"Confidence: {probs[best_idx]*100:.2f}%")

        st.info("📌 Image-based prediction is prioritized for better accuracy.")

    else:
        st.warning("⚠️ Please upload at least an image")
