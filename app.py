import streamlit as st
import numpy as np
import librosa
import joblib
from PIL import Image
import os

# ===============================
# LOAD MODELS
# ===============================
audio_model = joblib.load("bird_model.pkl")
le = joblib.load("label_encoder.pkl")

# If you have image model
try:
    from tensorflow.keras.models import load_model
    image_model = load_model("bird_image_model.h5")
    IMAGE_MODEL_AVAILABLE = True
except:
    IMAGE_MODEL_AVAILABLE = False

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="🐦 Bird Classifier", layout="wide")

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("🐦 Bird AI App")

app_mode = st.sidebar.radio(
    "Select Mode",
    ["🏠 Home", "🎧 Audio Prediction", "🖼 Image Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.info("Final Year Project\nBird Species Detection")

# ===============================
# FEATURE FUNCTIONS
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


def predict_audio(file):
    features = extract_audio_features(file)
    probs = audio_model.predict_proba([features])[0]
    idx = np.argmax(probs)

    return le.inverse_transform([idx])[0], probs[idx]


def predict_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = image_model.predict(img)
    idx = np.argmax(preds)

    return le.inverse_transform([idx])[0], preds[0][idx]


# ===============================
# HOME PAGE
# ===============================
if app_mode == "🏠 Home":

    st.title("🐦 Bird Species Prediction System")
    st.markdown("### AI-based Audio + Image Classification")

    col1, col2 = st.columns(2)

    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/616/616490.png", width=200)

    with col2:
        st.write("""
        This system can:
        - 🎧 Detect bird from audio
        - 🖼 Detect bird from image
        - ⚡ Fast & simple UI
        """)

    st.success("✅ Ready to use!")

# ===============================
# AUDIO PAGE
# ===============================
elif app_mode == "🎧 Audio Prediction":

    st.title("🎧 Bird Audio Prediction")

    audio_file = st.file_uploader("Upload Bird Sound", type=["wav", "mp3"])

    if audio_file:
        st.audio(audio_file)

        if st.button("🔍 Predict Bird"):
            label, confidence = predict_audio(audio_file)

            st.success(f"🐦 Prediction: {label}")
            st.progress(float(confidence))

            st.write(f"Confidence: {confidence*100:.2f}%")

# ===============================
# IMAGE PAGE
# ===============================
elif app_mode == "🖼 Image Prediction":

    st.title("🖼 Bird Image Prediction")

    if not IMAGE_MODEL_AVAILABLE:
        st.error("⚠️ Image model not found!")
    else:
        image_file = st.file_uploader("Upload Bird Image", type=["jpg", "png"])

        if image_file:
            img = Image.open(image_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)

            if st.button("🔍 Predict Bird"):
                label, confidence = predict_image(img)

                st.success(f"🐦 Prediction: {label}")
                st.progress(float(confidence))

                st.write(f"Confidence: {confidence*100:.2f}%")
