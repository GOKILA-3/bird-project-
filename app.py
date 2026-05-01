import streamlit as st
import numpy as np
from PIL import Image
import librosa
import joblib
from io import BytesIO
import requests

# PAGE
st.set_page_config(page_title="Bird Classifier", layout="wide")

st.title("🕊️ Bird Species Classifier")

# LOAD MODEL
model = joblib.load("model_compressed.pkl")

class_names = ["crow","sparrow","parrot","pigeon","peacock",
               "eagle","owl","kingfisher","woodpecker","duck"]

# IMAGE URL
def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

# AUDIO FEATURES
def extract_audio_features(file):
    signal, sr = librosa.load(file, sr=22050)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    if len(mfcc) < 256:
        mfcc = np.pad(mfcc, (0, 256 - len(mfcc)))
    else:
        mfcc = mfcc[:256]

    return mfcc

# UI
img_file = st.file_uploader("Upload Image", type=["jpg", "png"])
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])

if st.button("Predict"):

    if img_file and audio_file:

        img = Image.open(img_file)
        st.image(img)

        st.audio(audio_file)

        audio_features = extract_audio_features(audio_file)

        # fake image features
        image_features = np.random.rand(256)

        final_features = (image_features + audio_features) / 2
        final_features = final_features.reshape(1, -1)

        probs = model.predict_proba(final_features)[0]
        best = np.argmax(probs)

        st.success(f"Prediction: {class_names[best]}")
        st.write(f"Confidence: {probs[best]*100:.2f}%")

    else:
        st.warning("Upload both image and audio")
