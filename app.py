import streamlit as st
import numpy as np
import librosa
import joblib

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="🐦 Bird AI Vision",
    layout="centered",
    page_icon="🐦"
)

# ===============================
# BIRD IMAGE MAP (ADD YOUR OWN)
# ===============================
bird_images = {
    "sparrow": "https://upload.wikimedia.org/wikipedia/commons/5/5c/House_Sparrow_mar08.jpg",
    "crow": "https://upload.wikimedia.org/wikipedia/commons/0/0c/Corvus_brachyrhynchos.jpg",
    "peacock": "https://upload.wikimedia.org/wikipedia/commons/e/e0/Peacock_Plumage.jpg",
    "pigeon": "https://upload.wikimedia.org/wikipedia/commons/1/1d/Rock_Pigeon_Columba_livia.jpg",
}

# ===============================
# CUSTOM CSS (ANIMATED UI)
# ===============================
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

/* Title animation */
h1 {
    text-align: center;
    color: #00ffd5;
    animation: fadeIn 2s ease-in-out;
}

/* Card style */
.card {
    background: rgba(0,0,0,0.6);
    padding: 20px;
    border-radius: 20px;
    margin: 15px 0;
    box-shadow: 0 0 20px rgba(0,255,213,0.2);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

/* Hover animation */
.card:hover {
    transform: scale(1.03);
    box-shadow: 0 0 30px rgba(0,255,213,0.5);
}

/* Image styling */
img {
    border-radius: 15px;
    transition: transform 0.3s ease;
}

img:hover {
    transform: scale(1.05);
}

/* Button */
.stButton>button {
    background-color: #00ffd5;
    color: black;
    font-size: 18px;
    border-radius: 12px;
    padding: 10px 20px;
}

/* Fade animation */
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

</style>
""", unsafe_allow_html=True)

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
# HEADER
# ===============================
st.markdown("<h1>🐦 Bird Audio Intelligence AI</h1>", unsafe_allow_html=True)
st.markdown("### Upload bird sound and see species prediction with visuals 🎧✨")

# ===============================
# UPLOAD
# ===============================
file = st.file_uploader("🎵 Upload Bird Audio", type=["wav", "mp3"])

if file:
    st.audio(file)

# ===============================
# PREDICT
# ===============================
if file and st.button("🚀 Predict Bird Species"):

    with st.spinner("Listening to nature... 🌿"):
        results = predict_audio(file)

    st.success("Prediction Completed 🎉")

    # ===========================
    # RESULTS UI
    # ===========================
    for label, conf in results:

        img_url = bird_images.get(label.lower(), "https://upload.wikimedia.org/wikipedia/commons/3/3a/Bird_icon.png")

        st.markdown(f"""
        <div class="card">
            <h2>🐦 {label}</h2>
            <img src="{img_url}" width="250">
        </div>
        """, unsafe_allow_html=True)

        st.progress(conf)
        st.write(f"Confidence: **{conf*100:.2f}%**")
