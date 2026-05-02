import streamlit as st
import numpy as np
import librosa
import joblib

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="🐦 Bird Audio AI",
    layout="centered",
    page_icon="🐦"
)

# ===============================
# CUSTOM CSS (MAKE IT BEAUTIFUL)
# ===============================
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
}

.main {
    background-color: #0e1117;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.5);
}

h1 {
    text-align: center;
    color: #00ffcc;
    font-size: 40px;
}

.stButton>button {
    background-color: #00ffcc;
    color: black;
    border-radius: 10px;
    font-size: 18px;
    padding: 10px 20px;
}

.stProgress > div > div > div {
    background-color: #00ffcc;
}

.card {
    background: #1a1d25;
    padding: 15px;
    border-radius: 15px;
    margin-bottom: 10px;
    box-shadow: 0px 0px 10px rgba(0,255,204,0.2);
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
# UI HEADER
# ===============================
st.markdown("<h1>🐦 Bird Audio Species AI</h1>", unsafe_allow_html=True)
st.markdown("### Upload bird sound and discover species instantly 🎧")

# ===============================
# FILE UPLOAD
# ===============================
file = st.file_uploader("🎵 Upload Bird Audio", type=["wav", "mp3"])

if file:
    st.audio(file, format='audio/wav')

# ===============================
# PREDICTION BUTTON
# ===============================
if file and st.button("🚀 Predict Species"):

    with st.spinner("Analyzing bird sound... 🔍"):
        results = predict_audio(file)

    st.success("Prediction Complete! 🎉")

    # ===========================
    # RESULT CARDS
    # ===========================
    for label, conf in results:
        st.markdown(f"""
        <div class="card">
            <h3>🐦 {label}</h3>
        </div>
        """, unsafe_allow_html=True)

        st.progress(conf)
        st.write(f"Confidence: **{conf*100:.2f}%**")
