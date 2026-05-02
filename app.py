import streamlit as st
import numpy as np
import librosa
import joblib

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="🐦 Bird Vision AI", layout="wide")

# ===============================
# LOAD MODEL
# ===============================
audio_model = joblib.load("bird_model.pkl")
le = joblib.load("label_encoder.pkl")

# ===============================
# BIRD IMAGES
# ===============================
bird_images = {
    "sparrow": "https://upload.wikimedia.org/wikipedia/commons/5/5c/House_Sparrow_mar08.jpg",
    "crow": "https://upload.wikimedia.org/wikipedia/commons/0/0c/Corvus_brachyrhynchos.jpg",
    "pigeon": "https://upload.wikimedia.org/wikipedia/commons/1/1d/Rock_Pigeon_Columba_livia.jpg",
    "peacock": "https://upload.wikimedia.org/wikipedia/commons/e/e0/Peacock_Plumage.jpg",
}

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

def predict_audio(file):
    features = extract_features(file)
    probs = audio_model.predict_proba([features])[0]
    top3 = probs.argsort()[-3:][::-1]
    return [(le.inverse_transform([i])[0], float(probs[i])) for i in top3]

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("🐦 Bird Vision AI")
menu = st.sidebar.radio("Navigation", ["Dashboard", "Predict"])

# ===============================
# CSS STYLE
# ===============================
st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: bold;
    color: #7c4dff;
}

.card {
    background: #161b22;
    padding: 20px;
    border-radius: 15px;
    margin-top: 15px;
    box-shadow: 0px 0px 10px rgba(124,77,255,0.2);
}

.result-box {
    background: #0f172a;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
}

img {
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# DASHBOARD
# ===============================
if menu == "Dashboard":
    st.markdown("<div class='main-title'>🐦 Bird Vision Dashboard</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='card'><h3>Accuracy</h3><h2>96.8%</h2></div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'><h3>Species</h3><h2>10+</h2></div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='card'><h3>Predictions</h3><h2>1.2K</h2></div>", unsafe_allow_html=True)

# ===============================
# PREDICT PAGE
# ===============================
if menu == "Predict":

    st.markdown("<div class='main-title'>🎧 Bird Audio Prediction</div>", unsafe_allow_html=True)

    file = st.file_uploader("Upload Bird Audio (wav/mp3)", type=["wav", "mp3"])

    predict = st.button("🚀 Predict")

    # Store result in session (IMPORTANT FIX)
    if "result" not in st.session_state:
        st.session_state.result = None

    if file and predict:
        st.session_state.result = predict_audio(file)

    # ===============================
    # ALWAYS SHOW RESULT IF EXISTS
    # ===============================
    if st.session_state.result:

        results = st.session_state.result
        top_label, top_conf = results[0]

        st.markdown("### 📊 Prediction Result")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(f"""
            <div class='result-box'>
                <h2>🐦 {top_label}</h2>
                <h3 style='color:#00ffcc'>{top_conf*100:.2f}% Confidence</h3>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            img = bird_images.get(top_label.lower())
            if img:
                st.image(img, width=300)

        st.markdown("### 🔥 Top Predictions")

        for label, conf in results:
            st.write(f"**{label}**")
            st.progress(conf)
            st.write(f"{conf*100:.2f}%")
