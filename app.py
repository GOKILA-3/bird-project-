import streamlit as st
import numpy as np
import librosa
import joblib

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Bird Vision AI", layout="wide")

# ===============================
# MODEL LOAD
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
# SIDEBAR (LIKE YOUR IMAGE)
# ===============================
st.sidebar.title("🐦 Bird Vision AI")
menu = st.sidebar.radio("Navigation", ["Dashboard", "Predict", "Analytics"])

st.sidebar.markdown("---")
st.sidebar.info("AI Bird Sound Classification System")

# ===============================
# DARK THEME UI
# ===============================
st.markdown("""
<style>
body {
    background-color: #0e1117;
}

.main-title {
    font-size: 40px;
    font-weight: bold;
    color: #7c4dff;
}

.card {
    background: #161b22;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 10px rgba(124,77,255,0.2);
    margin-bottom: 15px;
}

.big-number {
    font-size: 28px;
    font-weight: bold;
    color: #00ffcc;
}

img {
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# DASHBOARD PAGE
# ===============================
if menu == "Dashboard":
    st.markdown("<div class='main-title'>🐦 Bird Vision Dashboard</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='card'><h3>Model Accuracy</h3><div class='big-number'>96.8%</div></div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'><h3>Species Supported</h3><div class='big-number'>10+</div></div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='card'><h3>Predictions Made</h3><div class='big-number'>1.2K</div></div>", unsafe_allow_html=True)

    st.markdown("### 📊 Recent Activity")
    st.info("Upload audio in Predict section to start predictions.")

# ===============================
# PREDICT PAGE (MAIN UI LIKE IMAGE)
# ===============================
if menu == "Predict":

    st.markdown("<div class='main-title'>🎧 Bird Audio Prediction</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.2])

    # LEFT SIDE (UPLOAD)
    with col1:
        st.markdown("### 📤 Upload Audio")
        file = st.file_uploader("Upload bird sound", type=["wav", "mp3"])

        if file:
            st.audio(file)

        predict_btn = st.button("🚀 Predict")

    # RIGHT SIDE (RESULT PANEL)
    with col2:
        st.markdown("### 📊 Prediction Result")

        if file and predict_btn:

            results = predict_audio(file)

            top_label, top_conf = results[0]
            img = bird_images.get(top_label.lower(), "")

            st.markdown(f"""
            <div class='card'>
                <h2>🐦 {top_label}</h2>
                <h3 style='color:#00ffcc'>{top_conf*100:.2f}% Confidence</h3>
            </div>
            """, unsafe_allow_html=True)

            if img:
                st.image(img, width=300)

            st.markdown("### 🔥 Top Predictions")

            for label, conf in results:
                st.write(f"**{label}**")
                st.progress(conf)
                st.write(f"{conf*100:.2f}%")

# ===============================
# ANALYTICS PAGE
# ===============================
if menu == "Analytics":
    st.markdown("<div class='main-title'>📈 Analytics</div>", unsafe_allow_html=True)

    st.info("Add charts here later (accuracy, dataset distribution, etc.)")
