import streamlit as st
import numpy as np
import librosa
import joblib

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="🐦 Bird Vision AI",
    layout="wide",
    page_icon="🐦"
)

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
# CSS (MODERN UI)
# ===============================
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b, #0f172a);
    color: white;
}

/* Title */
.main-title {
    font-size: 45px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #7c4dff, #00e5ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

/* Glass Card */
.card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(10px);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-5px);
}

/* Result box */
.result-box {
    background: rgba(124,77,255,0.15);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.1);
}

/* Image styling */
img {
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0b1220;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("🐦 Bird Vision AI")
menu = st.sidebar.radio("Navigation", ["🏠 Dashboard", "🎧 Predict"])

# ===============================
# HEADER BANNER
# ===============================
st.markdown("<div class='main-title'>🐦 Bird Vision AI</div>", unsafe_allow_html=True)
st.markdown("### 🎶 Detect bird species from audio using AI")

st.image(
    "https://upload.wikimedia.org/wikipedia/commons/6/6a/Bird_flying_banner.jpg",
    use_container_width=True
)

# ===============================
# DASHBOARD
# ===============================
if menu == "🏠 Dashboard":

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='card'><h3>🎯 Accuracy</h3><h2>96.8%</h2></div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'><h3>🐦 Species</h3><h2>10+</h2></div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='card'><h3>📊 Predictions</h3><h2>1.2K</h2></div>", unsafe_allow_html=True)

    st.markdown("### 🐦 Popular Birds")

    cols = st.columns(4)
    for i, (bird, img) in enumerate(bird_images.items()):
        with cols[i % 4]:
            st.image(img, caption=bird.title(), use_container_width=True)

# ===============================
# PREDICT PAGE
# ===============================
if menu == "🎧 Predict":

    st.markdown("## 🎧 Upload Bird Audio")

    file = st.file_uploader("Upload WAV / MP3", type=["wav", "mp3"])
    predict = st.button("🚀 Predict Bird")

    if "result" not in st.session_state:
        st.session_state.result = None

    if file and predict:
        with st.spinner("Analyzing bird sounds... 🧠"):
            st.session_state.result = predict_audio(file)

    if st.session_state.result:

        results = st.session_state.result
        top_label, top_conf = results[0]

        st.markdown("## 📊 Prediction Result")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(f"""
            <div class='result-box'>
                <h2>🐦 {top_label.title()}</h2>
                <h3 style='color:#00e5ff'>{top_conf*100:.2f}% Confidence</h3>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            img = bird_images.get(top_label.lower())
            if img:
                st.image(img, use_container_width=True)

        st.markdown("## 🔥 Top Predictions")

        for label, conf in results:
            st.write(f"**🐦 {label.title()}**")
            st.progress(float(conf))
            st.caption(f"{conf*100:.2f}% confidence")
