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
# MODERN CSS
# ===============================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b, #0b1220);
    color: white;
}

/* Title */
.main-title {
    font-size: 44px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #7c4dff, #00e5ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    opacity: 0.8;
}

/* Card */
.card {
    background: rgba(255,255,255,0.06);
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 0 15px rgba(124,77,255,0.2);
    text-align: center;
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-5px);
}

/* Result box */
.result-box {
    background: rgba(0, 229, 255, 0.08);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.1);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0b1220;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #7c4dff, #00e5ff);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("🐦 Bird Vision AI")
menu = st.sidebar.radio("Navigation", ["🏠 Dashboard", "🎧 Predict"])

# ===============================
# HEADER
# ===============================
st.markdown("<div class='main-title'>🐦 Bird Species Recognition AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered bird Species classification system</div>", unsafe_allow_html=True)

st.markdown("---")

# ===============================
# DASHBOARD (FIXED & CENTERED)
# ===============================
if menu == "🏠 Dashboard":

    st.markdown("### 📊 Model Overview")

    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("""
            <div class='card'>
                <h3>🎯 Accuracy</h3>
                <h2>98.67%</h2>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class='card'>
                <h3>🐦 Species</h3>
                <h2>5</h2>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📌 Supported Bird Classes")
    st.info("Sparrow • Crow • Pigeon • Peacock • Parrot • Dove")

# ===============================
# PREDICTION PAGE
# ===============================
if menu == "🎧 Predict":

    st.markdown("## 🎧 Upload Bird Audio File")

    file = st.file_uploader("Upload WAV / MP3 audio", type=["wav", "mp3"])
    predict = st.button("🚀 Predict Bird")

    if "result" not in st.session_state:
        st.session_state.result = None

    if file and predict:
        with st.spinner("Analyzing bird sound... 🧠"):
            st.session_state.result = predict_audio(file)

    if st.session_state.result:

        results = st.session_state.result
        top_label, top_conf = results[0]

        st.markdown("## 📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class='result-box'>
                <h2>🐦 {top_label.title()}</h2>
                <h3 style='color:#00e5ff'>{top_conf*100:.2f}% Confidence</h3>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.success(f"Top Prediction: {top_label.title()}")

        st.markdown("## 🔥 Top Predictions")

        for label, conf in results:
            st.write(f"🐦 **{label.title()}**")
            st.progress(float(conf))
            st.caption(f"{conf*100:.2f}% confidence")
