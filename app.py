import streamlit as st
import numpy as np
from PIL import Image
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
from io import BytesIO
import requests

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Bird Classifier", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🕊️ Bird Species Classifier</h1>
    <p style='text-align: center;'>Multimodal Deep Learning (Image + Audio)</p>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("model_compressed.pkl")

class_names = ["crow","sparrow","parrot","pigeon","peacock",
               "eagle","owl","kingfisher","woodpecker","duck"]

# ==============================
# REFERENCE IMAGES
# ==============================
reference_images = {
    "crow": "https://upload.wikimedia.org/wikipedia/commons/1/11/Crow_in_flight.jpg",
    "sparrow": "https://upload.wikimedia.org/wikipedia/commons/5/5e/House_sparrow04.jpg",
    "parrot": "https://upload.wikimedia.org/wikipedia/commons/0/05/Scarlet_Macaw_and_Blue-and-yellow_Macaw.jpg",
    "pigeon": "https://upload.wikimedia.org/wikipedia/commons/9/9b/Rock_Pigeon_01.jpg",
    "peacock": "https://upload.wikimedia.org/wikipedia/commons/e/e3/Peacock_Plumage.jpg",
    "eagle": "https://upload.wikimedia.org/wikipedia/commons/1/1a/Bald_Eagle_Portrait.jpg",
    "owl": "https://upload.wikimedia.org/wikipedia/commons/1/1c/Barn_Owl.jpg",
    "kingfisher": "https://upload.wikimedia.org/wikipedia/commons/5/5a/Common_Kingfisher.jpg",
    "woodpecker": "https://upload.wikimedia.org/wikipedia/commons/6/6e/Great_Spotted_Woodpecker.jpg",
    "duck": "https://upload.wikimedia.org/wikipedia/commons/7/74/Mallard2.jpg"
}

# ==============================
# DEVICE
# ==============================
device = torch.device("cpu")  # force CPU (safe for Streamlit)

# ==============================
# IMAGE TRANSFORM
# ==============================
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ==============================
# LIGHTWEIGHT MODELS (NO DOWNLOAD)
# ==============================
class VisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=None)  # no internet download
        self.model.fc = nn.Linear(self.model.fc.in_features, 128)

    def forward(self, x):
        return self.model(x)

class AudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 128)

    def forward(self, v, a):
        return self.fc(torch.cat((v, a), dim=1))

# Initialize models
vision_model = VisionModel().to(device)
audio_model = AudioModel().to(device)
fusion_model = FusionModel().to(device)

vision_model.eval()
audio_model.eval()
fusion_model.eval()

# ==============================
# LOAD IMAGE FROM URL
# ==============================
def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

# ==============================
# AUDIO FEATURE EXTRACTION
# ==============================
def extract_audio_features(file):
    signal, sr = librosa.load(file, sr=22050)

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    if len(mfcc) < 256:
        mfcc = np.pad(mfcc, (0, 256 - len(mfcc)))
    else:
        mfcc = mfcc[:256]

    return torch.tensor(mfcc).float()

# ==============================
# UI INPUTS
# ==============================
col1, col2 = st.columns(2)

with col1:
    img_file = st.file_uploader("📸 Upload Bird Image", type=["jpg", "png"])

with col2:
    audio_file = st.file_uploader("🎵 Upload Bird Audio", type=["wav", "mp3"])

# ==============================
# PREDICTION
# ==============================
if st.button("🔍 Predict", use_container_width=True):

    if img_file and audio_file:

        colA, colB = st.columns(2)

        with colA:
            img = Image.open(img_file).convert("RGB")
            st.image(img, caption="Uploaded Image")

        with colB:
            st.audio(audio_file)

        img_tensor = image_transform(img).unsqueeze(0).to(device)
        audio_features = extract_audio_features(audio_file).unsqueeze(0).to(device)

        # Fake feature extraction (since no trained DL weights)
        with torch.no_grad():
            v = vision_model(img_tensor)
            a = audio_model(audio_features)
            f = fusion_model(v, a).cpu().numpy()

        # Final ML prediction
        probs = model.predict_proba(f)[0]
        top3_idx = np.argsort(probs)[-3:][::-1]

        st.markdown("## 🧠 Prediction Results")

        for i in top3_idx:
            st.write(f"### {class_names[i].upper()}")
            st.progress(float(probs[i]))
            st.caption(f"Confidence: {probs[i]*100:.2f}%")

        best_idx = top3_idx[0]
        best = class_names[best_idx]
        best_conf = probs[best_idx]

        st.markdown("## 🎯 Final Prediction")

        if best_conf > 0.8:
            st.success(f"🔥 High Confidence: {best.upper()}")
        elif best_conf > 0.5:
            st.info(f"⚡ Medium Confidence: {best.upper()}")
        else:
            st.warning(f"⚠️ Low Confidence: {best.upper()}")

        st.markdown("## 🖼️ Reference Image")

        if best in reference_images:
            try:
                ref_img = load_image_from_url(reference_images[best])
                st.image(ref_img, caption=best.upper())
            except:
                st.warning("Could not load image")

    else:
        st.warning("⚠️ Upload both image and audio!")
