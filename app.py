import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from PIL import Image
import numpy as np
import os
import gdown
from groq import Groq

# --- Page config ---
st.set_page_config(page_title="♻️ Waste Classifier + Recycling Tips", layout="centered")

# --- Constants ---
MODEL_PATH = "efficientnet_waste_classifier.h5"
DRIVE_FILE_ID = "1tVjhrpLA7OzBa2FwymIq6JgxJnanYg6M"
IMG_HEIGHT, IMG_WIDTH = 224, 224
NUM_CLASSES = 6
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# --- Groq API key from secrets ---
GROQ_API_KEY = st.secrets.get("API", "")

# --- Download model from Drive if not exists ---
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        try:
            gdown.download(url, MODEL_PATH, quiet=False)
            st.success("✅ Model downloaded successfully.")
        except Exception as e:
            st.error(f"❌ Model download failed: {e}")

# --- Load model ---
@st.cache_resource
def load_model():
    download_model()
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_model()

# --- Image preprocessing ---
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")  # Make sure it's RGB (3 channels)
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

# --- Prediction ---
def predict(image: Image.Image):
    if model is None:
        return None, None
    img = preprocess_image(image)
    preds = model.predict(img)
    idx = np.argmax(preds)
    confidence = float(np.max(preds))
    return CLASS_NAMES[idx], confidence

# --- Groq recycling tips ---
@st.cache_data
def get_recycling_tips(waste_category, api_key):
    if not api_key:
        return "⚠️ Groq API Key not configured. Add it to `.streamlit/secrets.toml`."
    try:
        client = Groq(api_key=api_key)
        prompt = f"""You are an expert environmental advisor. Provide **three short, actionable, and easy-to-follow recycling tips** 
for the following type of waste: '{waste_category}'.
- Use **bullet points** only.
- Keep each tip **under 25 words**.
- Make the tips **practical** for households, offices, or small businesses.
- Do not add any extra commentary or explanations.
- Focus on **reducing waste, proper sorting, and safe disposal or reuse**."""

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"❌ Error from Groq: {e}"

# --- Streamlit UI ---
st.title("♻️ Waste Classifier + Recycling Tips")

uploaded_file = st.file_uploader("📷 Upload a waste image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    label, confidence = predict(image)
    if label:
        st.success(f"🧠 Predicted: **{label.capitalize()}** ({confidence*100:.2f}% confidence)")

        st.subheader(f"♻️ Recycling Tips for {label.capitalize()}")
        with st.spinner("💬 Generating tips using Groq..."):
            tips = get_recycling_tips(label, GROQ_API_KEY)
            st.markdown(tips)
    else:
        st.error("⚠️ Prediction failed. Please check your model file.")
