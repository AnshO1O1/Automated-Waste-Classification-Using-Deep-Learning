# app.py

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown
from groq import Groq

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="♻️ Waste Classifier + Recycling Tips",
    layout="centered"
)

# --- Constants ---
MODEL_PATH = "efficientnet_waste_classifier.h5"
DRIVE_FILE_ID = "1tVjhrpLA7OzBa2FwymIq6JgxJnanYg6M"  # Your Google Drive model ID
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# --- Groq API ---
GROQ_API_KEY = st.secrets.get("API", "")  # Add API key in .streamlit/secrets.toml

# --- Download Full Model from Drive ---
def download_model_from_drive(model_path, file_id):
    if not os.path.exists(model_path):
        st.info("📥 Downloading model from Google Drive...")
        try:
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)
            st.success("✅ Model downloaded.")
        except Exception as e:
            st.error(f"❌ Download failed: {e}")

# --- Load Full Model ---
@st.cache_resource
def load_model_from_drive(model_path, file_id):
    download_model_from_drive(model_path, file_id)
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("✅ Full model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

# --- Load model once ---
model = load_model_from_drive(MODEL_PATH, DRIVE_FILE_ID)

# --- Image Preprocessing ---
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

# --- Predict Function ---
def predict(image: Image.Image):
    if model is None:
        return None, None
    img = preprocess_image(image)
    preds = model.predict(img)
    idx = np.argmax(preds)
    confidence = float(np.max(preds))
    return CLASS_NAMES[idx], confidence

# --- Groq: Get Recycling Tips ---
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

    label, conf = predict(image)
    if label:
        st.success(f"🧠 Predicted: **{label.capitalize()}** ({conf*100:.2f}% confidence)")

        st.subheader(f"♻️ Recycling Tips for {label.capitalize()}")
        with st.spinner("💬 Generating tips using Groq..."):
            tips = get_recycling_tips(label, GROQ_API_KEY)
            st.markdown(tips)
    else:
        st.error("⚠️ Prediction failed. Please check the model file.")
