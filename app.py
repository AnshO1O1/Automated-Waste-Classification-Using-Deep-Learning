import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2

# ------------------------
# Model Settings
# ------------------------
MODEL_PATH = "waste_classifier_mobilenet.h5"  # your weights file
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 6
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# ------------------------
# Groq API Key
# ------------------------
api_key = st.secrets.get("API", "")  # Use "API" in secrets.toml

# ------------------------
# Build MobileNetV2 Model
# ------------------------
@st.cache_resource
def build_model():
    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    fine_tune_at = len(base_model.layers) - 20
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax', name='output_layer', kernel_regularizer=l2(0.001))
    ], name="MobileNetV2_Transfer_Learning")

    return model

# ------------------------
# Load Weights
# ------------------------
@st.cache_resource
def load_model_weights(model_path):
    model = build_model()
    if not os.path.exists(model_path):
        st.error(f"❌ Model weights file not found at {model_path}. Place your .h5 file here.")
        return None
    try:
        model.load_weights(model_path)
        st.success("✅ Model loaded successfully with weights!")
        return model
    except Exception as e:
        st.error(f"❌ Could not load weights: {e}")
        return None

model = load_model_weights(MODEL_PATH)

# ------------------------
# Image Preprocessing
# ------------------------
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# ------------------------
# Prediction
# ------------------------
def predict(image: Image.Image):
    if model is None:
        return None, None
    img = preprocess_image(image)
    preds = model.predict(img)
    idx = np.argmax(preds)
    confidence = float(np.max(preds))
    return CLASS_NAMES[idx], confidence

# ------------------------
# Groq API Integration
# ------------------------
def get_recycling_tips_groq(waste_category, api_key):
    if not api_key:
        return "Groq API Key not configured. Add it to `.streamlit/secrets.toml` as API = 'YOUR_KEY_HERE'."

    url = "https://api.groq.com/openai/v1/chat/completions"  # Groq endpoint
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = f"Provide three short, actionable, and easy-to-follow recycling tips for '{waste_category}' waste. Use bullet points."

    payload = {
        "model": "grok-16k",  # Groq model
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        tips = result["choices"][0]["message"]["content"]
        return tips
    except Exception as e:
        return f"Error fetching tips from Groq API: {e}"

# ------------------------
# Streamlit UI
# ------------------------
st.title("♻️ Waste Classifier + Recycling Tips (MobileNetV2 + Groq)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    label, conf = predict(image)
    if label:
        st.success(f"Predicted: **{label}** ({conf*100:.2f}% confidence)")

        st.subheader(f"♻️ Recycling Tips for {label.capitalize()}")
        with st.spinner("Generating tips..."):
            tips = get_recycling_tips_groq(label, api_key)
            st.markdown(tips)
    else:
        st.error("Prediction not available. Please check your model weights.")
