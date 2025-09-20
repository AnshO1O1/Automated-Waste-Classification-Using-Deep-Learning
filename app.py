import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- Page Configuration ---
st.set_page_config(
    page_title="Automated Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

# --- Secure API Key Configuration ---
api_key = st.secrets.get("API", "")

# --- Model Path ---
MODEL_PATH = "waste_classifier_mobilenet.h5"

# --- Class Names ---
class_names = {
    0: 'cardboard',
    1: 'glass',
    2: 'metal',
    3: 'paper',
    4: 'plastic',
    5: 'trash'
}

# --- Rebuild Model Architecture ---
def build_model(input_shape=(224, 224, 3), num_classes=len(class_names)):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=None
    )
    base_model.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=base_model.input, outputs=output)
    return model

# --- Cache Model Loading ---
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please make sure the model file is in the same directory.")
        return None
    try:
        model = build_model()
        model.load_weights(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Image Preprocessing ---
def preprocess_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.asarray(image)
    image_array = np.expand_dims(image_array, axis=0)
    return preprocess_input(image_array)

# --- LLM Integration for Recycling Tips ---
def get_recycling_tips(waste_category, api_key):
    if not api_key:
        return "Gemini API Key is not configured. Please add it to your `.streamlit/secrets.toml` file."

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    prompt = f"Provide three short, actionable, and easy-to-follow recycling tips for '{waste_category}' waste. Use bullet points."
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        tips = result['candidates'][0]['content']['parts'][0]['text']
        return tips
    except Exception as e:
        return f"Error with Gemini API: {e}"

# --- Main Application UI ---
st.title("♻️ Automated Waste Classification")
st.markdown("Leveraging Deep Learning and Generative AI to promote effective recycling.")

if not api_key:
    st.warning("The Gemini API key is not configured. Please add `API = 'YOUR_KEY_HERE'` to your `.streamlit/secrets.toml` file.", icon="🔑")

model = load_model(MODEL_PATH)

if model is not None:
    st.success(f"Model weights loaded successfully from '{os.path.basename(MODEL_PATH)}'!")

    uploaded_file = st.file_uploader("Upload an image of a waste item", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        image = Image.open(uploaded_file)
        processed_image = preprocess_image(image)

        with st.spinner("Classifying..."):
            prediction = model.predict(processed_image)
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = class_names.get(predicted_class_index, "Unknown")
            confidence = np.max(prediction) * 100

        with col2:
            st.success(f"**Prediction:** {predicted_class_name.capitalize()}")
            st.info(f"**Confidence:** {confidence:.2f}%")

        st.subheader(f"Recycling Tips for {predicted_class_name.capitalize()}", divider="rainbow")
        with st.spinner("Generating tips with Gemini..."):
            tips = get_recycling_tips(predicted_class_name, api_key)
            st.markdown(tips)
