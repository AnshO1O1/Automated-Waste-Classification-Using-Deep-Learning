import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2
from PIL import Image
import numpy as np
import os
from groq import Groq

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="♻️ Waste Classifier + Recycling Tips",
    layout="centered"
)

# --- SETTINGS ---
MODEL_PATH = "waste_classifier_mobilenet.h5"
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 6
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# --- Groq API Key from Streamlit Secrets ---
GROQ_API_KEY = st.secrets.get("API", "")

# --- BUILD TRANSFER LEARNING MODEL ---
@st.cache_resource
def build_model():
    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=l2(0.001))
    ])
    return model

# --- LOAD WEIGHTS ---
@st.cache_resource
def load_model_weights(model_path):
    model = build_model()
    if not os.path.exists(model_path):
        st.error(f"Model weights file not found at {model_path}. Place your .h5 file here.")
        return None
    try:
        model.load_weights(model_path)
        st.success(" Model loaded successfully !")
        return model
    except Exception as e:
        st.error(f" Could not load weights: {e}")
        return None

model = load_model_weights(MODEL_PATH)

# --- IMAGE PREPROCESSING ---
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# --- PREDICTION ---
def predict(image: Image.Image):
    if model is None:
        return None, None
    img = preprocess_image(image)
    preds = model.predict(img)
    idx = np.argmax(preds)
    confidence = float(np.max(preds))
    return CLASS_NAMES[idx], confidence

# --- GROQ RECYCLING TIPS ---
@st.cache_data
def get_recycling_tips(waste_category, api_key):
    if not api_key:
        return "Groq API Key not configured. Add it to `.streamlit/secrets.toml` as API = 'YOUR_KEY_HERE'."
    try:
        client = Groq(api_key=api_key)
        prompt = f"""You are an expert environmental advisor. Provide **three short, actionable, and easy-to-follow recycling tips** 
for the following type of waste: '{waste_category}'.
- Use **bullet points** only.
- Keep each tip **under 25 words**. 
- Make the tips **practical** for households, offices, or small businesses. 
- Do not add any extra commentary or explanations outside the bullet points. 
- Focus on **reducing waste, proper sorting, and safe disposal or reuse**."""
        
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"  # Updated model
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error fetching tips from Groq API: {e}"

# --- STREAMLIT UI ---
st.title("♻️ Automated Waste Classifier + Recycling Tips")

uploaded_file = st.file_uploader("Upload an image of waste", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    label, conf = predict(image)
    if label:
        st.success(f"Predicted: **{label.capitalize()}** ({conf*100:.2f}% confidence)")

        st.subheader(f"♻️ Recycling Tips for {label.capitalize()}")
        with st.spinner("Generating tips with Groq..."):
            tips = get_recycling_tips(label, GROQ_API_KEY)
            st.markdown(tips)
    else:
        st.error("Prediction not available. Check your model weights.")
