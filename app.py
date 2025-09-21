import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.regularizers import l2
from PIL import Image
import numpy as np
import os
import gdown
from groq import Groq

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="♻️ Waste Classifier (ResNet50 Transfer Learning)",
    layout="centered"
)

# --- SETTINGS ---
MODEL_PATH = "trained_resnet_model.h5"
DRIVE_FILE_ID = "1zG_6QvXZucGVv_C1cGmYHVsskAvc08am"  # Your ResNet model file ID
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 6
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# --- API KEY FROM SECRETS ---
GROQ_API_KEY = st.secrets.get("API", "")

# --- DOWNLOAD MODEL FROM GOOGLE DRIVE ---
def download_model_from_drive(model_path, file_id):
    if not os.path.exists(model_path):
        st.info("🔄 Downloading model weights from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            gdown.download(url, model_path, quiet=False)
            st.success("✅ Model downloaded successfully.")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")

# --- BUILD TRANSFER LEARNING MODEL (Your version) ---
@st.cache_resource
def build_transfer_learning_model3():
    base_model = ResNet50(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # freeze all layers

    # Fine-tune last 20 layers
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
    ], name="ResNet50_Transfer_Learning")
    return model

# --- LOAD MODEL + WEIGHTS ---
@st.cache_resource
def load_model():
    download_model_from_drive(MODEL_PATH, DRIVE_FILE_ID)
    model = build_transfer_learning_model3()
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model weights file not found at {MODEL_PATH}.")
        return None
    try:
        model.load_weights(MODEL_PATH)
        st.success("✅ Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model weights: {e}")
        return None

model3 = load_model()

# --- IMAGE PREPROCESSING ---
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# --- PREDICT ---
def predict(image: Image.Image):
    if model3 is None:
        return None, None
    img_array = preprocess_image(image)
    preds = model3.predict(img_array)
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
- Keep each tip **under 50 words**.
- Make the tips **practical** for households, offices, or small businesses.
- Do not add any extra commentary or explanations outside the bullet points.
- Focus on **reducing waste, proper sorting, and safe disposal or reuse**."""

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error fetching tips from Groq API: {e}"

# --- UI ---
st.title("♻️ Waste Classifier (ResNet50 TL) + Recycling Tips")

uploaded_file = st.file_uploader("Upload an image of waste", type=["jpg", "jpeg", "png"])

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
