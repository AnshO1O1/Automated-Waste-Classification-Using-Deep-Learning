import streamlit as st
import tensorflow as tf
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
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# --- Groq API key from secrets ---
# Ensure you have a secrets.toml file in a .streamlit folder with your API key
# [API]
# GROQ_API_KEY = "your-key-here"
try:
    GROQ_API_KEY = st.secrets["API"]
except (KeyError, FileNotFoundError):
    GROQ_API_KEY = ""

# --- Download model from Drive if not exists ---
def download_model():
    """Downloads the model from Google Drive if it's not already present."""
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading classification model... (this may take a moment)")
        with st.spinner("Fetching model from Google Drive..."):
            try:
                url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
                gdown.download(url, MODEL_PATH, quiet=False)
                st.success("✅ Model downloaded successfully.")
            except Exception as e:
                st.error(f"❌ Model download failed: {e}")
                st.stop()

# --- Load model ---
@st.cache_resource
def load_keras_model():
    """Loads the Keras model into memory, caching it for performance."""
    download_model()
    try:
        # The 'compile=False' argument can sometimes help with loading models
        # where the custom optimizer/loss functions aren't needed for inference.
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        # Re-compile the model if you need to evaluate metrics, but for prediction it's not essential.
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"""
        ❌ **Failed to load model:** {e}
        
        This error often occurs due to a version mismatch in TensorFlow or related libraries. 
        Please ensure your environment is set up using the provided `requirements.txt` file.
        """)
        return None

# --- Image preprocessing ---
def preprocess_image(image: Image.Image):
    """Converts a PIL Image to the format expected by the model."""
    image = image.convert("RGB")  # Ensure image is in RGB format
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

# --- Prediction ---
def predict(model, image: Image.Image):
    """Makes a prediction on a given image."""
    if model is None:
        return None, None
    
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    
    score = tf.nn.softmax(predictions[0])
    class_index = np.argmax(score)
    confidence = np.max(score)
    
    return CLASS_NAMES[class_index], confidence

# --- Groq recycling tips ---
@st.cache_data
def get_recycling_tips(waste_category: str, api_key: str):
    """Fetches actionable recycling tips from the Groq API."""
    if not api_key:
        return "⚠️ Groq API Key not configured. Please add it to your Streamlit secrets."
    try:
        client = Groq(api_key=api_key)
        prompt = f"""You are an expert environmental advisor. Provide three short, actionable, and easy-to-follow recycling tips
for the following type of waste: '{waste_category}'.
- Use bullet points only.
- Keep each tip under 25 words.
- Make the tips practical for households, offices, or small businesses.
- Do not add any extra commentary or explanations.
- Focus on reducing waste, proper sorting, and safe disposal or reuse."""

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192"
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"❌ Error fetching tips from Groq: {e}"

# --- Streamlit UI ---
st.title("♻️ Waste Classifier & Recycling Advisor")
st.markdown("Upload an image of a waste item, and the AI will classify it and provide practical recycling tips.")

# Load the model
model = load_keras_model()

# File uploader
uploaded_file = st.file_uploader("📷 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform prediction
    with st.spinner("🧠 Classifying the item..."):
        label, confidence = predict(model, image)

    if label:
        st.success(f"**Prediction:** `{label.capitalize()}` ({confidence*100:.2f}% confidence)")

        # Fetch and display recycling tips
        st.subheader(f"♻️ Recycling Tips for {label.capitalize()}")
        with st.spinner("💬 Generating tips using Groq..."):
            tips = get_recycling_tips(label, GROQ_API_KEY)
            st.markdown(tips)
    else:
        st.error("⚠️ Prediction failed. Please try a different image or check the model file.")

