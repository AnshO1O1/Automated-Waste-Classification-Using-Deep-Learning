import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown
from groq import Groq
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.regularizers import l2

# --- Page config ---
st.set_page_config(page_title="♻️ Waste Classifier + Recycling Tips", layout="centered")

# --- Constants ---
MODEL_PATH = "efficientnet_waste_classifier.h5"
DRIVE_FILE_ID = "1tVjhrpLA7OzBa2FwymIq6JgxJnanYg6M"
IMG_HEIGHT, IMG_WIDTH = 224, 224
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = 6

# --- Model building function (EXACTLY matches your training code) ---
def build_transfer_learning_model():
    base_model = EfficientNetB0(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),  # 3 channels for RGB
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False   # freeze all layers initially

    # Fine-tuning: unfreeze last 20 layers
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
    ], name="EfficientNetB0_Transfer_Learning")
    return model

# --- Groq API key from secrets ---
try:
    GROQ_API_KEY = st.secrets["API"]["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    GROQ_API_KEY = ""

# --- Download model from Drive if not exists ---
def download_model():
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
    download_model()
    try:
        # Build the model architecture first (EXACT match with training)
        model = build_transfer_learning_model()
        
        # Then load the weights
        model.load_weights(MODEL_PATH)
        
        # Compile the model (same as training)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"""
        ❌ **Failed to load model:** {e}
        
        This error often occurs due to a version mismatch in TensorFlow or related libraries. 
        Please ensure your environment is set up using the provided `requirements.txt` file.
        """)
        return None

# --- Image preprocessing for RGB (3 channels) ---
def preprocess_image(image: Image.Image):
    """Converts a PIL Image to the format expected by the model."""
    image = image.convert("RGB")  # Ensure image is in RGB format (3 channels)
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch
    # Use EfficientNet's preprocessing
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
