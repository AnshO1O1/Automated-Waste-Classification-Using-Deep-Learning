import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Lambda, Input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.regularizers import l2
from PIL import Image
import numpy as np
import os
import gdown
from groq import Groq

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="♻️ Waste Classifier + Recycling Tips",
    layout="centered"
)

# --- SETTINGS ---
MODEL_PATH = "trained_EN_model.h5"
DRIVE_FILE_ID = "1tVjhrpLA7OzBa2FwymIq6JgxJnanYg6M"
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 6
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# --- Groq API Key from Streamlit Secrets ---
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

# --- BUILD TRANSFER LEARNING MODEL FOR GRAYSCALE INPUT ---
@st.cache_resource
def build_model():
    # Input for grayscale images (1 channel)
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    
    # Convert grayscale to RGB by repeating the channel 3 times
    x = Lambda(lambda x: tf.repeat(x, 3, axis=-1))(inputs)
    
    base_model = EfficientNetB0(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    # Fine-tune last 20 layers
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=l2(0.001))(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_Grayscale")
    
    return model

# --- LOAD MODEL AND WEIGHTS ---
@st.cache_resource
def load_model_weights(model_path):
    download_model_from_drive(model_path, DRIVE_FILE_ID)
    model = build_model()
    if not os.path.exists(model_path):
        st.error(f"Model weights file not found at {model_path}.")
        return None
    try:
        model.load_weights(model_path)
        st.success("✅ Model loaded successfully with weights!")
        return model
    except Exception as e:
        st.error(f"❌ Could not load weights: {e}")
        return None

model = load_model_weights(MODEL_PATH)

# --- IMAGE PREPROCESSING FOR GRAYSCALE ---
def preprocess_image(image: Image.Image):
    # Convert to grayscale (1 channel)
    image = image.convert("L")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize to [0, 1] (don't use EfficientNet preprocessing for grayscale)
    img_array = img_array / 255.0
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
            model="llama-3.3-70b-versatile"
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error fetching tips from Groq API: {e}"

# --- STREAMLIT UI ---
st.title("♻️ Automated Waste Classifier + Recycling Tips")

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
