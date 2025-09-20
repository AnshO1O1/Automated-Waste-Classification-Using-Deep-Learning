import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests # To make the API call to Gemini
# Import the specific preprocessing function for MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- Page Configuration ---
st.set_page_config(
    page_title="Automated Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

# --- Secure API Key Configuration ---
# Fetches the API key from st.secrets.toml. This is the secure way to handle keys.
# The app will show a warning if the key is not found.
api_key = st.secrets.get("API", "")


# --- Model Path ---
# The model is expected to be in the same directory as this script.
MODEL_PATH = "waste_classifier_mobilenet.h5"


# Cache the model loading to improve performance
@st.cache_resource
def load_model(model_path):
    """Loads a Keras model from the specified path."""
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please make sure the model file is in the same directory.")
        return None
    try:
        # By adding compile=False, we tell Keras to only load the model's
        # architecture and weights. This bypasses issues with the saved
        # optimizer state and is the standard practice for inference.
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Image Preprocessing ---
def preprocess_image(image, target_size=(224, 224)):
    """Preprocesses the uploaded image to be model-ready."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.asarray(image)
    image_array = np.expand_dims(image_array, axis=0)
    # Use the correct preprocessing function for MobileNetV2
    # This scales pixel values to the range [-1, 1], as expected by the model.
    return preprocess_input(image_array)

# --- LLM Integration for Recycling Tips ---
def get_recycling_tips(waste_category, api_key):
    """Calls the Gemini API to get recycling tips."""
    if not api_key:
        return "Gemini API Key is not configured. Please add it to your `.streamlit/secrets.toml` file to enable this feature."
    
    # Use the gemini-2.5-flash-preview-05-20 model
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    
    prompt = f"Provide three short, actionable, and easy-to-follow recycling tips for '{waste_category}' waste. Use bullet points."
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status() # Raise an exception for bad status codes
        result = response.json()
        
        # Extract the text from the response
        tips = result['candidates'][0]['content']['parts'][0]['text']
        return tips
    except requests.exceptions.RequestException as e:
        return f"Error calling the API: {e}. Please check your API key and network connection."
    except (KeyError, IndexError) as e:
        return f"Error parsing API response: {e}. The response from the API was not in the expected format."


# --- Main Application UI ---
st.title("♻️ Automated Waste Classification")
st.markdown("Leveraging Deep Learning and Generative AI to promote effective recycling.")

# Show a warning if the API key is not configured.
if not api_key:
    st.warning("The Gemini API key is not configured. Please add `GEMINI_API_KEY = 'YOUR_KEY_HERE'` to your `.streamlit/secrets.toml` file to enable recycling tips.", icon="🔑")

# Load the model
model = load_model(MODEL_PATH)

# Only show the uploader and the rest of the app if the model loaded successfully
if model is not None:
    st.success(f"Model '{os.path.basename(MODEL_PATH)}' loaded successfully!")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image of a waste item",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Preprocess and predict
        image = Image.open(uploaded_file)
        processed_image = preprocess_image(image)
        
        # IMPORTANT: This dictionary maps the model's output index to a class name.
        # Ensure the indices (0, 1, 2, etc.) match the alphabetical order of your
        # training data subdirectories (e.g., how `flow_from_directory` found them).
        class_names = {
            0: 'cardboard',
            1: 'glass',
            2: 'metal',
            3: 'paper',
            4: 'plastic',
            5: 'trash'
        }

        with st.spinner("Classifying..."):
            prediction = model.predict(processed_image)
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = class_names.get(predicted_class_index, "Unknown")
            confidence = np.max(prediction) * 100

        with col2:
            st.success(f"**Prediction:** {predicted_class_name.capitalize()}")
            st.info(f"**Confidence:** {confidence:.2f}%")

        # --- Display Recycling Tips ---
        st.subheader(f"Recycling Tips for {predicted_class_name.capitalize()}", divider="rainbow")
        with st.spinner("Generating tips with Gemini..."):
            tips = get_recycling_tips(predicted_class_name, api_key)
            st.markdown(tips)

