import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests # To make the API call to Gemini

# --- Page Configuration ---
st.set_page_config(
    page_title="Automated Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

# --- Gemini API Configuration ---
# IMPORTANT: It's recommended to use st.secrets for API keys in deployed apps
# For local development, you can set it as an environment variable or directly here.
# For this example, we'll allow the user to input it in the sidebar.
st.sidebar.header("API Configuration")
api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")

# --- Model Loading ---
# Use a dictionary to map model names to their file paths
MODEL_PATHS = {
    "MobileNetV2": os.path.join("models", "waste_classifier_mobilenet.h5"),
    "EfficientNetB0": os.path.join("models", "waste_efficientnet.h5"),
    "ResNet50V2": os.path.join("models", "waste_resnet.h5")
}

# Cache the model loading to improve performance
@st.cache_resource
def load_model(model_path):
    """Loads a Keras model from the specified path."""
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please make sure the model files are in the 'models' directory.")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
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
    # The preprocessing function depends on the model.
    # For simplicity, we'll use a generic scaling here.
    # For best results, use the specific preprocess_input for each model.
    return image_array / 255.0

# --- LLM Integration for Recycling Tips ---
def get_recycling_tips(waste_category, api_key):
    """Calls the Gemini API to get recycling tips."""
    if not api_key:
        return "Please enter a Gemini API Key in the sidebar to get recycling tips."
    
    # Use the gemini-2.5-flash-preview-05-20 model
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    
    prompt = f"Provide three short, actionable, and easy-to-follow recycling tips for '{waste_category}' waste. Use bullet points."
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }]
    }
    
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status() # Raise an exception for bad status codes
        result = response.json()
        
        # Extract the text from the response
        tips = result['candidates'][0]['content']['parts'][0]['text']
        return tips
    except requests.exceptions.RequestException as e:
        return f"Error calling the API: {e}. Please check your API key and network connection."
    except (KeyError, IndexError) as e:
        return f"Error parsing API response. The response might not be in the expected format. Details: {e}"


# --- Main Application UI ---
st.title("♻️ Automated Waste Classification")
st.markdown("Leveraging Deep Learning and Generative AI to promote effective recycling.")

# Sidebar for model selection
st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Choose a classification model",
    list(MODEL_PATHS.keys())
)

# Load the selected model
model_path = MODEL_PATHS[selected_model_name]
model = load_model(model_path)

if model is not None:
    st.sidebar.success(f"Successfully loaded **{selected_model_name}**.")

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
        
        # Define class names (must match the order from your training)
        class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

        with st.spinner("Classifying..."):
            prediction = model.predict(processed_image)
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = class_names[predicted_class_index]
            confidence = np.max(prediction) * 100

        with col2:
            st.success(f"**Prediction:** {predicted_class_name.capitalize()}")
            st.info(f"**Confidence:** {confidence:.2f}%")

        # --- Display Recycling Tips ---
        st.subheader(f"Recycling Tips for {predicted_class_name.capitalize()}", divider="rainbow")
        with st.spinner("Generating tips with Gemini..."):
            tips = get_recycling_tips(predicted_class_name, api_key)
            st.markdown(tips)
else:
    st.warning("Please place your trained models in a folder named 'models' in the same directory as the app.")
