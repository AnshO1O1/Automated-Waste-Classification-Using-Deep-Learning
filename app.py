import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ------------------------
# Path to your trained model
# ------------------------
MODEL_PATH = "waste_classifier_mobilenet.h5"

# ------------------------
# Load Model
# ------------------------
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please upload/keep it in the same folder.")
        return None
    try:
        model = tf.keras.models.load_model(model_path, compile=False)  # load full model
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model(MODEL_PATH)

# ------------------------
# Class Names (adjust these to your dataset)
# ------------------------
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# ------------------------
# Image Preprocessing
# ------------------------
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")  # ensure 3 channels
    image = image.resize((224, 224))  # MobileNet input size
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# ------------------------
# Prediction Function
# ------------------------
def predict(image: Image.Image):
    if model is None:
        return None, None

    processed = preprocess_image(image)
    preds = model.predict(processed)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(tf.nn.softmax(preds)))
    return CLASS_NAMES[class_idx], confidence

# ------------------------
# Streamlit UI
# ------------------------
st.title("♻️ Waste Classifier using MobileNet")
st.write("Upload an image of waste, and the model will classify it into categories.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if model is not None:
        label, confidence = predict(image)
        if label:
            st.success(f"✅ Predicted: **{label}** with confidence {confidence:.2f}")
    else:
        st.error("Model could not be loaded. Please check your `.h5` file.")
