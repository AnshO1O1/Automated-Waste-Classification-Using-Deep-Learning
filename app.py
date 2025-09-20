import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# ------------------------
# Model settings
# ------------------------
MODEL_PATH = "waste_classifier_mobilenet.h5"
NUM_CLASSES = 6
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# ------------------------
# Build MobileNetV2 architecture
# ------------------------
@st.cache_resource
def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224,224,3),
        include_top=False,
        weights="imagenet"  # Use pretrained weights
    )
    base_model.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    return model

# ------------------------
# Load model weights
# ------------------------
@st.cache_resource
def load_model_weights(model_path):
    model = build_model()
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please place the .h5 file in the folder.")
        return None

    try:
        model.load_weights(model_path)
        st.success("✅ Weights loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Could not load weights: {e}")
        return None

model = load_model_weights(MODEL_PATH)

# ------------------------
# Image preprocessing
# ------------------------
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((224,224))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# ------------------------
# Prediction function
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
# Streamlit UI
# ------------------------
st.title("♻️ Waste Classifier")
st.write("Upload an image of waste, and the model will classify it.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    label, conf = predict(image)
    if label:
        st.success(f"Predicted: **{label}** ({conf*100:.2f}% confidence)")
    else:
        st.error("Prediction not available. Check model weights.")
