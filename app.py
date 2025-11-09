
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# ----------------------------
# Title and description
# ----------------------------
st.set_page_config(page_title="Emotion Detection App", page_icon="😊", layout="centered")

st.title("😊 Emotion Detection App")
st.write("Upload an image, and the model will predict the emotion displayed.")

# ----------------------------
# Load the trained model
# ----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

# ----------------------------
# Define emotion labels
# ----------------------------
emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}

# ----------------------------
# File uploader
# ----------------------------
uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert to grayscale if needed
    if img.mode != "L":
        img = img.convert("L")  # grayscale

    # Resize image to 48x48 (the model’s input size)
    img = img.resize((48, 48))

    # Convert to array
    img_array = image.img_to_array(img)

    # Ensure correct shape (48, 48, 1)
    if img_array.shape[-1] != 1:
        img_array = np.expand_dims(img_array[:, :, 0], axis=-1)

    # Expand batch dimension and normalize
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype("float32") / 255.0

    # Prediction
    with st.spinner("Analyzing emotion..."):
        predictions = model.predict(img_array)
        emotion_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

    st.success(f"**Predicted Emotion:** {emotion_labels[emotion_index]}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Developed by Davida Essen — Emotion Detection Project")
