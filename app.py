
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# ----------------------------
# Load your trained model
# ----------------------------
# Make sure your model file is in the same folder as app.py
model = load_model("model.h5")

# ----------------------------
# Streamlit App Layout
# ----------------------------
st.title("😊Emotion Detection App")
st.write("Upload an image and the AI will predict the emotion.")

# ----------------------------
# Image Uploader
# ----------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    img = Image.open(uploaded_file)
    
    # Convert to RGB if not already
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Resize to model input size
    img = img.resize((48, 48))
    
    # Convert image to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Display uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # ----------------------------
    # Make Prediction
    # ----------------------------
    predictions = model.predict(img_array)
    
    # Assuming your model outputs probabilities for 7 emotions
    # Replace these labels with your dataset's actual labels
    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    predicted_class = np.argmax(predictions)
    predicted_emotion = emotion_labels[predicted_class]
    
    st.write(f"Predicted Emotion: **{predicted_emotion}**")
