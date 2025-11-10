
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ==============================
# DATABASE SETUP
# ==============================
# Create a folder for storing uploaded images
if not os.path.exists("user_images"):
    os.makedirs("user_images")

# Database configuration
DATABASE_NAME = "emotion_users.db"
engine = create_engine(f"sqlite:///{DATABASE_NAME}")
Base = declarative_base()

# Table definition
class UserData(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    image_path = Column(String(100))
    emotion_result = Column(String(50))
    mode = Column(String(10))  # 'online' or 'offline'

# Create the table if it doesn’t exist
Base.metadata.create_all(engine)

# Setup database session
Session = sessionmaker(bind=engine)
session = Session()

# Function to save user data to database
def save_user(name, image_path, emotion_result, mode):
    new_user = UserData(
        name=name,
        image_path=image_path,
        emotion_result=emotion_result,
        mode=mode
    )
    session.add(new_user)
    session.commit()
    print("✅ User data saved successfully!")

# ==============================
# MODEL SETUP
# ==============================
# Load the trained emotion detection model
model = load_model("emotion_model.h5")

# Define emotion labels (modify to match your dataset)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="Emotion Detection App", page_icon="😊", layout="centered")
st.title("😊 Emotion Detection App")
st.write("Upload an image or take a picture to detect the emotion and save the result to the database.")

# User inputs
name = st.text_input("Enter your name:")
mode = st.radio("Select mode:", ["online", "offline"])

st.write("You can either upload an image or take a picture with your webcam:")

# Option 1: Upload image
uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])

# Option 2: Take picture with camera
camera_image = st.camera_input("Take a picture with your webcam:")

# Choose which image to use
if uploaded_file is not None:
    img = Image.open(uploaded_file)
elif camera_image is not None:
    img = Image.open(camera_image)
else:
    img = None

# Proceed if an image is provided
if img is not None:
    st.image(img, caption="Selected Image", use_column_width=True)

    if img.mode != "RGB":
        img = img.convert("RGB")

    # Preprocess image for model
    img_resized = img.resize((48, 48))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict emotion
    predictions = model.predict(img_array)
    predicted_label = emotion_labels[np.argmax(predictions)]
    st.write(f"### 😃 Predicted Emotion: **{predicted_label}**")

    # Save to database
    if st.button("💾 Save to Database"):
        if name.strip() == "":
            st.warning("Please enter your name before saving.")
        else:
            # Determine filename for saving
            if uploaded_file is not None:
                filename = uploaded_file.name
            else:
                filename = "camera_image.png"

            image_path = os.path.join("user_images", filename)
            img.save(image_path)

            save_user(name, image_path, predicted_label, mode)
            st.success("✅ Data saved successfully!")

# ==============================
# OPTIONAL: VIEW DATABASE LINK
# ==============================
st.write("---")
st.subheader("📁 Database Information")
st.text("Database Name: emotion_users.db")
st.text("Stored Path: ./emotion_users.db")
st.text("Images Folder: ./user_images/")
