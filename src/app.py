import os
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import subprocess

# Define paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets'))
train_dir = os.path.join(base_dir, 'train')
model_save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results/steel_microstructure_model.keras'))

# Create directories if they do not exist
os.makedirs(train_dir, exist_ok=True)

# Create subdirectories based on the screenshot
categories = ["Crazing", "Inclusion", "Patches", "Pitted", "Rolled", "Scratches"]
for category in categories:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)

st.title("Steel Microstructure Classification")
st.write("Upload a model file and an image to classify its microstructure.")


# Step 2: Upload images to categories
st.write("Upload images to categories for training the model.")
category = st.selectbox("Select category", categories)
uploaded_image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"], key="upload_image")

if uploaded_image_file is not None and category is not None:
    img_path = os.path.join(train_dir, category, uploaded_image_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_image_file.getbuffer())
    st.success(f"Image saved to: {img_path}")

# Step 3: Train the model
if st.button("Train Model"):
    if os.path.exists(model_save_path):
        os.remove(model_save_path)  # Remove old model file if exists
    st.write("Training the model...")
    result = subprocess.run(["python", "train.py"], capture_output=True, text=True)
    st.write(result.stdout)
    st.write("Model training completed.")

# Step 4: Analyze images if model exists
if os.path.exists(model_save_path):
    model = load_model(model_save_path, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def predict_image(img, model):
        img = img.resize((150, 150))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        predictions = model.predict(img_array)
        return predictions

    def get_class_labels(train_data_dir):
        return [d for d in os.listdir(train_data_dir) if os.path.isdir(os.path.join(train_data_dir, d))]

    class_labels = get_class_labels(train_dir)

    st.write("Analyze uploaded image.")
    analyze_image_file = st.file_uploader("Choose an image to analyze...", type=["jpg", "jpeg", "png", "bmp"], key="analyze_image")

    if analyze_image_file is not None:
        img = Image.open(analyze_image_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        predictions = predict_image(img, model)
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = np.max(predictions)

        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}")
else:
    st.error(f"Model file not found: {model_save_path}")
