import sys
import os
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Добавление пути к src, чтобы модули могли быть импортированы
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from dataset import SteelMicrostructureDataset
from model import SteelMicrostructureModel

# Путь к модели
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results/steel_microstructure_model.keras'))

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
else:
    model = load_model(model_path, compile=False)
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

    def get_class_labels(train_data):
        return list(train_data.class_indices.keys())

    # Путь к данным
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'datasets/train'))
    dataset = SteelMicrostructureDataset(data_dir)
    dataset.load_data()
    class_labels = get_class_labels(dataset.train_data)

    st.title("Steel Microstructure Classification")
    st.write("Upload an image to classify its microstructure.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        predictions = predict_image(img, model)
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = np.max(predictions)

        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}")
