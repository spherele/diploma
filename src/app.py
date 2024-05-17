import gdown
import os
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from dataset import SteelMicrostructureDataset
from model import SteelMicrostructureModel

# Определяем переменную окружения для локальной работы
IS_LOCAL = os.getenv('IS_LOCAL', 'False').lower() in ('true', '1', 't')

# Указываем путь к модели для локального окружения и ссылку для облачного
if IS_LOCAL:
    model_path = os.path.abspath('../results/steel_microstructure_model.keras')
else:
    url = 'https://drive.google.com/uc?id=1-e9pdmzMP2-cE7NcV1yplRYPEpwcaZfs'
    output = 'steel_microstructure_model.keras'
    model_path = os.path.abspath(output)

    # Загрузка файла модели, если он не существует
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

# Проверка существования файла модели
if not os.path.exists(model_path):
    st.error(f"Файл модели не найден: {model_path}")
else:
    try:
        model = load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        st.stop()

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

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/train'))

    if not os.path.exists(data_dir):
        st.error(f"Директория с данными не найдена: {data_dir}")
        st.stop()

    dataset = SteelMicrostructureDataset(data_dir)
    dataset.load_data()
    class_labels = get_class_labels(dataset.train_data)

    st.title("Классификация микроструктуры стали")
    st.write("Загрузите изображение для классификации его микроструктуры.")

    uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Загруженное изображение', use_column_width=True)

        predictions = predict_image(img, model)
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = np.max(predictions)

        st.write(f"Предсказанный класс: {predicted_class}")
        st.write(f"Уверенность: {confidence:.2f}")
