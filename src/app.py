import gdown
import os
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from dataset import SteelMicrostructureDataset
import pandas as pd
import random
import tempfile

# Определяем переменную окружения для локальной работы
IS_LOCAL = os.getenv('IS_LOCAL', 'False').lower() in ('true', '1', 't')

# Указываем путь к модели для локального окружения и ссылку для облачного
if IS_LOCAL:
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'steel_microstructure_model.keras'))
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

    def predict_image(img_path, model):
        img = Image.open(img_path)
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

    # Обновляем путь к директории с данными
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'train'))

    if not os.path.exists(data_dir):
        st.error(f"Директория с данными не найдена: {data_dir}")
        st.stop()

    dataset = SteelMicrostructureDataset(data_dir)
    dataset.load_data()
    class_labels = get_class_labels(dataset.train_data)

    st.title("Классификация микроструктуры стали")
    st.write("Загрузите изображение для классификации его микроструктуры или выберите одно из предложенных.")

    # Получаем список всех изображений в папке datasets/valid
    valid_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'valid'))
    all_images = []
    for root, dirs, files in os.walk(valid_dir):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp')):
                all_images.append(os.path.join(root, file))

    # Выбираем случайные изображения
    num_examples = 4
    if 'example_images' not in st.session_state:
        st.session_state.example_images = random.sample(all_images, num_examples)
    example_images = st.session_state.example_images
    example_image_labels = [os.path.basename(img_path) for img_path in example_images]

    # Отображение предложенных изображений в виде превью
    cols = st.columns(num_examples)
    for i, img_path in enumerate(example_images):
        with cols[i]:
            img = Image.open(img_path)
            st.image(img, caption=os.path.basename(img_path), use_column_width=True)

    selected_example = st.selectbox("Выбрать изображение из предложенных", ["Не задано"] + example_image_labels)

    uploaded_file = st.file_uploader("Загрузить своё изображение", type=["jpg", "jpeg", "png", "bmp"])

    img_path = None

    # Добавляем флаг для отслеживания источника изображения
    if 'selected_source' not in st.session_state:
        st.session_state.selected_source = None

    # Проверяем, изменился ли источник изображения
    if selected_example != "Не задано":
        st.session_state.selected_source = "example"
        st.session_state.uploaded_file_path = None  # Очищаем загруженный файл
        img_path = example_images[example_image_labels.index(selected_example)]
    elif uploaded_file is not None:
        st.session_state.selected_source = "uploaded"
        st.session_state.selected_example = "Не задано"  # Очищаем выбранное изображение из списка
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.read())
            st.session_state.uploaded_file_path = temp_file.name
        img_path = st.session_state.uploaded_file_path

    # Отображаем изображение, если оно выбрано или загружено
    if img_path:
        st.image(img_path, caption='Выбранное изображение', use_column_width=True)

    if img_path:
        try:
            predictions = predict_image(img_path, model)
            predictions = predictions[0]  # Убираем дополнительное измерение

            # Подготовка данных для таблицы
            data = {
                "Класс": class_labels,
                "Уверенность": [f"{confidence:.2f}" for confidence in predictions]
            }
            df = pd.DataFrame(data)  # Исправлено использование pandas

            # Вывод данных в виде таблицы
            st.table(df)

            # Предсказанный класс и уверенность для него
            predicted_class = class_labels[np.argmax(predictions)]
            confidence = np.max(predictions)

            st.write(f"Предсказанный класс: {predicted_class}")
            st.write(f"Уверенность: {confidence:.2f}")
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")