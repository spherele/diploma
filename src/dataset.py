import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image


class SteelMicrostructureDataset:
    def __init__(self, data_dir, img_height=150, img_width=150, batch_size=32):
        self.data_dir = data_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.train_data = None
        self.val_data = None

    def load_data(self):
        datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)

        def check_image(file_path):
            try:
                img = Image.open(file_path)
                img.verify()  # Проверка, что изображение не повреждено
                return True
            except (IOError, SyntaxError):
                return False

        valid_images = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if check_image(file_path):
                    valid_images.append(file_path)
                else:
                    print(f"Invalid image file: {file_path}")

        valid_data_dir = 'valid_data'
        if not os.path.exists(valid_data_dir):
            os.makedirs(valid_data_dir)

        for file_path in valid_images:
            rel_path = os.path.relpath(file_path, self.data_dir)
            dest_path = os.path.join(valid_data_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            if not os.path.exists(dest_path):
                os.symlink(file_path, dest_path)

        # Проверка, что valid_data_dir не пустой
        if not os.listdir(valid_data_dir):
            raise ValueError("No valid images found in the dataset directory.")

        self.train_data = datagen.flow_from_directory(
            valid_data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )

        self.val_data = datagen.flow_from_directory(
            valid_data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )

        # Дополнительная проверка количества образцов
        if self.train_data.samples == 0 or self.val_data.samples == 0:
            raise ValueError("No training or validation images found after processing.")
