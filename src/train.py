import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__)))

from dataset import SteelMicrostructureDataset
from model import SteelMicrostructureModel


def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/train'))
    dataset = SteelMicrostructureDataset(data_dir)
    dataset.load_data()

    # Добавьте отладочный вывод
    print(f"Number of training samples: {dataset.train_data.samples}")
    print(f"Number of validation samples: {dataset.val_data.samples}")

    input_shape = (dataset.img_height, dataset.img_width, 3)
    num_classes = len(dataset.train_data.class_indices)

    model = SteelMicrostructureModel(input_shape, num_classes)
    model.train(dataset.train_data, dataset.val_data)

    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results'))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    model.save_model(os.path.join(results_dir, 'steel_microstructure_model.keras'))


if __name__ == '__main__':
    main()
