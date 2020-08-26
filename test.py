import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D

from config import cfg
from dataset.dataset import Dataset, DatasetGenerator

model = tf.keras.models.load_model('./saved_model/{}'.format(cfg.backbone))


def main(dataset_path):
    dataset_generator = DatasetGenerator(dataset_path=dataset_path)
    dataset_train = Dataset(generator=dataset_generator, image_size=cfg.image_size, batch_size=cfg.batch_size,
                            buffer_size=cfg.buffer_size, prefetch_size=cfg.prefetch_size,
                            mode="train").create_dataset()

    embeddings = None
    true_label = None
    for x, y in dataset_train:
        predictions = model.predict(x)
        if embeddings is None:
            embeddings = predictions
            true_label = y
        else:
            embeddings = np.concatenate([embeddings, predictions], axis=1)
            true_label = np.concatenate([true_label, y], axis=0)

    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    # plot
    fig2 = plt.figure()
    ax2 = Axes3D(fig2)
    for c in range(len(np.unique(true_label))):
        ax2.plot(embeddings[true_label == c, 0], embeddings[true_label == c, 1], embeddings[true_label == c, 2],
                 '.', alpha=0.1)
    plt.title('ArcFace')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train detection model')
    parser.add_argument('--dataset_path', type=str)
    args = parser.parse_args()

    main(args.dataset_path)
