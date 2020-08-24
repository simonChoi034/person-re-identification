import argparse

import tensorflow as tf
from scipy.spatial.distance import cosine

from config import cfg

model = tf.keras.models.load_model('./saved_model/{}'.format(cfg.backbone))


def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_with_pad(img, cfg.image_size[0], cfg.image_size[1])
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = (img - 0.5) * 2
    return tf.expand_dims(img, 0)


def get_embedding(img):
    embedding = model.predict(img)
    return embedding


def get_score(embedding_1, embedding_2):
    distance = cosine(embedding_1, embedding_2)
    return 1 - distance


def main(img_1, img_2):
    img_1 = load_img(img_1)
    img_2 = load_img(img_2)

    embedding_1 = get_embedding(img_1)
    embedding_2 = get_embedding(img_2)

    score = get_score(embedding_1, embedding_2)

    print(score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train detection model')
    parser.add_argument('--img_1', type=str)
    parser.add_argument('--img_2', type=str)
    args = parser.parse_args()

    main(args.img_1, args.img_2)