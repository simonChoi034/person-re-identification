import argparse


import numpy as np
import tensorflow as tf

from config import cfg
from scipy.spatial.distance import cosine, euclidean

model = tf.keras.models.load_model('./saved_model/{}'.format(cfg.backbone))

def tf_imread(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (256, 128))
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = (img - 0.5) * 2
    return tf.expand_dims(img, 0)


person_A = tf_imread('/home/simon/Downloads/Market-1501-v15.09.15/bounding_box_test/-1_c1s1_022176_02.jpg')
person_B = tf_imread('/home/simon/Downloads/Market-1501-v15.09.15/bounding_box_test/-1_c1s1_031226_01.jpg')

persons = tf.concat([person_A, person_B], axis=0)
A = model(persons)

emb_a = A[0]
emb_b = A[1]

emb_a /= np.linalg.norm(emb_a)
emb_b /= np.linalg.norm(emb_b)

print(cosine(emb_a, emb_b))

embedding_score = float(np.clip(1 - cosine(emb_a, emb_b), a_min=0, a_max=1))