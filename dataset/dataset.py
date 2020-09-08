import glob
import os
import random
from typing import List, Tuple, Any

import tensorflow as tf


class LPWDatasetGenerator:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.image_paths = glob.glob(os.path.join(dataset_path, "*/*/*/*.jpg"))
        self.dataset_size = len(self.image_paths)
        self.train_ratio = 0.8
        random.shuffle(self.image_paths)
        self.set_labels()

    def set_labels(self):
        image_list = list(map(lambda x: x.replace(self.dataset_path, ""), self.image_paths))
        label_list = list(map(lambda x: x.split("/")[-4:], image_list))
        label_list = sorted(list(set(map(lambda x: x[0] + "_" + x[2], label_list))))
        self.num_classes = len(label_list)
        self.label_mapping = {key: i for i, key in enumerate(label_list)}

    def get_num_classes(self) -> int:
        return self.num_classes

    def get_label(self, img_path: str) -> int:
        img_path = img_path.replace(self.dataset_path, "")
        img_path = img_path.split("/")[-4:]
        label_key = img_path[0] + "_" + img_path[2]
        return self.label_mapping[label_key]

    def gen_next_pair_val(self):
        index = int(self.dataset_size * self.train_ratio)
        for img_path in self.image_paths[index:]:
            label = self.get_label(img_path)

            yield img_path, label

    def gen_next_pair_train(self):
        index = int(self.dataset_size * self.train_ratio)
        for img_path in self.image_paths[:index]:
            label = self.get_label(img_path)

            yield img_path, label


class Dataset:
    def __init__(self, generator: LPWDatasetGenerator, image_size: List, batch_size: int, buffer_size: int,
                 prefetch_size: int, mode="train"):
        self.generator = generator
        self.image_size = image_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.prefetch_size = prefetch_size
        self.mode = mode

    def preprocess_image(self, img_path: str) -> tf.Tensor:
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize_with_pad(img, self.image_size[0], self.image_size[1])
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.4)
        # img = tf.image.random_saturation(img, 0.6, 1.4)
        img = img / 127.5 - 1
        return img

    def preprocess_data(self, img_path, label) -> Tuple[tf.Tensor, Any]:
        img = self.preprocess_image(img_path)

        return img, label

    def create_dataset(self) -> tf.data.Dataset:
        gen_fn = self.generator.gen_next_pair_train if self.mode == "train" else self.generator.gen_next_pair_val
        dataset = tf.data.Dataset.from_generator(
            gen_fn,
            output_types=(tf.string, tf.int32)
        )
        dataset = dataset.shuffle(buffer_size=self.buffer_size)
        dataset = dataset.map(map_func=self.preprocess_data)
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.prefetch(buffer_size=self.prefetch_size)

        return dataset
