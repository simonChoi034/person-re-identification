from typing import List

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D

from model.layer import MyConv2D, OSBlock


class OSNet(tf.keras.Model):
    def __init__(self, layers: List[int], filters: List[int]):
        super(OSNet, self).__init__()

        self.conv1 = MyConv2D(kernel_size=7, filters=filters[0], strides=2)
        self.max_pool = MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.conv2 = Sequential([OSBlock(filters=filters[1]) for _ in range(layers[0])])
        self.reduction1 = Sequential([MyConv2D(kernel_size=1, filters=filters[1]), AveragePooling2D(2, strides=2)])
        self.conv3 = Sequential([OSBlock(filters=filters[2]) for _ in range(layers[1])])
        self.reduction2 = Sequential([MyConv2D(kernel_size=1, filters=filters[2]), AveragePooling2D(2, strides=2)])
        self.conv4 = Sequential([OSBlock(filters=filters[3]) for _ in range(layers[2])])
        self.conv5 = MyConv2D(kernel_size=1, filters=filters[3])

    def call(self, inputs: tf.Tensor, training: bool = False, **kwargs) -> tf.Tensor:
        x = self.conv1(inputs, training=training)
        x = self.max_pool(x, training=training)
        x = self.conv2(x, training=training)
        x = self.reduction1(x, training=training)
        x = self.conv3(x, training=training)
        x = self.reduction2(x, training=training)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)

        return x