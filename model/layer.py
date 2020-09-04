from typing import Union, List

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, LayerNormalization, Activation
from tensorflow_addons.layers import InstanceNormalization


class MyConv2D(Layer):
    def __init__(
            self,
            filters: int,
            kernel_size: Union[List, int],
            strides: int = 1,
            dilation_rate: float = 1,
            padding: str = "same",
            groups: int = 1,
            apply_activation: bool = True,
            apply_norm: bool = True,
            use_IN: bool = False,
            **kwargs):
        super(MyConv2D, self).__init__(**kwargs)
        self.conv2d = Conv2D(
            filters,
            kernel_size,
            strides,
            dilation_rate=dilation_rate,
            padding=padding,
            kernel_initializer=tf.initializers.GlorotNormal(),
            use_bias=False,
            groups=groups
        )
        self.activation = ReLU()
        self.apply_activation = apply_activation
        self.apply_norm = apply_norm
        self.norm = InstanceNormalization(
            axis=3,
            center=True,
            scale=True,
            beta_initializer="random_uniform",
            gamma_initializer="random_uniform") if use_IN else BatchNormalization()

    def call(self, inputs: tf.Tensor, training: bool = False, **kwargs) -> tf.Tensor:
        x = self.conv2d(inputs)
        if self.apply_norm:
            x = self.norm(x, training=training)

        if self.apply_activation:
            x = self.activation(x)

        return x


class LightConv2D(Layer):
    def __init__(
            self,
            filters: int,
            kernel_size: Union[List, int],
            **kwargs):
        super(LightConv2D, self).__init__(**kwargs)
        self.conv1 = MyConv2D(kernel_size=1, filters=filters, apply_activation=False, apply_norm=False)
        self.conv2 = MyConv2D(kernel_size=kernel_size, filters=filters, groups=filters)

    def call(self, inputs: tf.Tensor, training: bool = False, **kwargs) -> tf.Tensor:
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        return x


class LightConvStream(Layer):
    def __init__(self, kernel_size, filters, depth):
        super(LightConvStream, self).__init__()
        self.convs = Sequential([LightConv2D(filters=filters, kernel_size=kernel_size) for _ in range(depth)])

    def call(self, inputs: tf.Tensor, training: bool = False, **kwargs):
        return self.convs(inputs, training=training)


class ChannelGate(Layer):
    def __init__(self, filters: int, num_gates: int, return_gates: bool = False, gate_activation: str = 'sigmoid',
                 reduction: int = 16, layer_norm: bool = False):
        super(ChannelGate, self).__init__()
        self.num_gates = num_gates
        self.return_gates = return_gates

        self.fc1 = MyConv2D(kernel_size=1, filters=filters // reduction, apply_norm=False, apply_activation=False)
        self.norm = LayerNormalization(axis=1, center=True, scale=True) if layer_norm else None
        self.relu = ReLU()
        self.fc2 = MyConv2D(kernel_size=1, filters=num_gates, apply_norm=False, apply_activation=False)
        self.gate_activation = Activation(gate_activation)

    def call(self, inputs: tf.Tensor, training: bool = False, **kwargs) -> tf.Tensor:
        x = inputs
        x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        x = self.fc1(x, training=training)
        if self.norm is not None:
            x = self.norm(x, training=training)

        x = self.relu(x)
        x = self.fc2(x)
        x = self.gate_activation(x)
        if self.return_gates:
            return x
        else:
            return inputs * x


class OSBlock(Layer):
    def __init__(self, filters: int, bottleneck_reduction: int = 4, use_IN: bool = False):
        super(OSBlock, self).__init__()
        self.mid_filters = filters // bottleneck_reduction
        self.conv1 = MyConv2D(kernel_size=1, filters=self.mid_filters)
        self.conv2a = LightConvStream(kernel_size=3, filters=self.mid_filters, depth=1)
        self.conv2b = LightConvStream(kernel_size=3, filters=self.mid_filters, depth=2)
        self.conv2c = LightConvStream(kernel_size=3, filters=self.mid_filters, depth=3)
        self.conv2d = LightConvStream(kernel_size=3, filters=self.mid_filters, depth=4)
        self.gate = ChannelGate(self.mid_filters, self.mid_filters)
        self.conv3 = MyConv2D(kernel_size=1, filters=filters, apply_activation=False, use_IN=use_IN)
        self.down_sample = MyConv2D(kernel_size=1, filters=filters, apply_activation=False)
        self.relu = ReLU()

    def call(self, inputs: tf.Tensor, training: bool = False, **kwargs) -> tf.Tensor:
        identity = self.down_sample(inputs, training=training)

        x1 = self.conv1(inputs, training=training)
        x2a = self.conv2a(x1, training=training)
        x2b = self.conv2b(x1, training=training)
        x2c = self.conv2c(x1, training=training)
        x2d = self.conv2d(x1, training=training)
        x2 = self.gate(x2a, training=training) + self.gate(x2b, training=training) + self.gate(x2c, training=training) + self.gate(x2d, training=training)
        x3 = self.conv3(x2, training=training)

        out = x3 + identity

        out = self.relu(out)
        return out
