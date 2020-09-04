from typing import Union, List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, LayerNormalization, Activation
from tensorflow_addons.layers import InstanceNormalization


class CosineSimilarity(Layer):
    """
    Cosine similarity with classwise weights
    """

    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight(shape=(input_dim, self.num_classes),
                                 initializer='random_normal',
                                 trainable=True, name="weight")

    def call(self, inputs):
        x = tf.nn.l2_normalize(inputs, axis=-1)  # (batch_size, ndim)
        w = tf.nn.l2_normalize(self.W, axis=0)  # (ndim, nclass)
        cos = tf.matmul(x, w)  # (batch_size, nclass)
        return cos


class ArcFace(Layer):
    """
    Implementation of https://arxiv.org/pdf/1801.07698.pdf
    """

    def __init__(self, num_classes, margin=0.5, scale=64, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        self.cos_similarity = CosineSimilarity(num_classes)

    def call(self, inputs, training):
        # If not training (prediction), labels are ignored
        feature, labels = inputs
        cos = self.cos_similarity(feature)

        if training:
            theta = tf.acos(tf.clip_by_value(cos, -1, 1))
            cos_add = tf.cos(theta + self.margin)

            mask = tf.cast(labels, dtype=cos_add.dtype)
            logits = mask * cos_add + (1 - mask) * cos
            logits *= self.scale
            return logits
        else:
            return cos


class AdaCos(Layer):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

        self.cos_similarity = CosineSimilarity(num_classes)
        self.scale = tf.Variable(tf.sqrt(2.0) * tf.math.log(num_classes - 1.0),
                                 trainable=False)

    def call(self, inputs, training):
        # In inference, labels are ignored
        feature, labels = inputs
        cos = self.cos_similarity(feature)

        if training:
            mask = tf.cast(labels, dtype=cos.dtype)

            # Collect cosine values at only false labels
            B = (1 - mask) * tf.exp(self.scale * cos)
            B_avg = tf.reduce_mean(tf.reduce_sum(B, axis=-1), axis=0)

            theta = tf.acos(tf.clip_by_value(cos, -1, 1))
            # Collect cosine at true labels
            theta_true = tf.reduce_sum(mask * theta, axis=-1)
            # get median (=50-percentile)
            theta_med = tfp.stats.percentile(theta_true, q=50)

            scale = tf.math.log(B_avg) / tf.cos(tf.minimum(np.pi / 4, theta_med))
            scale = tf.stop_gradient(scale)
            logits = scale * cos

            self.scale.assign(scale)
            return logits
        else:
            return cos


class CircleLoss(Layer):
    """
    Implementation of https://arxiv.org/abs/2002.10857 (pair-level label version)
    """

    def __init__(self, margin=0.25, scale=256, **kwargs):
        """
        Args
          margin: a float value, margin for the true label (default 0.25)
          scale: a float value, final scale value,
            stated as gamma in the original paper (default 256)
        Returns:
          a tf.keras.layers.Layer object, outputs logit values of each class
        In the original paper, margin and scale (=gamma) are set depends on tasks
        - Face recognition: m=0.25, scale=256 (default)
        - Person re-identification: m=0.25, scale=256
        - Fine-grained image retrieval: m=0.4, scale=64
        """
        super().__init__(**kwargs)
        self.margin = margin
        self.scale = scale

        self._Op = 1 + margin  # O_positive
        self._On = -margin  # O_negative
        self._Dp = 1 - margin  # Delta_positive
        self._Dn = margin  # Delta_negative

    def call(self, inputs, training):
        feature, labels = inputs
        x = tf.nn.l2_normalize(feature, axis=-1)
        cos = tf.matmul(x, x, transpose_b=True)  # (batch_size, batch_size)

        if training:
            # pairwise version
            mask = tf.cast(labels, dtype=cos.dtype)
            mask_p = tf.matmul(mask, mask, transpose_b=True)
            mask_n = 1 - mask_p
            mask_p = mask_p - tf.eye(mask_p.shape[0])

            logits_p = - self.scale * tf.nn.relu(self._Op - cos) * (cos - self._Dp)
            logits_n = self.scale * tf.nn.relu(cos - self._On) * (cos - self._Dn)

            logits_p = tf.where(mask_p == 1, logits_p, -np.inf)
            logits_n = tf.where(mask_n == 1, logits_n, -np.inf)

            logsumexp_p = tf.reduce_logsumexp(logits_p, axis=-1)
            logsumexp_n = tf.reduce_logsumexp(logits_n, axis=-1)

            mask_p_row = tf.reduce_max(mask_p, axis=-1)
            mask_n_row = tf.reduce_max(mask_n, axis=-1)
            logsumexp_p = tf.where(mask_p_row == 1, logsumexp_p, 0)
            logsumexp_n = tf.where(mask_n_row == 1, logsumexp_n, 0)

            losses = tf.nn.softplus(logsumexp_p + logsumexp_n)

            mask_paired = mask_p_row * mask_n_row
            losses = mask_paired * losses
            return losses
        else:
            return cos


class CircleLossCL(Layer):
    """
    Implementation of https://arxiv.org/abs/2002.10857 (class-level label version)
    """

    def __init__(self, num_classes, margin=0.25, scale=256, **kwargs):
        """
        Args
          num_classes: an int value, number of target classes
          margin: a float value, margin for the true label (default 0.25)
          scale: a float value, final scale value,
            stated as gamma in the original paper (default 256)
        Returns:
          a tf.keras.layers.Layer object, outputs logit values of each class
        In the original paper, margin and scale (=gamma) are set depends on tasks
        - Face recognition: m=0.25, scale=256 (default)
        - Person re-identification: m=0.25, scale=256
        - Fine-grained image retrieval: m=0.4, scale=64
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        self._Op = 1 + margin  # O_positive
        self._On = -margin  # O_negative
        self._Dp = 1 - margin  # Delta_positive
        self._Dn = margin  # Delta_negative

        self.cos_similarity = CosineSimilarity(num_classes)

    def call(self, inputs, training):
        feature, labels = inputs
        cos = self.cos_similarity(feature)

        if training:
            # class-lebel version
            mask = tf.cast(labels, dtype=cos.dtype)

            alpha_p = tf.nn.relu(self._Op - cos)
            alpha_n = tf.nn.relu(cos - self._On)

            logits_p = self.scale * alpha_p * (cos - self._Dp)
            logits_n = self.scale * alpha_n * (cos - self._Dn)

            logits = mask * logits_p + (1 - mask) * logits_n
            return logits
        else:
            return cos


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
        x2 = self.gate(x2a, training=training) + self.gate(x2b, training=training) + self.gate(x2c,
                                                                                               training=training) + self.gate(
            x2d, training=training)
        x3 = self.conv3(x2, training=training)

        out = x3 + identity

        out = self.relu(out)
        return out
