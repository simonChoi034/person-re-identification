import math
from typing import Dict

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, Flatten, Dropout


class Backbone(tf.keras.layers.Layer):
    def __init__(self, model: str = "EfficientNetB0", freeze: bool = False, use_pretrain: bool = True):
        super(Backbone, self).__init__()

        weights = "imagenet" if use_pretrain else None

        assert model in dir(tf.keras.applications)
        self.backbone_model = getattr(tf.keras.applications, model)(include_top=False, input_shape=(None, None, 3),
                                                                    weights=weights)

        # freeze model for transfer learning
        if freeze:
            self.backbone_model.trainable = False

    def call(self, inputs: tf.Tensor, training: bool = None, **kwargs: Dict) -> tf.Tensor:
        x = self.backbone_model(inputs, training=training)
        return x


class OutputLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_shape: int, w_decay: float = 5e-4):
        super(OutputLayer, self).__init__()
        self.conv_batch_norm = BatchNormalization()
        self.dense_batch_norm = BatchNormalization()
        self.dropout = Dropout(rate=0.5)
        self.flatten = Flatten()
        self.dense = Dense(embedding_shape, kernel_regularizer=tf.keras.regularizers.l2(w_decay))

    def call(self, inputs: tf.Tensor, training: bool = None, **kwargs: Dict) -> tf.Tensor:
        x = self.conv_batch_norm(inputs, training=training)
        x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.dense(x, training=training)
        x = self.dense_batch_norm(x, training=training)
        return x


class ArcHead(tf.keras.layers.Layer):
    """ArcMarginPenaltyLogists"""

    def __init__(self, num_classes: int, margin: float = 0.5, logist_scale: int = 64, **kwargs: Dict):
        super(ArcHead, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale

    def build(self, input_shape):
        self.w = self.add_variable(
            "weights", shape=[int(input_shape[-1]), self.num_classes])
        self.cos_m = tf.identity(math.cos(self.margin), name='cos_m')
        self.sin_m = tf.identity(math.sin(self.margin), name='sin_m')
        self.th = tf.identity(math.cos(math.pi - self.margin), name='th')
        self.mm = tf.multiply(self.sin_m, self.margin, name='mm')

    def call(self, embds: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
        sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')

        cos_mt = tf.subtract(
            cos_t * self.cos_m, sin_t * self.sin_m, name='cos_mt')

        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)

        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                          name='one_hot_mask')

        logists = tf.where(mask == 1., cos_mt, cos_t)
        logists = tf.multiply(logists, self.logist_scale, 'arcface_logist')

        return logists


class ArcPersonModel(tf.keras.Model):
    def __init__(self, num_classes: int, margin: float = 0.5, logist_scale: int = 64, embd_shape: int = 512,
                 backbone: str = 'EfficientNetB0',
                 w_decay: float = 5e-4, use_pretrain: bool = True, freeze_backbone: bool = False):
        super(ArcPersonModel, self).__init__()
        self.backbone = Backbone(model=backbone, use_pretrain=use_pretrain, freeze=freeze_backbone)
        self.embedding = OutputLayer(embd_shape, w_decay)
        self.archead = ArcHead(num_classes=num_classes, margin=margin, logist_scale=logist_scale)

    def call(self, inputs: tf.Tensor, training: bool = None, mask: bool = None, **kwargs: Dict) -> tf.Tensor:
        x = self.backbone(inputs)
        embeddings = self.embedding(x)

        if training:
            labels = kwargs['labels']
            logist = self.archead(embds=embeddings, labels=labels)
            return logist
        else:
            return embeddings

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(inputs=x, labels=y, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(inputs=x, labels=y, training=True)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}