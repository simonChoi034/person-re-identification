import math
from typing import Dict

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, Flatten, Dropout
from model.layer import AdaCos, ArcFace, CircleLossCL

from model.backbone.osnet import OSNet


class BaseModel(tf.keras.Model):
    def __init__(self, embedding_shape: int, w_decay: float = 5e-4, model: str = "OSNet",
                 freeze_backbone: bool = False, use_pretrain: bool = True):
        super(BaseModel, self).__init__()

        weights = "imagenet" if use_pretrain else None

        if model in dir(tf.keras.applications):
            self.backbone_model = getattr(tf.keras.applications, model)(include_top=False, weights=weights)
        elif model.lower() == "osnet":
            self.backbone_model = OSNet(layers=[2, 2, 2], filters=[64, 256, 384, 512])

        # freeze model for transfer learning
        self.backbone_model.trainable = False if freeze_backbone else True

        self.fc = FCLayer(embedding_shape=embedding_shape, w_decay=w_decay)

    def call(self, inputs: tf.Tensor, training: bool = None, **kwargs: Dict) -> tf.Tensor:
        x = self.backbone_model(inputs, training=training)
        x = self.fc(x, training=training)
        return x


class FCLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_shape: int, w_decay: float = 5e-4):
        super(FCLayer, self).__init__()
        self.conv_batch_norm = BatchNormalization(axis=-1,
                                                  scale=True,
                                                  momentum=0.9,
                                                  epsilon=2e-5,
                                                  renorm=True,
                                                  renorm_clipping={'rmax': 3,
                                                                   'rmin': 0.3333,
                                                                   'dmax': 5},
                                                  renorm_momentum=0.9,
                                                  beta_regularizer=tf.keras.regularizers.l2(
                                                      l=5e-4),
                                                  gamma_regularizer=tf.keras.regularizers.l2(
                                                      l=5e-4),
                                                  gamma_initializer='ones',
                                                  name='bn1')
        self.dense_batch_norm = BatchNormalization(axis=-1,
                                                   scale=False,
                                                   momentum=0.9,
                                                   epsilon=2e-5,
                                                   renorm=True,
                                                   renorm_clipping={'rmax': 3,
                                                                    'rmin': 0.3333,
                                                                    'dmax': 5},
                                                   renorm_momentum=0.9,
                                                   beta_regularizer=tf.keras.regularizers.l2(
                                                       l=5e-4),
                                                   name='fc1')
        self.dropout = Dropout(rate=0.5)
        self.flatten = Flatten()
        self.dense = Dense(embedding_shape, kernel_regularizer=tf.keras.regularizers.l2(w_decay),
                           bias_regularizer=tf.keras.regularizers.l2(
                               l=5e-4), kernel_initializer='glorot_normal')

    def call(self, inputs: tf.Tensor, training: bool = None, **kwargs: Dict) -> tf.Tensor:
        x = self.conv_batch_norm(inputs, training=training)
        x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.dense(x, training=training)
        x = self.dense_batch_norm(x, training=training)
        return x


class ReIDModel(tf.keras.Model):
    def __init__(
            self,
            num_classes: int,
            margin: float = 0.25,
            logist_scale: int = 256,
            embd_shape: int = 512,
            backbone: str = 'OSNet',
            w_decay: float = 5e-4,
            use_pretrain: bool = True,
            freeze_backbone: bool = False,
            loss="circle_loss"):
        super(ReIDModel, self).__init__()
        self.num_classes = num_classes
        self.base_model = BaseModel(embd_shape, w_decay=w_decay, model=backbone,
                                    use_pretrain=use_pretrain, freeze_backbone=freeze_backbone)
        if loss.lower() == "circle_loss":
            self.head = CircleLossCL(num_classes=num_classes, margin=margin, scale=logist_scale, name="circle_loss")
        elif loss.lower() == "ada_cos":
            self.head = AdaCos(num_classes=num_classes, name="AdaCos")
        else:
            self.head = ArcFace(num_classes=num_classes, margin=margin, scale=logist_scale, name="ArcFace")

    def call(self, inputs: tf.Tensor, training: bool = None, mask: bool = None, **kwargs: Dict) -> tf.Tensor:
        embedding = self.base_model(inputs, training=training)
        return embedding

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        one_hot_label = tf.one_hot(y, depth=self.num_classes)

        with tf.GradientTape() as tape:
            embedding = self(inputs=x, training=True)  # Forward pass
            y_pred = self.head((embedding, one_hot_label), training=True)
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(one_hot_label, y_pred, regularization_losses=self.losses)
            reg_loss = tf.reduce_sum(self.losses)
            total_loss = loss + reg_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(one_hot_label, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        x, y = data
        one_hot_label = tf.one_hot(y, self.num_classes)
        # Compute predictions
        embedding = self(inputs=x, training=True)  # Forward pass
        y_pred = self.head((embedding, one_hot_label), training=True)
        # Updates the metrics tracking the loss
        self.compiled_loss(one_hot_label, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(one_hot_label, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
