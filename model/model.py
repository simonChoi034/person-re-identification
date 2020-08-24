import math
from typing import Dict

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, Flatten, Dropout


class BaseModel(tf.keras.layers.Layer):
    def __init__(self, embedding_shape: int, w_decay: float = 5e-4, model: str = "EfficientNetB0",
                 freeze_backbone: bool = False, use_pretrain: bool = True):
        super(BaseModel, self).__init__()

        weights = "imagenet" if use_pretrain else None

        assert model in dir(tf.keras.applications)
        self.backbone_model = getattr(tf.keras.applications, model)(include_top=False, weights=weights)

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
                                                  gamma_regularizer=tf.keras.regularizers.l2(
                                                      l=5e-4),
                                                  name='bn1')
        self.dense_batch_norm = BatchNormalization(axis=-1,
                                                   scale=False,
                                                   momentum=0.9,
                                                   epsilon=2e-5,
                                                   name='fc1')
        self.dropout = Dropout(rate=0.4)
        self.flatten = Flatten()
        self.dense = Dense(embedding_shape, kernel_regularizer=tf.keras.regularizers.l2(w_decay),
                           kernel_initializer='glorot_normal')

    def call(self, inputs: tf.Tensor, training: bool = None, **kwargs: Dict) -> tf.Tensor:
        x = self.conv_batch_norm(inputs, training=training)
        x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.dense(x, training=training)
        x = self.dense_batch_norm(x, training=training)
        return x


class ArcHead(tf.keras.layers.Layer):
    """ArcMarginPenaltyLogists"""

    def __init__(self, num_classes: int, margin: float = 0.5, logist_scale: int = 30, **kwargs: Dict):
        super(ArcHead, self).__init__(**kwargs)
        self.output_dim = num_classes
        self.margin = margin
        self.logist_scale = logist_scale

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1],
                                             self.output_dim),
                                      initializer='glorot_normal',
                                      regularizer=tf.keras.regularizers.l2(
                                          l=5e-4),
                                      trainable=True)
        super(ArcHead, self).build(input_shape)

    def call(self, embedding: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        cos_m = math.cos(self.margin)
        sin_m = math.sin(self.margin)
        mm = sin_m * self.margin  # issue 1
        threshold = math.cos(math.pi - self.margin)
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / embedding_norm
        weights_norm = tf.norm(self.kernel, axis=0, keepdims=True)
        weights = self.kernel / weights_norm
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = self.logist_scale * tf.subtract(tf.multiply(cos_t, cos_m),
                                                 tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = self.logist_scale * (cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = tf.one_hot(labels, depth=self.output_dim, name='one_hot_mask')
        # mask = tf.squeeze(mask, 1)
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        s_cos_t = tf.multiply(tf.cast(self.logist_scale, dtype=tf.float32), cos_t, name='scalar_cos_t')

        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(
            cos_mt_temp, mask), name='arcface_loss_output')

        return output


class NormHead(tf.keras.layers.Layer):
    def __init__(self, num_classes: int, w_decay: float = 5e-4, **kwargs: Dict):
        super(NormHead, self).__init__(**kwargs)
        self.dense = Dense(num_classes, kernel_regularizer=tf.keras.regularizers.l2(w_decay))

    def call(self, inputs: tf.Tensor, training: bool = None, **kwargs: Dict) -> tf.Tensor:
        x = self.dense(inputs, training=training)
        x = tf.keras.backend.l2_normalize(x, axis=-1)
        return x


class ArcPersonModel(tf.keras.Model):
    def __init__(self, num_classes: int, margin: float = 0.5, logist_scale: int = 30, embd_shape: int = 512,
                 backbone: str = 'EfficientNetB0',
                 w_decay: float = 5e-4, use_pretrain: bool = True, freeze_backbone: bool = False, train_arcloss=False):
        super(ArcPersonModel, self).__init__()
        self.num_classes = num_classes
        self.base_model = BaseModel(embd_shape, w_decay=w_decay, model=backbone,
                                    use_pretrain=use_pretrain, freeze_backbone=freeze_backbone)
        self.archead = ArcHead(num_classes=num_classes, margin=margin, logist_scale=logist_scale)
        self.normhead = NormHead(num_classes=num_classes, w_decay=w_decay)
        self.train_arcloss = train_arcloss

    def set_train_arcloss(self):
        self.train_arcloss = True

    def call(self, inputs: tf.Tensor, training: bool = None, mask: bool = None, **kwargs: Dict) -> tf.Tensor:
        embedding = self.base_model(inputs, training=training)
        return embedding

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        one_hot_label = tf.one_hot(y, depth=self.num_classes)

        with tf.GradientTape() as tape:
            y_pred = self(inputs=x, training=True)  # Forward pass
            if self.train_arcloss:
                y_pred = self.archead(embedding=y_pred, labels=y)
            else:
                y_pred = self.normhead(y_pred)
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
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        x, y = data
        one_hot_label = tf.one_hot(y, self.num_classes)
        # Compute predictions
        y_pred = self(inputs=x, training=True)  # Forward pass
        if self.train_arcloss:
            y_pred = self.archead(embedding=y_pred, labels=y)
        else:
            y_pred = self.normhead(y_pred)
        # Updates the metrics tracking the loss
        self.compiled_loss(one_hot_label, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
