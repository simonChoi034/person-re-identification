import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard


class LRTensorboard(TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super(LRTensorboard, self).__init__(log_dir=log_dir, **kwargs)
        self.lr_writer = tf.summary.create_file_writer(self.log_dir + '/train')

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs['lr'] = self.model.optimizer.lr(batch)

        super(LRTensorboard, self).on_train_batch_end(batch, logs)