import argparse

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, TerminateOnNaN, EarlyStopping
from tensorflow.keras.experimental import LinearCosineDecay
from tensorflow.keras.losses import CategoricalCrossentropy as CrossEntropy
from tensorflow.keras.metrics import SparseCategoricalCrossentropy, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam, SGD

from config import cfg
from dataset.dataset import Dataset, DatasetGenerator
from model.model import ArcPersonModel

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class Trainer:
    def __init__(self, dataset_path: str, batch_size: int):
        dataset_generator = DatasetGenerator(dataset_path=dataset_path)
        self.dataset_train = Dataset(generator=dataset_generator, image_size=cfg.image_size, batch_size=batch_size,
                                     buffer_size=cfg.buffer_size, prefetch_size=cfg.prefetch_size,
                                     mode="train").create_dataset()
        self.dataset_eval = Dataset(generator=dataset_generator, image_size=cfg.image_size, batch_size=batch_size,
                                    buffer_size=cfg.buffer_size, prefetch_size=cfg.prefetch_size,
                                    mode="eval").create_dataset()
        self.num_classes = dataset_generator.get_num_classes()
        self.model = ArcPersonModel(num_classes=self.num_classes, backbone=cfg.backbone, use_pretrain=False, logist_scale=10)
        self.loss_fn = CrossEntropy(from_logits=True)
        self.lr_scheduler = LinearCosineDecay(initial_learning_rate=cfg.lr,
                                              decay_steps=dataset_generator.dataset_size * cfg.warmup_epochs / batch_size)
        self.optimizer = Adam(learning_rate=self.lr_scheduler, clipnorm=1) if cfg.optimizer == "Adam" else SGD(learning_rate=self.lr_scheduler, momentum=0.9, nesterov=True, clipnorm=1)

        self.tensorboard_callback = TensorBoard(log_dir="./logs/{}".format(cfg.backbone), write_graph=True,
                                                write_images=True, update_freq=cfg.step_to_log,
                                                embeddings_freq=cfg.step_to_log, histogram_freq=cfg.step_to_log)
        self.checkpoint_callback = ModelCheckpoint(filepath="./checkpoint/{}/cp.ckpt".format(cfg.backbone), verbose=1, save_freq="epoch")

    def train(self):
        self.model.compile(run_eagerly=True, optimizer=self.optimizer, loss=self.loss_fn,
                           metrics=[SparseCategoricalCrossentropy(from_logits=True), SparseCategoricalAccuracy()])
        self.model.fit(self.dataset_train, validation_data=self.dataset_eval, epochs=5,
                       callbacks=[self.tensorboard_callback, self.checkpoint_callback, TerminateOnNaN(), EarlyStopping()])

        self.model.set_train_arcloss()
        self.model.fit(self.dataset_train, validation_data=self.dataset_eval, epochs=cfg.train_epochs,
                       callbacks=[self.tensorboard_callback, self.checkpoint_callback, TerminateOnNaN(),
                                  EarlyStopping()])
        self.model.save('./saved_model/{}'.format(cfg.backbone))

    def evaluate(self):
        self.model.evaluate(self.dataset_eval, return_dict=True, callbacks=[self.tensorboard_callback])

    def main(self):
        self.train()
        self.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train detection model')
    parser.add_argument('-b', '--batch_size', type=int, default=cfg.batch_size, help="Batch size")
    parser.add_argument('-d', '--dataset_path', type=str, help="path of dataset")
    args = parser.parse_args()

    trainer = Trainer(
        batch_size=args.batch_size,
        dataset_path=args.dataset_path
    )

    trainer.main()
