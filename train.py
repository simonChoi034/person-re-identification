import argparse

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TerminateOnNaN, EarlyStopping, TensorBoard
from tensorflow.keras.experimental import LinearCosineDecay
from tensorflow.keras.losses import CategoricalCrossentropy as CrossEntropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam, SGD

from config import cfg
from dataset.dataset import Dataset, LPWDatasetGenerator
from model.model import ReIDModel

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
        dataset_generator = LPWDatasetGenerator(dataset_path=dataset_path)
        self.dataset_train = Dataset(generator=dataset_generator, image_size=cfg.image_size, batch_size=batch_size,
                                     buffer_size=cfg.buffer_size, prefetch_size=cfg.prefetch_size,
                                     mode="train").create_dataset()
        self.dataset_eval = Dataset(generator=dataset_generator, image_size=cfg.image_size, batch_size=batch_size,
                                    buffer_size=cfg.buffer_size, prefetch_size=cfg.prefetch_size,
                                    mode="eval").create_dataset()
        self.num_classes = dataset_generator.get_num_classes()
        self.model = ReIDModel(num_classes=self.num_classes, backbone=cfg.backbone, use_pretrain=False, loss=cfg.loss, logist_scale=cfg.loss_scale, margin=cfg.margin)
        self.loss_fn = CrossEntropy(from_logits=True)
        self.lr_scheduler = LinearCosineDecay(initial_learning_rate=cfg.lr,
                                              decay_steps=dataset_generator.dataset_size * cfg.train_epochs / batch_size)
        self.optimizer = Adam(learning_rate=self.lr_scheduler, clipnorm=1) if cfg.optimizer == "Adam" else SGD(
            learning_rate=self.lr_scheduler, momentum=0.9, nesterov=True, clipnorm=1)

        self.tensorboard_callback = TensorBoard(log_dir="./logs/{}_arcface".format(cfg.backbone),
                                                write_graph=True,
                                                write_images=True, update_freq=cfg.step_to_log,
                                                embeddings_freq=1, histogram_freq=1)
        self.checkpoint_callback = ModelCheckpoint(filepath="./checkpoint/{}/cp.ckpt".format(cfg.backbone),
                                                   save_freq="epoch", period=5)

    def compile(self):
        self.model.compile(run_eagerly=False, optimizer=self.optimizer, loss=self.loss_fn,
                           metrics=[CategoricalAccuracy()])

        # load latest checkpoint
        latest = tf.train.latest_checkpoint("./checkpoint/{}".format(cfg.backbone))
        if latest:
            print("Load from checkpoint {}".format(latest))
            checkpoint_manager = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
            checkpoint_manager.restore(latest)

    def train_arcloss(self):
        self.model.fit(self.dataset_train, validation_data=self.dataset_eval, epochs=cfg.train_epochs,
                       callbacks=[self.tensorboard_callback, self.checkpoint_callback, TerminateOnNaN(),
                                  EarlyStopping(patience=5, restore_best_weights=True)])
        self.model.base_model.save_weights('./weights/{}_arcface'.format(cfg.backbone))
        self.model.base_model.save('./saved_model/{}_arcface'.format(cfg.backbone))

    def evaluate(self):
        self.model.evaluate(self.dataset_eval, return_dict=True, callbacks=[self.tensorboard_callback])
        self.model.base_model.save('./saved_model/{}'.format(cfg.backbone))

    def main(self):
        self.compile()
        self.train_arcloss()
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
