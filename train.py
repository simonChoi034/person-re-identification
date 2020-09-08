import argparse

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TerminateOnNaN, EarlyStopping, TensorBoard
from tensorflow.keras.experimental import LinearCosineDecay
from tensorflow.keras.losses import CategoricalCrossentropy as CrossEntropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam, SGD

from config import cfg
from dataset.dataset import Dataset, LPWDatasetGenerator, MSMT17DatasetGenerator
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
    def __init__(self, source_dataset_path: str, target_dataset_path: str, batch_size: int):
        # dataset for training
        source_dataset_generator = MSMT17DatasetGenerator(dataset_path=source_dataset_path, combine_all=True)
        self.dataset_train = Dataset(generator=source_dataset_generator, image_size=cfg.image_size,
                                     batch_size=batch_size,
                                     buffer_size=cfg.buffer_size, prefetch_size=cfg.prefetch_size,
                                     mode="train").create_dataset()
        self.num_classes = source_dataset_generator.get_num_classes()

        # dataset for evaluation
        target_dataset_generator = LPWDatasetGenerator(dataset_path=target_dataset_path, combine_all=True)
        self.dataset_eval = Dataset(generator=target_dataset_generator, image_size=cfg.image_size,
                                    batch_size=batch_size,
                                    buffer_size=cfg.buffer_size, prefetch_size=cfg.prefetch_size,
                                    mode="train").create_dataset()

        # setup model
        self.model = ReIDModel(num_classes=self.num_classes, backbone=cfg.backbone, use_pretrain=False, loss=cfg.loss,
                               logist_scale=cfg.loss_scale, margin=cfg.margin)
        self.loss_fn = CrossEntropy(from_logits=True, label_smoothing=0.1)
        self.lr_scheduler = LinearCosineDecay(initial_learning_rate=cfg.lr,
                                              decay_steps=source_dataset_generator.dataset_size * cfg.train_epochs / batch_size)
        self.optimizer = Adam(learning_rate=self.lr_scheduler, clipnorm=1) if cfg.optimizer == "Adam" else SGD(
            learning_rate=self.lr_scheduler, momentum=0.9, nesterov=True, clipnorm=1)

        # setup callback
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

    def train(self):
        self.model.fit(self.dataset_train, epochs=cfg.train_epochs,
                       callbacks=[self.tensorboard_callback, self.checkpoint_callback, TerminateOnNaN(),
                                  EarlyStopping(patience=5, restore_best_weights=True)])
        self.model.base_model.save_weights('./weights/{}_arcface'.format(cfg.backbone))
        self.model.base_model.save('./saved_model/{}_arcface'.format(cfg.backbone))

    def evaluate(self):
        embeddings = None
        labels = None
        # get predictions for all data
        for eval_images, eval_labels in self.dataset_eval:
            pred_embeddings = self.model.predict(eval_images)
            if embeddings is None:
                embeddings = pred_embeddings
                labels = eval_labels
            else:
                embeddings = tf.concat([embeddings, pred_embeddings], axis=0)
                labels = tf.concat([labels, eval_labels], axis=0)

    def main(self):
        self.compile()
        self.train()
        # self.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train detection model')
    parser.add_argument('-b', '--batch_size', type=int, default=cfg.batch_size, help="Batch size")
    parser.add_argument('-s', '--source_dataset_path', type=str, help="path of source dataset")
    parser.add_argument('-t', '--target_dataset_path', type=str, help="path of source dataset")
    args = parser.parse_args()

    trainer = Trainer(
        batch_size=args.batch_size,
        source_dataset_path=args.source_dataset_path,
        target_dataset_path=args.target_dataset_path
    )

    trainer.main()
