import argparse

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, TerminateOnNaN, EarlyStopping
from tensorflow.keras.experimental import LinearCosineDecay
from tensorflow.keras.losses import SparseCategoricalCrossentropy as CrossEntropy
from tensorflow.keras.metrics import Precision, Recall, SparseCategoricalCrossentropy, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam

from config import cfg
from dataset.dataset import Dataset, DatasetGenerator
from model.model import ArcPersonModel


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
        self.model = ArcPersonModel(num_classes=self.num_classes, backbone=cfg.backbone)
        self.loss_fn = CrossEntropy(from_logits=True)
        self.lr_scheduler = LinearCosineDecay(initial_learning_rate=cfg.lr,
                                              decay_steps=dataset_generator.dataset_size * cfg.warmup_epochs)
        self.optimizer = Adam(learning_rate=self.lr_scheduler)

        self.tensorboard_callback = TensorBoard(log_dir="./logs/{}".format(cfg.backbone), write_graph=True,
                                                write_images=True, update_freq=cfg.step_to_log,
                                                embeddings_freq=cfg.step_to_log)
        self.checkpoint_callback = ModelCheckpoint(filepath="./checkpoint/{}".format(cfg.backbone), save_best_only=True,
                                                   mode='max', monitor='val_acc', save_freq=cfg.step_to_log)


    def train(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn,
                           metrics=[SparseCategoricalCrossentropy(from_logits=True), SparseCategoricalAccuracy()])
        self.model.fit(self.dataset_train, epochs=cfg.train_epochs,
                       callbacks=[self.tensorboard_callback, self.checkpoint_callback, TerminateOnNaN(),
                                  EarlyStopping()])

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
