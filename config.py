from easydict import EasyDict

cfg = EasyDict()

cfg.backbone = "OSNet"
cfg.optimizer = "SGD"
cfg.image_size = [256, 128]
cfg.buffer_size = 1000
cfg.batch_size = 32
cfg.prefetch_size = 5
cfg.lr = 0.1
cfg.warmup_epochs = 10
cfg.train_epochs = 100
cfg.step_to_log = 500
