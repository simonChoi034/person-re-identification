from easydict import EasyDict

cfg = EasyDict()

cfg.backbone = "OSNet"
cfg.optimizer = "SGD"
cfg.image_size = [256, 128]
cfg.buffer_size = 1000
cfg.batch_size = 32
cfg.prefetch_size = 5
cfg.lr = 0.1
cfg.warmup_epochs = 5
cfg.train_epochs = 300
cfg.step_to_log = 500
