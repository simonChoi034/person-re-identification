from easydict import EasyDict

cfg = EasyDict()

cfg.backbone = "EfficientNetB0"
cfg.optimizer = "Adam"
cfg.image_size = [224, 224]
cfg.buffer_size = 1000
cfg.batch_size = 32
cfg.prefetch_size = 5
cfg.lr = 0.1
cfg.warmup_epochs = 5
cfg.train_epochs = 50
cfg.step_to_log = 500
