from easydict import EasyDict

cfg = EasyDict()

cfg.backbone = "EfficientNetB0"
cfg.image_size = [245, 128]
cfg.buffer_size = 1000
cfg.batch_size = 32
cfg.prefetch_size = 5
cfg.lr = 0.01
cfg.warmup_epochs = 5
cfg.train_epochs = 300
cfg.step_to_log = 100