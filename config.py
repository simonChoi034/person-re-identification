from easydict import EasyDict

cfg = EasyDict()

cfg.backbone = "OSNet"
cfg.optimizer = "SGD"
cfg.loss = "circle_loss"
cfg.loss_scale = 256
cfg.margin = 0.25
cfg.image_size = [256, 128]
cfg.buffer_size = 1000
cfg.batch_size = 32
cfg.prefetch_size = 5
cfg.lr = 1e-1
cfg.warmup_epochs = 10
cfg.train_epochs = 200
cfg.step_to_log = 500
