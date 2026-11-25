"""P2 专用加载器：独立 train/val 参数，不改变模型或训练算法。"""
from __future__ import annotations

from copy import copy

from ultralytics.data.build import build_dataloader, close_dataloader
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import torch_distributed_zero_first


def log_loader(loader, mode):
    LOGGER.info(
        f"P2_LOADER mode={mode} batch={loader.batch_size} workers={loader.num_workers} "
        f"prefetch_factor={loader.prefetch_factor} persistent_workers={loader.persistent_workers} "
        f"cache={loader.dataset.cache} pin_memory={loader.pin_memory}"
    )
    return loader


def make_validator(options, *, standalone=False):
    class P2Validator(DetectionValidator):
        def get_dataloader(self, dataset_path, batch_size):
            dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
            workers = self.args.workers if standalone else options.val_workers
            prefetch = options.prefetch_factor if standalone else options.val_prefetch_factor
            return log_loader(build_dataloader(
                dataset, batch_size, workers, shuffle=False, rank=-1,
                drop_last=self.args.compile, pin_memory=self.training,
                prefetch_factor=prefetch, infinite=False,
            ), "eval" if standalone else "val")

        def __call__(self, trainer=None, model=None):
            try:
                return super().__call__(trainer=trainer, model=model)
            finally:
                # 训练期间保留有限验证 loader；最终/独立验证后显式释放。
                if trainer is None and self.dataloader is not None:
                    close_dataloader(self.dataloader)

    return P2Validator


def make_trainer(options):
    class P2Trainer(DetectionTrainer):
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            with torch_distributed_zero_first(rank):
                dataset = self.build_dataset(dataset_path, mode, batch_size)
            train = mode == "train"
            shuffle = train and not getattr(dataset, "rect", False)
            return log_loader(build_dataloader(
                dataset, batch_size,
                self.args.workers if train else options.val_workers,
                shuffle=shuffle, rank=rank,
                drop_last=self.args.compile and train,
                prefetch_factor=options.prefetch_factor if train else options.val_prefetch_factor,
                infinite=train,
            ), mode)

        def get_validator(self):
            self.loss_names = "box_loss", "cls_loss", "dfl_loss"
            return make_validator(options)(
                self.test_loader, save_dir=self.save_dir,
                args=copy(self.args), _callbacks=self.callbacks,
            )

        def train(self):
            try:
                return super().train()
            finally:
                for name in ("train_loader", "test_loader"):
                    loader = getattr(self, name, None)
                    if loader is not None:
                        close_dataloader(loader)

    return P2Trainer
