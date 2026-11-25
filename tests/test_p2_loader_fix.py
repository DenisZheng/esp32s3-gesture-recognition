"""P2 内存修复的最小回归；真实 DataLoader，只用合成 Tensor。"""
import gc
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


HAS_ML = importlib.util.find_spec("ultralytics") is not None


@unittest.skipUnless(HAS_ML, "需要已应用补丁的 P2 镜像")
class LoaderFixTests(unittest.TestCase):
    def test_cache_gc_restored_on_success_and_errors(self):
        import numpy as np
        from ultralytics.data.utils import load_dataset_cache_file

        original = gc.isenabled()
        try:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                good, bad, missing = root / "good.cache", root / "bad.cache", root / "missing.cache"
                with good.open("wb") as f:
                    np.save(f, {"value": 7})
                bad.write_bytes(b"broken cache")
                for enabled in (True, False):
                    for path in (good, bad, missing):
                        with self.subTest(enabled=enabled, path=path.name):
                            (gc.enable if enabled else gc.disable)()
                            if path == good:
                                self.assertEqual(load_dataset_cache_file(path), {"value": 7})
                            else:
                                with self.assertRaises(Exception):
                                    load_dataset_cache_file(path)
                            self.assertEqual(gc.isenabled(), enabled)
        finally:
            (gc.enable if original else gc.disable)()

    def test_workers_prefetch_and_validation_lifecycle(self):
        from torch.utils.data import TensorDataset
        import torch
        from scripts.p2_loaders import make_trainer
        from ultralytics.data.build import InfiniteDataLoader, close_dataloader

        data = TensorDataset(torch.arange(32))
        data.cache, data.rect = False, False
        for train_workers, val_workers in ((0, 0), (2, 0), (4, 2)):
            with self.subTest(train=train_workers, val=val_workers):
                options = SimpleNamespace(val_workers=val_workers, prefetch_factor=2, val_prefetch_factor=1)
                trainer_cls = make_trainer(options)
                trainer = trainer_cls.__new__(trainer_cls)
                trainer.args = SimpleNamespace(workers=train_workers, compile=False)
                trainer.build_dataset = lambda *a: data
                train = trainer.get_dataloader("unused", 8, -1, "train")
                val = trainer.get_dataloader("unused", 16, -1, "val")
                self.assertIsInstance(train, InfiniteDataLoader)
                self.assertNotIsInstance(val, InfiniteDataLoader)
                self.assertEqual(train.num_workers, train_workers)
                self.assertEqual(val.num_workers, val_workers)
                self.assertEqual(train.prefetch_factor, 2 if train_workers else None)
                self.assertEqual(val.prefetch_factor, 1 if val_workers else None)
                self.assertEqual(val.persistent_workers, bool(val_workers))
                try:
                    self.assertEqual(len(list(train)), 4)
                    self.assertEqual(len(list(val)), 2)
                    self.assertEqual(len(list(val)), 2)
                    if val_workers:
                        self.assertEqual(val._iterator._tasks_outstanding, 0)
                    saved = list(val._iterator._workers) if val_workers else []
                finally:
                    close_dataloader(train)
                    close_dataloader(val)
                self.assertTrue(all(not p.is_alive() for p in saved))


class CliDefaultsTests(unittest.TestCase):
    def test_cli_worker_defaults(self):
        from scripts.p2_train import build_parser
        args = build_parser().parse_args(["train", "--data", "d.yaml", "--model-yaml", "m.yaml", "--run-dir", "runs/new"])
        self.assertEqual((args.workers, args.val_workers, args.prefetch_factor, args.val_prefetch_factor), (4, 2, 2, 1))
        self.assertEqual((args.batch, args.imgsz), (8, 224))


if __name__ == "__main__":
    unittest.main(verbosity=2)
