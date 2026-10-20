"""P0 contract tests; no heavy ML package or external data needed."""

import unittest
from pathlib import Path
import json

from scripts.check_contract import check
from scripts.data_contract import CLASS_NAMES, SOURCE_TO_INDEX
from scripts.prepare_model_config import render


class ContractTests(unittest.TestCase):
    def test_file_contract(self):
        check()

    def test_classes_and_sources(self):
        self.assertEqual(len(CLASS_NAMES), 10)
        self.assertEqual(len(SOURCE_TO_INDEX), 10)
        self.assertEqual(SOURCE_TO_INDEX["two_up"], 1)
        self.assertEqual(SOURCE_TO_INDEX["palm"], 4)
        self.assertEqual(SOURCE_TO_INDEX["no_gesture"], 7)
        self.assertNotIn("no_hand", SOURCE_TO_INDEX)

    def test_single_model_gesture_recognition_baseline(self):
        root = Path(__file__).resolve().parents[1]
        classes = json.loads((root / "configs/classes.json").read_text(encoding="utf-8"))
        self.assertEqual(classes["input_hw"], [224, 224])
        self.assertEqual(classes["inference_model_count"], 1)
        self.assertEqual(classes["input_type"], "whole_camera_frame")
        self.assertFalse(classes["separate_mobile_net_classifier"])
        self.assertIn("Gesture Recognition", (root / "README.md").read_text(encoding="utf-8").splitlines()[0])
        self.assertTrue((root / "docs/INPUT_RESOLUTION.md").is_file())
        deps = json.loads((root / "upstreams.lock.json").read_text(encoding="utf-8"))["dependencies"]
        dl = next(d for d in deps if d["id"] == "esp-dl")
        self.assertEqual(dl["ref"], "v3.3.14")
        self.assertEqual(dl["commit"], "5acd1c9992af4afb0fa80c221a676982b11381b8")

    def test_only_upstream_nc_is_changed(self):
        source = "# AGPL-3.0\nnc: 1 # upstream\nhead:\n  - [[16,19,22],1,ESPDetect, [nc]]\n"
        self.assertIn("nc: 10", render(source))
        with self.assertRaises(ValueError):
            render("nc: 10\nESPDetect, [nc]")
        with self.assertRaises(ValueError):
            render("nc: 1\nNotAHead: true")


if __name__ == "__main__":
    unittest.main()
