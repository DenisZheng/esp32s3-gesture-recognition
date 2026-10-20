"""P0 contract tests; no heavy ML package or external data needed."""

import unittest

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

    def test_only_upstream_nc_is_changed(self):
        source = "# AGPL-3.0\nnc: 1 # upstream\nhead:\n  - [[16,19,22],1,ESPDetect, [nc]]\n"
        self.assertIn("nc: 10", render(source))
        with self.assertRaises(ValueError):
            render("nc: 10\nESPDetect, [nc]")
        with self.assertRaises(ValueError):
            render("nc: 1\nNotAHead: true")


if __name__ == "__main__":
    unittest.main()
