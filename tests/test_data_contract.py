"""Synthetic-only boundary checks for future licensed P1 conversion."""

import unittest

from scripts.data_contract import (
    AnnotationError,
    assert_subject_disjoint,
    strict_hand_labels,
    xywh_to_yolo,
)


class DatasetContractTests(unittest.TestCase):
    def test_xywh_to_yolo_normalized(self):
        self.assertEqual(xywh_to_yolo([0.1, 0.2, 0.3, 0.4]), (0.25, 0.4, 0.3, 0.4))

    def test_box_bounds_are_strict(self):
        for bbox in ([0, 0, 0, 0.5], [0.9, 0.2, 0.2, 0.4], [-0.1, 0, 0.2, 0.3], [float("nan"), 0, 0.2, 0.3]):
            with self.assertRaises(AnnotationError):
                xywh_to_yolo(bbox)

    def test_two_hands_two_boxes(self):
        rows = strict_hand_labels(
            [[0.1, 0.2, 0.2, 0.25], [0.55, 0.5, 0.2, 0.2]], ["peace", "no_gesture"]
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0].startswith("1 "))
        self.assertTrue(rows[1].startswith("7 "))

    def test_unreviewed_noncommand_rejects(self):
        with self.assertRaises(AnnotationError):
            strict_hand_labels([[0.1, 0.1, 0.4, 0.4]], ["fist"])
        rows = strict_hand_labels(
            [[0.1, 0.1, 0.4, 0.4]], ["fist"], non_command_sources=frozenset({"fist"})
        )
        self.assertTrue(rows[0].startswith("7 "))

    def test_unlabeled_hand_fails_closed(self):
        with self.assertRaises(AnnotationError):
            strict_hand_labels([], [])
        with self.assertRaises(AnnotationError):
            strict_hand_labels([[0.1, 0.1, 0.2, 0.3]], [])
        self.assertEqual(strict_hand_labels([], [], verified_no_hand=True), [])
        with self.assertRaises(AnnotationError):
            strict_hand_labels([[0.1, 0.1, 0.2, 0.3]], ["palm"], verified_no_hand=True)

    def test_user_split_leakage(self):
        assert_subject_disjoint({"train": {"u-a", "u-b"}, "val": {"u-c"}, "test": {"u-d"}})
        with self.assertRaises(AnnotationError):
            assert_subject_disjoint({"train": {"u-a"}, "val": {"u-a"}, "test": {"u-d"}})


if __name__ == "__main__":
    unittest.main()
