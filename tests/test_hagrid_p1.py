"""Synthetic-only tests: no HaGRID download, images or people are used."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.data_contract import CLASS_NAMES, SOURCE_TO_INDEX
from scripts.hagrid_p1 import PreparationError, audit, convert, _check_sample, OFFICIAL_LICENSE_SHA256


class HaGRIDP1Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.annotations = self.root / "annotation"
        self.images = self.root / "images"
        self.output = self.root / "prepared"
        self.license = self.root / "research_license_ack.json"
        self.mapping = self.root / "class_review.json"
        self.sources = ("one", "palm")
        self.by_split = {}
        for split in ("train", "val", "test"):
            for source in self.sources:
                stem = f"synthetic_{split}_{source}"
                subject = f"synthetic_subject_{split}_{source}"
                record = {
                    "bboxes": [[0.10, 0.20, 0.40, 0.50]],
                    "labels": [source],
                    "united_bbox": None, "united_label": None,
                    "user_id": subject,
                    "meta": {"age": [99], "race": ["never disclose"]},
                }
                path = self.annotations / split / f"{source}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({stem: record}), encoding="utf-8")
                img = self.images / source / f"{stem}.jpg"
                img.parent.mkdir(parents=True, exist_ok=True)
                img.write_bytes(b"\xff\xd8\xff" + f"{split}:{source}:{stem}".encode())
                self.by_split[(split, source)] = (stem, record)
        self.license.write_text(json.dumps({
            "reviewer": "synthetic only", "reviewed_on": "2026-10-20",
            "license_pdf_sha256": OFFICIAL_LICENSE_SHA256,
            "permitted_scope": "research_only", "accept_terms": True,
        }), encoding="utf-8")
        self.mapping.write_text(json.dumps({
            "reviewer": "synthetic only", "reviewed_on": "2026-10-20",
            "classes": [
                {"source": source, "output": CLASS_NAMES[i],
                 "approved": True, "samples_reviewed": 3, "official_reference_checked": True}
                for source, i in SOURCE_TO_INDEX.items()
            ],
        }), encoding="utf-8")

    def _rewrite(self, split, source, mutate):
        path = self.annotations / split / f"{source}.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        stem = next(iter(doc))
        mutate(doc[stem])
        path.write_text(json.dumps(doc), encoding="utf-8")

    def _convert(self):
        return convert(self.annotations, self.images, self.output, self.license,
                       self.mapping, self.sources, limit_per_source=1)

    def test_metadata_audit_aggregates_without_identities(self):
        report = audit(self.annotations, self.sources)
        self.assertEqual(report["samples_by_split"], {"train": 2, "val": 2, "test": 2})
        self.assertEqual(report["valid_by_split"]["train"], 2)
        self.assertEqual(report["boxes_by_class"]["one"], 3)
        self.assertEqual(report["boxes_by_class"]["five"], 3)
        serialized = json.dumps(report)
        self.assertNotIn("synthetic_subject", serialized)
        self.assertNotIn("never disclose", serialized)

    def test_micro_conversion_writes_per_hand_yolo_rows(self):
        result = self._convert()
        self.assertEqual(result["selected_images_by_split"], {"train": 2, "val": 2, "test": 2})
        self.assertEqual(result["distinct_selected_image_hashes"], 6)
        yolo = (self.output / "labels/train/palm/synthetic_train_palm.txt").read_text()
        fields = yolo.split()
        self.assertEqual(fields[0], "4")
        self.assertEqual(len(fields), 5)
        self.assertAlmostEqual(float(fields[1]), 0.3)
        self.assertTrue((self.output / "images/train/palm/synthetic_train_palm.jpg").is_file())
        aggregate = (self.output / "manifest.aggregate.json").read_text()
        self.assertNotIn("synthetic_subject", aggregate)
        self.assertNotIn("never disclose", aggregate)

    def test_multiple_independent_hands_have_separate_labels(self):
        self._rewrite("train", "one", lambda item: item.update({
            "bboxes": [
                [0.05, 0.10, 0.30, 0.35],
                [0.60, 0.50, 0.20, 0.20],
            ],
            "labels": ["one", "no_gesture"],
        }))
        self._convert()
        rows = (self.output / "labels/train/one/synthetic_train_one.txt").read_text().splitlines()
        self.assertEqual(len(rows), 2)
        self.assertEqual([int(row.split()[0]) for row in rows], [0, 7])

    def test_hand_without_class_and_boolean_bbox_rejected(self):
        self._rewrite("train", "one", lambda item: item.update({"labels": ["no_hand"]}))
        self._rewrite("val", "one", lambda item: item.update({
            "bboxes": [[False, 0.1, 0.3, 0.3]]
        }))
        result = audit(self.annotations, self.sources)
        self.assertEqual(result["rejected_by_split"]["train"], 1)
        self.assertEqual(result["rejected_by_split"]["val"], 1)

    def test_untrusted_mapping_approval_types_fail_closed(self):
        doc = json.loads(self.mapping.read_text())
        doc["classes"][0]["samples_reviewed"] = "three"
        self.mapping.write_text(json.dumps(doc))
        with self.assertRaises(PreparationError):
            self._convert()

    def test_nonexistent_calendar_date_fails_closed(self):
        doc = json.loads(self.license.read_text())
        doc["reviewed_on"] = "2026-99-99"
        self.license.write_text(json.dumps(doc))
        with self.assertRaises(PreparationError):
            self._convert()

    def test_incorrect_source_pdf_hash_is_rejected(self):
        record = json.loads(self.license.read_text())
        record["license_pdf_sha256"] = hashlib.sha256(b"wrong license").hexdigest()
        self.license.write_text(json.dumps(record))
        with self.assertRaises(PreparationError):
            self._convert()

    def test_license_ack_required(self):
        self.license.unlink()
        with self.assertRaises(PreparationError):
            self._convert()
        self.assertFalse(self.output.exists())

    def test_mapping_visual_approval_required(self):
        doc = json.loads(self.mapping.read_text())
        doc["classes"][1]["approved"] = False
        self.mapping.write_text(json.dumps(doc))
        with self.assertRaises(PreparationError):
            self._convert()
        self.assertFalse(self.output.exists())

    def test_compound_rejected_even_with_individual_boxes(self):
        self._rewrite("train", "one", lambda item: item.update({
            "united_bbox": [[0.1, 0.1, 0.8, 0.8]],
            "united_label": ["hand_heart"],
        }))
        report = audit(self.annotations, self.sources)
        self.assertEqual(report["compound_samples_rejected"], 1)
        self.assertEqual(report["valid_by_split"]["train"], 1)

    def test_unknown_hand_is_not_implicit_background(self):
        self._rewrite("train", "one", lambda item: item.update({"labels": ["fist"]}))
        report = audit(self.annotations, self.sources)
        self.assertEqual(report["rejected_by_split"]["train"], 1)
        self.assertEqual(report["boxes_by_class"]["one"], 2)

    def test_missing_hand_label_and_no_hand_empty_rejected(self):
        for split, source in (("train", "one"), ("val", "palm")):
            self._rewrite(split, source, lambda item: item.update({"labels": []}))
        self._rewrite("test", "palm", lambda item: item.update({
            "bboxes": [], "labels": []
        }))
        report = audit(self.annotations, self.sources)
        self.assertEqual(report["rejected_by_split"], {"train": 1, "val": 1, "test": 1})

    def test_cross_split_user_leak_is_fatal_and_not_disclosed(self):
        self._rewrite("val", "palm", lambda item: item.update({
            "user_id": "synthetic_subject_train_one"
        }))
        with self.assertRaisesRegex(PreparationError, "Subject overlap") as raised:
            self._convert()
        self.assertNotIn("synthetic_subject_train_one", str(raised.exception))
        self.assertFalse(self.output.exists())

    def test_identical_image_content_is_rejected_atomically(self):
        image = self.images / "one/synthetic_train_one.jpg"
        other = self.images / "palm/synthetic_val_palm.jpg"
        other.write_bytes(image.read_bytes())
        with self.assertRaisesRegex(PreparationError, "Duplicate image"):
            self._convert()
        self.assertFalse(self.output.exists())

    def test_unsafe_source_stem_is_rejected(self):
        path = self.annotations / "train/one.json"
        doc = json.loads(path.read_text())
        record = next(iter(doc.values()))
        path.write_text(json.dumps({"../bad": record}), encoding="utf-8")
        report = audit(self.annotations, self.sources)
        self.assertEqual(report["rejected_by_split"]["train"], 1)

    def test_bad_jpg_signature_is_not_used(self):
        (self.images / "one/synthetic_train_one.jpg").write_bytes(b"not-jpeg")
        result = self._convert()
        self.assertEqual(result["missing_or_invalid_jpg_by_split"]["train"], 1)
        self.assertEqual(result["selected_images_by_split"]["train"], 1)

    def test_overwrite_refused(self):
        self.output.mkdir()
        with self.assertRaisesRegex(PreparationError, "already exists"):
            self._convert()

    def test_limit_bounds_enforced(self):
        with self.assertRaises(PreparationError):
            convert(self.annotations, self.images, self.output, self.license,
                    self.mapping, self.sources, limit_per_source=100000)

    def test_individual_noncommand_hand_is_class_seven(self):
        subject, labels = _check_sample("synthetic", {
            "user_id": "fake", "bboxes": [[0, 0, 0.5, 0.5]],
            "labels": ["no_gesture"], "united_bbox": None, "united_label": None
        })
        self.assertEqual(subject, "fake")
        self.assertTrue(labels[0].startswith("7 "))


if __name__ == "__main__":
    unittest.main()
