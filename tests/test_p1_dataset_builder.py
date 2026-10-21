"""P1-05 synthetic-only integration: 10 target labels, manual ZIPs/folders, resumes.

Every person ID, image and signoff below is fictional. No HaGRID datasets,
actual downloads, trained checkpoints or external services in CI.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
import warnings
import zipfile
import zlib
from pathlib import Path
from unittest.mock import patch

from scripts.data_contract import CLASS_NAMES, SOURCE_TO_INDEX
from scripts.hagrid_p1 import OFFICIAL_LICENSE_SHA256
from scripts.hagrid_image_zip64 import ImageMember
from scripts.p1_dataset_builder import (
    DatasetBuildError, build_dataset, offline_plan, verify_output,
)
from scripts.p1_dataset_sources import (
    ByteBudget, RemoteImages, SourceError, _atomic_private_write
)

SOURCES = tuple(SOURCE_TO_INDEX)
COUNTS = {"train": 2, "val": 1, "test": 1}
ALIASES = {
    "two_up": ["peace", "peace_inverted", "two_up_inverted"],
    "palm": ["stop", "stop_inverted"],
    "three": ["three2", "three3", "three_gun"],
    "one": ["point", "mute"],
}


def fictional_jpeg(key):
    return b"\xff\xd8\xff" + key.encode("ascii") * 2 + b"\xff\xd9"


@patch.dict(os.environ, {"CI": "false", "GITHUB_ACTIONS": "false"})
class TenClassManualDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ann = self.root / "annotations"
        self.photos = self.root / "images"
        self.zips = self.root / "zips"
        self.annzip = self.root / "annotations.zip"
        self.output = self.root / "processed"
        self.license = self.root / "license.private.json"
        self.mapping = self.root / "mapping.private.json"
        self.license.write_text(json.dumps({
            "reviewer": "SYNTHETIC TEST ONLY",
            "reviewed_on": "2026-10-20",
            "license_pdf_sha256": OFFICIAL_LICENSE_SHA256,
            "permitted_scope": "research_only",
            "accept_terms": True,
        }))
        self.mapping.write_text(json.dumps({
            "reviewer": "FAKE REVIEWER, NOT ACTUAL AUTHORIZATION",
            "reviewed_on": "2026-10-20",
            "classes": [
                {
                    "source": source, "output": CLASS_NAMES[index], "approved": True,
                    "samples_reviewed": 5, "official_reference_checked": True,
                    "contrasts_reviewed": ALIASES.get(source, []),
                }
                for source, index in SOURCE_TO_INDEX.items()
            ],
        }))
        self.synthetic = {}
        for split in ("train", "val", "test"):
            for source in SOURCES:
                rows = {}
                for i in range(3):
                    stem = f"SYNTHETIC_{split}_{source}_{i}"
                    rows[stem] = {
                        "user_id": f"FICTIONAL_ID_{split}_{source}_{i}",
                        "bboxes": [[0.1, 0.15, 0.3, 0.4]],
                        "labels": [source],
                        "united_bbox": None, "united_label": None,
                        "meta": {"private": "FAKE_NEVER_EXPOSE"},
                    }
                    picture = self.photos / source / (stem + ".jpg")
                    picture.parent.mkdir(parents=True, exist_ok=True)
                    picture.write_bytes(fictional_jpeg(stem))
                path = self.ann / split / (source + ".json")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(rows))
                self.synthetic[(split, source)] = rows

    def make_archives(self, *, malicious=None):
        self.zips.mkdir(exist_ok=True)
        with zipfile.ZipFile(self.annzip, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for split in ("train", "val", "test"):
                for source in SOURCES:
                    file = self.ann / split / (source + ".json")
                    z.write(file, f"annotations/{split}/{source}.json")
        for source in SOURCES:
            with zipfile.ZipFile(
                self.zips / (source + ".zip"), "w",
                compression=zipfile.ZIP_DEFLATED, allowZip64=True
            ) as z:
                for file in (self.photos / source).glob("*.jpg"):
                    entry = file.name if source == "no_gesture" else f"{source}/{file.name}"
                    z.write(file, entry)
                if source == "no_gesture":
                    z.writestr("__MACOSX/._ignored.jpg", b"not a source JPEG")
                if malicious and source == "one":
                    z.writestr(malicious, b"\xff\xd8\xffnotme\xff\xd9")

    def build(self, mode="local-dir", **extra):
        return build_dataset(
            mode, self.output, self.license, self.mapping,
            annotations=self.ann if mode == "local-dir" else self.annzip,
            images=self.photos if mode == "local-dir" else self.zips,
            counts=COUNTS, **extra,
        )

    def test_offline_plan_keeps_exact_ten_class_contract(self):
        report = offline_plan(COUNTS)
        self.assertEqual(report["requested_images_total"], 40)
        self.assertEqual(report["class_names"], list(CLASS_NAMES))
        self.assertEqual(report["sources"], list(SOURCES))
        self.assertTrue(report["no_gesture_is_bbox_class_7_not_no_hand"])
        self.assertFalse(report["actual_download_or_training_started"])

    def test_manual_directory_builds_three_splits_all_ten_classes(self):
        result = self.build()
        self.assertEqual(result["status"], "TEN_CLASS_CANDIDATE_BUILT_PENDING_HUMAN_QA")
        self.assertEqual(result["by_split"], {"train": 20, "val": 10, "test": 10})
        self.assertFalse(result["human_qa_and_legal_pass"])
        self.assertTrue((self.output / "dataset.yaml").is_file())
        self.assertEqual(verify_output(self.output)["samples"], 40)
        m = json.loads((self.output / "manifest.aggregate.json").read_text())
        self.assertEqual(m["no_hand_negative_images_auto_created"], 0)
        self.assertEqual(set(m["box_counts"]), set(CLASS_NAMES))
        self.assertTrue(all(count == 4 for count in m["box_counts"].values()))
        self.assertEqual(m["model_classes"], 10)
        self.assertNotIn("FICTIONAL_ID_", json.dumps(m))
        self.assertNotIn("FAKE_NEVER_EXPOSE", json.dumps(m))
        cls7 = next((self.output / "labels/train/no_gesture").glob("*.txt"))
        self.assertTrue(cls7.read_text().startswith("7 "))

    def test_manual_local_zip_reads_only_selected_jpgs_not_40gb_extractall(self):
        self.make_archives()
        result = self.build(mode="local-zip")
        self.assertEqual(result["by_split"]["train"], 20)
        self.assertEqual(verify_output(self.output)["samples"], 40)
        data = json.loads((self.output / "manifest.private.json").read_text())
        self.assertEqual(len(data["sample_files"]), 40)
        self.assertNotIn("FICTIONAL_ID_", json.dumps(data))
        self.assertNotIn("FAKE_NEVER_EXPOSE", json.dumps(data))

    def test_bad_mapping_review_prevents_private_media_read(self):
        m = json.loads(self.mapping.read_text())
        m["classes"][1]["approved"] = False
        self.mapping.write_text(json.dumps(m))
        with self.assertRaisesRegex(DatasetBuildError, "approvals"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_missing_license_refuses_before_file_scan(self):
        self.license.unlink()
        with self.assertRaisesRegex(DatasetBuildError, "approvals"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_public_ci_cannot_process_actual_local_or_remote_media(self):
        with patch.dict(os.environ, {"CI": "true", "GITHUB_ACTIONS": "true"}):
            with self.assertRaisesRegex(DatasetBuildError, "forbidden in CI"):
                self.build()
        self.assertFalse(self.output.exists())

    def test_undeclared_second_hand_rejects_entire_frame(self):
        path = self.ann / "train/one.json"
        doc = json.loads(path.read_text())
        poisoned = next(iter(doc))
        doc[poisoned]["labels"] = ["one", "peace"]
        doc[poisoned]["bboxes"].append([0.6, 0.1, 0.2, 0.3])
        path.write_text(json.dumps(doc))
        result = self.build()
        self.assertEqual(result["by_split"]["train"], 20)
        # Fails closed by skipping contaminated candidate, never dropping its
        # unapproved second hand from output labels.
        self.assertFalse((self.output / f"images/train/one/{poisoned}.jpg").exists())

    def test_duplicate_subject_across_train_test_blocks_everything(self):
        path = self.ann / "test/one.json"
        doc = json.loads(path.read_text())
        key = next(iter(doc))
        doc[key]["user_id"] = next(
            iter(self.synthetic[("train", "one")].values())
        )["user_id"]
        path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(DatasetBuildError, "overlap"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_too_few_licensed_jpgs_rolls_back_complete_output(self):
        for path in list((self.photos / "one").glob("SYNTHETIC_train_one_*.jpg"))[:2]:
            path.unlink()
        with self.assertRaisesRegex(DatasetBuildError, "Insufficient"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_corrupted_label_fails_integrity_recheck(self):
        self.build()
        label = next((self.output / "labels/train/one").glob("*.txt"))
        label.write_text("9 0.5 0.5 0.2 0.2\n")
        with self.assertRaisesRegex(DatasetBuildError, "label changed"):
            verify_output(self.output)

    def test_dataset_output_is_never_overwritten(self):
        self.build()
        with self.assertRaisesRegex(DatasetBuildError, "already exists"):
            self.build()

    def test_quota_guards_reject_more_than_2000_or_boolean(self):
        for value in (0, 2001, True):
            with self.assertRaises(DatasetBuildError):
                offline_plan(dict(COUNTS, train=value))

    def test_zip_path_traversal_is_not_extracted(self):
        self.make_archives(malicious="../escaped.jpg")
        with self.assertRaises(DatasetBuildError):
            self.build(mode="local-zip")
        self.assertFalse(self.output.exists())
        self.assertFalse((self.root / "escaped.jpg").exists())

    def test_existing_cache_location_can_never_be_the_output(self):
        with self.assertRaises(DatasetBuildError):
            build_dataset("auto", self.output, self.license, self.mapping,
                          cache=self.output, counts=COUNTS)

    def test_cache_requires_license_even_in_auto_mode(self):
        self.license.unlink()
        with self.assertRaisesRegex(DatasetBuildError, "approvals"):
            build_dataset("auto", self.output, self.license, self.mapping,
                          cache=self.root / "auto-cache", counts=COUNTS)


class OfflineRemoteResumeTests(unittest.TestCase):
    def test_private_resumable_member_hits_crc_checked_cache(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            data = fictional_jpeg("one_sample")
            m = ImageMember(
                "one/SYNTHETIC.jpg", offset=0, compressed=10,
                uncompressed=len(data), crc32=zlib.crc32(data),
                method=8, flags=0,
            )
            reader = RemoteImages(cache, ByteBudget(1024 * 1024))
            reader.index["one"] = {"SYNTHETIC": m}
            reader.transport["one"] = object()
            with patch("scripts.p1_dataset_sources.imagezip.read_jpeg", return_value=data) as fetch:
                self.assertEqual(reader.get("one", "SYNTHETIC"), data)
                self.assertEqual(reader.get("one", "SYNTHETIC"), data)
                fetch.assert_called_once()
            p = cache / "images/one/SYNTHETIC.jpg"
            self.assertTrue(p.is_file())
            p.write_bytes(b"tampered")
            with self.assertRaisesRegex(SourceError, "CRC invalid"):
                reader.get("one", "SYNTHETIC")

    def test_atomic_cache_write_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cache/one/photo.jpg"
            _atomic_private_write(p, fictional_jpeg("one"))
            with self.assertRaisesRegex(SourceError, "overwrite"):
                _atomic_private_write(p, fictional_jpeg("two"))
            self.assertTrue(p.is_file())


if __name__ == "__main__":
    unittest.main()
