"""Tests for scripts/p1_dataset_qa.py — the human QA sign-off path.

Synthetic datasets only. The point of these tests is that the *guard rails* work:
a blank form, a rubber-stamped form, a swapped image or a stale manifest must all
fail closed.
"""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import p1_dataset_qa as qa  # noqa: E402

CLASSES = qa.CLASS_NAMES


def _jpeg(tag: str) -> bytes:
    return b"\xff\xd8\xff" + tag.encode() * 4 + b"\xff\xd9"


def make_dataset(root: Path, per_class: int = 6) -> Path:
    """Smallest dataset that satisfies the QA tool's expectations."""
    dataset = root / "data" / "processed" / "ds"
    for split in ("train", "val", "test"):
        for name in CLASSES:
            (dataset / "images" / split / name).mkdir(parents=True, exist_ok=True)
            (dataset / "labels" / split / name).mkdir(parents=True, exist_ok=True)

    rows = []
    counts = {"train": 0, "val": 0, "test": 0}
    for index, name in enumerate(CLASSES):
        for i in range(per_class):
            split = "train"
            filename = f"{name}_{i}.jpg"
            img = dataset / "images" / split / name / filename
            img.write_bytes(_jpeg(f"{name}{i}"))
            label = dataset / "labels" / split / name / (filename[:-4] + ".txt")
            label.write_text(f"{index} 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            counts[split] += 1
            rows.append({
                "image": f"images/{split}/{name}/{filename}",
                "sha256": hashlib.sha256(img.read_bytes()).hexdigest(),
                "label": f"labels/{split}/{name}/{filename[:-4]}.txt",
                "label_sha256": hashlib.sha256(label.read_bytes()).hexdigest(),
            })

    aggregate = {
        "schema_version": 1,
        "status": "TEN_CLASS_CANDIDATE_BUILT_PENDING_HUMAN_QA",
        "model_classes": 10,
        "by_split": counts,
        "human_visual_decode_and_legal_qa_status": "PENDING",
        "P1_pass": False,
    }
    aggregate_bytes = json.dumps(aggregate, indent=2, sort_keys=True).encode()
    (dataset / "manifest.aggregate.json").write_bytes(aggregate_bytes)
    private = {
        "schema_version": 1,
        "sample_files": rows,
        "aggregate_sha256": hashlib.sha256(
            json.dumps(aggregate, sort_keys=True).encode()
        ).hexdigest(),
    }
    (dataset / "manifest.private.json").write_text(
        json.dumps(private, indent=2, sort_keys=True), encoding="utf-8"
    )
    return dataset


class QAPacketTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = make_dataset(self.root)
        self.workspace = self.root / "data" / "reviews" / "qa"

    def _prepare(self, per_class=6):
        return qa.prepare(
            self.dataset, self.workspace, per_class, "test-seed",
            data_root=self.root / "data",
        )

    def _form(self, decision="ok", reviewer="synthetic reviewer"):
        form = json.loads((self.workspace / "review.json").read_text(encoding="utf-8"))
        form["reviewer"] = reviewer
        form["reviewed_on"] = "2026-10-26"
        for row in form["decisions"]:
            row["decision"] = decision
        return form

    def _write(self, form, name="filled.json"):
        path = self.root / name
        path.write_text(json.dumps(form), encoding="utf-8")
        return path

    def test_packet_lists_every_class(self):
        result = self._prepare()
        self.assertEqual(result["status"], "PACKET_GENERATED_NOT_APPROVED")
        self.assertFalse(result["approves_anything"])
        self.assertEqual(set(result["counts_by_class"]), set(CLASSES))
        self.assertTrue((self.workspace / "index.html").is_file())

    def test_blank_form_is_rejected(self):
        self._prepare()
        form = self._form()
        for row in form["decisions"]:
            row["decision"] = ""
        with self.assertRaises(qa.QAError):
            qa.verify(self.dataset, self.workspace, self._write(form), data_root=self.root / "data")

    def test_missing_reviewer_is_rejected(self):
        self._prepare()
        form = self._form()
        form["reviewer"] = ""
        with self.assertRaises(qa.QAError):
            qa.verify(self.dataset, self.workspace, self._write(form), data_root=self.root / "data")

    def test_partial_ok_rate_is_rejected(self):
        """Marking most samples bad must not yield an ack."""
        self._prepare()
        form = self._form()
        for row in form["decisions"][: len(form["decisions"]) // 2]:
            row["decision"] = "bad_gesture"
        with self.assertRaises(qa.QAError):
            qa.verify(self.dataset, self.workspace, self._write(form), data_root=self.root / "data")

    def test_one_class_lacking_ok_is_rejected(self):
        """A class with too few clean samples blocks the whole sign-off."""
        self._prepare()
        form = self._form()
        target = CLASSES[3]
        for row in form["decisions"]:
            if row["class"] == target:
                row["decision"] = "bad_box"
        with self.assertRaises(qa.QAError):
            qa.verify(self.dataset, self.workspace, self._write(form), data_root=self.root / "data")

    def test_duplicate_decision_is_rejected(self):
        self._prepare()
        form = self._form()
        form["decisions"].append(dict(form["decisions"][0]))
        with self.assertRaises(qa.QAError):
            qa.verify(self.dataset, self.workspace, self._write(form), data_root=self.root / "data")

    def test_swapped_packet_image_is_rejected(self):
        self._prepare()
        form = self._form()
        review = self._write(form)
        target = self.workspace / "images" / "0000.jpg"
        target.write_bytes(_jpeg("tampered"))
        with self.assertRaises(qa.QAError):
            qa.verify(self.dataset, self.workspace, review, data_root=self.root / "data")

    def test_wrong_dataset_manifest_is_rejected(self):
        self._prepare()
        form = self._form()
        agg = self.dataset / "manifest.aggregate.json"
        agg.write_bytes(agg.read_bytes() + b" ")
        with self.assertRaises(qa.QAError):
            qa.verify(self.dataset, self.workspace, self._write(form), data_root=self.root / "data")

    def test_clean_review_produces_ack_and_apply_records_it(self):
        self._prepare()
        form = self._form()
        result = qa.verify(self.dataset, self.workspace, self._write(form), data_root=self.root / "data")
        self.assertEqual(result["status"], "QA_ACKNOWLEDGED_NOT_P1_PASS")
        ack = Path(result["ack"])
        self.assertTrue(ack.is_file())

        applied = qa.apply_ack(self.dataset, ack, data_root=self.root / "data")
        self.assertEqual(applied["human_visual_decode_and_legal_qa_status"], "PASS")
        # A visual review alone must not claim full P1 acceptance.
        self.assertFalse(applied["P1_pass"])

        state = qa.status(self.dataset)
        self.assertEqual(state["human_visual_decode_and_legal_qa_status"], "PASS")
        self.assertEqual(state["human_visual_qa"]["samples_reviewed"],
                         result["samples_reviewed"])

    def test_ack_cannot_be_applied_to_a_different_dataset(self):
        self._prepare()
        form = self._form()
        result = qa.verify(self.dataset, self.workspace, self._write(form), data_root=self.root / "data")
        ack = Path(result["ack"])

        other = make_dataset(self.root / "other")
        with self.assertRaises(qa.QAError):
            qa.apply_ack(other, ack, data_root=self.root / "data")

    def test_workspace_outside_data_is_refused(self):
        with self.assertRaises(qa.QAError):
            qa.prepare(
                self.dataset, self.root / "not-private", 6, "s",
                data_root=self.root / "data",
            )

    def test_existing_workspace_is_not_overwritten(self):
        self._prepare()
        with self.assertRaises(qa.QAError):
            qa.prepare(
                self.dataset, self.workspace, 6, "s",
                data_root=self.root / "data",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
