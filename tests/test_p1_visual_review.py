"""Offline P1 visual QA tests. Only invented subjects and fake JPEG signatures."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.hagrid_p1 import OFFICIAL_LICENSE_SHA256
from scripts.p1_visual_review import (
    DEFAULT_SOURCES, VisualReviewError, build_packet, verify_review,
)


class VisualReviewPacketTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.annotations = self.root / "annotations"
        self.images = self.root / "images"
        self.output = self.root / "visual-private"
        self.license = self.root / "review-license.json"
        self.sources = tuple(DEFAULT_SOURCES)
        self.license.write_text(json.dumps({
            "reviewer": "SYNTHETIC fixture",
            "reviewed_on": "2026-10-20",
            "license_pdf_sha256": OFFICIAL_LICENSE_SHA256,
            "permitted_scope": "research_only",
            "accept_terms": True,
        }), encoding="utf-8")
        for source in self.sources:
            records = {}
            for i in range(6):
                # Source stems and subject IDs must never enter generated HTML or
                # packet manifests. They are fictional synthetic data here.
                stem = f"SECRET_sample_{source}_{i}"
                record = {
                    "user_id": f"SECRET_person_{source}_{i}",
                    "bboxes": [[0.1, 0.2, 0.3, 0.4]],
                    "labels": [source],
                    "united_bbox": None,
                    "united_label": None,
                    "meta": {"age": 99, "private": "DONOTLEAK-FAKE"},
                }
                records[stem] = record
                path = self.images / source / (stem + ".jpg")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"\xff\xd8\xff" + stem.encode() + b"\xff\xd9")
            d = self.annotations / "train"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{source}.json").write_text(
                json.dumps(records), encoding="utf-8"
            )
        # A poison test split verifies review code never reads evaluation data.
        evaluation = self.annotations / "test"
        evaluation.mkdir(parents=True, exist_ok=True)
        (evaluation / "two_up.json").write_text("THIS IS INVALID TEST JSON")

    def packet(self, sources=None, per_source=3):
        return build_packet(
            self.annotations, self.images, self.output, self.license,
            tuple(sources or self.sources), per_source,
        )

    def sign(self, decisions=None):
        template = json.loads(
            (self.output / "review.template.json").read_text(encoding="utf-8")
        )
        template["reviewer"] = "synthetic human placeholder"
        template["reviewed_on"] = "2026-10-20"
        manifest = json.loads((self.output / "packet.manifest.json").read_text())
        for row in template["decisions"]:
            source = row["source"]
            ids = [x["packet_id"] for x in manifest["samples"] if x["source"] == source]
            row["reviewed_packet_ids"] = ids
            row["official_reference_checked"] = True
            row["comparative_notes"] = "Synthetic test explicitly compares finger separation, no real people."
            row["decision"] = "match" if source == "two_up" else "mismatch"
        if decisions:
            for source, value in decisions.items():
                next(row for row in template["decisions"] if row["source"] == source)[
                    "decision"
                ] = value
        path = self.root / "signed-local-review.json"
        path.write_text(json.dumps(template), encoding="utf-8")
        return path

    def test_private_packet_contains_four_sources_and_no_subjects(self):
        result = self.packet()
        self.assertEqual(result["status"], "PACKET_GENERATED_NOT_APPROVED")
        self.assertEqual(result["private_files_written"], 12)
        self.assertTrue(result["train_only"])
        self.assertTrue((self.output / "index.html").is_file())
        self.assertTrue((self.output / "review.template.json").is_file())
        page = (self.output / "index.html").read_text(encoding="utf-8")
        private = (self.output / "packet.manifest.json").read_text(encoding="utf-8")
        for text in ("SECRET_person", "SECRET_sample", "DONOTLEAK-FAKE"):
            self.assertNotIn(text, page)
            self.assertNotIn(text, private)
        self.assertIn("two_up", page)
        self.assertIn("peace", page)
        self.assertIn("left:10.00000%;top:20.00000%", page)
        self.assertNotIn("THIS IS INVALID TEST JSON", private)

    def test_packet_is_deterministic_with_same_seed(self):
        first = self.packet()
        second = build_packet(
            self.annotations, self.images, self.root / "visual-private-two",
            self.license, self.sources, 3,
        )
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
        self.assertEqual(
            (self.output / "packet.manifest.json").read_bytes(),
            (self.root / "visual-private-two" / "packet.manifest.json").read_bytes(),
        )

    def test_missing_license_blocks_before_output(self):
        self.license.unlink()
        with self.assertRaises(VisualReviewError):
            self.packet()
        self.assertFalse(self.output.exists())

    def test_nonexisting_source_json_blocks_atomically(self):
        with self.assertRaises(VisualReviewError):
            self.packet(("two_up", "stop"))
        self.assertFalse(self.output.exists())

    def test_output_cannot_be_overwritten(self):
        self.packet()
        with self.assertRaisesRegex(VisualReviewError, "already exists"):
            self.packet()

    def test_out_of_range_per_source_rejected(self):
        for n in (0, 21, -1):
            with self.assertRaises(VisualReviewError):
                self.packet(per_source=n)
        self.assertFalse(self.output.exists())

    def test_rejects_subjectless_or_broken_bbox(self):
        p = self.annotations / "train/two_up.json"
        doc = json.loads(p.read_text())
        for n, value in enumerate(doc.values()):
            if n == 0:
                value["user_id"] = ""
            elif n == 1:
                value["bboxes"] = [[0.99, 0.01, 0.5, 0.5]]
            elif n == 2:
                value["labels"] = ["<script>unsafe</script>"]
        p.write_text(json.dumps(doc))
        result = self.packet(("two_up",), per_source=2)
        self.assertEqual(result["private_files_written"], 2)
        manifest = json.loads((self.output / "packet.manifest.json").read_text())
        self.assertGreaterEqual(
            manifest["statistics"]["two_up"]["invalid_per_hand_records"], 3
        )
        self.assertNotIn("<script>", (self.output / "index.html").read_text())

    def test_real_multihand_boxes_and_union_are_review_only(self):
        p = self.annotations / "train/two_up.json"
        doc = json.loads(p.read_text())
        for value in doc.values():
            value["bboxes"] = [[0.1, 0.2, 0.3, 0.4], [0.61, 0.5, 0.2, 0.2]]
            value["labels"] = ["two_up", "no_gesture"]
            value["united_bbox"] = [[0.1, 0.1, 0.7, 0.8]]
        p.write_text(json.dumps(doc))
        self.packet(("two_up",), per_source=3)
        page = (self.output / "index.html").read_text()
        self.assertIn("联合框", page)
        self.assertIn("多只手", page)
        self.assertIn("no_gesture", page)
        self.assertEqual(
            len(json.loads((self.output / "packet.manifest.json").read_text())["samples"]), 3
        )

    def test_signed_review_verifies_image_integrity_and_records_no_conflict(self):
        self.packet()
        review = self.sign()
        result = verify_review(self.output, review)
        self.assertEqual(result["sources_checked"], 4)
        self.assertEqual(result["unresolved_sources"], [])
        self.assertEqual(result["potential_command_alias_conflicts"], [])
        self.assertEqual(result["status"], "HUMAN_REVIEW_RECORDED_NOT_P1_PASS")
        self.assertTrue(result["still_requires_ten_class_final_signoff"])

    def test_peace_matching_official_two_is_reported_as_alias_conflict(self):
        self.packet()
        review = self.sign({"peace": "match"})
        result = verify_review(self.output, review)
        self.assertIn("peace", result["potential_command_alias_conflicts"])
        self.assertEqual(result["status"], "HUMAN_REVIEW_RECORDED_WITH_UNRESOLVED_MAPPINGS")
        self.assertEqual(result["model_outputs_unchanged"], 10)

    def test_uncertain_peace_variant_cannot_be_silently_approved(self):
        self.packet()
        review = self.sign({"peace_inverted": "uncertain"})
        result = verify_review(self.output, review)
        self.assertIn("peace_inverted", result["potential_command_alias_conflicts"])
        self.assertEqual(result["status"], "HUMAN_REVIEW_RECORDED_WITH_UNRESOLVED_MAPPINGS")

    def test_unsigned_template_cannot_be_verified(self):
        self.packet()
        with self.assertRaises(VisualReviewError):
            verify_review(self.output, self.output / "review.template.json")

    def test_tampered_jpeg_breaks_review_integrity(self):
        self.packet()
        review = self.sign()
        sample = next((self.output / "images").glob("*.jpg"))
        sample.write_bytes(b"tampered")
        with self.assertRaisesRegex(VisualReviewError, "Image bytes changed"):
            verify_review(self.output, review)

    def test_invented_review_evidence_id_rejected(self):
        self.packet()
        path = self.sign()
        doc = json.loads(path.read_text())
        doc["decisions"][0]["reviewed_packet_ids"].append("invented:99")
        path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(VisualReviewError, "invented image"):
            verify_review(self.output, path)

    def test_two_mapping_requires_its_contrast_samples(self):
        self.packet(("two_up",), per_source=3)
        path = self.sign()
        with self.assertRaisesRegex(VisualReviewError, "contrast missing"):
            verify_review(self.output, path)

    def test_changing_packet_manifest_invalidates_review(self):
        self.packet()
        path = self.sign()
        manifest = self.output / "packet.manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b" \n")
        with self.assertRaisesRegex(VisualReviewError, "different or invalid"):
            verify_review(self.output, path)

    def test_review_cannot_change_output_class_or_decide_without_rationale(self):
        self.packet()
        path = self.sign()
        data = json.loads(path.read_text())
        data["decisions"][0]["proposed_output"] = "four"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(VisualReviewError, "silently change"):
            verify_review(self.output, path)
        data["decisions"][0]["proposed_output"] = "two"
        data["decisions"][0]["comparative_notes"] = ""
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(VisualReviewError, "rationale"):
            verify_review(self.output, path)

    def test_local_review_restricts_symlinks(self):
        img = next((self.images / "two_up").glob("*.jpg"))
        img.unlink()
        img.symlink_to(self.license)
        self.packet(("two_up",), per_source=3)
        manifest = json.loads((self.output / "packet.manifest.json").read_text())
        self.assertTrue(all(x["image"] != str(img) for x in manifest["samples"]))


if __name__ == "__main__":
    unittest.main()
