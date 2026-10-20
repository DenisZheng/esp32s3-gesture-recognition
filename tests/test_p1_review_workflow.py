"""Offline synthetic end-to-end P1 research review orchestration tests.

No real HaGRID data, real people, HTTP access, or genuine legal signoff exists
in these fixtures. Review records created here are obviously fake tests.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.hagrid_p1 import OFFICIAL_LICENSE_SHA256
from scripts.p1_review_workflow import (
    SOURCES, WorkflowError, offline_plan, prepare, verify,
)


@patch.dict("os.environ", {"CI": "false", "GITHUB_ACTIONS": "false"})
class LocalReviewWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.license = self.root / "fictional-license.json"
        self.workspace = self.root / "review-workspace"
        self.license.write_text(json.dumps({
            "reviewer": "SYNTHETIC UNIT TEST ONLY",
            "reviewed_on": "2026-10-20",
            "license_pdf_sha256": OFFICIAL_LICENSE_SHA256,
            "permitted_scope": "research_only",
            "accept_terms": True,
        }), encoding="utf-8")
        self.calls = []

    def fake_annotations(self, sources, splits, output, license_ack, max_transfer_mib):
        self.calls.append(("metadata", tuple(sources), tuple(splits)))
        self.assertEqual(tuple(sources), SOURCES)
        self.assertEqual(tuple(splits), ("train",))
        folder = output / "train"
        folder.mkdir(parents=True, exist_ok=True)
        manifest = {"files": {}}
        for source in sources:
            data = {}
            for n in range(5):
                stem = f"FAKE_{source}_{n}"
                data[stem] = {
                    "user_id": f"PRIVATE_FAKE_USER_{source}_{n}",
                    "bboxes": [[0.05, 0.1, 0.4, 0.5]],
                    "labels": [source],
                    "united_bbox": None,
                    "united_label": None,
                    "meta": {"private": "NEVER_PUBLISH_FAKE_METADATA"},
                }
            (folder / f"{source}.json").write_text(json.dumps(data), encoding="utf-8")
            manifest["files"][f"annotations/train/{source}.json"] = {"sha256": "fake"}
        return manifest

    def fake_images(
        self, source, annotations, output, license_ack, *,
        per_source, max_transfer_mib, seed,
    ):
        self.calls.append(("images", source, per_source))
        output_dir = output / source
        output_dir.mkdir(parents=True, exist_ok=True)
        obj = json.loads((annotations / "train" / f"{source}.json").read_text())
        for stem in obj:
            (output_dir / f"{stem}.jpg").write_bytes(
                b"\xff\xd8\xff" + stem.encode("ascii") + b"\xff\xd9"
            )
        return {"source": source, "jpeg_count": per_source, "train_only": True}

    def prepare(self, **kwargs):
        with patch("scripts.p1_review_workflow.extract_annotations",
                   side_effect=self.fake_annotations):
            with patch("scripts.p1_review_workflow.extract_images",
                       side_effect=self.fake_images):
                return prepare(self.workspace, self.license, per_source=3, **kwargs)

    def sign_fake_review(self, alias="mismatch"):
        packet = self.workspace / "packet"
        form = json.loads((packet / "review.template.json").read_text())
        manifest = json.loads((packet / "packet.manifest.json").read_text())
        form["reviewer"] = "SYNTHETIC REVIEWER - NOT REAL APPROVAL"
        form["reviewed_on"] = "2026-10-20"
        for decision in form["decisions"]:
            source = decision["source"]
            decision["reviewed_packet_ids"] = [
                row["packet_id"] for row in manifest["samples"] if row["source"] == source
            ]
            decision["official_reference_checked"] = True
            decision["comparative_notes"] = (
                "Fake test comparison; no licensed images or real fingers inspected"
            )
            decision["decision"] = "match" if source == "two_up" else "mismatch"
            if source == "peace":
                decision["decision"] = alias
        review = self.root / "fake-signed-review.json"
        review.write_text(json.dumps(form))
        return review

    def test_offline_plan_has_no_license_or_network_dependency(self):
        with patch("scripts.p1_review_workflow.extract_annotations") as fetched:
            report = offline_plan()
            fetched.assert_not_called()
        self.assertEqual(report["status"], "PLAN_ONLY_NO_DATA_ACCESSED")
        self.assertEqual(report["source_categories"], list(SOURCES))
        self.assertEqual(report["model_output_classes"], 10)
        self.assertEqual(report["official_train_json_compressed_bytes"], 54_388_405)
        self.assertEqual(report["official_train_json_uncompressed_bytes"], 323_899_528)
        self.assertTrue(report["requires_real_owner_license_acceptance"])

    def test_synthetic_end_to_end_packet_is_atomic_and_private(self):
        result = self.prepare()
        self.assertEqual(result["status"], "PRIVATE_TRAIN_REVIEW_PACKET_READY_NOT_HUMAN_APPROVED")
        self.assertEqual(result["selected_per_source"], dict.fromkeys(SOURCES, 3))
        self.assertFalse(result["p1_pass"])
        self.assertTrue((self.workspace / "packet/index.html").is_file())
        self.assertEqual(len(list((self.workspace / "packet/images").glob("*.jpg"))), 12)
        self.assertEqual(len(self.calls), 5)
        self.assertEqual(self.calls[0], ("metadata", SOURCES, ("train",)))
        receipt = (self.workspace / "workflow.private.json").read_text()
        packet = (self.workspace / "packet/packet.manifest.json").read_text()
        html = (self.workspace / "packet/index.html").read_text()
        for forbidden in ("PRIVATE_FAKE_USER_", "NEVER_PUBLISH_FAKE_METADATA"):
            self.assertNotIn(forbidden, receipt + packet + html)
        self.assertEqual(json.loads(receipt)["split"], "train")
        self.assertEqual(json.loads(receipt)["source_model_classes"], 10)

    def test_explicit_human_review_record_still_not_full_p1_acceptance(self):
        self.prepare()
        evidence = self.sign_fake_review()
        result = verify(self.workspace, evidence)
        self.assertEqual(result["unresolved_sources"], [])
        self.assertEqual(result["sources_checked"], 4)
        self.assertEqual(result["status"], "HUMAN_REVIEW_RECORDED_NOT_P1_PASS")
        self.assertTrue(result["ten_class_P1_signoff_still_required"])
        self.assertFalse(result["p1_pass"])

    def test_peace_that_looks_like_command_is_reported_as_conflict(self):
        self.prepare()
        review = self.sign_fake_review(alias="match")
        result = verify(self.workspace, review)
        self.assertIn("peace", result["potential_command_alias_conflicts"])
        self.assertEqual(result["status"], "HUMAN_REVIEW_RECORDED_WITH_UNRESOLVED_MAPPINGS")
        self.assertFalse(result["p1_pass"])

    def test_unsigned_review_cannot_pass(self):
        self.prepare()
        with self.assertRaises(WorkflowError):
            verify(self.workspace, self.workspace / "packet/review.template.json")

    def test_missing_license_blocks_before_network_and_workspace(self):
        self.license.unlink()
        with patch("scripts.p1_review_workflow.extract_annotations") as fake:
            with self.assertRaisesRegex(WorkflowError, "permission"):
                prepare(self.workspace, self.license)
            fake.assert_not_called()
        self.assertFalse(self.workspace.exists())

    def test_public_ci_denied_before_authorization_or_network(self):
        with patch.dict("os.environ", {"GITHUB_ACTIONS": "true", "CI": "true"}):
            with patch("scripts.p1_review_workflow.extract_annotations") as fake:
                with self.assertRaisesRegex(WorkflowError, "forbidden in CI"):
                    prepare(self.workspace, self.license)
                fake.assert_not_called()
        self.assertFalse(self.workspace.exists())

    def test_partial_image_failure_rolls_back_all_staged_private_data(self):
        def fail_after_one(source, *args, **kwargs):
            if source == "peace":
                raise ValueError("FICTIONAL_ANNOTATION_CORRUPTED")
            return self.fake_images(source, *args, **kwargs)
        # The orchestrator intentionally wraps only its declared data-contract
        # errors; unexpected faults still unwind TemporaryDirectory cleanly.
        with patch("scripts.p1_review_workflow.extract_annotations",
                   side_effect=self.fake_annotations):
            with patch("scripts.p1_review_workflow.extract_images", side_effect=fail_after_one):
                with self.assertRaises(ValueError):
                    prepare(self.workspace, self.license, per_source=3)
        self.assertFalse(self.workspace.exists())
        self.assertEqual(list(self.root.glob(".p1-review-stage-*")), [])

    def test_missing_or_wrong_archive_members_prevent_image_download(self):
        def incomplete(sources, splits, output, license_ack, budget):
            return {"files": {"annotations/train/two_up.json": {}}}
        with patch("scripts.p1_review_workflow.extract_annotations", side_effect=incomplete):
            with patch("scripts.p1_review_workflow.extract_images") as fake:
                with self.assertRaisesRegex(WorkflowError, "incomplete"):
                    prepare(self.workspace, self.license, per_source=3)
                fake.assert_not_called()
        self.assertFalse(self.workspace.exists())

    def test_budgets_and_sample_caps_validated_without_network(self):
        for args in (
            {"per_source": 2},
            {"per_source": 21},
            {"annotation_budget_mib": 0},
            {"annotation_budget_mib": 300},
            {"jpg_budget_mib": 0},
            {"jpg_budget_mib": 513},
            {"seed": ""},
        ):
            with patch("scripts.p1_review_workflow.extract_annotations") as fake:
                with self.assertRaises(WorkflowError):
                    prepare(self.workspace, self.license, **args)
                fake.assert_not_called()
        self.assertFalse(self.workspace.exists())

    def test_existing_workspace_never_overwritten(self):
        self.prepare()
        with patch("scripts.p1_review_workflow.extract_annotations") as fake:
            with self.assertRaisesRegex(WorkflowError, "already exists"):
                prepare(self.workspace, self.license, per_source=3)
            fake.assert_not_called()

    def test_tampering_packet_image_invalidates_signed_review(self):
        self.prepare()
        evidence = self.sign_fake_review()
        first = next((self.workspace / "packet/images").glob("*.jpg"))
        first.write_bytes(b"tampered")
        with self.assertRaisesRegex(WorkflowError, "cannot be matched"):
            verify(self.workspace, evidence)

    def test_nonexistent_workspace_cannot_verify(self):
        with self.assertRaisesRegex(WorkflowError, "Missing/invalid"):
            verify(self.workspace, self.root / "fictional.json")


if __name__ == "__main__":
    unittest.main()
