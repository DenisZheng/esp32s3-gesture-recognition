"""P1-06 campaign synthetic-only local tests: no real HaGRID images or network."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.data_contract import CLASS_NAMES, SOURCE_TO_INDEX
from scripts.hagrid_p1 import OFFICIAL_LICENSE_SHA256, PreparationError, _approval
from scripts.p1_visual_review import build_packet
from scripts.p1_ten_review import (
    ALL_SOURCES, GROUPS, SCHEMA, TenReviewError,
    draft, offline_plan, prepare, validate_groups, verify,
)
from scripts.hagrid_image_zip64 import PINNED_INDEX_SHA256
from scripts.hagrid_policy import policy_sha256


def fictional_jpeg(name: str) -> bytes:
    return b"\xff\xd8\xff" + name.encode("ascii") * 2 + b"\xff\xd9"


class ReviewCampaignContractTests(unittest.TestCase):
    def test_plan_is_offline_and_ten_outputs(self):
        with patch("scripts.p1_ten_review.fetch_train_annotations") as fetch:
            plan = offline_plan()
            fetch.assert_not_called()
        self.assertEqual(plan["unique_source_folders"], 20)
        self.assertEqual(plan["gesture_output_classes"], list(CLASS_NAMES))
        self.assertEqual(plan["groups"]["two"], [
            "two_up", "peace", "peace_inverted", "two_up_inverted"
        ])
        self.assertFalse(plan["dataset_and_model_training_authorized"])

    def test_all_ten_outputs_and_exact_comparison_variants_present(self):
        validate_groups()
        self.assertEqual(len(ALL_SOURCES), 20)
        self.assertEqual(len(set(ALL_SOURCES)), 20)
        self.assertEqual(set(SOURCE_TO_INDEX) & set(
            ("peace", "three2", "stop", "point")), set()
        )
        self.assertEqual(set(GROUPS), {
            "one", "two", "three", "five", "remaining"
        })


@patch.dict(os.environ, {"CI": "false", "GITHUB_ACTIONS": "false"})
class PrivateReviewCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.license = self.root / "private-license.json"
        self.license.write_text(json.dumps({
            "reviewer": "SYNTHETIC TEST ONLY",
            "reviewed_on": "2026-10-21",
            "license_pdf_sha256": OFFICIAL_LICENSE_SHA256,
            "permitted_scope": "research_only",
            "accept_terms": True,
        }))
        self.all = []

    def make_group(self, group: str, *, alias_conflict: str | None = None):
        workspace = self.root / ("workspace-" + group)
        photos = workspace / "images"
        anns = workspace / "annotations/train"
        packet = workspace / "packet"
        photos.mkdir(parents=True)
        anns.mkdir(parents=True)
        for source in GROUPS[group]:
            records = {}
            folder = photos / source
            folder.mkdir()
            for i in range(3):
                stem = f"SYNTHETIC_{group}_{source}_{i}"
                records[stem] = {
                    "user_id": f"SYNTHETIC_SUBJECT_{group}_{source}_{i}",
                    "bboxes": [[0.1, 0.2, 0.3, 0.4]], "labels": [source],
                    "united_bbox": None, "united_label": None,
                    "meta": {"private": "NEVER_INCLUDE"},
                }
                (folder / (stem + ".jpg")).write_bytes(fictional_jpeg(stem))
            (anns / (source + ".json")).write_text(json.dumps(records))
        result = build_packet(
            workspace / "annotations", photos, packet, self.license,
            sources=GROUPS[group], per_source=3, seed="SYNTHETIC"
        )
        self.assertEqual(set(result["counts_by_source"]), set(GROUPS[group]))
        self.assertTrue(all(n == 3 for n in result["counts_by_source"].values()))
        receipt = {
            "schema_version": SCHEMA, "group": group,
            "status": "GROUP_PACKET_CREATED_NOT_APPROVED",
            "sources": list(GROUPS[group]), "split": "train", "model_classes": 10,
            "packet_manifest_sha256": result["manifest_sha256"],
            "policy_sha256": policy_sha256(), "p1_pass": False,
            "official_index_sha256": {
                name: PINNED_INDEX_SHA256[name] for name in GROUPS[group]
            },
        }
        (workspace / "campaign.private.json").write_text(json.dumps(receipt))
        signed = json.loads((packet / "review.template.json").read_text())
        signed["reviewer"] = "SYNTHETIC reviewer NOT human"
        signed["reviewed_on"] = "2026-10-21"
        manifest = json.loads((packet / "packet.manifest.json").read_text())
        by_source = {}
        for sample in manifest["samples"]:
            by_source.setdefault(sample["source"], []).append(sample["packet_id"])
        for row in signed["decisions"]:
            source = row["source"]
            row["decision"] = (
                "match" if source in SOURCE_TO_INDEX or source == alias_conflict
                else "mismatch"
            )
            row["reviewed_packet_ids"] = by_source[source]
            row["official_reference_checked"] = True
            row["comparative_notes"] = (
                "Synthetic fake evidence; this is NOT an actual visual assessment"
            )
        review = self.root / (group + "-signed-SYNTHETIC.json")
        review.write_text(json.dumps(signed))
        self.all.append((group, workspace, review))
        return workspace, review

    def test_group_packet_binds_signed_choices_to_private_files(self):
        w, r = self.make_group("two")
        result = verify("two", w, r)
        self.assertEqual(result["sources_checked"], 4)
        self.assertEqual(result["reviewed_images"], 12)
        self.assertEqual(result["unresolved_sources"], [])
        self.assertEqual(result["potential_command_alias_conflicts"], [])
        self.assertFalse(result["p1_pass"])

    def test_image_modification_invalidates_group_receipt(self):
        w, r = self.make_group("one")
        image = next((w / "packet/images").glob("*.jpg"))
        image.write_bytes(b"tampered")
        with self.assertRaises(TenReviewError):
            verify("one", w, r)

    def test_unsigned_template_cannot_be_verified(self):
        w, _ = self.make_group("five")
        with self.assertRaises(TenReviewError):
            verify("five", w, w / "packet/review.template.json")

    def test_all_five_groups_yield_explicitly_invalid_final_mapping_draft(self):
        for group in GROUPS:
            self.make_group(group)
        out = self.root / "final-campaign"
        result = draft(self.all, out)
        self.assertEqual(result["source_categories_checked"], 20)
        self.assertEqual(result["review_groups"], 5)
        self.assertEqual(result["unresolved_sources"], [])
        self.assertEqual(result["lookalike_alias_questions"], [])
        self.assertFalse(result["p1_pass"])
        ackpath = out / "hagrid-mapping-ack.UNSIGNED.json"
        document = json.loads(ackpath.read_text())
        self.assertEqual(len(document["classes"]), 10)
        self.assertEqual({x["source"] for x in document["classes"]}, set(SOURCE_TO_INDEX))
        self.assertTrue(all(x["approved"] is False for x in document["classes"]))
        self.assertTrue(all(x["official_reference_checked"] is False
                            for x in document["classes"]))
        with self.assertRaises(PreparationError):
            _approval(ackpath, "mapping")
        private = (out / "campaign.evidence.private.json").read_text()
        self.assertNotIn("NEVER_INCLUDE", private)
        self.assertNotIn("SYNTHETIC_SUBJECT", private)

    def test_alias_conflict_appears_as_unresolved_and_never_approves(self):
        for group in GROUPS:
            self.make_group(group, alias_conflict="peace" if group == "two" else None)
        res = draft(self.all, self.root / "candidate-draft")
        self.assertIn("peace", res["lookalike_alias_questions"])
        self.assertEqual(res["status"], "REVIEW_CONFLICTS_REQUIRE_ADJUDICATION")
        signed = json.loads((self.root / "candidate-draft/hagrid-mapping-ack.UNSIGNED.json").read_text())
        self.assertTrue(all(x["approved"] is False for x in signed["classes"]))

    def test_missing_group_refuses_draft(self):
        for group in ("one", "two", "three", "five"):
            self.make_group(group)
        with self.assertRaisesRegex(TenReviewError, "five groups"):
            draft(self.all, self.root / "missing")
        self.assertFalse((self.root / "missing").exists())

    def test_duplicate_group_refuses_draft(self):
        w, r = self.make_group("two")
        with self.assertRaises(TenReviewError):
            draft([("two", w, r)] * 5, self.root / "duplicate")

    def test_preparation_fails_in_public_ci_before_real_data(self):
        with patch.dict(os.environ, {"CI": "true", "GITHUB_ACTIONS": "true"}):
            with patch("scripts.p1_ten_review.fetch_train_annotations") as call:
                with self.assertRaisesRegex(TenReviewError, "cannot be acquired in CI"):
                    prepare("two", self.root / "ci-forbidden", self.license)
                call.assert_not_called()

    def test_no_legal_signoff_fails_before_network_or_media(self):
        self.license.unlink()
        with patch("scripts.p1_ten_review.fetch_train_annotations") as call:
            with self.assertRaisesRegex(TenReviewError, "license acceptance"):
                prepare("one", self.root / "no-permission", self.license)
            call.assert_not_called()

    def test_bad_jpg_count_stops_before_network(self):
        for amount in (0, 2, 21, True):
            with patch("scripts.p1_ten_review.fetch_train_annotations") as call:
                with self.assertRaises(TenReviewError):
                    prepare("two", self.root / "bad-count", self.license,
                            per_source=amount)
                call.assert_not_called()

    def test_source_receipt_mismatch_fails_closed_and_discards_stage(self):
        out = self.root / "rollback"
        fake = {"files": {"annotations/train/one.json": {"sha256": "fake"}}}
        with patch("scripts.p1_ten_review.fetch_train_annotations", return_value=fake):
            with patch("scripts.p1_ten_review.fetch_image_subset") as images:
                with self.assertRaises(TenReviewError):
                    prepare("two", out, self.license)
                images.assert_not_called()
        self.assertFalse(out.exists())

    def test_no_existing_workspace_is_overwritten(self):
        out = self.root / "exists"
        out.mkdir()
        with self.assertRaisesRegex(TenReviewError, "exists"):
            prepare("two", out, self.license)


if __name__ == "__main__":
    unittest.main()
