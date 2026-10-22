"""Synthetic policy tests: no HaGRID images, identities, annotations or downloads."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.data_contract import SOURCE_TO_INDEX
from scripts.hagrid_policy import (
    GROUPS, OFFICIAL_SOURCES, PolicyError, approve_hard_negatives,
    policy_sha256, read_policy, validate_requested_sources,
)


class SourcePolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "private-approval.json"
        self.record = {
            "reviewer": "synthetic reviewer",
            "reviewed_on": "2026-10-20",
            "policy_sha256": policy_sha256(),
            "approved_sources": [{
                "source": "fist",
                "output": "no_gesture",
                "approved": True,
                "samples_reviewed": 3,
                "licensed_images_reviewed": True,
                "per_hand_boxes_reviewed": True,
            }],
        }
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.record), encoding="utf-8")

    def test_official_34_partition_and_exact_ten_model_outputs(self):
        p = read_policy()
        self.assertEqual(len(OFFICIAL_SOURCES), 34)
        self.assertEqual(set(p["primary_sources"]), set(SOURCE_TO_INDEX))
        self.assertEqual([len(p[k]) for k in GROUPS], [10, 7, 10, 7])
        self.assertEqual(set.union(*(set(p[k]) for k in GROUPS)), OFFICIAL_SOURCES)
        self.assertFalse(p["no_hand_is_model_class"])

    def test_ten_output_mapping_remains_unchanged(self):
        mapping = SOURCE_TO_INDEX
        self.assertEqual(mapping["no_gesture"], 7)
        # Owner visual review 2026-10-22: ESP-DL "two" is a separated V sign.
        self.assertEqual(mapping["peace"], 1)
        self.assertNotIn("two_up", mapping)
        self.assertEqual(mapping["palm"], 4)

    def test_peace_is_primary_but_inverted_and_two_up_quarantined(self):
        p = read_policy()
        self.assertIn("peace", p["primary_sources"])
        for source in ("palm", "peace", "one", "three", "four", "like", "ok",
                       "no_gesture", "call", "dislike"):
            self.assertIn(source, p["primary_sources"])
        self.assertIn("peace_inverted", p["lookalike_quarantine"])
        self.assertIn("two_up", p["lookalike_quarantine"])
        self.assertIn("two_up_inverted", p["lookalike_quarantine"])
        with self.assertRaises(PolicyError):
            validate_requested_sources(("peace_inverted",), conversion=True)
        with self.assertRaises(PolicyError):
            validate_requested_sources(("two_up",), conversion=True)

    def test_compound_hand_heart_cannot_be_auto_converted(self):
        self.assertIn("hand_heart", read_policy()["compound_quarantine"])
        with self.assertRaises(PolicyError):
            validate_requested_sources(("hand_heart",), conversion=True)

    def test_audit_can_inspect_quarantined_source_without_converting(self):
        self.assertEqual(validate_requested_sources(("peace",)), frozenset())
        self.assertEqual(validate_requested_sources(("fist",), conversion=True),
                         frozenset({"fist"}))
        self.assertEqual(validate_requested_sources(("one", "fist"), conversion=True),
                         frozenset({"fist"}))

    def test_unknown_and_duplicate_sources_rejected(self):
        for names in [("made_up",), (), ("fist", "fist")]:
            with self.assertRaises(PolicyError):
                validate_requested_sources(names)

    def test_unsigned_negative_rejected(self):
        self.path.unlink()
        with self.assertRaises(PolicyError):
            approve_hard_negatives(self.path, frozenset({"fist"}))

    def test_valid_signed_single_source(self):
        self.assertEqual(
            approve_hard_negatives(self.path, frozenset({"fist"})),
            frozenset({"fist"}),
        )

    def test_mismatched_requested_sources_rejected(self):
        with self.assertRaises(PolicyError):
            approve_hard_negatives(self.path, frozenset({"fist", "rock"}))

    def test_stale_policy_hash_rejected(self):
        self.record["policy_sha256"] = "0" * 64
        self.save()
        with self.assertRaises(PolicyError):
            approve_hard_negatives(self.path, frozenset({"fist"}))

    def test_boolean_or_string_review_count_rejected(self):
        for value in (True, "3", 0, -1, None):
            self.record["approved_sources"][0]["samples_reviewed"] = value
            self.save()
            with self.assertRaises(PolicyError):
                approve_hard_negatives(self.path, frozenset({"fist"}))

    def test_nonapproved_visual_signoff_rejected(self):
        for field in ("approved", "licensed_images_reviewed",
                      "per_hand_boxes_reviewed"):
            previous = self.record["approved_sources"][0][field]
            self.record["approved_sources"][0][field] = False
            self.save()
            with self.assertRaises(PolicyError):
                approve_hard_negatives(self.path, frozenset({"fist"}))
            self.record["approved_sources"][0][field] = previous

    def test_relabeling_lookalike_as_negative_is_forbidden(self):
        self.record["approved_sources"][0]["source"] = "peace"
        self.save()
        with self.assertRaises(PolicyError):
            approve_hard_negatives(self.path, frozenset({"peace"}))

    def test_relabeling_two_hand_compound_as_negative_is_forbidden(self):
        self.record["approved_sources"][0]["source"] = "xsign"
        self.save()
        with self.assertRaises(PolicyError):
            approve_hard_negatives(self.path, frozenset({"xsign"}))

    def test_duplicate_approval_rows_rejected(self):
        self.record["approved_sources"].append(dict(self.record["approved_sources"][0]))
        self.save()
        with self.assertRaises(PolicyError):
            approve_hard_negatives(self.path, frozenset({"fist"}))

    def test_review_date_must_be_actual_date(self):
        self.record["reviewed_on"] = "2026-02-30"
        self.save()
        with self.assertRaises(PolicyError):
            approve_hard_negatives(self.path, frozenset({"fist"}))


if __name__ == "__main__":
    unittest.main()
