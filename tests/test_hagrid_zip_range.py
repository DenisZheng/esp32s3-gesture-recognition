"""Offline-only tests; build tiny synthetic ZIPs with no real people or metadata."""
import io
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.hagrid_p1 import PreparationError
from scripts.hagrid_zip_range import (
    ARCHIVE_BYTES, Member, extract, get_range, parse_directory,
    read_member, select_members,
)


@patch.dict("os.environ", {"CI": "false", "GITHUB_ACTIONS": "false"})
class HaGRIDRangeZipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.synthetic_data = {}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for split in ("train", "val", "test"):
                for cls in ("one", "no_gesture", "fist"):
                    key = f"annotations/{split}/{cls}.json"
                    content = json.dumps({"synthetic": {
                        "user_id": f"FAKE-NEVER-UPLOAD-{split}-{cls}",
                        "bboxes": [[0.1, 0.2, 0.3, 0.4]],
                        "labels": [cls],
                    }}).encode()
                    self.synthetic_data[key] = content
                    z.writestr(key, content)
            z.writestr("README.txt", b"irrelevant")
            z.writestr("../unsafe.json", b"NEVER")
        self.raw_zip = buffer.getvalue()

    def _members(self):
        return parse_directory(self.raw_zip[-65536:], len(self.raw_zip))

    def _getter(self, start, end):
        return self.raw_zip[start:end+1]

    def test_directory_reads_only_canonical_classes(self):
        members = self._members()
        self.assertEqual(len(members), 9)
        self.assertNotIn("../unsafe.json", members)
        self.assertTrue(all(m.method == 8 for m in members.values()))

    def test_selection_uses_requested_splits_and_sources(self):
        chosen = select_members(self._members(), ("train", "test"), ("no_gesture",))
        self.assertEqual([m.path for m in chosen], [
            "annotations/train/no_gesture.json",
            "annotations/test/no_gesture.json",
        ])

    def test_select_reviewable_negative_metadata_without_changing_models(self):
        chosen = select_members(self._members(), ("train", "val"), ("fist",))
        self.assertEqual([m.path for m in chosen], [
            "annotations/train/fist.json", "annotations/val/fist.json"
        ])

    def test_selected_member_recovers_original_json_with_crc(self):
        for name, member in self._members().items():
            self.assertEqual(read_member(member, getter=self._getter), self.synthetic_data[name])

    def test_corrupt_compressed_payload_fails(self):
        member = self._members()["annotations/train/one.json"]
        def bad_getter(start, end):
            data = bytearray(self._getter(start, end))
            if end-start+1 == member.compressed:
                data[len(data)//2] ^= 0x55
            return bytes(data)
        with self.assertRaises(PreparationError):
            read_member(member, getter=bad_getter)

    def test_invalid_or_duplicate_class_selections_rejected(self):
        for splits, sources in [
            (("train", "train"), ("one",)),
            (("train",), ("one", "one")),
            (("train",), ("bad",)),
            (("bad",), ("one",)),
            ((), ("one",)),
        ]:
            with self.assertRaises(PreparationError):
                select_members(self._members(), splits, sources)

    def test_unselected_large_annotation_does_not_break_zip_index(self):
        b = bytearray(self.raw_zip)
        i = b.find(b"PK\x01\x02")
        self.assertGreaterEqual(i, 0)
        # Simulates a large *other class* in the official 121-entry archive.
        struct.pack_into("<I", b, i+20, 40 * 1024 * 1024)
        members = parse_directory(bytes(b)[-65536:], len(b))
        chosen = select_members(members, ("train",), ("no_gesture",))
        self.assertEqual(len(chosen), 1)
        with self.assertRaisesRegex(PreparationError, "Selected ZIP member exceeds"):
            select_members(members, ("train",), ("one",))
        with patch("scripts.hagrid_zip_range.get_range") as fake:
            with self.assertRaisesRegex(PreparationError, "Selected ZIP member exceeds"):
                read_member(members["annotations/train/one.json"])
            fake.assert_not_called()

    def test_truncated_tail_rejected(self):
        with self.assertRaises(PreparationError):
            parse_directory(b"PK", len(self.raw_zip))

    def test_ci_fails_before_any_private_annotation_download(self):
        with patch.dict("os.environ", {"CI": "true", "GITHUB_ACTIONS": "true"}):
            with patch("scripts.hagrid_zip_range.fetch_directory") as fetch:
                with patch("scripts.hagrid_zip_range._approval") as signed:
                    with self.assertRaisesRegex(PreparationError, "forbidden in CI"):
                        extract(("one",), ("train",), self.root / "no",
                                self.root / "review.json", 5)
                    fetch.assert_not_called()
                    signed.assert_not_called()

    def test_unlicensed_extraction_refuses_before_fetch(self):
        with patch("scripts.hagrid_zip_range.fetch_directory") as fake:
            with self.assertRaises(PreparationError):
                extract(("no_gesture",), ("train",), self.root/"local", self.root/"missing", 5)
            fake.assert_not_called()

    def test_local_only_atomic_manifest_contains_no_raw_subjects(self):
        output = self.root / "prepared"
        with patch("scripts.hagrid_zip_range._approval", return_value={}):
            with patch("scripts.hagrid_zip_range.fetch_directory", return_value=self._members()):
                with patch("scripts.hagrid_zip_range.read_member",
                           side_effect=lambda m: self.synthetic_data[m.path]):
                    result = extract(
                        ("no_gesture",), ("train", "val", "test"),
                        output, self.root/"approved.json", 5
                    )
        self.assertEqual(len(result["files"]), 3)
        self.assertIsNone(result["archive_sha256"])
        self.assertEqual(
            (output/"train/no_gesture.json").read_bytes(),
            self.synthetic_data["annotations/train/no_gesture.json"]
        )
        self.assertNotIn("FAKE-NEVER-UPLOAD", json.dumps(result))
        self.assertNotIn("FAKE-NEVER-UPLOAD", (output/"_provenance.aggregate.json").read_text())
        with patch("scripts.hagrid_zip_range._approval", return_value={}):
            with self.assertRaises(PreparationError):
                extract(("no_gesture",), ("train",), output, self.root/"approved.json", 5)

    def test_selection_budget_refuses_bulk_download(self):
        too_large = Member("annotations/train/no_gesture.json", 6 * 1024 * 1024,
                           10_000_000, 0, 0, 8, 0)
        with patch("scripts.hagrid_zip_range._approval", return_value={}):
            with patch("scripts.hagrid_zip_range.fetch_directory",
                       return_value={too_large.path: too_large}):
                with patch("scripts.hagrid_zip_range.read_member") as fake:
                    with self.assertRaises(PreparationError):
                        extract(("no_gesture",), ("train",),
                                self.root/"no", self.root/"approved.json", 1)
                    fake.assert_not_called()

    def test_non_partial_response_cannot_download_whole_zip(self):
        class Dummy:
            status = 200
            headers = {"Content-Range": f"bytes 0-9/{ARCHIVE_BYTES}"}
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read(self, count): return b"abc"
        with patch("scripts.hagrid_zip_range.urllib.request.urlopen", return_value=Dummy()):
            with self.assertRaisesRegex(PreparationError, "ignored Range"):
                get_range(0, 9)


if __name__ == "__main__":
    unittest.main()
