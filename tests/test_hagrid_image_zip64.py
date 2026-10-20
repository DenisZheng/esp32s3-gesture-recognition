"""Synthetic-only ZIP64 tests. No real HaGRID images or annotations are fetched."""
import hashlib
import io
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.hagrid_p1 import OFFICIAL_LICENSE_SHA256
from scripts.hagrid_image_zip64 import (
    IMAGE_URLS, PINNED_INDEX_SHA256, PINNED_SIZES, HTTPRange, ImageRangeError,
    extract, inspect, parse_image_index, parse_zip64_footer, read_jpeg, _zip64_values
)


def fake_jpeg(label: str) -> bytes:
    # Handwritten sentinel bytes are NOT a real person, nor even a decoded image.
    return b"\xff\xd8\xff" + label.encode("ascii") * 20 + b"\xff\xd9"


def tiny_zip64(source="two_up", count=7, *, sizes_zip64=False):
    """Build a true ZIP64 EOCD + locator + central directory offset extras.

    zipfile writes small archives in ZIP32 by default, so transform their
    central index into ZIP64 structures without allocating massive data.
    """
    fp = io.BytesIO()
    samples = {}
    with zipfile.ZipFile(fp, "w", allowZip64=True, compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(source + "/", b"", compress_type=zipfile.ZIP_STORED)
        for i in range(count):
            stem = f"synthetic_{i:02d}"
            samples[stem] = fake_jpeg(stem)
            z.writestr(f"{source}/{stem}.jpg", samples[stem])
    original = fp.getvalue()
    eocd = original.rfind(b"PK\x05\x06")
    old = struct.unpack_from("<4s4H2IH", original, eocd)
    count_total, old_cd_size, old_cd_offset = old[4], old[5], old[6]
    old_cd = original[old_cd_offset:old_cd_offset + old_cd_size]
    pos, new_cd = 0, bytearray()
    for _ in range(count_total):
        fields = list(struct.unpack_from("<4s6H3I5H2I", old_cd, pos))
        nlen, xlen, clen = fields[10:13]
        name = old_cd[pos+46:pos+46+nlen]
        extra = old_cd[pos+46+nlen:pos+46+nlen+xlen]
        comment = old_cd[pos+46+nlen+xlen:pos+46+nlen+xlen+clen]
        if name.endswith(b".jpg"):
            packed = bytearray()
            if sizes_zip64:
                packed += struct.pack("<QQ", fields[9], fields[8])
                fields[9] = fields[8] = 0xFFFFFFFF
            packed += struct.pack("<Q", fields[16])
            fields[16] = 0xFFFFFFFF
            extra += struct.pack("<HH", 1, len(packed)) + packed
            fields[11] = len(extra)
        new_cd.extend(struct.pack("<4s6H3I5H2I", *fields) + name + extra + comment)
        pos += 46+nlen+xlen+clen
    assert pos == len(old_cd)
    prefix = original[:old_cd_offset]
    new_cd_offset = len(prefix)
    eocd64_offset = new_cd_offset + len(new_cd)
    record = struct.pack(
        "<4sQHHIIQQQQ", b"PK\x06\x06", 44, 45, 45, 0, 0,
        count_total, count_total, len(new_cd), new_cd_offset
    )
    locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, eocd64_offset, 1)
    eocd_classic = struct.pack(
        "<4s4H2IH", b"PK\x05\x06", 0, 0, count_total, count_total,
        0xFFFFFFFF, 0xFFFFFFFF, 0
    )
    return prefix + new_cd + record + locator + eocd_classic, samples


class FakeTransport:
    def __init__(self, data):
        self.data = data
        self.size = len(data)
    def get(self, start, end):
        if start < 0 or end >= self.size:
            raise ImageRangeError("Invalid test byte range")
        return self.data[start:end+1]


class ZIP64ParsingTests(unittest.TestCase):
    def setUp(self):
        self.raw, self.samples = tiny_zip64()
        self.transport = FakeTransport(self.raw)

    def test_all_four_image_archives_have_immutable_index_metadata(self):
        self.assertEqual(set(IMAGE_URLS), set(PINNED_SIZES))
        self.assertEqual(set(IMAGE_URLS), set(PINNED_INDEX_SHA256))
        self.assertEqual(len(IMAGE_URLS), 4)
        self.assertEqual(PINNED_SIZES["two_up"], 42_733_450_409)
        for source, digest in PINNED_INDEX_SHA256.items():
            self.assertEqual(len(digest), 64)
            self.assertTrue(all(ch in "0123456789abcdef" for ch in digest))
            self.assertGreater(PINNED_SIZES[source], 40_000_000_000)

    def test_modified_public_zip64_index_fails_immutable_baseline(self):
        # Force the real HTTP transport type with synthetic archive bytes:
        # the signature+ZIP structure is valid, but it is not the pinned
        # official directory and must be rejected.
        transport = HTTPRange.__new__(HTTPRange)
        transport.size = len(self.raw)
        transport.get = lambda start, end: self.transport.get(start, end)
        with self.assertRaisesRegex(ImageRangeError, "index changed"):
            inspect("two_up", transport)

    def test_fake_archive_has_real_zip64_record_and_offsets(self):
        self.assertIn(b"PK\x06\x06", self.raw)
        self.assertIn(b"PK\x06\x07", self.raw)
        cd_start, cd_size, count = parse_zip64_footer(self.raw, len(self.raw))
        self.assertEqual(count, 8)
        self.assertGreater(cd_size, 0)
        self.assertEqual(self.raw[cd_start:cd_start+4], b"PK\x01\x02")

    def test_parse_and_extract_seven_jpegs_with_crc(self):
        index, status = inspect("two_up", self.transport)
        self.assertEqual(len(index), 7)
        self.assertEqual(status["zip64_entries"], 8)
        self.assertEqual(status["jpeg_entries"], 7)
        self.assertEqual(len(status["central_directory_sha256"]), 64)
        self.assertEqual(
            {stem: read_jpeg(member, self.transport) for stem, member in index.items()},
            self.samples
        )

    def test_zip64_uncompressed_and_compressed_sizes(self):
        image_zip, _ = tiny_zip64(sizes_zip64=True)
        index, _ = inspect("two_up", FakeTransport(image_zip))
        self.assertEqual(len(index), 7)
        self.assertTrue(all(m.compressed < 4000 for m in index.values()))

    def test_another_pinned_source_archive(self):
        raw, _ = tiny_zip64(source="peace", count=3)
        index, out = inspect("peace", FakeTransport(raw))
        self.assertEqual(len(index), 3)
        self.assertEqual(out["source"], "peace")

    def test_bad_crc_trips_even_if_image_starts_like_jpeg(self):
        idx, _ = inspect("two_up", self.transport)
        member = next(iter(idx.values()))
        class Mutant:
            size = self.transport.size
            def get(self2, start, end):
                data = bytearray(self.transport.get(start, end))
                if end-start+1 == member.compressed:
                    data[len(data)//2] ^= 0x1
                return bytes(data)
        with self.assertRaises(ImageRangeError):
            read_jpeg(member, Mutant())

    def test_zip64_tail_missing_is_rejected(self):
        with self.assertRaises(ImageRangeError):
            parse_zip64_footer(b"PK" * 30, 60)

    def test_source_traversal_rejected(self):
        with self.assertRaises(ImageRangeError):
            inspect("../unsafe", self.transport)

    def test_missing_zip64_extra_is_rejected(self):
        with self.assertRaises(ImageRangeError):
            _zip64_values(b"", 1, 2, 0xFFFFFFFF, 0)

    def test_wrong_archive_size_or_partial_directory_trips(self):
        with self.assertRaises(ImageRangeError):
            parse_zip64_footer(self.raw, len(self.raw)+1000)

    def test_zip64_member_does_not_need_full_40gb(self):
        t = self.transport
        calls = []
        class CountingTransport:
            size = t.size
            def get(self2, a, b):
                calls.append(b-a+1)
                return t.get(a,b)
        idx, _ = inspect("two_up", CountingTransport())
        result = read_jpeg(next(iter(idx.values())), CountingTransport())
        self.assertTrue(result.startswith(b"\xff\xd8\xff"))
        self.assertLess(sum(calls), 15_000)
        # For a tiny synthetic ZIP the footer read includes the whole ZIP;
        # production archives read just the 64KiB tail + 3.92MiB index.
        self.assertLess(sum(calls), len(self.raw) + 5000)


class ExtractPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.anno = self.root / "annotations"
        self.output = self.root / "images"
        self.license = self.root / "approved-local-license.json"
        self.license.write_text(json.dumps({
            "reviewer": "SYNTHETIC TEST ONLY",
            "reviewed_on": "2026-10-20",
            "license_pdf_sha256": OFFICIAL_LICENSE_SHA256,
            "permitted_scope": "research_only",
            "accept_terms": True,
        }), encoding="utf-8")
        self.raw, self.samples = tiny_zip64()
        self.transport = FakeTransport(self.raw)
        folder = self.anno / "train"
        folder.mkdir(parents=True)
        records = {
            stem: {
                "user_id": f"FAKE_PERSON_{stem}",
                "bboxes": [[0.1,0.2,0.25,0.3]],
                "labels": ["two_up"],
                "united_bbox": None, "united_label": None,
                "meta": {"demographic": "SYNTHETIC_NEVER_EXPORTED"}
            } for stem in self.samples
        }
        (folder / "two_up.json").write_text(json.dumps(records), encoding="utf-8")
        (self.anno / "test").mkdir()
        (self.anno / "test" / "two_up.json").write_text("BANNED_TEST_DATA")

    def run_extract(self, **kwargs):
        return extract("two_up", self.anno, self.output, self.license,
                       per_source=3, transport=self.transport, **kwargs)

    def test_extract_exactly_three_private_images(self):
        result = self.run_extract()
        self.assertEqual(result["status"], "PRIVATE_SAMPLE_EXTRACTED_NOT_VISUALLY_APPROVED")
        self.assertTrue(result["train_only"])
        names = list((self.output / "two_up").glob("*.jpg"))
        self.assertEqual(len(names), 3)
        doc = json.loads((self.output / "two_up/_provenance.private.json").read_text())
        self.assertEqual(len(doc["selected_images"]), 3)
        self.assertIsNone(doc["archive_sha256"])
        self.assertEqual(len(doc["zip64_central_directory_sha256"]), 64)
        serialized = json.dumps(result) + json.dumps(doc)
        self.assertNotIn("FAKE_PERSON_", serialized)
        self.assertNotIn("SYNTHETIC_NEVER_EXPORTED", serialized)
        self.assertNotIn("BANNED_TEST_DATA", serialized)
        for f in names:
            self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(),
                             doc["selected_images"][f.name]["sha256"])

    def test_repeat_selection_is_deterministic(self):
        first = self.run_extract()
        second = extract("two_up", self.anno, self.root / "other-images",
                         self.license, 3, transport=self.transport)
        a = json.loads((self.output / "two_up/_provenance.private.json").read_text())
        b = json.loads((self.root / "other-images/two_up/_provenance.private.json").read_text())
        self.assertEqual(sorted(a["selected_images"]), sorted(b["selected_images"]))
        self.assertEqual(first["jpeg_count"], second["jpeg_count"])

    def test_unlicensed_source_cannot_trigger_network_or_write(self):
        self.license.unlink()
        class FakeNetwork:
            def get(self, *_):
                raise AssertionError("Must not fetch")
        with self.assertRaises(ImageRangeError):
            extract("two_up", self.anno, self.output, self.license,
                    3, transport=FakeNetwork())
        self.assertFalse(self.output.exists())

    def test_budget_rejected_before_any_jpg_fetch(self):
        fake = self.transport
        class CountGet:
            size=fake.size
            def get(self, start, end):
                # Footer/index GET okay; member HTTP GET must never happen
                if start < 1000:
                    raise AssertionError("Budget must reject before JPEG payload")
                return fake.get(start,end)
        with patch("scripts.hagrid_image_zip64.inspect") as inspected:
            index, details = inspect("two_up", self.transport)
            big = {k: type(m)(m.filename,m.offset,8*1024*1024,
                              m.uncompressed,m.crc32,m.method,m.flags)
                   for k,m in index.items()}
            inspected.return_value = big, details
            with self.assertRaisesRegex(ImageRangeError, "transfer budget"):
                extract("two_up", self.anno, self.output, self.license,
                        3, max_transfer_mib=1, transport=CountGet())
        self.assertFalse((self.output / "two_up").exists())

    def test_output_overwrite_never_allowed(self):
        self.run_extract()
        with self.assertRaisesRegex(ImageRangeError, "overwrite"):
            self.run_extract()

    def test_insufficient_distinct_subjects_rejected(self):
        p=self.anno/"train/two_up.json"
        doc=json.loads(p.read_text())
        for item in doc.values():
            item["user_id"]="SAME_FAKE_SUBJECT"
        p.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ImageRangeError, "Not enough distinct"):
            self.run_extract()
        self.assertFalse((self.output/"two_up").exists())

    def test_no_test_split_inspected(self):
        self.run_extract()
        self.assertTrue((self.anno/"test/two_up.json").is_file())
        self.assertEqual((self.anno/"test/two_up.json").read_text(), "BANNED_TEST_DATA")

    def test_uncorrelated_archive_key_cannot_be_used(self):
        p=self.anno/"train/two_up.json"
        doc=json.loads(p.read_text())
        for item in doc.values():
            item["labels"]=["fist"]
        p.write_text(json.dumps(doc))
        with self.assertRaises(ImageRangeError):
            self.run_extract()

    def test_unknown_url_is_rejected_before_HEAD(self):
        with self.assertRaises(ImageRangeError):
            HTTPRange("https://evil.invalid/steal.zip")

    def test_public_github_ci_blocks_real_image_fetch_before_network(self):
        # Injected fake transport is allowed for synthetic unit tests; actual
        # authenticated image acquisition must never run on public GitHub CI.
        with patch.dict("os.environ", {"GITHUB_ACTIONS": "true"}):
            with patch("scripts.hagrid_image_zip64.HTTPRange") as backend:
                with self.assertRaisesRegex(ImageRangeError, "forbidden in CI"):
                    extract("two_up", self.anno, self.output, self.license,
                            per_source=3, transport=None)
                backend.assert_not_called()
        self.assertFalse((self.output / "two_up").exists())

    def test_http_200_ignored_range_never_reads_image(self):
        class InvalidResponse:
            status=200
            headers={"Content-Range":"bytes 0-9/42733450409"}
            def __enter__(self): return self
            def __exit__(self,*_): return None
            def read(self,n): raise AssertionError("Must not call read on 200")
        # Bypass HEAD via __new__ so the test has no actual HTTP call
        client=HTTPRange.__new__(HTTPRange)
        client.url=IMAGE_URLS["two_up"]
        client.size=42_733_450_409
        client.etag=None
        client.modified=None
        with patch("scripts.hagrid_image_zip64.urllib.request.urlopen",
                   return_value=InvalidResponse()):
            with self.assertRaisesRegex(ImageRangeError, "ignored HTTP Range"):
                client.get(0,9)


if __name__ == "__main__":
    unittest.main()
