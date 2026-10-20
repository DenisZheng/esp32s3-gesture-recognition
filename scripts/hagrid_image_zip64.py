#!/usr/bin/env python3
"""Private, bounded HTTP Range extraction of HaGRIDv2 JPGs in a ZIP64 archive.

Never fetches a whole 40+ GB image ZIP or any personal images in GitHub CI.
Inspection reads only a 64 KiB ZIP64 footer and <=8 MiB central index.
Image extraction requires *local*, signed HaGRID research-license acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import tempfile
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path

from scripts.hagrid_p1 import ROOT, _approval
from scripts.p1_visual_review import (
    VisualReviewError, _candidate_rank, _clean_record, _load_annotation
)

HOST = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru"
IMAGE_URLS = {
    source: (HOST + "/datasets/hagrid/hagrid_dataset_new_554800/"
             "hagrid_dataset/" + source + ".zip")
    for source in ("two_up", "peace", "peace_inverted", "two_up_inverted")
}
# This particular archive size was measured via HEAD and checked against
# Content-Range in real GitHub Actions on 2026-10-20. Other source archives
# are pinned by their fixed URL + per-session HTTP validators/CRC.
PINNED_SIZES = {"two_up": 42_733_450_409}
MAX_ARCHIVE_BYTES = 80 * 1024 ** 3
MAX_CENTRAL_BYTES = 8 * 1024 ** 2
MAX_RANGE_BYTES = 24 * 1024 ** 2
MAX_JPG_COMPRESSED = 16 * 1024 ** 2
MAX_JPG_DECOMPRESSED = 20 * 1024 ** 2
MAX_SELECTED = 20
MIN_SELECTED = 3
TAIL_BYTES = 65_536
STEM_RE = re.compile(r"^[A-Za-z0-9_-]+$")
RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


class ImageRangeError(ValueError):
    """Invalid archive, unauthorized access, or unsafe local media extraction."""


@dataclass(frozen=True)
class ImageMember:
    filename: str  # canonical source/stem.jpg, never a filesystem input
    offset: int
    compressed: int
    uncompressed: int
    crc32: int
    method: int
    flags: int


class HTTPRange:
    """Fixed HTTPS provenance with fail-closed single-session byte-range reads."""

    def __init__(self, url: str, expected_size: int | None = None):
        if url not in IMAGE_URLS.values():
            raise ImageRangeError("Archive URL is not an official pinned source URL")
        self.url = url
        req = urllib.request.Request(url, method="HEAD", headers={
            "Accept-Encoding": "identity", "User-Agent": "ESPDetP1-Metadata/1"
        })
        try:
            with urllib.request.urlopen(req, timeout=45) as res:
                if res.status != 200:
                    raise ImageRangeError("Unexpected HTTP status on image archive HEAD")
                self.size = int(res.headers["Content-Length"])
                self.etag = res.headers.get("ETag")
                self.modified = res.headers.get("Last-Modified")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ImageRangeError("Image archive HEAD failed") from exc
        if not 100_000 <= self.size <= MAX_ARCHIVE_BYTES:
            raise ImageRangeError("Unsafe source archive size")
        if expected_size is not None and self.size != expected_size:
            raise ImageRangeError("Source archive differs from measured pinned size")
        if self.etag and self.etag.startswith("W/"):
            self.etag = None  # weak ETags are unsuitable for If-Match

    def get(self, start: int, end: int) -> bytes:
        length = end - start + 1
        if start < 0 or end >= self.size or length < 1 or length > MAX_RANGE_BYTES:
            raise ImageRangeError("Invalid or over-budget HTTP Range request")
        headers = {
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
            "User-Agent": "ESPDetP1-Metadata/1",
        }
        if self.etag:
            headers["If-Match"] = self.etag
        elif self.modified:
            headers["If-Unmodified-Since"] = self.modified
        req = urllib.request.Request(self.url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as res:
                # Check this BEFORE reading even one payload byte. HTTP 200
                # could otherwise start downloading tens of gigabytes.
                if res.status != 206:
                    raise ImageRangeError("Server ignored HTTP Range; refusing bulk download")
                match = RANGE_RE.fullmatch(res.headers.get("Content-Range", "").strip())
                if (not match or tuple(int(n) for n in match.groups())
                        != (start, end, self.size)):
                    raise ImageRangeError("Archive length or Content-Range changed")
                if self.etag and res.headers.get("ETag") not in (None, self.etag):
                    raise ImageRangeError("Archive ETag changed during extraction")
                if self.modified and res.headers.get("Last-Modified") not in (
                    None, self.modified
                ):
                    raise ImageRangeError("Archive last-modified changed during extraction")
                result = res.read(length + 1)
        except ImageRangeError:
            raise  # Preserve safety failure: HTTP 200/invalid Range must be explicit.
        except (OSError, ValueError) as exc:
            raise ImageRangeError("Image archive HTTP byte-range read failed") from exc
        if len(result) != length:
            raise ImageRangeError("Short or oversized HTTP byte-range payload")
        return result


def parse_zip64_footer(tail: bytes, archive_size: int) -> tuple[int, int, int]:
    """Return central (offset, size, entry count) without exposing image names."""
    start = archive_size - len(tail)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 20 or eocd + 22 != len(tail):
        raise ImageRangeError("Missing or commented ZIP64 end-of-directory")
    marker, disk, cdisk, entries_disk, entries, cd_size32, cd_start32, comment = (
        struct.unpack_from("<4s4H2IH", tail, eocd)
    )
    if marker != b"PK\x05\x06" or disk or cdisk or comment:
        raise ImageRangeError("Multidisk or nonstandard image archive")
    loc_pos = eocd - 20
    locator, loc_disk, eocd64_offset, total_disks = struct.unpack_from(
        "<4sIQI", tail, loc_pos
    )
    if locator != b"PK\x06\x07" or loc_disk or total_disks != 1:
        raise ImageRangeError("Missing valid ZIP64 locator")
    if eocd64_offset < start or eocd64_offset + 56 > start + loc_pos:
        raise ImageRangeError("ZIP64 record unavailable in bounded footer")
    fields = struct.unpack_from("<4sQHHIIQQQQ", tail, eocd64_offset - start)
    (signature, record_len, _made, _need, disk64, cdisk64, local_entries,
     total_entries, cd_size, cd_offset) = fields
    if (
        signature != b"PK\x06\x06" or record_len < 44 or disk64 or cdisk64
        or local_entries != total_entries or not 1 <= total_entries <= 200_000
        or not 0 < cd_size <= MAX_CENTRAL_BYTES
        or cd_offset + cd_size > eocd64_offset
        or eocd64_offset + record_len + 12 > start + loc_pos
    ):
        raise ImageRangeError("Invalid or over-budget ZIP64 directory")
    return cd_offset, cd_size, total_entries


def _zip64_values(extra: bytes, regular_usize: int, regular_csize: int,
                  regular_offset: int, regular_disk: int):
    expected = (
        regular_usize == 0xFFFFFFFF, regular_csize == 0xFFFFFFFF,
        regular_offset == 0xFFFFFFFF, regular_disk == 0xFFFF,
    )
    if not any(expected):
        return regular_usize, regular_csize, regular_offset, regular_disk
    pos, payload = 0, None
    while pos + 4 <= len(extra):
        kind, size = struct.unpack_from("<HH", extra, pos)
        pos += 4
        if pos + size > len(extra):
            raise ImageRangeError("Malformed ZIP extra data")
        if kind == 1:
            if payload is not None:
                raise ImageRangeError("Duplicate ZIP64 extra record")
            payload = extra[pos:pos+size]
        pos += size
    if payload is None:
        raise ImageRangeError("ZIP64 extra record is missing")
    cursor = 0
    values = []
    regular = (regular_usize, regular_csize, regular_offset, regular_disk)
    for needed, original in zip(expected, regular):
        if not needed:
            values.append(original)
            continue
        width = 4 if len(values) == 3 else 8
        if cursor + width > len(payload):
            raise ImageRangeError("Truncated ZIP64 extra values")
        values.append(int.from_bytes(payload[cursor:cursor+width], "little"))
        cursor += width
    return tuple(values)


def parse_image_index(central: bytes, count: int, source: str) -> dict[str, ImageMember]:
    """Only accepts source/<safe-stem>.jpg entries (plus source/ directory)."""
    if source not in IMAGE_URLS or count < 1 or count > 200_000:
        raise ImageRangeError("Unknown archive image class")
    cursor, entries = 0, {}
    for _ in range(count):
        if cursor + 46 > len(central):
            raise ImageRangeError("Truncated image central-directory entry")
        f = struct.unpack_from("<4s6H3I5H2I", central, cursor)
        if f[0] != b"PK\x01\x02":
            raise ImageRangeError("Unexpected central-directory signature")
        flags, method, crc = f[3], f[4], f[7]
        name_len, extra_len, comment_len = f[10:13]
        end = cursor + 46 + name_len + extra_len + comment_len
        if end > len(central):
            raise ImageRangeError("Invalid ZIP central entry lengths")
        try:
            name = central[cursor + 46:cursor + 46 + name_len].decode(
                "utf-8" if flags & 0x800 else "cp437"
            )
        except UnicodeError as exc:
            raise ImageRangeError("Bad ZIP image entry name") from exc
        extra = central[cursor + 46 + name_len:cursor + 46 + name_len + extra_len]
        # Container metadata is not a photo: never use paths from the archive.
        if name == source + "/":
            cursor = end
            continue
        prefix = source + "/"
        if (
            not name.startswith(prefix) or not name.endswith(".jpg")
            or not STEM_RE.fullmatch(name[len(prefix):-4])
            or flags & 1 or method not in (0, 8)
        ):
            raise ImageRangeError("Unexpected or unsafe member in gesture image archive")
        # Unix external attributes identifying symbolic links are forbidden.
        if (f[15] >> 16) & 0o170000 == 0o120000:
            raise ImageRangeError("Symlink inside ZIP is forbidden")
        uncompressed, compressed, offset, disk = _zip64_values(
            extra, f[9], f[8], f[16], f[13]
        )
        if disk != 0 or not (0 < compressed <= MAX_JPG_COMPRESSED) or not (
            0 < uncompressed <= MAX_JPG_DECOMPRESSED
        ):
            raise ImageRangeError("Oversized/non-local JPEG; manual archive review required")
        stem = name[len(prefix):-4]
        if stem in entries:
            raise ImageRangeError("Duplicate original image stem in ZIP")
        entries[stem] = ImageMember(name, offset, compressed, uncompressed,
                                    crc, method, flags)
        cursor = end
    if cursor != len(central):
        raise ImageRangeError("Unparsed central-directory trailing bytes")
    if not entries:
        raise ImageRangeError("Image ZIP has no supported JPEG members")
    return entries


def read_jpeg(member: ImageMember, transport: HTTPRange) -> bytes:
    header = transport.get(member.offset, member.offset + 29)
    fields = struct.unpack("<I5H3I2H", header)
    if fields[0] != 0x04034B50 or fields[2] != member.flags or fields[3] != member.method:
        raise ImageRangeError("ZIP local header differs from central directory")
    nlen, xlen = fields[9], fields[10]
    if nlen < 1 or nlen > 1024 or xlen > 65_000:
        raise ImageRangeError("Unsafe ZIP local filename/extra lengths")
    name_data = transport.get(member.offset + 30, member.offset + 29 + nlen + xlen)
    name = name_data[:nlen].decode("utf-8" if member.flags & 0x800 else "cp437")
    if name != member.filename:
        raise ImageRangeError("ZIP local name mismatch")
    payload_start = member.offset + 30 + nlen + xlen
    packed = transport.get(payload_start, payload_start + member.compressed - 1)
    if member.method == 8:
        dec = zlib.decompressobj(-15)
        try:
            raw = dec.decompress(packed, member.uncompressed + 1)
        except zlib.error as exc:
            raise ImageRangeError("Corrupt DEFLATE image payload") from exc
        if not dec.eof or dec.unconsumed_tail or dec.unused_data:
            raise ImageRangeError("Invalid/oversized JPEG decompression stream")
    else:
        raw = packed
    if (len(raw) != member.uncompressed
            or zlib.crc32(raw) & 0xFFFFFFFF != member.crc32):
        raise ImageRangeError("JPEG size/CRC32 mismatch")
    if len(raw) < 4 or not raw.startswith(b"\xff\xd8\xff") or not raw.endswith(b"\xff\xd9"):
        raise ImageRangeError("Invalid JPEG header/end marker")
    # This does not decode or aesthetically verify the picture. Human QA is mandatory.
    return raw


def inspect(source: str, transport: HTTPRange | None = None):
    if source not in IMAGE_URLS:
        raise ImageRangeError("Source not in approved four-way P1 contrast group")
    if transport is None:
        transport = HTTPRange(IMAGE_URLS[source], PINNED_SIZES.get(source))
    tail = transport.get(max(0, transport.size - TAIL_BYTES), transport.size - 1)
    offset, size, count = parse_zip64_footer(tail, transport.size)
    central = transport.get(offset, offset + size - 1)
    entries = parse_image_index(central, count, source)
    return entries, {
        "status": "INDEX_ONLY_NO_PERSONAL_IMAGES_FETCHED",
        "source": source,
        "archive_size": transport.size,
        "zip64_entries": count,
        "jpeg_entries": len(entries),
        "central_directory_bytes": size,
        "central_directory_sha256": hashlib.sha256(central).hexdigest(),
    }


def extract(source: str, annotations: Path, output: Path, license_ack: Path,
            per_source: int = 6, max_transfer_mib: int = 80,
            seed: str = "p1-visual-review-v1", transport: HTTPRange | None = None):
    """Read a tiny deterministic train-only subset. Atomic, never overwrite."""
    # CI uses injected synthetic transports for tests, but may never acquire
    # real person media with network requests or generate public artifacts.
    if transport is None and (os.environ.get("GITHUB_ACTIONS") == "true"
                              or os.environ.get("CI") == "true"):
        raise ImageRangeError("Real HaGRID JPG extraction is forbidden in CI")
    try:
        _approval(license_ack, "license")
    except ValueError as exc:
        raise ImageRangeError("Owner research license approval is missing/invalid") from exc
    if source not in IMAGE_URLS:
        raise ImageRangeError("Only four initial number-two comparison sources are available")
    if not MIN_SELECTED <= per_source <= MAX_SELECTED:
        raise ImageRangeError("Require 3–20 different images for visual review")
    if not 1 <= max_transfer_mib <= 512 or not seed or len(seed) > 200:
        raise ImageRangeError("Invalid download byte budget or deterministic seed")
    dest_root = output.resolve()
    local_root = ROOT.resolve()
    if dest_root == local_root or (
        dest_root.is_relative_to(local_root)
        and not dest_root.is_relative_to(local_root / "data")
    ):
        raise ImageRangeError("Within the Git repository, outputs must be under ignored data/")
    final = dest_root / source
    if final.exists():
        raise ImageRangeError("Refusing to overwrite an existing class image folder")
    # Read only the train/source.json: never look at held-out val/test.
    try:
        records = _load_annotation(annotations / "train" / (source + ".json"))
    except VisualReviewError as exc:
        raise ImageRangeError("Authorized train annotation JSON missing/invalid") from exc
    if transport is None:
        transport = HTTPRange(IMAGE_URLS[source], PINNED_SIZES.get(source))
    index, details = inspect(source, transport)
    # Do not serialize original user_id or original image stem into public logs.
    candidates = []
    for stem, record in records.items():
        if stem not in index:
            continue
        check = _clean_record(stem, record)
        if check is None:
            continue
        if source not in {hand["label"] for hand in check["hands"]}:
            continue
        candidates.append((_candidate_rank(seed, source, stem), stem, check["subject"]))
    used_subjects, selected = set(), []
    for _rank, stem, subject in sorted(candidates):
        if subject in used_subjects:
            continue
        member = index[stem]
        used_subjects.add(subject)
        selected.append((stem, member))
        if len(selected) == per_source:
            break
    if len(selected) < per_source:
        raise ImageRangeError("Not enough distinct subjects with source-labeled train JPEGs")
    # Count all requests conservatively, including ZIP metadata and local headers.
    budget_bytes = (TAIL_BYTES + details["central_directory_bytes"]
                    + sum(m.compressed + 30 + 1024 + 65_000 for _, m in selected))
    if budget_bytes > max_transfer_mib * 1024 ** 2:
        raise ImageRangeError("Selected image metadata/payload exceeds explicit transfer budget")
    dest_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    hashes = set()
    provenance = {
        "source": source,
        "official_archive": IMAGE_URLS[source],
        "archive_bytes": transport.size,
        "archive_sha256": None,  # Cannot hash a 40+GB ZIP without downloading it.
        "zip64_central_directory_sha256": details["central_directory_sha256"],
        "source_annotation_sha256": hashlib.sha256(
            (annotations / "train" / (source + ".json")).read_bytes()
        ).hexdigest(),
        "selected_train_only": True,
        "sampling_seed_sha256": hashlib.sha256(seed.encode("utf-8")).hexdigest(),
        "selected_images": {},
    }
    with tempfile.TemporaryDirectory(prefix=".hagrid-images-", dir=dest_root) as tmp:
        stage = Path(tmp) / source
        stage.mkdir(mode=0o700)
        for stem, member in selected:
            data = read_jpeg(member, transport)
            digest = hashlib.sha256(data).hexdigest()
            if digest in hashes:
                raise ImageRangeError("Duplicate selected image bytes across distinct image keys")
            hashes.add(digest)
            (stage / (stem + ".jpg")).write_bytes(data)
            provenance["selected_images"][stem + ".jpg"] = {
                "sha256": digest, "length": len(data), "zip_crc32": f"{member.crc32:08x}"
            }
        (stage / "_provenance.private.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if final.exists():
            raise ImageRangeError("Another task created this class folder; abort")
        stage.rename(final)
    return {
        "status": "PRIVATE_SAMPLE_EXTRACTED_NOT_VISUALLY_APPROVED",
        "source": source,
        "jpeg_count": len(selected),
        "archive_bytes": transport.size,
        "index_bytes": details["central_directory_bytes"],
        "local_folder": str(final),
        "train_only": True,
        "no_real_image_ci_used": True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("inspect", "extract"))
    p.add_argument("--source", required=True, choices=sorted(IMAGE_URLS))
    p.add_argument("--annotations", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--license-ack", type=Path)
    p.add_argument("--per-source", type=int, default=6)
    p.add_argument("--seed", default="p1-visual-review-v1")
    p.add_argument("--max-transfer-mib", type=int, default=80)
    args = p.parse_args()
    try:
        if args.mode == "inspect":
            _, result = inspect(args.source)
        else:
            if args.annotations is None or args.output is None or args.license_ack is None:
                p.error("extract requires --annotations, --output and --license-ack")
            result = extract(args.source, args.annotations, args.output,
                             args.license_ack, args.per_source,
                             args.max_transfer_mib, args.seed)
    except ImageRangeError as exc:
        p.exit(2, f"FAIL CLOSED: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
