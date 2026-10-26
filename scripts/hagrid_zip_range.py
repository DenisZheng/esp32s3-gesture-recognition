#!/usr/bin/env python3
"""Safely inspect or selectively extract HaGRIDv2 annotation JSON using byte ranges.

inspect: reads the final 64 KiB of the official 719 MB ZIP and its directory.
extract: requires a signed local license acknowledgement; downloads only chosen
JSON entries, never the whole archive. Never execute real-data extraction in CI.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import random
import re
import socket
import struct
import tempfile
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path

from scripts.data_contract import SOURCE_TO_INDEX
from scripts.hagrid_policy import OFFICIAL_SOURCES
from scripts.hagrid_p1 import PreparationError, ROOT, SPLITS, _approval

URL = ("https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/"
       "hagrid_v2/annotations_with_landmarks/annotations.zip")
ARCHIVE_BYTES = 719_041_314  # Verified official HEAD/Content-Range 2026-10-20
SOURCE_COMMIT = "cade4dd4ba162832ef2b62b92d89d4d7a2dc2b75"
TAIL_BYTES = 65_536
ENTRY_PATTERN = re.compile(r"^annotations/(train|val|test)/([a-z_]+)\.json$")
RANGE_PATTERN = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
MAX_COMPRESSED = 24 * 1024 * 1024
MAX_UNCOMPRESSED = 120 * 1024 * 1024


@dataclass(frozen=True)
class Member:
    path: str
    compressed: int
    uncompressed: int
    offset: int
    crc: int
    method: int
    flags: int


def get_range(start: int, end: int) -> bytes:
    if start < 0 or end < start or end >= ARCHIVE_BYTES:
        raise PreparationError("Out-of-bounds archive range")
    size = end - start + 1
    if size > MAX_COMPRESSED + 128 * 1024:
        raise PreparationError("Range transfer exceeds the per-file safety cap")
    req = urllib.request.Request(URL, headers={
        "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"
    })
    # The HaGRID object store drops long-lived connections
    # (RemoteDisconnected / Connection reset by peer). Retry those transient
    # faults with backoff; never retry an HTTP 200 (Range ignored, would fetch
    # the whole archive) or a size/Content-Range mismatch.
    attempts = 5
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                if response.status != 206:
                    raise PreparationError(
                        "Server ignored Range; refusing full ZIP download")
                match = RANGE_PATTERN.fullmatch(response.headers.get("Content-Range", ""))
                if (not match
                        or tuple(map(int, match.groups())) != (start, end, ARCHIVE_BYTES)):
                    raise PreparationError(
                        "Archive response does not match pinned size/range")
                data = response.read(size + 1)
        except PreparationError:
            raise  # Surface explicit HTTP 200/invalid-range failure.
        except (OSError, ValueError) as exc:
            transient = isinstance(
                exc, (http.client.HTTPException, ConnectionError,
                      TimeoutError, socket.timeout, urllib.error.URLError)
            )
            if not transient or attempt == attempts - 1:
                raise PreparationError("HTTP Range transfer failed") from exc
            time.sleep(min(30.0, 0.5 * (2 ** attempt)) * (1 + random.random() * 0.5))
            continue
        if len(data) != size:
            raise PreparationError("Incomplete/excessive byte-range transfer")
        return data
    raise PreparationError("HTTP Range transfer retry limit reached")


def parse_directory(tail: bytes, total_bytes: int) -> dict[str, Member]:
    if len(tail) < 22:
        raise PreparationError("ZIP tail too short")
    pos = tail.rfind(b"PK\x05\x06")
    if pos < 0 or pos + 22 > len(tail):
        raise PreparationError("ZIP end directory not found")
    sig, disk, cd_disk, entries_disk, count, cd_size, cd_offset, comment = (
        struct.unpack_from("<4s4H2IH", tail, pos)
    )
    if (sig != b"PK\x05\x06" or disk or cd_disk or entries_disk != count
            or not 1 <= count <= 1000 or comment != 0):
        raise PreparationError("ZIP64/multidisk/nonstandard archive refused")
    base = total_bytes - len(tail)
    if cd_offset < base or cd_offset + cd_size > total_bytes:
        raise PreparationError("Central directory outside bounded tail")
    cursor = cd_offset - base
    limit = cursor + cd_size
    members: dict[str, Member] = {}
    for _ in range(count):
        if cursor + 46 > limit:
            raise PreparationError("Truncated central directory")
        f = struct.unpack_from("<4s6H3I5H2I", tail, cursor)
        if f[0] != b"PK\x01\x02":
            raise PreparationError("Bad ZIP central signature")
        name_len, extra_len, comment_len = f[10:13]
        next_pos = cursor + 46 + name_len + extra_len + comment_len
        if next_pos > limit:
            raise PreparationError("Truncated ZIP file metadata")
        try:
            name = tail[cursor+46:cursor+46+name_len].decode(
                "utf-8" if f[3] & 0x800 else "cp437"
            )
        except UnicodeError as exc:
            raise PreparationError("Invalid ZIP filename encoding") from exc
        match = ENTRY_PATTERN.fullmatch(name)
        if match and match.group(2) in OFFICIAL_SOURCES:
            if name in members or f[3] & 1 or f[4] not in (0, 8):
                raise PreparationError("Duplicate, encrypted, or unsupported JSON member")
            # Enumerate all official class JSON metadata even when *unselected*
            # classes are too big for this intentionally bounded extractor.
            # Enforce per-member memory/network caps only on selected entries.
            if not (0 < f[8] <= ARCHIVE_BYTES and 0 < f[9] <= 0xFFFFFFFF):
                raise PreparationError("Malformed ZIP member size")
            members[name] = Member(name, f[8], f[9], f[16], f[7], f[4], f[3])
        cursor = next_pos
    if cursor != limit:
        raise PreparationError("ZIP directory byte count mismatch")
    return members


def select_members(all_members: dict[str, Member], splits: tuple[str, ...],
                   sources: tuple[str, ...]) -> list[Member]:
    if (not splits or not sources or len(set(splits)) != len(splits)
            or len(set(sources)) != len(sources)):
        raise PreparationError("Empty/duplicate split or class selection")
    result = []
    for split in splits:
        if split not in SPLITS:
            raise PreparationError("Unknown requested split")
        for source in sources:
            if source not in OFFICIAL_SOURCES:
                raise PreparationError("Unknown requested gesture source")
            name = f"annotations/{split}/{source}.json"
            if name not in all_members:
                raise PreparationError("Requested class/split does not exist in archive")
            member = all_members[name]
            if not (0 < member.compressed <= MAX_COMPRESSED and
                    0 < member.uncompressed <= MAX_UNCOMPRESSED):
                raise PreparationError("Selected ZIP member exceeds transfer/decode safety cap")
            result.append(member)
    return result


def read_member(member: Member, getter=get_range) -> bytes:
    if not (0 < member.compressed <= MAX_COMPRESSED and
            0 < member.uncompressed <= MAX_UNCOMPRESSED):
        raise PreparationError("Selected ZIP member exceeds transfer/decode safety cap")
    header = getter(member.offset, member.offset + 29)
    sig, _version, flags, method, _time, _date, _crc, _cs, _us, nl, el = (
        struct.unpack("<I5H3I2H", header)
    )
    if sig != 0x04034B50 or flags != member.flags or method != member.method:
        raise PreparationError("Local ZIP header does not match central directory")
    if nl == 0 or nl > 1024 or el > 65535:
        raise PreparationError("Bad local header name/extra length")
    extra = getter(member.offset+30, member.offset+29+nl+el)
    try:
        name = extra[:nl].decode("utf-8" if flags & 0x800 else "cp437")
    except UnicodeError as exc:
        raise PreparationError("Invalid local ZIP member name") from exc
    if name != member.path:
        raise PreparationError("ZIP filename mismatch")
    data_start = member.offset + 30 + nl + el
    compressed = getter(data_start, data_start + member.compressed - 1)
    if member.method == 8:
        dec = zlib.decompressobj(-15)
        try:
            raw = dec.decompress(compressed, member.uncompressed + 1)
        except zlib.error as exc:
            raise PreparationError("Invalid compressed JSON stream") from exc
        if not dec.eof or dec.unconsumed_tail:
            raise PreparationError("DEFLATE size/end mismatch")
    else:
        raw = compressed
    if (len(raw) != member.uncompressed
            or zlib.crc32(raw) & 0xFFFFFFFF != member.crc):
        raise PreparationError("ZIP member CRC32 or size mismatch")
    if not raw.lstrip().startswith(b"{"):
        raise PreparationError("ZIP JSON payload malformed")
    return raw


def fetch_directory() -> dict[str, Member]:
    return parse_directory(get_range(ARCHIVE_BYTES - TAIL_BYTES, ARCHIVE_BYTES - 1),
                           ARCHIVE_BYTES)


def extract(sources: tuple[str, ...], splits: tuple[str, ...], output: Path,
            license_ack: Path, max_transfer_mib: int) -> dict:
    if os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("CI") == "true":
        raise PreparationError("Real HaGRID annotation extraction is forbidden in CI")
    _approval(license_ack, "license")
    if not (1 <= max_transfer_mib <= 256):
        raise PreparationError("Invalid requested byte budget")
    output = output.resolve()
    if output == ROOT or (output.is_relative_to(ROOT)
                          and not output.is_relative_to(ROOT / "data")):
        raise PreparationError("Output inside git must be under ignored data/")
    if output.exists():
        raise PreparationError("Output already exists")
    selected = select_members(fetch_directory(), splits, sources)
    bytes_required = sum(m.compressed + 128 for m in selected)
    if bytes_required > max_transfer_mib * 1024 * 1024:
        raise PreparationError("Requested classes exceed transfer budget")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": URL, "source_commit": SOURCE_COMMIT,
        "archive_size": ARCHIVE_BYTES,
        "archive_sha256": None,  # Cannot know SHA256 of an archive not downloaded.
        "scope": "selected JSON files only, private user_id metadata",
        "files": {},
    }
    with tempfile.TemporaryDirectory(prefix=".hagrid-p1-", dir=output.parent) as temp:
        stage = Path(temp) / "annotations"
        for member in selected:
            data = read_member(member)
            _, split, filename = member.path.split("/")
            local = stage / split / filename
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
            manifest["files"][member.path] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "compressed_bytes": member.compressed,
                "uncompressed_bytes": member.uncompressed,
                "crc32_hex": f"{member.crc:08x}",
            }
        (stage / "_provenance.aggregate.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8"
        )
        stage.rename(output)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("inspect", "extract"))
    p.add_argument("--sources", nargs="+", default=list(SOURCE_TO_INDEX))
    p.add_argument("--splits", nargs="+", default=list(SPLITS))
    p.add_argument("--output", type=Path)
    p.add_argument("--license-ack", type=Path)
    p.add_argument("--max-transfer-mib", type=int, default=5)
    a = p.parse_args()
    try:
        splits, sources = tuple(a.splits), tuple(a.sources)
        if a.mode == "inspect":
            selected = select_members(fetch_directory(), splits, sources)
            result = {
                "archive_bytes": ARCHIVE_BYTES,
                "selected_json_files": len(selected),
                "compressed_bytes": sum(m.compressed for m in selected),
                "uncompressed_bytes": sum(m.uncompressed for m in selected),
                "individual_annotations_downloaded": False,
            }
        else:
            if a.output is None or a.license_ack is None:
                p.error("extract requires --output and --license-ack")
            result = extract(sources, splits, a.output, a.license_ack,
                             a.max_transfer_mib)
    except PreparationError as exc:
        p.exit(2, f"FAIL CLOSED: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
