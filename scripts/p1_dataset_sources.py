#!/usr/bin/env python3
"""P1-05 private, bounded HaGRIDv2 sources: local folders, ZIPs or HTTP Range.

The same ten-class dataset builder uses these accessors; no whole 40GB image
archive is extracted. Remote JPEG cache is resumable by exact CRC+byte length,
and both raw data and cache are PRIVATE (ignored data/, NEVER GitHub CI).
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import http.client
import json
import os
import random
import re
import socket
import stat
import threading
import time
import urllib.error
import zipfile
import zlib
from pathlib import Path

from scripts import hagrid_image_zip64 as imagezip
from scripts import hagrid_zip_range as annzip
from scripts.data_contract import SOURCE_TO_INDEX

SOURCES = tuple(SOURCE_TO_INDEX)
SPLITS = ("train", "val", "test")
STEM = re.compile(r"^[A-Za-z0-9_-]+$")
MAX_JPEG_BYTES = imagezip.MAX_JPG_DECOMPRESSED
MAX_JSON_BYTES = annzip.MAX_UNCOMPRESSED


class SourceError(ValueError):
    """Refuse ambiguous, private, stale, oversized or corrupted source media."""


class BudgetError(SourceError):
    """Declared transfer/output byte budget was exhausted.

    Separate subclass so concurrent prefetch can distinguish an authoritative
    budget stop (must abort the build) from a recoverable per-image transport
    failure (safe to skip and retry later).
    """


# Transport faults that mean "this connection died, try again" rather than
# "the archive is wrong". A long range-download session against the HaGRID
# object store does get connections closed mid-flight; treating that as a
# permanent failure aborts hours of progress.
_TRANSIENT_HTTP_STATUS = frozenset((408, 425, 429, 500, 502, 503, 504))


def _is_transient(exc: BaseException | None) -> bool:
    if exc is None:
        return False
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in _TRANSIENT_HTTP_STATUS
    if isinstance(exc, urllib.error.URLError):
        return True
    if isinstance(exc, (http.client.HTTPException, ConnectionError,
                        TimeoutError, socket.timeout)):
        return True
    return False


class ByteBudget:
    def __init__(self, total: int):
        if not isinstance(total, int) or total < 1:
            raise SourceError("Positive byte budget required")
        self.total = total
        self.used = 0

    def charge(self, amount: int):
        if amount < 0 or self.used + amount > self.total:
            raise BudgetError("Transfer/output byte budget exceeded; private cache retained")
        self.used += amount


class _ThreadSafeBudget:
    """Serialize budget accounting across prefetch workers.

    Concurrent charging must stay atomic or the transfer cap could be exceeded.
    """

    def __init__(self, budget: ByteBudget):
        self._budget = budget
        self._lock = threading.Lock()

    @property
    def used(self) -> int:
        return self._budget.used

    @property
    def total(self) -> int:
        return self._budget.total

    def charge(self, amount: int) -> None:
        with self._lock:
            self._budget.charge(amount)


def _check_name(name: str):
    if not isinstance(name, str) or not name or name.startswith("/") or "\\" in name:
        raise SourceError("Unsafe input ZIP member")
    if any(segment in ("", ".", "..") for segment in name.rstrip("/").split("/")):
        raise SourceError("Input ZIP traversal/empty path segment")


def _open_archive(path: Path) -> zipfile.ZipFile:
    if path.is_symlink() or not path.is_file():
        raise SourceError("Local ZIP missing or symlinked")
    try:
        archive = zipfile.ZipFile(path, "r", allowZip64=True)
        seen = set()
        for member in archive.infolist():
            _check_name(member.filename)
            if member.filename in seen:
                raise SourceError("Duplicate name in ZIP central directory")
            seen.add(member.filename)
            mode = (member.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK or member.flag_bits & 1:
                raise SourceError("Symlink/encrypted input ZIP member forbidden")
            if not member.is_dir() and member.compress_type not in (
                zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED
            ):
                raise SourceError("Unsupported input ZIP compression")
        return archive
    except (OSError, zipfile.BadZipFile, UnicodeError, ValueError) as exc:
        raise SourceError("Cannot safely inspect local ZIP") from exc


def _read_zip_member(archive: zipfile.ZipFile, name: str, cap: int) -> bytes:
    try:
        member = archive.getinfo(name)
        if (member.is_dir() or not 0 < member.file_size <= cap
                or member.compress_size <= 0):
            raise SourceError("Selected ZIP member missing/over size cap")
        # ZipFile.open verifies per-entry CRC32 on full read; never extractall().
        with archive.open(member) as stream:
            raw = stream.read(cap + 1)
    except (KeyError, OSError, RuntimeError, zipfile.BadZipFile, EOFError) as exc:
        raise SourceError("Selected local ZIP member unreadable or CRC invalid") from exc
    if len(raw) != member.file_size or len(raw) > cap:
        raise SourceError("Oversize/short selected ZIP member")
    return raw


def _safe_local(path: Path, root: Path) -> Path:
    # Parent directory may be outside repo, but never silently follow symlinks.
    if path.is_symlink() or not path.is_file():
        raise SourceError("Local file missing or symlinked")
    if not path.resolve().is_relative_to(root.resolve()):
        raise SourceError("Local file escapes authorized image/annotation root")
    return path


class LocalAnnotations:
    def __init__(self, mode: str, root: Path):
        self.mode = mode
        self.root = root
        self.archive = _open_archive(root) if mode == "local-zip" else None
        if self.archive is None and (root.is_symlink() or not root.is_dir()):
            raise SourceError("Local annotation folder missing/symlinked")

    def get(self, split: str, source: str) -> bytes:
        if split not in SPLITS or source not in SOURCES:
            raise SourceError("Unknown split or gesture annotation")
        if self.archive:
            paths = (f"annotations/{split}/{source}.json", f"{split}/{source}.json",
                     f"hagrid_annotations/{split}/{source}.json")
            present = [n for n in paths if n in self.archive.namelist()]
            if len(present) != 1:
                raise SourceError("Expected exactly one recognized annotation ZIP member")
            raw = _read_zip_member(self.archive, present[0], MAX_JSON_BYTES)
        else:
            file = _safe_local(self.root / split / f"{source}.json", self.root)
            if not 0 < file.stat().st_size <= MAX_JSON_BYTES:
                raise SourceError("Unreasonable local JSON annotation size")
            raw = file.read_bytes()
        return raw

    def close(self):
        if self.archive:
            self.archive.close()


class RemoteAnnotations:
    def __init__(self, cache: Path, budget: ByteBudget):
        self.cache, self.budget = cache, budget
        self.budget.charge(annzip.TAIL_BYTES)
        self.members = annzip.fetch_directory()

    def get(self, split: str, source: str) -> bytes:
        if split not in SPLITS or source not in SOURCES:
            raise SourceError("Unknown split/source in remote annotation request")
        m = annzip.select_members(self.members, (split,), (source,))[0]
        file = self.cache / "annotations" / split / f"{source}.json"
        if file.exists() or file.is_symlink():
            _safe_local(file, self.cache)
            if file.stat().st_size != m.uncompressed:
                raise SourceError("Cached annotation length changed")
            raw = file.read_bytes()
            if len(raw) != m.uncompressed or zlib.crc32(raw) & 0xFFFFFFFF != m.crc:
                raise SourceError("Cached annotation CRC/size changed; refuse poisoned cache")
            return raw
        self.budget.charge(m.compressed + 66000)
        raw = annzip.read_member(m)
        if len(raw) != m.uncompressed:
            raise SourceError("Remote JSON member length mismatch")
        _atomic_private_write(file, raw)
        return raw

    def close(self):
        return None


def _jpeg_checked(raw: bytes) -> bytes:
    if (not isinstance(raw, bytes) or not 4 <= len(raw) <= MAX_JPEG_BYTES
            or not raw.startswith(b"\xff\xd8\xff") or not raw.endswith(b"\xff\xd9")):
        raise SourceError("JPEG signature/end marker invalid (human decode QA still required)")
    return raw


def _atomic_private_write(path: Path, data: bytes):
    if not path.parent.exists():
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise SourceError("Refusing to overwrite original private cache")
    import tempfile
    with tempfile.NamedTemporaryFile(
        prefix=".p1-fetch-", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temp = Path(handle.name)
        os.fchmod(handle.fileno(), 0o600)
        try:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        if path.exists() or path.is_symlink():
            raise SourceError("Concurrent cache file appeared")
        temp.rename(path)
    finally:
        temp.unlink(missing_ok=True)


class LocalImages:
    def __init__(self, mode: str, root: Path):
        self.mode, self.root = mode, root
        self.archives = {}
        self.index = {}
        if mode == "local-dir" and (root.is_symlink() or not root.is_dir()):
            raise SourceError("Local image root missing or symlinked")
        if mode == "local-zip" and (root.is_symlink() or not root.is_dir()):
            raise SourceError("Local per-class ZIP directory missing/symlinked")

    def _index(self, source: str) -> dict[str, str]:
        if source in self.index:
            return self.index[source]
        if source not in SOURCES:
            raise SourceError("Only ten currently approved candidate source categories")
        archive = _open_archive(self.root / (source + ".zip"))
        seen = {}
        for item in archive.infolist():
            if item.is_dir() or not item.filename.endswith(".jpg"):
                continue
            parts = item.filename.split("/")
            expected = (
                [source, parts[-1]] if len(parts) == 2 else
                ["hagrid_dataset", source, parts[-1]] if len(parts) == 3 else
                [parts[-1]] if source == "no_gesture" and len(parts) == 1 else []
            )
            if not expected or parts != expected:
                # Ignore non-sample vendor metadata (e.g. __MACOSX/._name.jpg).
                # Never guess which physical image corresponds to a stem.
                continue
            stem = parts[-1][:-4]
            if not STEM.fullmatch(stem) or stem in seen:
                raise SourceError("Invalid/duplicate canonical JPEG stems")
            if not (0 < item.file_size <= MAX_JPEG_BYTES):
                raise SourceError("Oversized JPEG in local ZIP")
            seen[stem] = item.filename
        if not seen:
            archive.close()
            raise SourceError("No recognized hand images in source ZIP")
        self.archives[source] = archive
        self.index[source] = seen
        return seen

    def has(self, source: str, stem: str) -> bool:
        if source not in SOURCES or not STEM.fullmatch(stem):
            return False
        if self.mode == "local-zip":
            return stem in self._index(source)
        p = self.root / source / (stem + ".jpg")
        return p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(
            self.root.resolve()
        )

    def get(self, source: str, stem: str) -> bytes:
        if not self.has(source, stem):
            raise SourceError("Local JPEG absent")
        if self.mode == "local-zip":
            raw = _read_zip_member(self.archives[source], self.index[source][stem],
                                   MAX_JPEG_BYTES)
        else:
            p = _safe_local(self.root / source / (stem + ".jpg"), self.root)
            if not 0 < p.stat().st_size <= MAX_JPEG_BYTES:
                raise SourceError("Local JPG oversized")
            raw = p.read_bytes()
        return _jpeg_checked(raw)

    def close(self):
        for a in self.archives.values():
            a.close()


class MeteredHTTPRange(imagezip.HTTPRange):
    def __init__(self, url: str, size: int, budget: ByteBudget):
        self.budget = budget
        super().__init__(url, size)

    def get(self, start: int, end: int) -> bytes:
        # Retry transient transport faults with exponential backoff. The HaGRID
        # object store closes long-lived connections (RemoteDisconnected); that
        # is a recoverable per-request fault, not corrupt archive bytes. Safety
        # failures (HTTP 200 instead of 206, changed ETag/Content-Range, short
        # payload) still fail immediately and are never retried.
        attempts = 5
        for attempt in range(attempts):
            self.budget.charge(end - start + 1)
            try:
                return super().get(start, end)
            except imagezip.ImageRangeError as exc:
                if not _is_transient(exc.__cause__) or attempt == attempts - 1:
                    raise
                delay = min(30.0, 0.5 * (2 ** attempt)) * (1 + random.random() * 0.5)
                time.sleep(delay)
        raise SourceError("Unreachable retry limit")


class RemoteImages:
    def __init__(self, cache: Path, budget: ByteBudget):
        self.cache, self.budget = cache, budget
        self.budget_meter = _ThreadSafeBudget(budget)
        self.index, self.transport = {}, {}

    def prefetch(self, requests: list[tuple[str, str, str]], *, workers: int = 8) -> dict:
        """Warm the private cache concurrently, then let get() re-verify.

        Downloads only bytes not already cached. Every fetched JPEG still passes
        the size/CRC/signature checks inside get(); a per-image transport failure
        is recorded and skipped here because get() re-raises the real error when
        that sample is actually needed. Only an exhausted byte budget aborts.

        Transfer accounting: prefetch is the single place bytes actually come off
        the wire, so it charges the shared budget exactly once per downloaded
        range. get() then finds the CRC-checked cache file and charges nothing.
        """
        if not 3 <= workers <= 32:
            raise SourceError("Prefetch workers must be 3..32")
        pending = []
        for source, stem, _split in ((s, st, sp) for sp, s, st in requests):
            file = self.cache / "images" / source / (stem + ".jpg")
            if file.exists() or file.is_symlink():
                continue
            pending.append((source, stem))
        if not pending:
            return {"prefetched": 0, "skipped_cached": len(requests)}
        # Group by source because a transport (and its pinned index) is per class.
        by_source: dict[str, list[str]] = {}
        for source, stem in pending:
            by_source.setdefault(source, []).append(stem)
        done = failed = 0
        for source, stems in by_source.items():
            self._load(source)
            index = self.index[source]
            url = imagezip.IMAGE_URLS[source]
            size = self.transport[source].size
            local = threading.local()
            print(
                f"[prefetch] {source}: {len(stems)} stems, budget "
                f"{self.budget_meter.used / (1024 ** 2):.0f}/"
                f"{self.budget_meter.total / (1024 ** 2):.0f} MiB",
                flush=True,
            )

            def transport_for_thread():
                # HTTPRange keeps per-instance request state (ETag/If-Match) and
                # is not thread-safe, so each worker owns its own transport.
                net = getattr(local, "net", None)
                if net is None:
                    net = MeteredHTTPRange(url, size, self.budget_meter)
                    local.net = net
                return net

            def fetch(stem: str) -> str:
                """Return 'ok', 'failed' or 'missing'; never raise except budget."""
                member = index.get(stem)
                if member is None:
                    return "missing"
                try:
                    data = imagezip.read_jpeg(member, transport_for_thread())
                except BudgetError:
                    raise
                except (imagezip.ImageRangeError, SourceError, OSError) as exc:
                    print(f"[prefetch] {source}/{stem[:8]}: {type(exc).__name__}",
                          flush=True)
                    return "failed"
                data = _jpeg_checked(data)
                try:
                    _atomic_private_write(
                        self.cache / "images" / source / (stem + ".jpg"), data
                    )
                except SourceError:
                    # Another worker won the race; an existing cache file is fine.
                    pass
                return "ok"

            with concurrent.futures.ThreadPoolExecutor(workers) as pool:
                for status in pool.map(fetch, stems):
                    if status == "ok":
                        done += 1
                    elif status == "failed":
                        failed += 1
            print(f"[prefetch] {source}: done={done} failed={failed}", flush=True)
        return {"prefetched": done, "failed": failed,
                "skipped_cached": len(requests) - len(pending),
                "budget_used_mib": round(self.budget_meter.used / (1024 ** 2), 2),
                "budget_total_mib": round(self.budget_meter.total / (1024 ** 2), 2)}

    def _load(self, source: str):
        if source in self.index:
            return
        if source not in SOURCES:
            raise SourceError("Remote source is not one of the ten output candidate sources")
        net = MeteredHTTPRange(
            imagezip.IMAGE_URLS[source], imagezip.PINNED_SIZES[source],
            self.budget,
        )
        # inspect() checks the exact previously audited directory SHA256
        index, info = imagezip.inspect(source, net)
        if info["central_directory_sha256"] != imagezip.PINNED_INDEX_SHA256[source]:
            raise SourceError("Unpinned upstream image ZIP index")
        self.index[source], self.transport[source] = index, net

    def has(self, source: str, stem: str) -> bool:
        if source not in SOURCES or not STEM.fullmatch(stem):
            return False
        self._load(source)
        return stem in self.index[source]

    def get(self, source: str, stem: str) -> bytes:
        if not self.has(source, stem):
            raise SourceError("Selected image stem absent from pinned remote ZIP")
        member = self.index[source][stem]
        file = self.cache / "images" / source / (stem + ".jpg")
        if file.exists() or file.is_symlink():
            _safe_local(file, self.cache)
            if (file.stat().st_size != member.uncompressed
                    or file.stat().st_size > MAX_JPEG_BYTES):
                raise SourceError("Resumable image cache changed/CRC invalid")
            data = file.read_bytes()
            if (len(data) != member.uncompressed
                    or zlib.crc32(data) & 0xFFFFFFFF != member.crc32):
                raise SourceError("Resumable image cache changed/CRC invalid")
            return _jpeg_checked(data)
        data = imagezip.read_jpeg(member, self.transport[source])
        data = _jpeg_checked(data)
        _atomic_private_write(file, data)
        return data

    def close(self):
        return None


def make_sources(mode: str, annotations: Path | None, images: Path | None,
                 cache: Path | None, budget: ByteBudget):
    if mode == "local-dir":
        if annotations is None or images is None:
            raise SourceError("local-dir requires annotation/image roots")
        return LocalAnnotations(mode, annotations), LocalImages(mode, images)
    if mode == "local-zip":
        if annotations is None or images is None:
            raise SourceError("local-zip requires annotation ZIP and folder of per-class ZIPs")
        return LocalAnnotations(mode, annotations), LocalImages(mode, images)
    if mode == "auto":
        if cache is None:
            raise SourceError("auto requires private persistent cache for resume")
        return RemoteAnnotations(cache, budget), RemoteImages(cache, budget)
    raise SourceError("Unknown source mode")
