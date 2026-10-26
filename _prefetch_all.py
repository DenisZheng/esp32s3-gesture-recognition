#!/usr/bin/env python3
"""Weekend cache warmer for the ten-class HaGRID gesture dataset.

Purpose: fill ``data/cache`` with every JPEG the dataset builder will need, so
the final ``p1_dataset_builder build`` runs offline from cache in minutes and can
be repeated without touching the network.

Why a separate tool instead of just running ``build``:

* ``build`` is all-or-nothing. If one class cannot reach its quota it discards
  the staged output, so hours of downloading look like a failure.
* ``build`` prefetches in fixed ``SOURCES`` order. ``call``/``dislike`` are last,
  which hid a 3x slower source until the very end.
* This tool is resumable at any interruption: every cached JPEG is length- and
  CRC-checked against the pinned ZIP central directory, so re-running continues
  exactly where it stopped.

It performs no conversion, no training and writes no dataset. It only downloads
into the ignored private cache with the exact same integrity checks the builder
uses (pinned archive size, pinned central-directory SHA256, HTTP 206, ETag /
Last-Modified pinning, CRC32, JPEG signature).

Usage
-----
    python3 _prefetch_all.py --train 500 --val 50 --test 50
    python3 _prefetch_all.py --train 500 --val 50 --test 50 --rounds 1
    python3 _prefetch_all.py --report-only
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts.p1_dataset_builder import (  # noqa: E402
    _gather_candidates,
    MAX_PER_SOURCE,
)
from scripts.hagrid_p1 import PreparationError, _approval  # noqa: E402
from scripts.p1_dataset_sources import (  # noqa: E402
    ByteBudget,
    RemoteAnnotations,
    RemoteImages,
    SOURCES,
    SPLITS,
    SourceError,
)

SEED = "espdet-p1-05-v1"
DEFAULT_LICENSE_ACK = Path("data/reviews/hagrid-license-ack.json")
DEFAULT_MAPPING_ACK = Path("data/reviews/hagrid-mapping-ack.json")


def require_owner_approvals(license_ack: Path, mapping_ack: Path) -> None:
    """Fail closed unless the owner's research licence and ten-class mapping
    acknowledgements are present and valid.

    This warmer reads HaGRID annotations and downloads person-bearing JPEGs.
    ``p1_dataset_builder`` gates that behind these two private files; warming the
    cache must not become a side channel that skips the same gate. Reviewer
    identity is never printed.
    """
    try:
        _approval(license_ack, "license")
    except PreparationError as exc:
        raise SystemExit(
            f"BLOCKED: research licence acknowledgement missing or invalid: {exc}"
        )
    try:
        _approval(mapping_ack, "mapping")
    except PreparationError as exc:
        raise SystemExit(
            f"BLOCKED: ten-class visual mapping acknowledgement missing or "
            f"invalid: {exc}"
        )


def cache_counts(cache: Path) -> dict[str, int]:
    out = {}
    for source in SOURCES:
        folder = cache / "images" / source
        out[source] = sum(1 for _ in folder.glob("*.jpg")) if folder.is_dir() else 0
    return out


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_request_list(cache: Path, counts: dict[str, int]) -> list[tuple[str, str, str]]:
    """Return (split, source, stem) for every sample the final build may need.

    The window mirrors the builder: quota plus a margin, so a handful of
    unusable samples cannot strand the quota loop short.
    """
    # Budget only has to cover one annotation-ZIP index read (~171 MB) plus the
    # directory reads; JPEG transfer is metered separately in main().
    budget = ByteBudget(8 * 1024 ** 3)
    annotations = RemoteAnnotations(cache, budget)
    images = RemoteImages(cache, budget)
    try:
        candidates, _ = _gather_candidates(annotations, images, SEED)
    finally:
        images.close()
        annotations.close()

    margin = max(10, counts["train"] // 10)
    want = {s: min(MAX_PER_SOURCE, counts[s] + margin) for s in SPLITS}
    requests: list[tuple[str, str, str]] = []
    for split in SPLITS:
        for source in SOURCES:
            for row in candidates[(split, source)][: want[split]]:
                requests.append((split, source, row[1]))
    return requests


def pending_by_source(requests, cache: Path) -> dict[str, list[tuple[str, str, str]]]:
    """Return samples still missing *or corrupt*, so damaged cache entries are
    re-downloaded instead of being counted as complete.

    A plain ``file.exists()`` check treats a truncated or corrupted JPEG as done.
    The failure then surfaces hours later during dataset assembly, or worse, a
    subtly damaged image reaches training. Validate structure here: JPEG magic,
    EOI marker, and a non-trivial length.
    """
    grouped: dict[str, list[tuple[str, str, str]]] = {}
    for split, source, stem in requests:
        path = cache / "images" / source / (stem + ".jpg")
        if _cache_entry_ok(path):
            continue
        grouped.setdefault(source, []).append((split, source, stem))
    return grouped


def _cache_entry_ok(path: Path) -> bool:
    """Cheap integrity probe for one cached JPEG."""
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size < 4:
        return False
    try:
        with path.open("rb") as handle:
            head = handle.read(3)
            handle.seek(-2, os.SEEK_END)
            tail = handle.read(2)
    except OSError:
        return False
    return head == b"\xff\xd8\xff" and tail == b"\xff\xd9"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=int, default=500)
    ap.add_argument("--val", type=int, default=50)
    ap.add_argument("--test", type=int, default=50)
    ap.add_argument("--cache", type=Path, default=Path("data/cache/hagrid10-p1-05"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--license-ack", type=Path, default=DEFAULT_LICENSE_ACK,
                    help="owner research-licence acknowledgement (required)")
    ap.add_argument("--mapping-ack", type=Path, default=DEFAULT_MAPPING_ACK,
                    help="ten-class visual mapping acknowledgement (required)")
    ap.add_argument("--rounds", type=int, default=0,
                    help="download passes; 0 keeps retrying until complete")
    ap.add_argument("--sleep", type=int, default=60,
                    help="seconds between retry passes")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    counts = {"train": args.train, "val": args.val, "test": args.test}
    if not 1 <= args.workers <= 32:
        return fail("workers must be 1..32")

    cache = args.cache
    if not cache.is_dir():
        return fail(f"cache directory missing: {cache} (run the dataset builder once)")
    # The cache must stay inside the git-ignored data/ tree so personal images
    # can never be committed. Compare resolved paths without assuming POSIX.
    try:
        cache.resolve().relative_to((ROOT / "data").resolve())
    except ValueError:
        return fail("cache must live under the repository data/ directory")

    # Authorisation gate BEFORE any annotation or JPEG access.
    require_owner_approvals(args.license_ack, args.mapping_ack)

    before = cache_counts(cache)
    log("cached before: " + " ".join(f"{k}={v}" for k, v in before.items()))
    log(f"target: train={args.train} val={args.val} test={args.test} per class")

    log("resolving annotations and pinned ZIP indexes (network-light)...")
    requests = build_request_list(cache, counts)
    grouped = pending_by_source(requests, cache)
    total_pending = sum(len(v) for v in grouped.values())
    log(f"needed samples={len(requests)} pending={total_pending}")
    if not grouped:
        log("CACHE_COMPLETE: every required sample is already cached")
        return 0
    if args.report_only:
        for source in SOURCES:
            if source in grouped:
                log(f"  {source}: {len(grouped[source])} pending (cached {before[source]})")
        return 1

    # Meter JPEG transfer generously: the warm-up may run across several days of
    # retries, and the hard cap exists only to stop a runaway download. The
    # builder's own 60 GiB cap is what governs the published dataset.
    budget = ByteBudget(200 * 1024 ** 3)
    images = RemoteImages(cache, budget)
    # Largest gap first: surfaces a slow source early instead of at the end.
    order = sorted(grouped, key=lambda s: -len(grouped[s]))
    log("download order: " + ", ".join(f"{s}({len(grouped[s])})" for s in order))

    rnd = 0
    started = time.time()
    last_remaining = total_pending
    stalls = 0
    STALL_LIMIT = 3
    try:
        while True:
            rnd += 1
            log(f"=== pass {rnd} ===")
            for source in order:
                todo = [r for r in grouped.get(source, [])
                        if not _cache_entry_ok(
                            cache / "images" / source / (r[2] + ".jpg"))]
                if not todo:
                    continue
                t0 = time.time()
                try:
                    stats = images.prefetch(todo, workers=args.workers)
                except SourceError as exc:
                    log(f"  {source}: pass aborted ({type(exc).__name__}: {exc})")
                    continue
                elapsed = max(1e-6, time.time() - t0)
                left = sum(
                    1 for r in todo
                    if not _cache_entry_ok(cache / "images" / source / (r[2] + ".jpg"))
                )
                done = len(todo) - left
                rate = done / elapsed * 60
                log(f"  {source}: +{done}/{len(todo)} at {rate:.1f} img/min, "
                    f"{left} still missing, failures={stats.get('failed', 0)}")

            grouped = pending_by_source(requests, cache)
            remaining = sum(len(v) for v in grouped.values())
            after = cache_counts(cache)
            log("cached now: " + " ".join(f"{k}={v}" for k, v in after.items()))
            log(f"remaining={remaining} budget_used="
                f"{budget.used / (1024 ** 3):.2f}/60 GiB "
                f"elapsed={(time.time() - started) / 3600:.2f}h")
            if remaining == 0:
                log("CACHE_COMPLETE")
                return 0
            if args.rounds and rnd >= args.rounds:
                log(f"STOPPED_AFTER_{rnd}_PASSES remaining={remaining}")
                return 2

            # Never spin all weekend on a source that yields nothing: count
            # consecutive passes with no new bytes and stop loudly instead.
            if remaining < last_remaining:
                stalls = 0
            else:
                stalls += 1
                log(f"no progress this pass ({stalls}/{STALL_LIMIT})")
                if stalls >= STALL_LIMIT:
                    log("STALLED: no new images for "
                        f"{STALL_LIMIT} consecutive passes; remaining="
                        f"{remaining}. Re-run to resume from cache.")
                    return 3
            last_remaining = remaining

            log(f"sleeping {args.sleep}s before next pass")
            time.sleep(args.sleep)
    finally:
        images.close()


def fail(msg: str) -> int:
    print(f"FAIL: {msg}", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
