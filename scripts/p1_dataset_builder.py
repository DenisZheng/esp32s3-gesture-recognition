#!/usr/bin/env python3
"""P1-05 ten-class private dataset builder: local-dir, local-zip or auto Range.

P1 owner license + full ten-class visual mapping are mandatory BEFORE reading
actual person data. Auto uses only official index-SHA-pinned HaGRID sources and
a resumable PRIVATE CRC-checked cache. Manual modes consume already downloaded
source JPEG folders or per-class ZIPs. No YOLO training is launched.

Result is a *technical dataset candidate*, NEVER P1 legal/visual QA PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path

from scripts.data_contract import AnnotationError, CLASS_NAMES, SOURCE_TO_INDEX, assert_subject_disjoint
from scripts.hagrid_p1 import ROOT, PreparationError, _approval, _check_sample
from scripts.hagrid_policy import policy_sha256
from scripts.hagrid_image_zip64 import ImageRangeError
from scripts.p1_dataset_sources import (
    ByteBudget, RemoteImages, SOURCES, SPLITS, SourceError, STEM, make_sources,
)

SCHEMA = 1
DEFAULT_COUNTS = {"train": 500, "val": 50, "test": 50}
MAX_PER_SOURCE = 2000
ANNOTATION_BYTES_OFFICIAL = 171_848_608  # public ZIP metadata, ten classes / three splits
MIN_LOCAL_JPEG_BYTES = 4
MAX_CANDIDATE_RECORDS = 100_000


class DatasetBuildError(ValueError):
    """Refuse incomplete, ambiguous, unauthorized or non-reproducible dataset."""


def offline_plan(counts: dict[str, int] | None = None) -> dict:
    counts = DEFAULT_COUNTS if counts is None else counts
    _validate_counts(counts)
    return {
        "status": "P1_05_PLAN_ONLY_NO_PERSONAL_DATA_ACCESSED",
        "sources": list(SOURCES),
        "class_count": 10,
        "class_names": list(CLASS_NAMES),
        "input_hw": [224, 224],
        "model_architecture": "single ESPDet-Pico",
        "methods": ["local-dir", "local-zip", "auto"],
        "auto_annotation_index_compressed_bytes": ANNOTATION_BYTES_OFFICIAL,
        "requested_per_class_split": dict(counts),
        "requested_images_total": 10 * sum(counts.values()),
        "auto_never_downloads_whole_image_archives": True,
        "subject_disjoint_original_splits_required": True,
        "license_and_ten_gesture_mapping_signoff_required": True,
        "no_gesture_is_bbox_class_7_not_no_hand": True,
        "p1_human_data_qa_still_required": True,
        "actual_download_or_training_started": False,
    }


def _validate_counts(counts: dict[str, int]):
    if set(counts) != set(SPLITS):
        raise DatasetBuildError("Exactly train, val and test selection caps required")
    if any(type(value) is not int or not 1 <= value <= MAX_PER_SOURCE
           for value in counts.values()):
        raise DatasetBuildError("Per-class split quotas must be 1..2000")


def _private_target(path: Path) -> Path:
    result = path.resolve()
    repo = ROOT.resolve()
    if path.is_symlink() or result == repo or (
        result.is_relative_to(repo) and
        not result.is_relative_to(repo / "data")
    ):
        raise DatasetBuildError("Git repository output/cache must be under ignored data/")
    return result


def _read_source_json(raw: bytes) -> dict:
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetBuildError("Licensed source annotation JSON invalid") from exc
    if not isinstance(doc, dict) or len(doc) > MAX_CANDIDATE_RECORDS:
        raise DatasetBuildError("Annotation JSON is not an expected bounded source object")
    return doc


def _rank(seed: str, split: str, source: str, stem: str) -> str:
    return hashlib.sha256(
        (seed + "\0" + split + "\0" + source + "\0" + stem).encode("utf-8")
    ).hexdigest()


def _prioritized_candidates(rows):
    """Stable subject diversity first, then fill from remaining valid samples."""
    ordered = sorted(rows, key=lambda r: (r[0], r[1]))
    first, remaining, seen = [], [], set()
    for record in ordered:
        if record[2] in seen:
            remaining.append(record)
        else:
            first.append(record)
            seen.add(record[2])
    return first + remaining


def _gather_candidates(annotations, images, seed: str):
    """Scan all original splits BEFORE fetching a single personal JPEG."""
    users = {split: set() for split in SPLITS}
    selected = {}
    rejection = Counter()
    annotation_hashes = {}
    valid_records = Counter()
    for split in SPLITS:
        for source in SOURCES:
            raw = annotations.get(split, source)
            annotation_hashes[f"{split}/{source}.json"] = hashlib.sha256(raw).hexdigest()
            doc = _read_source_json(raw)
            candidates = []
            for stem, record in doc.items():
                # Cross-split identity leakage counts even otherwise rejected
                # annotation records; no source ID is printed or serialized.
                if isinstance(record, dict):
                    subject = record.get("user_id")
                    if isinstance(subject, str) and subject.strip():
                        users[split].add(subject)
                try:
                    subject, yolo = _check_sample(stem, record)
                except PreparationError:
                    rejection[split] += 1
                    continue
                if source not in record.get("labels", ()):
                    rejection[split] += 1
                    continue
                valid_records[split] += 1
                if not images.has(source, stem):
                    continue
                candidates.append((_rank(seed, split, source, stem), stem, subject, yolo))
            selected[(split, source)] = _prioritized_candidates(candidates)
    try:
        assert_subject_disjoint(users)
    except AnnotationError as exc:
        raise DatasetBuildError("Original subject IDs overlap across train/val/test") from exc
    return selected, {
        "original_annotation_sha256": annotation_hashes,
        "subjects_by_split": {key: len(s) for key, s in users.items()},
        "invalid_annotation_records_by_split": {s: rejection[s] for s in SPLITS},
        "valid_annotation_records_by_split": {s: valid_records[s] for s in SPLITS},
    }


def build_dataset(
    mode: str, output: Path, license_ack: Path, mapping_ack: Path,
    *, annotations: Path | None = None, images: Path | None = None,
    cache: Path | None = None, counts: dict[str, int] | None = None,
    full: bool = False, seed: str = "espdet-p1-05-v1", max_transfer_gib: int = 12,
    max_output_gib: int = 12, prefetch_workers: int = 8,
):
    # Important: no user-submitted/official image or JSON reads before gate.
    if os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("CI") == "true":
        raise DatasetBuildError("Real HaGRID dataset processing is forbidden in CI")
    try:
        _approval(license_ack, "license")
        _approval(mapping_ack, "mapping")
    except PreparationError as exc:
        raise DatasetBuildError("Owner license and all-ten mapping approvals are required") from exc
    if full:
        if mode != "local-dir":
            raise DatasetBuildError("Full mode is only supported for a local directory")
        if counts is not None:
            raise DatasetBuildError("Full mode must not receive bounded split quotas")
    else:
        counts = DEFAULT_COUNTS if counts is None else counts
        _validate_counts(counts)
    if mode not in ("local-dir", "local-zip", "auto"):
        raise DatasetBuildError("Choose local-dir, local-zip or auto mode")
    if not isinstance(seed, str) or not 1 <= len(seed) <= 120:
        raise DatasetBuildError("Nonempty deterministic seed (max 120) required")
    if (type(max_transfer_gib) is not int or not 1 <= max_transfer_gib <= 60
            or type(max_output_gib) is not int or not 1 <= max_output_gib <= 100):
        raise DatasetBuildError("Unsafe byte budget; choose explicit limits")
    if type(prefetch_workers) is not int or not 3 <= prefetch_workers <= 32:
        raise DatasetBuildError("Prefetch workers must be 3..32")
    dest = _private_target(output)
    if dest.exists():
        raise DatasetBuildError("Output dataset already exists; never overwrite")
    if mode == "auto":
        if cache is None:
            raise DatasetBuildError("auto mode needs an ignored persistent --cache for resume")
        cache = _private_target(cache)
        if cache == dest or cache.is_relative_to(dest) or dest.is_relative_to(cache):
            raise DatasetBuildError("Cache and final dataset folders must be separate")
        if cache.is_symlink():
            raise DatasetBuildError("Private cache cannot be a symlink")
    else:
        if annotations is None or images is None:
            raise DatasetBuildError("Manual input needs --annotations and --images")
    # Auto caches valid JPEGs across failures. Only the FINAL dataset is
    # transactional; no incomplete training dataset gets published.
    budget = ByteBudget(max_transfer_gib * (1024 ** 3))
    dest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    a = im = None
    try:
        a, im = make_sources(mode, annotations, images, cache, budget)
        candidates, provenance = _gather_candidates(a, im, seed)
        with tempfile.TemporaryDirectory(
            prefix=".p1-05-dataset-", dir=dest.parent
        ) as tmp:
            stage = Path(tmp) / "dataset"
            seen_hashes = {}
            selected_stats = Counter()
            class_boxes = Counter()
            manifest = []
            output_bytes = 0
            skipped_media = 0
            duplicate_across_splits = 0
            # Prefetch remote JPEGs concurrently. The upstream range reader is
            # strictly serial: each image costs two round trips (local header
            # then payload), which measured ~6 images/min on the owner host.
            # Eight workers reach roughly 1.3 MB/s on the same pinned URLs while
            # keeping every CRC/length/signature check in RemoteImages.get.
            if mode == "auto" and isinstance(im, RemoteImages):
                # Over-fetch per split, not globally: a single shared window sized
                # by the train quota made val/test prefetch as many images as
                # train (measured 5x wasted transfer on the 500/50/50 build).
                margin = max(10, counts["train"] // 10)

                def _want(split: str) -> int:
                    return min(MAX_PER_SOURCE, counts[split] + margin)

                im.prefetch(
                    [(split, source, row[1])
                     for split in SPLITS for source in SOURCES
                     for row in candidates[(split, source)][:_want(split)]],
                    workers=prefetch_workers,
                )
            for split in SPLITS:
                for source in SOURCES:
                    quota = (len(candidates[(split, source)]) if full
                             else counts[split])
                    selected_count = 0
                    source_skipped = 0
                    source_duplicates = 0
                    for _score, stem, _subject, rows in candidates[(split, source)]:
                        if selected_count >= quota:
                            break
                        try:
                            image = im.get(source, stem)
                        except (SourceError, ImageRangeError) as exc:
                            # A single sample that still fails after the transport's
                            # own retries must not discard an hours-long build. Skip
                            # it, keep the failure visible, and let the per-source
                            # quota check below still fail closed if too few landed.
                            skipped_media += 1
                            source_skipped += 1
                            print(
                                f"[build] skip {split}/{source}/{stem[:8]}: "
                                f"{type(exc).__name__}",
                                flush=True,
                            )
                            continue
                        sha = hashlib.sha256(image).hexdigest()
                        if sha in seen_hashes:
                            if seen_hashes[sha] != split:
                                # The same bytes already landed in another split.
                                # Publishing them again would leak a held-out
                                # sample into training, so drop this candidate
                                # and let the quota loop take the next one; the
                                # per-source quota check still fails closed if
                                # too few distinct images remain.
                                duplicate_across_splits += 1
                                source_duplicates += 1
                                print(
                                    f"[build] drop duplicate bytes "
                                    f"{split}/{source}/{stem[:8]} "
                                    f"(already in {seen_hashes[sha]})",
                                    flush=True,
                                )
                                continue
                            # Same-frame multi-gesture duplicate: drop within split
                            # rather than poison source class balancing.
                            source_duplicates += 1
                            continue
                        label = ("\n".join(rows) + "\n").encode("utf-8")
                        output_bytes += len(image) + len(label)
                        if output_bytes > max_output_gib * (1024 ** 3):
                            raise DatasetBuildError("Local dataset output size budget exceeded")
                        rel = Path(split) / source / (stem + ".jpg")
                        photo = stage / "images" / rel
                        label_path = stage / "labels" / rel.with_suffix(".txt")
                        photo.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                        label_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                        photo.write_bytes(image)
                        label_path.write_bytes(label)
                        seen_hashes[sha] = split
                        selected_count += 1
                        selected_stats[split] += 1
                        for row in rows:
                            class_boxes[CLASS_NAMES[int(row.split()[0])]] += 1
                        # File names/hashes are PRIVATE. Do not copy to GitHub.
                        manifest.append({
                            "image": str(Path("images") / rel),
                            "sha256": sha,
                            "label": str(Path("labels") / rel.with_suffix(".txt")),
                            "label_sha256": hashlib.sha256(label).hexdigest(),
                        })
                    if selected_count != quota and (
                        not full or source_skipped or not source_duplicates
                    ):
                        raise DatasetBuildError(
                            "Insufficient licensed, distinct valid images for requested "
                            "split/source quota"
                        )
            aggregate = {
                "schema_version": SCHEMA,
                "status": "TEN_CLASS_CANDIDATE_BUILT_PENDING_HUMAN_QA",
                "model_classes": 10,
                "model_input_hw": [224, 224],
                "sources": list(SOURCES),
                "source_mode": mode,
                "source_policy_sha256": policy_sha256(),
                "classes_config_sha256": hashlib.sha256(
                    (ROOT / "configs/classes.json").read_bytes()
                ).hexdigest(),
                "annotation_file_sha256_private": "See manifest.private.json",
                "by_split": {s: selected_stats[s] for s in SPLITS},
                "selection_mode": (
                    "full_available_valid_records" if full
                    else "bounded_per_source_quota"
                ),
                "per_source_split_target": (
                    {
                        split: {
                            source: len(candidates[(split, source)])
                            for source in SOURCES
                        }
                        for split in SPLITS
                    }
                    if full else dict(counts)
                ),
                "box_counts": {label: class_boxes[label] for label in CLASS_NAMES},
                "source_subject_counts": provenance["subjects_by_split"],
                "invalid_records_by_split": provenance["invalid_annotation_records_by_split"],
                "discarded_cross_label_records_not_silently_ignored": True,
                "seen_image_sha256_count": len(seen_hashes),
                "dropped_cross_split_duplicate_bytes": duplicate_across_splits,
                "skipped_unreadable_media": skipped_media,
                "no_hand_negative_images_auto_created": 0,
                "human_visual_decode_and_legal_qa_status": "PENDING",
                "P1_pass": False,
                "P2_training_started": False,
            }
            dataset_yaml = "path: " + json.dumps(str(dest)) + "\n"
            dataset_yaml += "train: images/train\nval: images/val\ntest: images/test\n"
            dataset_yaml += "nc: 10\nnames:\n"
            dataset_yaml += "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES))
            # The YAML is part of the dataset contract, not just a pointer: it
            # fixes class order and split paths. Record its hash in the manifest
            # so a later edit cannot keep the image/label hashes valid while
            # silently changing what the model is trained on.
            aggregate["dataset_yaml_sha256"] = hashlib.sha256(
                dataset_yaml.encode("utf-8")
            ).hexdigest()
            aggregate["class_names_ordered"] = list(CLASS_NAMES)
            private = {
                "schema_version": SCHEMA,
                "created_on": date.today().isoformat(),
                "annotation_file_sha256": provenance["original_annotation_sha256"],
                "sample_files": manifest,
                "aggregate_sha256": hashlib.sha256(
                    json.dumps(aggregate, sort_keys=True).encode("utf-8")
                ).hexdigest(),
                "no_raw_subject_ids_exported": True,
            }
            (stage / "manifest.aggregate.json").write_text(
                json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (stage / "manifest.private.json").write_text(
                json.dumps(private, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (stage / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")
            if dest.exists():
                raise DatasetBuildError("Output directory appeared during build")
            stage.rename(dest)
    except (SourceError, PreparationError, ImageRangeError) as exc:
        raise DatasetBuildError("Private source failed integrity/size/permission checks") from exc
    finally:
        if im:
            im.close()
        if a:
            a.close()
    # Aggregate only; no original filename, source key, ID or image digest.
    return {
        "status": "TEN_CLASS_CANDIDATE_BUILT_PENDING_HUMAN_QA",
        "output": str(dest),
        "source_mode": mode,
        "by_split": aggregate["by_split"],
        "box_counts": aggregate["box_counts"],
        "network_bytes_accounted": budget.used if mode == "auto" else 0,
        "human_qa_and_legal_pass": False,
        "P2_training_started": False,
    }


def verify_output(folder: Path) -> dict:
    """Local byte-for-byte integrity, not an image/gesture semantic oracle."""
    base = folder.resolve()
    try:
        aggregate_file = base / "manifest.aggregate.json"
        private_file = base / "manifest.private.json"
        if aggregate_file.is_symlink() or private_file.is_symlink():
            raise DatasetBuildError("Symlinked manifest is forbidden")
        aggregate = json.loads(aggregate_file.read_text(encoding="utf-8"))
        manifest = json.loads(private_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetBuildError("Missing/invalid private dataset manifests") from exc
    if (aggregate.get("status") != "TEN_CLASS_CANDIDATE_BUILT_PENDING_HUMAN_QA"
            or aggregate.get("model_classes") != 10
            or aggregate.get("P1_pass") not in (False, True)
            or manifest.get("schema_version") != SCHEMA):
        raise DatasetBuildError("Not a P1-05 candidate dataset")
    calculated = hashlib.sha256(json.dumps(aggregate, sort_keys=True).encode()).hexdigest()
    if calculated != manifest.get("aggregate_sha256"):
        raise DatasetBuildError("Aggregate dataset manifest changed")
    sample_files = manifest.get("sample_files")
    if not isinstance(sample_files, list) or not sample_files:
        raise DatasetBuildError("Empty/malformed private sample manifest")
    counts = Counter()
    for row in sample_files:
        if not isinstance(row, dict):
            raise DatasetBuildError("Malformed private sample row")
        rel_img, rel_lbl = row.get("image"), row.get("label")
        for rel, stem in ((rel_img, "images"), (rel_lbl, "labels")):
            if (not isinstance(rel, str) or "\\" in rel or
                    Path(rel).is_absolute() or ".." in Path(rel).parts or
                    not rel.startswith(stem + "/")):
                raise DatasetBuildError("Unsafe dataset manifest relative path")
        img, label = base / rel_img, base / rel_lbl
        if (not img.is_file() or not label.is_file() or img.is_symlink()
                or label.is_symlink() or not img.resolve().is_relative_to(base)
                or not label.resolve().is_relative_to(base)):
            raise DatasetBuildError("Missing/symlinked dataset image/label")
        if (hashlib.sha256(img.read_bytes()).hexdigest() != row.get("sha256")
                or hashlib.sha256(label.read_bytes()).hexdigest()
                != row.get("label_sha256")):
            raise DatasetBuildError("Dataset JPEG or YOLO label changed after preparation")
        parts = Path(rel_img).parts
        if len(parts) != 4 or parts[1] not in SPLITS or parts[2] not in SOURCES:
            raise DatasetBuildError("Malformed output split or source path")
        counts[parts[1]] += 1
    if {s: counts[s] for s in SPLITS} != aggregate.get("by_split"):
        raise DatasetBuildError("Manifest split count differs from file evidence")
    return {
        "status": "PRIVATE_DATASET_CHECKSUMS_PASS_HUMAN_QA_PENDING",
        "samples": len(sample_files),
        "splits": {s: counts[s] for s in SPLITS},
        "actual_images_visually_reviewed": False,
        "P1_pass": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    preview = sub.add_parser("plan", help="show ten-class quotas and source modes offline")
    for x in (preview,):
        x.add_argument("--train-per-source", type=int, default=500)
        x.add_argument("--val-per-source", type=int, default=50)
        x.add_argument("--test-per-source", type=int, default=50)
    b = sub.add_parser("build", help="license-gated local ZIP/folder or ranged download")
    b.add_argument("--mode", required=True, choices=("local-dir", "local-zip", "auto"))
    b.add_argument("--annotations", type=Path, help="annotation folder or local annotations ZIP")
    b.add_argument("--images", type=Path, help="image source folders or per-class ZIP folder")
    b.add_argument("--cache", type=Path, help="persistent private auto download cache")
    b.add_argument("--output", required=True, type=Path)
    b.add_argument("--license-ack", required=True, type=Path)
    b.add_argument("--mapping-ack", required=True, type=Path)
    for key, default in DEFAULT_COUNTS.items():
        b.add_argument(f"--{key}-per-source", type=int, default=default)
    b.add_argument("--full", action="store_true",
                   help="use every valid image in each source/split; local-dir only")
    b.add_argument("--seed", default="espdet-p1-05-v1")
    b.add_argument("--max-transfer-gib", type=int, default=12)
    b.add_argument("--max-output-gib", type=int, default=12)
    b.add_argument("--prefetch-workers", type=int, default=16,
                   help="Concurrent range prefetch workers for auto mode (3..32)")
    v = sub.add_parser("verify", help="verify private image/label SHA256s")
    v.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        if args.command == "verify":
            out = verify_output(args.output)
        else:
            full = getattr(args, "full", False)
            counts = None if full else {
                "train": args.train_per_source,
                "val": args.val_per_source,
                "test": args.test_per_source,
            }
            out = offline_plan(counts or DEFAULT_COUNTS) if args.command == "plan" else build_dataset(
                args.mode, args.output, args.license_ack, args.mapping_ack,
                annotations=args.annotations, images=args.images, cache=args.cache,
                counts=counts, full=full, seed=args.seed,
                max_transfer_gib=args.max_transfer_gib,
                max_output_gib=args.max_output_gib,
                prefetch_workers=args.prefetch_workers,
            )
    except DatasetBuildError as exc:
        p.exit(2, f"FAIL CLOSED: {exc}\n")
    print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
