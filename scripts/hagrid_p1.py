#!/usr/bin/env python3
"""Offline HaGRIDv2 annotation audit and guarded *micro-sample* YOLO conversion.

Only JSON metadata and explicitly supplied local JPGs are read. Never downloads
images, weights or annotations. Raw user IDs remain in memory only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from datetime import date
from collections import Counter
from pathlib import Path

from scripts.data_contract import (
    AnnotationError, CLASS_NAMES, SOURCE_TO_INDEX,
    assert_subject_disjoint, strict_hand_labels,
)
from scripts.hagrid_policy import (
    PolicyError, approve_hard_negatives, validate_requested_sources,
)

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_LICENSE_SHA256 = "14f4845e9c8d3de875cbac4139491da368ab1040a7f565e02894477279134d22"
SPLITS = ("train", "val", "test")
SAFE_NAME = re.compile(r"[A-Za-z0-9_-]+")
HEX_SHA = re.compile(r"[a-f0-9]{64}")


class PreparationError(ValueError):
    """A fail-closed data, provenance, identity or licensing violation."""


def _load_json(path: Path) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise PreparationError(f"Cannot read required JSON: {path.name}") from exc
    if not isinstance(result, dict):
        raise PreparationError(f"JSON object required: {path.name}")
    return result


def _approval(path: Path, kind: str) -> dict:
    """Private evidence; never emit reviewer/identity information into reports."""
    doc = _load_json(path)
    if not isinstance(doc.get("reviewer"), str) or not doc["reviewer"].strip():
        raise PreparationError(f"{kind}: reviewer required")
    if not isinstance(doc.get("reviewed_on"), str):
        raise PreparationError(f"{kind}: ISO review date required")
    try:
        date.fromisoformat(doc["reviewed_on"])
    except ValueError as exc:
        raise PreparationError(f"{kind}: valid ISO review date required") from exc
    if kind == "license":
        sha = doc.get("license_pdf_sha256")
        if (
            not isinstance(sha, str)
            or not HEX_SHA.fullmatch(sha)
            or sha != OFFICIAL_LICENSE_SHA256
            or doc.get("permitted_scope") != "research_only"
            or doc.get("accept_terms") is not True
        ):
            raise PreparationError("License PDF review is missing or not approved for research")
    else:
        rows = doc.get("classes")
        if not isinstance(rows, list) or len(rows) != 10:
            raise PreparationError("All ten visual class mappings must be signed off")
        expected = {(source, CLASS_NAMES[idx]) for source, idx in SOURCE_TO_INDEX.items()}
        found = set()
        for item in rows:
            if not isinstance(item, dict):
                raise PreparationError("Malformed mapping review")
            source, output = item.get("source"), item.get("output")
            review_count = item.get("samples_reviewed")
            if (
                not isinstance(source, str) or not isinstance(output, str)
                or type(review_count) is not int or review_count < 1
            ):
                raise PreparationError("Malformed or incomplete mapping approval")
            key = (source, output)
            if (
                key not in expected or key in found or item.get("approved") is not True
                or item.get("official_reference_checked") is not True
            ):
                raise PreparationError("One or more visual class mappings are unreviewed")
            # Figure comparison exposed near-duplicate numeric/palm poses.
            # Require the reviewer to explicitly compare alternatives instead of
            # accepting a plausible source solely by its folder name.
            near_aliases = {
                "two_up": {"peace", "peace_inverted", "two_up_inverted"},
                "palm": {"stop", "stop_inverted"},
                "three": {"three2", "three3", "three_gun"},
                "one": {"point", "mute"},
            }
            required = near_aliases.get(source, set())
            if required:
                contrasts = item.get("contrasts_reviewed")
                if (
                    not isinstance(contrasts, list)
                    or any(not isinstance(v, str) for v in contrasts)
                    or not required <= set(contrasts)
                ):
                    raise PreparationError("Required visual gesture variants not reviewed")
            found.add(key)
        if found != expected:
            raise PreparationError("Ten signed-off source/output mappings required")
    return doc


def _paths(annotation_root: Path, sources: tuple[str, ...]):
    """Iterate split/source files; no untrusted annotation key becomes a path."""
    for split in SPLITS:
        folder = annotation_root / split
        if not folder.is_dir():
            raise PreparationError(f"Missing annotation split: {split}")
        for source in sources:
            path = folder / f"{source}.json"
            if not path.is_file():
                raise PreparationError(f"Missing annotation file for {split}/{source}")
            yield split, source, path


def _check_sample(
    stem: str, item: object, non_command_sources: frozenset[str] = frozenset()
) -> tuple[str, list[str]]:
    if not isinstance(stem, str) or not SAFE_NAME.fullmatch(stem):
        raise PreparationError("Unsafe or malformed image key")
    if not isinstance(item, dict):
        raise PreparationError("Malformed annotation record")
    subject = item.get("user_id")
    if not isinstance(subject, str) or not subject.strip():
        raise PreparationError("Missing subject ID")
    # A compound/union box is not a physical-hand annotation. Do not infer.
    if item.get("united_bbox"):
        raise PreparationError("Compound two-hand annotation requires manual review")
    if item.get("united_label"):
        raise PreparationError("Compound two-hand annotation requires manual review")
    boxes, labels = item.get("bboxes"), item.get("labels")
    if not isinstance(boxes, list) or not isinstance(labels, list):
        raise PreparationError("Missing or malformed per-hand boxes/labels")
    try:
        rows = strict_hand_labels(boxes, labels, non_command_sources=non_command_sources)
    except (AnnotationError, TypeError, ValueError) as exc:
        raise PreparationError("Invalid, unknown, or incomplete per-hand annotation") from exc
    return subject, rows


def _scan(
    annotation_root: Path,
    sources: tuple[str, ...],
    non_command_sources: frozenset[str] = frozenset(),
):
    """Compute only aggregate metadata; never expose IDs, sample keys or images."""
    try:
        validate_requested_sources(sources, conversion=bool(non_command_sources))
    except PolicyError as exc:
        raise PreparationError(str(exc)) from exc
    users = {split: set() for split in SPLITS}
    stats = {
        "source": "HaGRIDv2 Hagrid_v2-1M (local supplied annotations)",
        "scope": "requested source files only; not a whole-dataset audit",
        "samples_by_split": {s: 0 for s in SPLITS},
        "valid_by_split": {s: 0 for s in SPLITS},
        "rejected_by_split": {s: 0 for s in SPLITS},
        "boxes_by_class": dict.fromkeys(CLASS_NAMES, 0),
        "compound_samples_rejected": 0,
        "annotation_files_sha256": {},
    }
    for split, source, path in _paths(annotation_root, sources):
        stats["annotation_files_sha256"][f"{split}/{source}.json"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        doc = _load_json(path)
        for stem, item in sorted(doc.items()):
            stats["samples_by_split"][split] += 1
            # Detect leaked subject IDs even for records excluded by class policy.
            if isinstance(item, dict) and isinstance(item.get("user_id"), str) and item["user_id"]:
                users[split].add(item["user_id"])
            try:
                _subject, rows = _check_sample(stem, item, non_command_sources)
            except PreparationError:
                stats["rejected_by_split"][split] += 1
                if isinstance(item, dict) and (item.get("united_bbox") or item.get("united_label")):
                    stats["compound_samples_rejected"] += 1
                continue
            stats["valid_by_split"][split] += 1
            for row in rows:
                stats["boxes_by_class"][CLASS_NAMES[int(row.split()[0])]] += 1
    try:
        assert_subject_disjoint(users)
    except AnnotationError as exc:
        raise PreparationError("Subject overlap detected across train/val/test") from exc
    stats["subjects_by_split"] = {split: len(ids) for split, ids in users.items()}
    return stats


def audit(annotation_root: Path, sources: tuple[str, ...]) -> dict:
    return _scan(annotation_root, sources)


def convert(
    annotation_root: Path, images_root: Path, output: Path,
    license_ack: Path, mapping_ack: Path, sources: tuple[str, ...],
    limit_per_source: int = 5,
    *,
    hard_negative_sources: frozenset[str] = frozenset(),
    hard_negative_ack: Path | None = None,
) -> dict:
    try:
        requested_candidates = validate_requested_sources(sources, conversion=True)
        if not requested_candidates <= hard_negative_sources:
            raise PolicyError("Source selection contains unreviewed negative candidates")
        if hard_negative_sources:
            if hard_negative_ack is None:
                raise PolicyError("Private signed hard-negative review required")
            approved = approve_hard_negatives(hard_negative_ack, hard_negative_sources)
        else:
            if hard_negative_ack is not None:
                raise PolicyError("Unexpected negative review file without selected negatives")
            approved = frozenset()
    except PolicyError as exc:
        raise PreparationError(str(exc)) from exc
    if not (1 <= limit_per_source <= 100):
        raise PreparationError("Micro-sample cap must be between 1 and 100 per source per split")
    _approval(license_ack, "license")
    _approval(mapping_ack, "mapping")
    stats = _scan(annotation_root, sources, approved)
    output = output.resolve()
    local_repo = ROOT.resolve()
    if output == local_repo or (
        output.is_relative_to(local_repo) and
        not output.is_relative_to(local_repo / "data")
    ):
        raise PreparationError("Inside the Git repo, output must be under ignored data/")
    if output.exists():
        raise PreparationError("Output already exists; never silently overwrite data")
    output.parent.mkdir(parents=True, exist_ok=True)
    taken_digests: set[str] = set()
    selected = Counter()
    labels_written = Counter()
    missing_images = Counter()
    # All output is prepared in a staging directory. Errors leave no final dataset.
    with tempfile.TemporaryDirectory(prefix=".hagrid10-stage-", dir=output.parent) as staging:
        stage = Path(staging) / "dataset"
        for split, source, path in _paths(annotation_root, sources):
            count, used_subjects = 0, set()
            for stem, item in sorted(_load_json(path).items()):
                if count >= limit_per_source:
                    break
                try:
                    subject, rows = _check_sample(stem, item, approved)
                except PreparationError:
                    continue
                if subject in used_subjects:
                    continue
                src = images_root / source / f"{stem}.jpg"
                if not src.is_file() or src.is_symlink():
                    missing_images[split] += 1
                    continue
                # JPEG signature is only a cheap sanity check, not visual QA/decode.
                with src.open("rb") as handle:
                    magic = handle.read(3)
                if magic != b"\xff\xd8\xff":
                    missing_images[split] += 1
                    continue
                digest = hashlib.sha256(src.read_bytes()).hexdigest()
                if digest in taken_digests:
                    raise PreparationError("Duplicate image content in selected micro-sample")
                taken_digests.add(digest)
                rel = Path(split) / source / f"{stem}.jpg"
                img_target = stage / "images" / rel
                label_target = stage / "labels" / rel.with_suffix(".txt")
                img_target.parent.mkdir(parents=True, exist_ok=True)
                label_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, img_target)
                label_target.write_text("\n".join(rows) + "\n", encoding="utf-8")
                count += 1
                used_subjects.add(subject)
                selected[split] += 1
                for row in rows:
                    labels_written[CLASS_NAMES[int(row.split()[0])]] += 1
        if not all(selected[s] for s in SPLITS):
            raise PreparationError("Each split must contain at least one licensed, approved image")
        stats["selected_images_by_split"] = {s: selected[s] for s in SPLITS}
        stats["selected_boxes_by_class"] = {name: labels_written[name] for name in CLASS_NAMES}
        stats["missing_or_invalid_jpg_by_split"] = {s: missing_images[s] for s in SPLITS}
        stats["distinct_selected_image_hashes"] = len(taken_digests)
        stats["approved_hard_negative_sources"] = sorted(approved)
        stats["license_review_present"] = True
        stats["visual_mapping_approval_present"] = True
        stats["limitations"] = (
            "Micro-sample only; JPEG signature not full decoding, no visual unlabeled-hand "
            "check, no verified negative no_hand set, no human image QA. NOT P1 PASS."
        )
        (stage / "manifest.aggregate.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stage.rename(output)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "convert"))
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument(
        "--sources", nargs="+", default=list(SOURCE_TO_INDEX),
        help="HaGRID source JSON stems; default is the canonical ten"
    )
    parser.add_argument("--images", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--license-ack", type=Path)
    parser.add_argument("--mapping-ack", type=Path)
    parser.add_argument("--limit-per-source", type=int, default=5)
    parser.add_argument("--hard-negative-sources", nargs="+", default=[])
    parser.add_argument("--hard-negative-ack", type=Path)
    args = parser.parse_args()
    sources = tuple(args.sources)
    try:
        validate_requested_sources(sources)
    except PolicyError as exc:
        parser.error(str(exc))
    try:
        if args.mode == "audit":
            stats = audit(args.annotations, sources)
        else:
            if not all((args.images, args.output, args.license_ack, args.mapping_ack)):
                parser.error("convert requires --images --output --license-ack --mapping-ack")
            stats = convert(
                args.annotations, args.images, args.output, args.license_ack,
                args.mapping_ack, sources, args.limit_per_source,
                hard_negative_sources=frozenset(args.hard_negative_sources),
                hard_negative_ack=args.hard_negative_ack,
            )
    except PreparationError as exc:
        parser.exit(2, f"FAIL CLOSED: {exc}\n")
    # Aggregate-only output, safe for issue/PR comments; no source identities.
    print(json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
