#!/usr/bin/env python3
"""P1-03/04 one-command LOCAL (NOT CI) four-way HaGRID visual review prep.

This is an orchestrator, not a rights-grant, model training, or a human review.
- plan: print fully offline intended inputs / caps, without network access.
- prepare: require signed *local* license scope, download train-only metadata
  and 3-20 JPGs per source by bounded HTTP Range; create private HTML packet.
- verify: cryptographically bind an existing real human review file to the
  prepared private packet. Never assert actual semantic or stage P1 PASS.

No real images, annotation records or reviewer names belong in public GitHub.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from scripts.hagrid_p1 import PreparationError, ROOT, _approval
from scripts.hagrid_zip_range import extract as extract_annotations
from scripts.hagrid_image_zip64 import (
    PINNED_INDEX_SHA256, PINNED_SIZES, ImageRangeError,
    extract as extract_images,
)
from scripts.p1_visual_review import (
    DEFAULT_SOURCES, VisualReviewError, build_packet, verify_review,
)

SOURCES = DEFAULT_SOURCES
WORKFLOW_SCHEMA = 1


class WorkflowError(ValueError):
    """Local review preparation was not committed as a complete workspace."""


def offline_plan() -> dict:
    """Truly offline: this function does not inspect, open or fetch images."""
    return {
        "status": "PLAN_ONLY_NO_DATA_ACCESSED",
        "stage": "P1-03/P1-04",
        "source_split": "train",
        "source_categories": list(SOURCES),
        "model_input": [224, 224],
        "model_output_classes": 10,
        "requires_real_owner_license_acceptance": True,
        "requires_human_visual_adjudication": True,
        "defaults": {
            "jpgs_per_source": 6,
            "annotation_budget_mib_total": 80,
            "jpg_budget_mib_per_source": 80,
        },
        "verified_image_zip64_byte_lengths": dict(PINNED_SIZES),
        "pinned_archive_indexes": True,
        "stages": [
            "Read local signed research-use license acknowledgment",
            "Download only 4 train JSON annotation ZIP members",
            "Download 3-20 train JPGs/source, with strict byte budget",
            "Generate private HTML card packet and unsigned review template",
            "Review actual image poses/boxes locally, then verify signed evidence",
        ],
        "no_real_data_downloaded": True,
        "p1_pass_claimed": False,
    }


def _check_env():
    # Nothing that fetches real personal annotations or JPGs can run on public CI.
    if os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("CI") == "true":
        raise WorkflowError("Actual HaGRID data acquisition is forbidden in CI")


def _validate_target(output: Path) -> Path:
    real = output.resolve()
    repo = ROOT.resolve()
    if output.is_symlink() or real == repo or (
        real.is_relative_to(repo) and
        not real.is_relative_to(repo / "data")
    ):
        raise WorkflowError("Inside this Public Git repository the output must be under ignored data/")
    if real.exists():
        raise WorkflowError("Private workspace already exists; refusing overwrite")
    if real.parent == real:
        raise WorkflowError("Unsafe output root")
    return real


def prepare(
    output: Path, license_ack: Path,
    per_source: int = 6, annotation_budget_mib: int = 80,
    jpg_budget_mib: int = 80, seed: str = "p1-two-review-v1",
) -> dict:
    """Fail closed and publish a new workspace *only after every step succeeds*."""
    _check_env()
    try:
        _approval(license_ack, "license")
    except PreparationError as exc:
        raise WorkflowError("Owner-signed local HaGRID research permission is missing") from exc
    if not 3 <= per_source <= 20:
        raise WorkflowError("Must prepare 3-20 train JPGs per source for visual adjudication")
    if not 1 <= annotation_budget_mib <= 256:
        raise WorkflowError("Annotation transfer budget must be 1-256 MiB")
    if not 1 <= jpg_budget_mib <= 512:
        raise WorkflowError("Per-source JPG transfer budget must be 1-512 MiB")
    if not isinstance(seed, str) or not seed or len(seed) > 200:
        raise WorkflowError("A bounded deterministic sampling seed is required")

    target = _validate_target(output)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    # No final folder appears until all four sources, review packet and receipt
    # are ready. TemporaryDirectory removes *everything* upon any exception.
    with tempfile.TemporaryDirectory(prefix=".p1-review-stage-", dir=target.parent) as temp:
        stage = Path(temp) / "workspace"
        stage.mkdir(mode=0o700)
        annotations = stage / "annotations"
        images = stage / "images"
        html_packet = stage / "packet"
        try:
            annotations_receipt = extract_annotations(
                SOURCES, ("train",), annotations, license_ack,
                annotation_budget_mib,
            )
            if sorted(annotations_receipt["files"]) != sorted(
                f"annotations/train/{source}.json" for source in SOURCES
            ):
                raise WorkflowError("Unexpected or incomplete train-only annotation extraction")
            counts = {}
            for source in SOURCES:
                image_receipt = extract_images(
                    source, annotations, images, license_ack,
                    per_source=per_source, max_transfer_mib=jpg_budget_mib,
                    seed=seed,
                )
                if (image_receipt["source"] != source
                        or image_receipt["jpeg_count"] != per_source
                        or image_receipt["train_only"] is not True):
                    raise WorkflowError("Incomplete or unsafe per-source JPG acquisition")
                counts[source] = image_receipt["jpeg_count"]
            packet_receipt = build_packet(
                annotations, images, html_packet, license_ack,
                sources=SOURCES, per_source=per_source, seed=seed,
            )
            if (packet_receipt["sources"] != list(SOURCES)
                    or packet_receipt["train_only"] is not True
                    or set(packet_receipt["counts_by_source"]) != set(SOURCES)
                    or any(n < 3 for n in packet_receipt["counts_by_source"].values())):
                raise WorkflowError("Incomplete private visual packet")
        except (PreparationError, ImageRangeError, VisualReviewError) as exc:
            raise WorkflowError(
                "P1 local media preparation failed; staged private files were discarded"
            ) from exc

        packet_manifest = (html_packet / "packet.manifest.json").read_bytes()
        packet_digest = hashlib.sha256(packet_manifest).hexdigest()
        if packet_digest != packet_receipt["manifest_sha256"]:
            raise WorkflowError("Private packet manifest was modified while staging")
        receipt = {
            "schema_version": WORKFLOW_SCHEMA,
            "status": "PRIVATE_TRAIN_REVIEW_PACKET_READY_NOT_HUMAN_APPROVED",
            "split": "train",
            "sources": list(SOURCES),
            "per_source_requested": per_source,
            "actual_image_counts": packet_receipt["counts_by_source"],
            "source_model_input": [224, 224],
            "source_model_classes": 10,
            "annotation_sources": len(annotations_receipt["files"]),
            "image_zip64_index_sha256": dict(PINNED_INDEX_SHA256),
            "packet_manifest_sha256": packet_digest,
            "license_acceptance_checked": True,
            "no_p1_semantic_or_licensing_pass_inferred": True,
            "instructions": (
                "Open packet/index.html locally, compare each physical hand,"
                " copy packet/review.template.json outside this directory,"
                " record true decisions and verify via the verify subcommand."
            ),
        }
        (stage / "workflow.private.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if target.exists():
            raise WorkflowError("Another process created the target workspace")
        stage.rename(target)

    # Return aggregate-only results; never emit raw sample IDs or subject IDs.
    return {
        "status": receipt["status"],
        "sources": list(SOURCES),
        "selected_per_source": receipt["actual_image_counts"],
        "workspace": str(target),
        "private_html": str(target / "packet" / "index.html"),
        "human_signoff_pending": True,
        "p1_pass": False,
    }


def verify(output: Path, review: Path) -> dict:
    """Integrity check, NOT a legal acceptance, identity check or gesture oracle."""
    workspace = output.resolve()
    try:
        data = json.loads(
            (workspace / "workflow.private.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, UnicodeError) as exc:
        raise WorkflowError("Missing/invalid private review workspace receipt") from exc
    if (
        not isinstance(data, dict) or data.get("schema_version") != WORKFLOW_SCHEMA
        or data.get("status") != "PRIVATE_TRAIN_REVIEW_PACKET_READY_NOT_HUMAN_APPROVED"
        or data.get("sources") != list(SOURCES) or data.get("split") != "train"
        or data.get("source_model_classes") != 10
    ):
        raise WorkflowError("Workspace provenance does not match P1 four-way review contract")
    try:
        report = verify_review(workspace / "packet", review)
    except VisualReviewError as exc:
        raise WorkflowError("Review cannot be matched to the private image packet") from exc
    # A fully valid review for *four sources* still cannot approve all ten.
    return {
        "status": report["status"],
        "sources_checked": report["sources_checked"],
        "reviewed_images": report["reviewed_images"],
        "unresolved_sources": report["unresolved_sources"],
        "potential_command_alias_conflicts": report["potential_command_alias_conflicts"],
        "human_gesture_verdict_recorded": True,
        "mapping_change_requires_new_reviewed_PR": True,
        "ten_class_P1_signoff_still_required": True,
        "p1_pass": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("plan", help="Offline-only requirements and byte-budget preview")
    make = sub.add_parser("prepare", help="Acquire signed/licensed train media into private workspace")
    make.add_argument("--license-ack", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--per-source", type=int, default=6)
    make.add_argument("--annotation-budget-mib", type=int, default=80)
    make.add_argument("--jpg-budget-mib", type=int, default=80)
    make.add_argument("--seed", default="p1-two-review-v1")
    check = sub.add_parser("verify", help="Validate existing locally reviewed packet integrity")
    check.add_argument("--workspace", type=Path, required=True)
    check.add_argument("--review", type=Path, required=True)
    args = p.parse_args()
    try:
        if args.action == "plan":
            out = offline_plan()
        elif args.action == "prepare":
            out = prepare(
                args.output, args.license_ack, args.per_source,
                args.annotation_budget_mib, args.jpg_budget_mib, args.seed,
            )
        else:
            out = verify(args.workspace, args.review)
    except WorkflowError as exc:
        p.exit(2, f"FAIL CLOSED: {exc}\n")
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
