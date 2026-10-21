#!/usr/bin/env python3
"""P1-06: five-group *private* train-only ten-gesture visual review campaign.

This script never signs license terms, approves a pose, or trains a network.
A user-authorized research-license receipt is required before any real
HaGRID annotation/JPG download. Every group has a separate private, atomic
workspace so review can proceed in bounded steps on a Linux desktop.

Modes:
  plan                         NO I/O/network/personal information
  prepare --group ...          3-20 train pictures/source via pinned Range
  verify --group ...           bind human review JSON to unchanged sample hashes
  draft --evidence GROUP WORKSPACE REVIEW (repeat for each of 5 groups)
                               produce *UNSIGNED INVALID* ten-class mapping draft
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

from scripts.data_contract import CLASS_NAMES, SOURCE_TO_INDEX
from scripts.hagrid_p1 import ROOT, PreparationError, _approval, OFFICIAL_LICENSE_SHA256
from scripts.hagrid_image_zip64 import (
    IMAGE_URLS, PINNED_SIZES, PINNED_INDEX_SHA256, ImageRangeError,
    extract as fetch_image_subset,
)
from scripts.hagrid_zip_range import extract as fetch_train_annotations
from scripts.hagrid_policy import policy_sha256
from scripts.p1_visual_review import (
    NEEDED_CONTRASTS, VisualReviewError, build_packet, verify_review,
)

SCHEMA = 1
# Alias negatives cannot be used for training without a separate policy and
# real visible-hand review. Here they are comparison references ONLY.
GROUPS = {
    "one": ("one", "point", "mute"),
    "two": ("two_up", "peace", "peace_inverted", "two_up_inverted"),
    "three": ("three", "three2", "three3", "three_gun"),
    "five": ("palm", "stop", "stop_inverted"),
    "remaining": ("four", "like", "ok", "no_gesture", "call", "dislike"),
}
EXPECTED = set(SOURCE_TO_INDEX) | {
    variant for variants in NEEDED_CONTRASTS.values() for variant in variants
}
ALL_SOURCES = tuple(source for group in GROUPS.values() for source in group)


class TenReviewError(ValueError):
    """A P1-06 evidence or per-group source invariant was not established."""


def validate_groups() -> None:
    if len(ALL_SOURCES) != len(set(ALL_SOURCES)) or set(ALL_SOURCES) != EXPECTED:
        raise TenReviewError("Ten official mappings plus exact required contrasts expected")
    if set(ALL_SOURCES) - set(IMAGE_URLS):
        raise TenReviewError("A required official photo ZIP index has not been pinned")
    if set(IMAGE_URLS) != set(PINNED_INDEX_SHA256) or set(IMAGE_URLS) != set(PINNED_SIZES):
        raise TenReviewError("Pinned photo ZIP URL/index/length registries differ")


def offline_plan() -> dict:
    validate_groups()
    return {
        "status": "P1_06_OFFLINE_PLAN_NO_MEDIA_OR_NETWORK",
        "groups": {k: list(v) for k, v in GROUPS.items()},
        "unique_source_folders": len(ALL_SOURCES),
        "gesture_output_classes": list(CLASS_NAMES),
        "source_split": "train_only",
        "unresolved_two_up_peace_mapping": True,
        "images_per_source_initial": 6,
        "licensed_jpeg_download_now": False,
        "requires_owner_research_license_signoff_before_prepare": True,
        "requires_human_photo_qa": True,
        "dataset_and_model_training_authorized": False,
    }


def _require_private_new_path(path: Path) -> Path:
    if path.is_symlink():
        raise TenReviewError("Symlinked workspace/draft is forbidden")
    resolved = path.resolve()
    repo = ROOT.resolve()
    if (resolved == repo or (resolved.is_relative_to(repo)
                             and not resolved.is_relative_to(repo / "data"))):
        raise TenReviewError("Inside a Public Git checkout, use only ignored data/")
    if resolved.exists():
        raise TenReviewError("Output exists; never overwrite signed/visual evidence")
    return resolved


def _require_not_ci() -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("CI") == "true":
        raise TenReviewError("Licensed person images/annotations cannot be acquired in CI")


def _receipt(workspace: Path, group: str) -> dict:
    if workspace.is_symlink():
        raise TenReviewError("Private workspace symlink refused")
    try:
        path = workspace / "campaign.private.json"
        if path.is_symlink():
            raise TenReviewError("Workspace receipt symlink refused")
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise TenReviewError("Missing/malformed P1-06 private group receipt") from exc
    if (not isinstance(body, dict) or body.get("schema_version") != SCHEMA
            or body.get("group") != group
            or body.get("sources") != list(GROUPS[group])
            or body.get("split") != "train"
            or body.get("status") != "GROUP_PACKET_CREATED_NOT_APPROVED"
            or body.get("model_classes") != 10
            or body.get("p1_pass") is not False):
        raise TenReviewError("Group receipt does not match versioned ten-class contract")
    manifest = workspace / "packet/packet.manifest.json"
    if manifest.is_symlink():
        raise TenReviewError("Private packet manifest symlink refused")
    try:
        actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
    except OSError as exc:
        raise TenReviewError("Group visual packet manifest missing") from exc
    if actual != body.get("packet_manifest_sha256"):
        raise TenReviewError("Packet has changed after workspace publication")
    return body


def prepare(group: str, output: Path, license_ack: Path, *,
            per_source: int = 6, annotation_budget_mib: int = 120,
            jpeg_budget_mib: int = 80, seed: str = "p1-06-gestures-v1") -> dict:
    validate_groups()
    if group not in GROUPS:
        raise TenReviewError("Unknown visual review group")
    _require_not_ci()
    try:
        _approval(license_ack, "license")
    except PreparationError as exc:
        raise TenReviewError("Human research-license acceptance is required first") from exc
    if type(per_source) is not int or not 3 <= per_source <= 20:
        raise TenReviewError("Require 3..20 reviewed training photos/source")
    if (type(annotation_budget_mib) is not int
            or not 1 <= annotation_budget_mib <= 256
            or type(jpeg_budget_mib) is not int or not 1 <= jpeg_budget_mib <= 512):
        raise TenReviewError("Invalid bounded per-group annotation or JPEG budget")
    if not isinstance(seed, str) or not 1 <= len(seed) <= 200:
        raise TenReviewError("Invalid deterministic sampler seed")
    target = _require_private_new_path(output)
    target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    group_sources = GROUPS[group]
    with tempfile.TemporaryDirectory(prefix=".p1-06-", dir=target.parent) as temp:
        stage = Path(temp) / "workspace"
        stage.mkdir(mode=0o700)
        anns, imgs, packet = (
            stage / "annotations", stage / "images", stage / "packet"
        )
        try:
            doc = fetch_train_annotations(
                group_sources, ("train",), anns,
                license_ack, annotation_budget_mib
            )
            expected = {f"annotations/train/{source}.json" for source in group_sources}
            if set(doc.get("files", {})) != expected:
                raise TenReviewError("Missing or different train JSON members")
            image_counts = {}
            for source in group_sources:
                image_receipt = fetch_image_subset(
                    source, anns, imgs, license_ack,
                    per_source=per_source, max_transfer_mib=jpeg_budget_mib,
                    seed=seed,
                )
                if (image_receipt.get("source") != source
                        or image_receipt.get("train_only") is not True
                        or image_receipt.get("jpeg_count") != per_source):
                    raise TenReviewError("Source train-only image acquisition incomplete")
                image_counts[source] = per_source
            packet_receipt = build_packet(
                anns, imgs, packet, license_ack,
                sources=group_sources, per_source=per_source, seed=seed
            )
            if (packet_receipt.get("sources") != list(group_sources)
                    or packet_receipt.get("train_only") is not True
                    or set(packet_receipt.get("counts_by_source", {})) != set(group_sources)
                    or min(packet_receipt["counts_by_source"].values()) < 3):
                raise TenReviewError("Private group contact sheet incomplete")
        except (PreparationError, ImageRangeError, VisualReviewError) as exc:
            raise TenReviewError(
                "Official train-only source acquisition/packet build failed; no public data"
            ) from exc

        manifest = (packet / "packet.manifest.json").read_bytes()
        if hashlib.sha256(manifest).hexdigest() != packet_receipt["manifest_sha256"]:
            raise TenReviewError("Packet hash changed during construction")
        receipt = {
            "schema_version": SCHEMA,
            "status": "GROUP_PACKET_CREATED_NOT_APPROVED",
            "group": group, "sources": list(group_sources),
            "split": "train", "model_classes": 10,
            "model_input_hw": [224, 224],
            "requested_images_per_source": per_source,
            "actual_images_per_source": packet_receipt["counts_by_source"],
            "packet_manifest_sha256": packet_receipt["manifest_sha256"],
            "official_index_sha256": {s: PINNED_INDEX_SHA256[s] for s in group_sources},
            "policy_sha256": policy_sha256(),
            "pdf_identity_sha256": OFFICIAL_LICENSE_SHA256,
            "license_record_verified_without_publication": True,
            "p1_pass": False,
        }
        (stage / "campaign.private.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if target.exists():
            raise TenReviewError("Another task occupied this private group output")
        stage.rename(target)
    return {
        "status": receipt["status"],
        "group": group,
        "sources": list(group_sources),
        "counts_by_source": receipt["actual_images_per_source"],
        "private_review_page": str(target / "packet/index.html"),
        "private_invalid_review_template": str(target / "packet/review.template.json"),
        "mapping_signoff_pending": True,
        "p1_pass": False,
    }


def verify(group: str, workspace: Path, review: Path) -> dict:
    validate_groups()
    if group not in GROUPS:
        raise TenReviewError("Unknown visual review group")
    receipt = _receipt(workspace, group)
    if receipt.get("policy_sha256") != policy_sha256():
        raise TenReviewError("Source policy changed after packet generation")
    if receipt.get("official_index_sha256") != {
        s: PINNED_INDEX_SHA256[s] for s in GROUPS[group]
    }:
        raise TenReviewError("Archived official image index identity changed")
    try:
        result = verify_review(workspace / "packet", review)
    except VisualReviewError as exc:
        raise TenReviewError("Human decisions do not match original private JPEG evidence") from exc
    if result["sources_checked"] != len(GROUPS[group]):
        raise TenReviewError("Incomplete group or wrong sample count")
    return {
        "status": result["status"],
        "group": group,
        "sources_checked": result["sources_checked"],
        "reviewed_images": result["reviewed_images"],
        "unresolved_sources": result["unresolved_sources"],
        "potential_command_alias_conflicts": result["potential_command_alias_conflicts"],
        "human_decisions_structurally_recorded": True,
        "semantic_and_data_rights_owner_acceptance_still_required": True,
        "p1_pass": False,
    }


def draft(evidence: list[tuple[str, Path, Path]], output: Path) -> dict:
    """Combine valid group reviews into an explicitly UNSIGNED/INVALID mapping."""
    validate_groups()
    if len(evidence) != len(GROUPS) or set(x[0] for x in evidence) != set(GROUPS):
        raise TenReviewError("A uniquely reviewed packet for EACH of five groups is required")
    target = _require_private_new_path(output)
    rows = {}
    details = {}
    unresolved, conflicts = set(), set()
    for group, workspace, review in evidence:
        report = verify(group, workspace, review)  # Integrity of all images again
        unresolved.update(report["unresolved_sources"])
        conflicts.update(report["potential_command_alias_conflicts"])
        try:
            form = json.loads(review.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise TenReviewError("Private human review JSON unreadable") from exc
        if not isinstance(form, dict) or not isinstance(form.get("decisions"), list):
            raise TenReviewError("Private decision list malformed")
        for decision in form["decisions"]:
            name = decision["source"]
            if name in rows or name not in GROUPS[group]:
                raise TenReviewError("Duplicate/unexpected campaign source")
            rows[name] = decision
        details[group] = {
            "packet_manifest_sha256": _receipt(workspace, group)["packet_manifest_sha256"],
            "review_file_sha256": hashlib.sha256(review.read_bytes()).hexdigest(),
            "source_count": len(GROUPS[group]),
            "reviewed_images": report["reviewed_images"],
            "issues_remaining": sorted(set(report["unresolved_sources"])
                                       | set(report["potential_command_alias_conflicts"])),
        }
    if set(rows) != EXPECTED:
        raise TenReviewError("Campaign reviewed sources not exactly ten+contrasts")
    candidate = []
    for source, cls_id in SOURCE_TO_INDEX.items():
        decision = rows[source]
        candidate.append({
            "source": source,
            "output": CLASS_NAMES[cls_id],
            "approved": False,  # Mandatory private final human action.
            "samples_reviewed": len(decision["reviewed_packet_ids"]),
            "official_reference_checked": False,  # Reaffirm in final owner signoff.
            "contrasts_reviewed": list(NEEDED_CONTRASTS.get(source, ())),
            "review_status": decision["decision"],
        })
    source_status = "REVIEW_CONFLICTS_REQUIRE_ADJUDICATION" if (
        unresolved or conflicts
    ) else "STRUCTURAL_EVIDENCE_COMPLETE_FINAL_HUMAN_SIGNOFF_PENDING"
    ack = {
        "note": ("INVALID UNSIGNED draft; this is NOT HaGRID permission, legal clearance "
                 "or actual semantic mapping acceptance. Do not change approved/checked "
                 "without reviewing the private evidence."),
        "reviewer": "", "reviewed_on": "",
        "classes": candidate,
    }
    private = {
        "schema_version": SCHEMA,
        "status": source_status,
        "source_policy_sha256": policy_sha256(),
        "review_groups": details,
        "unresolved_primary_sources": sorted(unresolved),
        "possible_lookalike_aliases": sorted(conflicts),
        "all_20_source_review_files_structurally_verified": True,
        "mapping_approved": False,
        "research_license_owner_acceptance_not_inferred": True,
        "p1_pass": False,
    }
    target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".p1-06-draft-", dir=target.parent) as tmp:
        stage = Path(tmp) / "campaign"
        stage.mkdir(mode=0o700)
        (stage / "hagrid-mapping-ack.UNSIGNED.json").write_text(
            json.dumps(ack, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (stage / "campaign.evidence.private.json").write_text(
            json.dumps(private, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if target.exists():
            raise TenReviewError("Concurrent attempt to overwrite private campaign")
        stage.rename(target)
    return {
        "status": source_status,
        "review_groups": len(GROUPS),
        "source_categories_checked": len(rows),
        "unresolved_sources": sorted(unresolved),
        "lookalike_alias_questions": sorted(conflicts),
        "unsigned_mapping_draft": str(target / "hagrid-mapping-ack.UNSIGNED.json"),
        "real_mapping_signoff_still_required": True,
        "p1_pass": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("plan")
    p = sub.add_parser("prepare")
    p.add_argument("--group", choices=GROUPS, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--license-ack", type=Path, required=True)
    p.add_argument("--per-source", type=int, default=6)
    p.add_argument("--annotation-budget-mib", type=int, default=120)
    p.add_argument("--jpg-budget-mib", type=int, default=80)
    p.add_argument("--seed", default="p1-06-gestures-v1")
    v = sub.add_parser("verify")
    v.add_argument("--group", choices=GROUPS, required=True)
    v.add_argument("--workspace", type=Path, required=True)
    v.add_argument("--review", type=Path, required=True)
    d = sub.add_parser("draft")
    d.add_argument("--evidence", nargs=3, action="append", required=True,
                   metavar=("GROUP", "WORKSPACE", "REVIEW"))
    d.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "plan":
            result = offline_plan()
        elif args.action == "prepare":
            result = prepare(
                args.group, args.output, args.license_ack,
                per_source=args.per_source,
                annotation_budget_mib=args.annotation_budget_mib,
                jpeg_budget_mib=args.jpg_budget_mib, seed=args.seed,
            )
        elif args.action == "verify":
            result = verify(args.group, args.workspace, args.review)
        else:
            result = draft(
                [(group, Path(w), Path(r)) for group, w, r in args.evidence],
                args.output,
            )
    except (TenReviewError, KeyError) as exc:
        parser.exit(2, f"FAIL CLOSED: {exc if isinstance(exc, TenReviewError) else 'Malformed private evidence'}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
