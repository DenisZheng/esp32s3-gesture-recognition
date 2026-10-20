"""Pinned HaGRIDv2 34-source inventory and fail-closed hard-negative review gate.

The policy is a quarantine/eligibility partition, never an automatic conversion
approval. No subject IDs, sample images, labels or signed reviews live in Git.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

from scripts.data_contract import SOURCE_TO_INDEX

ROOT = Path(__file__).resolve().parents[1]
POLICY_FILE = ROOT / "configs" / "hagrid_v2_source_policy.json"

# The set was audited against HaGRIDv2's pinned constants.py. Deliberately
# explicit: upstream source additions/renames must require a policy PR.
OFFICIAL_SOURCES = frozenset((
    "grabbing", "grip", "holy", "point", "call", "three3", "timeout", "xsign",
    "hand_heart", "hand_heart2", "little_finger", "middle_finger",
    "take_picture", "dislike", "fist", "four", "like", "mute", "ok",
    "one", "palm", "peace", "peace_inverted", "rock", "stop",
    "stop_inverted", "three", "three2", "two_up", "two_up_inverted",
    "three_gun", "thumb_index", "thumb_index2", "no_gesture",
))
GROUPS = (
    "primary_sources",
    "reviewable_noncommand_candidates",
    "lookalike_quarantine",
    "compound_quarantine",
)


class PolicyError(ValueError):
    """Invalid 34-class inventory or unsigned/unapproved negative selection."""


def read_policy(path: Path = POLICY_FILE) -> dict:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise PolicyError("Cannot read HaGRIDv2 source taxonomy policy") from exc
    if not isinstance(doc, dict):
        raise PolicyError("Policy root must be a JSON object")
    if (
        doc.get("schema_version") != 1
        or doc.get("source_category_count") != 34
        or doc.get("model_class_count") != 10
        or doc.get("no_hand_is_model_class") is not False
        or doc.get("hard_negative_output") != "no_gesture"
        or doc.get("source_commit") != "cade4dd4ba162832ef2b62b92d89d4d7a2dc2b75"
    ):
        raise PolicyError("Source/model contract changed without a policy review")
    raw = doc.get("raw_sources")
    if (not isinstance(raw, list) or len(raw) != 34
            or not all(isinstance(s, str) for s in raw)
            or set(raw) != OFFICIAL_SOURCES):
        raise PolicyError("HaGRIDv2 34-source catalog differs from the pinned upstream")
    groups = []
    for group in GROUPS:
        names = doc.get(group)
        if (not isinstance(names, list) or not names
                or not all(isinstance(s, str) for s in names)
                or len(set(names)) != len(names)):
            raise PolicyError(f"Invalid class policy partition: {group}")
        groups.append(set(names))
    if (
        set.union(*groups) != OFFICIAL_SOURCES
        or sum(len(group) for group in groups) != len(OFFICIAL_SOURCES)
        or groups[0] != set(SOURCE_TO_INDEX)
        or len(groups[0]) != 10
        or len(groups[1]) != 7
        or len(groups[2]) != 10
        or len(groups[3]) != 7
    ):
        raise PolicyError("Class policy has missing, repeated or invalid sources")
    # These particular aliases MUST NOT be silently relabeled to class 7:
    if not {"peace", "peace_inverted", "two_up_inverted"}.issubset(groups[2]):
        raise PolicyError("Potential number-two aliases must stay quarantined")
    if not {"hand_heart", "hand_heart2", "xsign"}.issubset(groups[3]):
        raise PolicyError("Compound two-hand classes must stay quarantined")
    return doc


def policy_sha256(path: Path = POLICY_FILE) -> str:
    read_policy(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_requested_sources(sources: tuple[str, ...], *,
                               conversion: bool = False) -> frozenset[str]:
    doc = read_policy()
    if not sources or len(set(sources)) != len(sources):
        raise PolicyError("Select nonempty, unique HaGRIDv2 source categories")
    unknown = set(sources) - OFFICIAL_SOURCES
    if unknown:
        raise PolicyError("Unknown source outside HaGRIDv2 34-class catalog")
    if conversion:
        forbidden = set(sources) & (
            set(doc["lookalike_quarantine"]) | set(doc["compound_quarantine"])
        )
        if forbidden:
            raise PolicyError(
                "Quarantined lookalike/compound source may not be converted"
            )
    return frozenset(set(sources) & set(doc["reviewable_noncommand_candidates"]))


def approve_hard_negatives(path: Path, requested: frozenset[str]) -> frozenset[str]:
    """Read local-only, signed visual review. No implied permission from class name.

    A reviewed entry says this single-hand pose is a noncommand for THIS
    experiment; it does not grant use rights or override the separate 10-class
    visual signoff and dataset license.
    """
    if not requested:
        raise PolicyError("Hard-negative approval requested with no class selected")
    inventory = read_policy()
    permitted = set(inventory["reviewable_noncommand_candidates"])
    if not requested <= permitted:
        raise PolicyError("Unapproved negative candidate is quarantined")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise PolicyError("Missing or unreadable private hard-negative signoff") from exc
    if not isinstance(doc, dict):
        raise PolicyError("Private hard-negative signoff must be an object")
    reviewer = doc.get("reviewer")
    when = doc.get("reviewed_on")
    if not isinstance(reviewer, str) or not reviewer.strip() or not isinstance(when, str):
        raise PolicyError("Signed reviewer and date are required")
    try:
        date.fromisoformat(when)
    except ValueError as exc:
        raise PolicyError("Review date is not a valid calendar date") from exc
    if doc.get("policy_sha256") != policy_sha256():
        raise PolicyError("Policy was revised after review; signoff is stale")
    entries = doc.get("approved_sources")
    if not isinstance(entries, list) or not entries:
        raise PolicyError("No approved hard-negative entries supplied")
    names = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise PolicyError("Malformed hard-negative approval row")
        source = entry.get("source")
        n = entry.get("samples_reviewed")
        if (
            not isinstance(source, str)
            or source not in permitted
            or entry.get("output") != "no_gesture"
            or entry.get("approved") is not True
            or entry.get("per_hand_boxes_reviewed") is not True
            or entry.get("licensed_images_reviewed") is not True
            or type(n) is not int or n < 1
        ):
            raise PolicyError("Hard-negative mapping not independently reviewed")
        names.append(source)
    if len(set(names)) != len(names) or set(names) != requested:
        raise PolicyError("Approved sources do not exactly match requested class set")
    return requested
