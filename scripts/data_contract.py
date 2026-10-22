"""Pure standard-library data contract functions for a future P1 converter.

No HaGRID image, subject, or annotation data is loaded by P0 or CI.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

CLASS_NAMES = (
    "one", "two", "three", "four", "five", "like", "ok", "no_gesture", "call", "dislike"
)
SOURCE_TO_INDEX = {
    "one": 0,
    # ESP-DL v3.3.14 reference "two" is a separated V sign. Owner visual
    # review on 2026-10-22 (P1-06 group "two", 24 licensed train samples across
    # two_up / peace / peace_inverted / two_up_inverted) selected HaGRID
    # `peace`; the previous tentative `two_up` (index+collinear fingers) does
    # not match. Changing this requires the P1-06 re-review and a policy PR.
    "peace": 1,
    "three": 2,
    "four": 3,
    "palm": 4,
    "like": 5,
    "ok": 6,
    "no_gesture": 7,
    "call": 8,
    "dislike": 9,
}


class AnnotationError(ValueError):
    """Source annotation violates the strict per-hand label contract."""


def xywh_to_yolo(xywh: Iterable[float]) -> tuple[float, float, float, float]:
    """Convert a *normalized*, per-physical-hand top-left xywh box to YOLO cxcywh.

    HaGRID sample boxes are normalized. Do not pass pixel coordinates or a
    `united_bbox` grouping multiple hands here.
    """
    box = tuple(xywh)
    if len(box) != 4 or any(type(v) not in (int, float) for v in box):
        raise AnnotationError("Box must contain four numeric normalized coordinates")
    x, y, w, h = map(float, box)
    if not all(map(math.isfinite, (x, y, w, h))):
        raise AnnotationError("Box must be finite")
    epsilon = 1e-8
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 1 + epsilon or y + h > 1 + epsilon:
        raise AnnotationError("Box must have positive area and lie within [0,1]")
    # Clamp only tiny floating-point overflows, never a materially invalid box.
    return (min(1.0, x + w / 2), min(1.0, y + h / 2), w, h)


def strict_hand_labels(
    boxes: list[list[float]],
    labels: list[str],
    *,
    verified_no_hand: bool = False,
    non_command_sources: frozenset[str] = frozenset(),
) -> list[str]:
    """Return YOLO class/bbox rows or FAIL CLOSED.

    Unknown source classes are NOT automatically background. An explicitly
    reviewed `non_command_sources` list can map otherwise-known noncommands
    to ID 7; source data provenance and manual review remain required at P1.
    """
    if len(boxes) != len(labels):
        raise AnnotationError("Every physical hand must have its own box and label")
    if not boxes:
        if not verified_no_hand:
            raise AnnotationError("Empty annotation is not proof of no_hand")
        return []
    if verified_no_hand:
        raise AnnotationError("An image with a labeled hand cannot be no_hand")

    rows = []
    for box, label in zip(boxes, labels):
        if label in SOURCE_TO_INDEX:
            class_id = SOURCE_TO_INDEX[label]
        elif label in non_command_sources and label != "no_hand":
            class_id = 7
        else:
            raise AnnotationError(f"Unreviewed or unknown hand label: {label!r}")
        c_x, c_y, width, height = xywh_to_yolo(box)
        rows.append(f"{class_id} {c_x:.9f} {c_y:.9f} {width:.9f} {height:.9f}")
    return rows


def assert_subject_disjoint(split_users: dict[str, set[str]]) -> None:
    """Fail on user overlap between all train/val/test splits; no IDs printed."""
    expected = {"train", "val", "test"}
    if set(split_users) != expected:
        raise AnnotationError("All three split groups must be provided")
    if not all(isinstance(v, set) for v in split_users.values()):
        raise AnnotationError("Split user sets must be sets")
    if any(not user for users in split_users.values() for user in users):
        raise AnnotationError("Empty user ID is invalid")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if split_users[a] & split_users[b]:
            raise AnnotationError(f"user_id leakage between {a} and {b}")
