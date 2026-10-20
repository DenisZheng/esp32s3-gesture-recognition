#!/usr/bin/env python3
"""Validate 10-class/output model schema and locked upstream metadata (offline)."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = [
    "one", "two", "three", "four", "five", "like", "ok", "no_gesture", "call", "dislike"
]
SOURCES = [
    "one", "two_up", "three", "four", "palm", "like", "ok", "no_gesture", "call", "dislike"
]
UPSTREAM_IDS = {"esp-detection", "esp-dl", "esp-ppq", "hagrid", "esp-idf"}


def check() -> None:
    cfg = json.loads((ROOT / "configs/classes.json").read_text(encoding="utf-8"))
    assert cfg["num_classes"] == 10
    assert cfg["input_hw"] == [224, 224]
    assert [c["id"] for c in cfg["classes"]] == list(range(10))
    assert [c["name"] for c in cfg["classes"]] == NAMES
    assert [c["hagrid_source"] for c in cfg["classes"]] == SOURCES
    assert cfg["negative_background_label"] == "no_hand"
    assert cfg["negative_background_is_detection_class"] is False
    assert cfg["two_hand_joint_box_allowed"] is False
    assert all(c["review"] in {"pending", "pending_visual_confirmation", "approved"} for c in cfg["classes"]), (
        "Unknown class mapping review state"
    )

    text = (ROOT / "configs/hagrid10.yaml").read_text(encoding="utf-8")
    assert re.search(r"^nc:\s*10\s*$", text, re.M)
    found = re.findall(r"^\s{2}(\d+):\s+([a-z_]+)\s*$", text, re.M)
    assert [(int(i), name) for i, name in found] == list(enumerate(NAMES))
    assert "no_hand" not in [v for _, v in found]

    deps = json.loads((ROOT / "upstreams.lock.json").read_text(encoding="utf-8"))["dependencies"]
    assert {d["id"] for d in deps} == UPSTREAM_IDS
    assert len(deps) == len(UPSTREAM_IDS)
    for dep in deps:
        assert re.fullmatch(r"[0-9a-f]{40}", dep["commit"]), dep["id"]
        assert dep["url"].startswith("https://github.com/") and dep["url"].endswith(".git")
    assert next(d for d in deps if d["id"] == "esp-dl")["ref"] == "v3.3.13"
    assert next(d for d in deps if d["id"] == "esp-idf")["ref"] == "v5.4.4"
    exp = json.loads((ROOT / "experiments/templates/experiment.json").read_text(encoding="utf-8"))
    assert exp["num_classes"] == 10
    assert exp["metrics"]["map50"] is None, "Never invent experiment metrics"
    print("PASS: exact class indices, source mappings, negative semantics, source SHAs, experiment schema")


if __name__ == "__main__":
    check()
