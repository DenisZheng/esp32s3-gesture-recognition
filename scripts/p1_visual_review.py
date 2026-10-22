#!/usr/bin/env python3
"""Create a PRIVATE local HTML gesture-mapping review packet from licensed data.

This tool intentionally does NOT fetch HaGRID files, use cloud APIs, use test
images, train a model, or grant dataset rights. Only the train split may be
reviewed here, so final held-out evaluation remains untouched.

CLI: python -m scripts.p1_visual_review packet --help
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import html
import json
import re
import shutil
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path

from scripts.data_contract import CLASS_NAMES, SOURCE_TO_INDEX, AnnotationError, xywh_to_yolo
from scripts.hagrid_p1 import PreparationError, ROOT, _approval
from scripts.hagrid_policy import OFFICIAL_SOURCES, read_policy, validate_requested_sources

SAFE_STEM = re.compile(r"^[A-Za-z0-9_-]+$")
# Owner visual review 2026-10-22: `peace` leads the group because it is now
# the reviewed class-1 `two` source; the other shapes are audit contrasts.
DEFAULT_SOURCES = ("peace", "two_up", "peace_inverted", "two_up_inverted")
NEEDED_CONTRASTS = {
    # Owner visual review 2026-10-22 selected `peace` as class-1 `two`; the
    # other number-two shapes remain mandatory contrast references.
    "peace": ("two_up", "two_up_inverted", "peace_inverted"),
    "palm": ("stop", "stop_inverted"),
    "three": ("three2", "three3", "three_gun"),
    "one": ("point", "mute"),
}
MAX_IMAGES_PER_SOURCE = 20
MAX_SINGLE_JPEG = 12 * 1024 * 1024
PROTOCOL_VERSION = 1


class VisualReviewError(ValueError):
    """No unsafe or misleading review output was produced."""


def _load_annotation(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise VisualReviewError("Expected local train/source.json is missing or symlinked")
    if path.stat().st_size > 125 * 1024 * 1024:
        raise VisualReviewError("Source JSON too large for the bounded visual sampling workflow")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise VisualReviewError("Cannot decode licensed annotation JSON") from exc
    if not isinstance(data, dict):
        raise VisualReviewError("Annotation root must be a JSON object")
    return data


def _clean_record(stem: str, record: object):
    """Return in-memory selection details; no subject IDs are exported."""
    if not isinstance(stem, str) or not SAFE_STEM.fullmatch(stem):
        return None
    if not isinstance(record, dict):
        return None
    subject = record.get("user_id")
    boxes, labels = record.get("bboxes"), record.get("labels")
    if not isinstance(subject, str) or not subject.strip():
        return None
    if not isinstance(boxes, list) or not isinstance(labels, list):
        return None
    if not boxes or len(boxes) != len(labels) or len(boxes) > 8:
        return None
    cleaned = []
    for box, label in zip(boxes, labels):
        if not isinstance(box, (tuple, list)) or not isinstance(label, str):
            return None
        if label not in OFFICIAL_SOURCES:
            return None
        try:
            xywh_to_yolo(box)
        except (AnnotationError, ValueError, TypeError):
            return None
        cleaned.append({"bbox": [float(v) for v in box], "label": label})
    return {
        "stem": stem,
        "subject": subject,
        "hands": cleaned,
        "compound": bool(record.get("united_bbox") or record.get("united_label")),
        "multi_hand": len(boxes) > 1,
    }


def _read_local_jpeg(images: Path, source: str, stem: str):
    """No directory traversal/symlinks; JPG only, copied into ignored private packet."""
    p = images / source / (stem + ".jpg")
    if not p.is_file() or p.is_symlink():
        return None
    if not p.resolve().is_relative_to(images.resolve()):
        return None
    try:
        if not 4 <= p.stat().st_size <= MAX_SINGLE_JPEG:
            return None
        with p.open("rb") as fh:
            if fh.read(3) != b"\xff\xd8\xff":
                return None
    except OSError:
        return None
    return p


def _candidate_rank(seed: str, source: str, stem: str) -> str:
    return hashlib.sha256((seed + "\0" + source + "\0" + stem).encode("utf-8")).hexdigest()


def _safe_output(output: Path) -> Path:
    dest = output.resolve()
    root = ROOT.resolve()
    if dest == root or (dest.is_relative_to(root) and
                        not dest.is_relative_to(root / "data")):
        raise VisualReviewError("Within this Git repo review packets must be under ignored data/")
    if dest.exists():
        raise VisualReviewError("Packet output already exists; never overwrite reviewer evidence")
    return dest


def _preview_html(manifest: dict) -> str:
    """Pure local static page; src paths are generated names, never untrusted paths."""
    chunks = ["""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PRIVATE P1 gesture mapping review</title>
<style>
body{font:15px/1.5 system-ui,-apple-system,sans-serif;background:#f5f7fb;color:#1c2835;margin:0}
header{padding:24px clamp(16px,4vw,56px);background:#182d43;color:white}
header p{max-width:850px;color:#d8e4f1}
main{max-width:1500px;margin:24px auto;padding:0 20px}
h2{margin:36px 0 8px;border-bottom:2px solid #dbe4eb;padding-bottom:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:16px}
.card{background:white;border:1px solid #dce3ed;border-radius:12px;padding:10px;overflow:hidden}
.image{position:relative;width:100%;background:#e0e6ee}
.image img{display:block;width:100%;height:auto}
.box{position:absolute;border:3px solid #39bf89;box-sizing:border-box;pointer-events:none}
.box i{position:absolute;left:-2px;top:-25px;background:#146949;color:white;padding:2px 6px;
font:12px system-ui;white-space:nowrap;font-style:normal}
.meta{font-size:13px;color:#58677b;overflow-wrap:anywhere}
.warn{background:#fff0cb;color:#734d0d;padding:7px;border-radius:6px}
footer{padding:30px 20px;color:#64748b}
a{color:#8ccafa}
</style></head><body>
<header><h1>P1 / 手势映射人工验收（仅本地）</h1>
<p>仅使用 train 划分与已授权样本。绿色框表示原始标注的每只手；不代表标签正确。
任何差异或漏标都应人工记录。勿将此页面、JPEG、manifest 或评论截图提交到 Public GitHub。</p>
<p>官方对照：
<a href="https://github.com/espressif/esp-dl/blob/v3.3.14/examples/hand_gesture_recognition/gestures.png">ESP-DL v3.3.14</a>
&nbsp;|&nbsp;
<a href="https://github.com/hukenovs/hagrid/blob/Hagrid_v2-1M/images/gestures.png">HaGRIDv2</a></p></header><main>
"""]
    by_source = {}
    for sample in manifest["samples"]:
        by_source.setdefault(sample["source"], []).append(sample)
    for source in manifest["requested_sources"]:
        samples = by_source.get(source, [])
        official = CLASS_NAMES[SOURCE_TO_INDEX[source]] if source in SOURCE_TO_INDEX else "review-only contrast"
        chunks.append("<h2>" + html.escape(source) + " → " + html.escape(official) +
                      f"（{len(samples)} 张）</h2><div class=\"grid\">")
        for row in samples:
            filename = html.escape(row["image"], quote=True)
            boxes = []
            for hand in row["hands"]:
                x, y, w, h = hand["bbox"]
                label = html.escape(hand["label"])
                boxes.append(
                    f'<div class="box" style="left:{x*100:.5f}%;top:{y*100:.5f}%;'
                    f'width:{w*100:.5f}%;height:{h*100:.5f}%"><i>{label}</i></div>'
                )
            flags = []
            if row["compound"]:
                flags.append("包含联合框；不可当作单手标签")
            if row["multi_hand"]:
                flags.append("多只手：逐框审核")
            if any(h["label"] != source for h in row["hands"]):
                flags.append("包含其它类别的手：核查完整标注")
            warning = ("<p class=\"warn\">" + html.escape("；".join(flags)) + "</p>") if flags else ""
            chunks.append(
                '<article class="card"><div class="image">'
                f'<img loading="lazy" src="{filename}" alt="local private review image"/>'
                + "".join(boxes) + '</div><p class="meta">样本 '
                + html.escape(row["packet_id"]) + " · 逐手原始标注（未批准）</p>" + warning + "</article>"
            )
        chunks.append("</div>")
    chunks.append("""</main><footer>此文件是离线审核工作材料。程序只检查来源/框格式，
不会自动确定姿势语义、肖像许可或模型准确率。</footer></body></html>""")
    return "\n".join(chunks)


def build_packet(annotations: Path, images: Path, output: Path,
                 license_ack: Path, sources: tuple[str, ...] = DEFAULT_SOURCES,
                 per_source: int = 6, seed: str = "p1-visual-review-v1") -> dict:
    """License-gated, atomic, train-only packet. No network or test data read."""
    try:
        _approval(license_ack, "license")
        validate_requested_sources(sources)  # Audit may review quarantined aliases.
        read_policy()
    except (PreparationError, ValueError) as exc:
        raise VisualReviewError("Dataset permission or source policy not satisfied") from exc
    if not 1 <= per_source <= MAX_IMAGES_PER_SOURCE:
        raise VisualReviewError("per-source must be between 1 and 20")
    if not seed or len(seed) > 200:
        raise VisualReviewError("Invalid deterministic selection seed")
    target = _safe_output(output)
    if not images.is_dir() or images.is_symlink():
        raise VisualReviewError("Local JPEG root missing or symlinked")
    root = annotations / "train"
    if not root.is_dir() or root.is_symlink():
        raise VisualReviewError("Only the train split is eligible for mapping review")
    meta = {
        "schema_version": PROTOCOL_VERSION,
        "purpose": "P1 TRAIN-SPLIT VISUAL MAPPING QA ONLY; NOT A TRAINING DATASET OR ACCEPTANCE",
        "source_split": "train",
        "requested_sources": list(sources),
        "per_source_limit": per_source,
        "sampling_seed_sha256": hashlib.sha256(seed.encode("utf-8")).hexdigest(),
        "annotation_file_sha256": {},
        "statistics": {},
        "samples": [],
        "contains_private_jpeg_copies": True,
        "data_rights_owner_signoff_not_auto_inferred": True,
    }
    selected_files = []
    used_hashes = set()
    for source in sources:
        jsonpath = root / f"{source}.json"
        records = _load_annotation(jsonpath)
        meta["annotation_file_sha256"][f"train/{source}.json"] = hashlib.sha256(
            jsonpath.read_bytes()
        ).hexdigest()
        counters = Counter()
        candidates = []
        image_dir = images / source
        if not image_dir.is_dir() or image_dir.is_symlink():
            raise VisualReviewError(f"Missing or symlinked local JPG source folder: {source}")
        # Users typically stage just 3-20 JPGs per source from a 40+GB
        # archive. Sample from those *available* JPEG stems first, not from
        # tens of thousands of annotation rows with no local media!
        available_stems = {
            file.stem for file in image_dir.glob("*.jpg")
            if file.is_file() and not file.is_symlink() and SAFE_STEM.fullmatch(file.stem)
        }
        for stem, record in records.items():
            counters["annotation_records"] += 1
            if stem not in available_stems:
                counters["annotation_without_local_jpeg"] += 1
                continue
            candidate = _clean_record(stem, record)
            if candidate is None:
                counters["invalid_per_hand_records"] += 1
                continue
            counters["well_formed_per_hand_records"] += 1
            key = _candidate_rank(seed, source, stem)
            candidates.append((key, stem, candidate))
        picked = 0
        subjects = set()
        for _score, _stem, candidate in heapq.nsmallest(
            min(len(candidates), max(40, per_source * 50)), candidates
        ):
            if picked >= per_source:
                break
            subject = candidate["subject"]
            if subject in subjects:
                counters["repeat_subject_skipped"] += 1
                continue
            picture = _read_local_jpeg(images, source, candidate["stem"])
            if picture is None:
                counters["missing_or_invalid_local_jpeg"] += 1
                continue
            binary = picture.read_bytes()
            digest = hashlib.sha256(binary).hexdigest()
            if digest in used_hashes:
                counters["duplicate_image_skipped"] += 1
                continue
            used_hashes.add(digest)
            subjects.add(subject)
            picked += 1
            name = f"{source}-{picked:03d}.jpg"
            sample = {
                "packet_id": f"{source}:{picked:03d}",
                "source": source,
                "image": f"images/{name}",
                "image_sha256": digest,
                "hands": candidate["hands"],
                "multi_hand": candidate["multi_hand"],
                "compound": candidate["compound"],
            }
            meta["samples"].append(sample)
            selected_files.append((picture, name))
            counters["selected"] += 1
            if candidate["multi_hand"]:
                counters["multi_hand_selected"] += 1
            if candidate["compound"]:
                counters["compound_selected_for_review_only"] += 1
        if not picked:
            raise VisualReviewError(
                f"No valid local train JPGs available for review source {source}"
            )
        meta["statistics"][source] = dict(sorted(counters.items()))
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix=".gesture-review-", dir=target.parent) as staging:
        stage = Path(staging) / "packet"
        stage.mkdir(mode=0o700)
        (stage / "images").mkdir(mode=0o700)
        for original, name in selected_files:
            shutil.copyfile(original, stage / "images" / name)
        content = json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        (stage / "packet.manifest.json").write_text(content, encoding="utf-8")
        (stage / "index.html").write_text(_preview_html(meta), encoding="utf-8")
        decisions = [
            {
                "source": source,
                "proposed_output": (
                    CLASS_NAMES[SOURCE_TO_INDEX[source]] if source in SOURCE_TO_INDEX else None
                ),
                "decision": "pending",
                "reviewed_packet_ids": [],
                "official_reference_checked": False,
                "comparative_notes": "",
            }
            for source in sources
        ]
        form = {
            "note": "PRIVATE INVALID UNSIGNED REVIEW TEMPLATE; not itself a mapping signoff.",
            "reviewer": "",
            "reviewed_on": "",
            "packet_manifest_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "decisions": decisions,
        }
        (stage / "review.template.json").write_text(
            json.dumps(form, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        stage.rename(target)
    return {
        "status": "PACKET_GENERATED_NOT_APPROVED",
        "sources": list(sources),
        "counts_by_source": {
            name: meta["statistics"][name]["selected"] for name in sources
        },
        "train_only": True,
        "private_files_written": len(selected_files),
        "output_path": str(target),
        "manifest_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def verify_review(packet: Path, review: Path) -> dict:
    """Integrity + signed completeness checks, never automatic gesture adjudication."""
    base = packet.resolve()
    try:
        bytes_manifest = (base / "packet.manifest.json").read_bytes()
        manifest = json.loads(bytes_manifest.decode("utf-8"))
        form = json.loads(review.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise VisualReviewError("Missing/malformed local packet or review JSON") from exc
    if (
        not isinstance(manifest, dict) or manifest.get("schema_version") != PROTOCOL_VERSION
        or manifest.get("source_split") != "train"
        or not isinstance(form, dict)
        or form.get("packet_manifest_sha256") != hashlib.sha256(bytes_manifest).hexdigest()
    ):
        raise VisualReviewError("Review references a different or invalid packet manifest")
    reviewer = form.get("reviewer")
    review_date = form.get("reviewed_on")
    if not isinstance(reviewer, str) or not reviewer.strip() or not isinstance(review_date, str):
        raise VisualReviewError("Human reviewer and date are missing")
    try:
        date.fromisoformat(review_date)
    except ValueError as exc:
        raise VisualReviewError("Signed review date must be valid") from exc
    if not isinstance(manifest.get("samples"), list) or not isinstance(
        manifest.get("requested_sources"), list
    ):
        raise VisualReviewError("Malformed packet samples or source list")
    by_source = {source: {} for source in manifest["requested_sources"]}
    for record in manifest["samples"]:
        if not isinstance(record, dict):
            raise VisualReviewError("Malformed image evidence record")
        source, sample_id = record.get("source"), record.get("packet_id")
        image_name = record.get("image")
        if source not in by_source or not isinstance(sample_id, str) or (
            sample_id in by_source[source]
        ):
            raise VisualReviewError("Duplicate or unknown packet sample")
        if not isinstance(image_name, str) or not re.fullmatch(
            rf"images/{re.escape(source)}-[0-9]{{3}}\.jpg", image_name
        ):
            raise VisualReviewError("Unsafe sample image path in review packet")
        image = base / image_name
        if not image.is_file() or image.is_symlink() or not image.resolve().is_relative_to(base):
            raise VisualReviewError("Missing or symlinked private JPEG evidence")
        if hashlib.sha256(image.read_bytes()).hexdigest() != record.get("image_sha256"):
            raise VisualReviewError("Image bytes changed after packet generation")
        by_source[source][sample_id] = record
    decisions = form.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != len(by_source):
        raise VisualReviewError("Review must include exactly the packet's source categories")
    seen = set()
    unresolved = []
    alias_conflicts = []
    for decision in decisions:
        if not isinstance(decision, dict):
            raise VisualReviewError("Malformed review decision row")
        source = decision.get("source")
        choice = decision.get("decision")
        if source not in by_source or source in seen:
            raise VisualReviewError("Duplicate or unexpected review source")
        seen.add(source)
        primary = source in SOURCE_TO_INDEX
        expected_output = CLASS_NAMES[SOURCE_TO_INDEX[source]] if primary else None
        if decision.get("proposed_output") != expected_output:
            raise VisualReviewError("Reviewer attempted to silently change model class mapping")
        if choice not in ("match", "mismatch", "uncertain"):
            raise VisualReviewError("An explicit match/mismatch/uncertain decision is required")
        packet_ids = decision.get("reviewed_packet_ids")
        if (
            not isinstance(packet_ids, list) or len(packet_ids) == 0
            or not all(isinstance(sample_id, str) for sample_id in packet_ids)
            or len(packet_ids) != len(set(packet_ids))
            or not set(packet_ids) <= set(by_source[source])
        ):
            raise VisualReviewError("Review contains missing, duplicate or invented image IDs")
        # A positive mapping decision needs >=3 independent train examples.
        # Fewer photos can support exploratory/uncertain review, never PASS.
        if primary and choice == "match" and len(packet_ids) < 3:
            raise VisualReviewError("At least three primary-source photos required for mapping match")
        if decision.get("official_reference_checked") is not True:
            raise VisualReviewError("Official visual reference must be checked")
        note = decision.get("comparative_notes")
        if not isinstance(note, str) or len(note.strip()) < 12:
            raise VisualReviewError("A specific written visual rationale is required")
        # For primary rows, match = matches the proposed official output.
        # For lookalike rows, match = looks like the nearby official command;
        # this is a possible *conflict*, never evidence for a safe negative.
        if choice != "match" and primary:
            unresolved.append(source)
        if not primary and choice in ("match", "uncertain"):
            alias_conflicts.append(source)
        if primary and choice == "match" and source in NEEDED_CONTRASTS:
            for contrast in NEEDED_CONTRASTS[source]:
                if contrast not in by_source:
                    raise VisualReviewError("Required gesture contrast missing from review packet")
                if contrast not in seen and not any(
                    d.get("source") == contrast and d.get("decision") in ("match", "mismatch", "uncertain")
                    for d in decisions if isinstance(d, dict)
                ):
                    raise VisualReviewError("Required gesture variant comparison unresolved")
    if seen != set(by_source):
        raise VisualReviewError("Not all source categories have review decisions")
    return {
        "status": (
            "HUMAN_REVIEW_RECORDED_WITH_UNRESOLVED_MAPPINGS"
            if unresolved or alias_conflicts else "HUMAN_REVIEW_RECORDED_NOT_P1_PASS"
        ),
        "sources_checked": len(seen),
        "reviewed_images": sum(len(d["reviewed_packet_ids"]) for d in decisions),
        "unresolved_sources": sorted(unresolved),
        "potential_command_alias_conflicts": sorted(alias_conflicts),
        "still_requires_ten_class_final_signoff": True,
        "model_outputs_unchanged": 10,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    make = sub.add_parser("packet", help="Build private HTML review packet (licensed train images only)")
    make.add_argument("--annotations", type=Path, required=True)
    make.add_argument("--images", type=Path, required=True)
    make.add_argument("--license-ack", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    make.add_argument("--per-source", type=int, default=6)
    make.add_argument("--seed", default="p1-visual-review-v1")
    check = sub.add_parser("verify", help="Check private human signoff against packet image hashes")
    check.add_argument("--packet", type=Path, required=True)
    check.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "packet":
            result = build_packet(
                args.annotations, args.images, args.output, args.license_ack,
                tuple(args.sources), args.per_source, args.seed,
            )
        else:
            result = verify_review(args.packet, args.review)
    except (VisualReviewError, PreparationError) as exc:
        parser.exit(2, f"FAIL CLOSED: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
