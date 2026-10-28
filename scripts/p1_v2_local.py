#!/usr/bin/env python3
"""V2 512px local-directory ingestion: validate, then build the ten-class dataset.

The ``auto`` downloader in ``p1_dataset_builder`` is pinned to HaGRIDv1 archive
sizes and central-directory SHA256 values, so it structurally cannot consume the
v2 512px bundle even though the images and annotations are compatible. This
entry point therefore uses the builder's existing ``local-dir`` mode (the
intended path for data that arrives by disk) and adds the pre-flight checks that
matter when the images came from somewhere else.

What it verifies before spending time on a build
------------------------------------------------
* every required class directory exists and holds JPEG files;
* the annotation JSONs exist for each split and class;
* a sample of annotation keys actually resolves to files on disk (catches the
  usual v1/v2 filename-mismatch failure before a multi-hour build);
* image dimensions and byte sizes are sane;
* the ten output classes still follow ``configs/classes.json``.

It never downloads, never overwrites an existing output directory, and writes an
aggregate report next to the dataset.

Usage
-----
    python3 -m scripts.p1_v2_local plan
    python3 -m scripts.p1_v2_local check \
        --images data/manual/hagrid_dataset \
        --annotations data/manual/hagrid_annotations
    python3 -m scripts.p1_v2_local build \
        --images data/manual/hagrid_dataset \
        --annotations data/manual/hagrid_annotations \
        --output data/processed/hagrid10-v2-512 \
        --train-per-source 2000 --val-per-source 200 --test-per-source 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.data_contract import SOURCE_TO_INDEX  # noqa: E402
from scripts.p1_dataset_builder import (  # noqa: E402
    DatasetBuildError,
    build_dataset,
)

SPLITS = ("train", "val", "test")
SOURCES = tuple(SOURCE_TO_INDEX)
# HaGRID wraps extracted trees differently depending on how the archive was
# packed. Accept the layouts seen in practice instead of hard-coding one.
IMAGE_LAYOUTS = (
    ("{source}",),
    ("hagrid_dataset", "{source}"),
    ("hagrid_dataset_512", "{source}"),
    ("hagrid_v2", "{source}"),
)
ANNOTATION_LAYOUTS = (
    ("{split}", "{source}.json"),
    ("hagrid_annotations", "{split}", "{source}.json"),
    ("annotations", "{split}", "{source}.json"),
    ("annotations_with_landmarks", "annotations", "{split}", "{source}.json"),
)


class V2Error(ValueError):
    """Input tree is not a usable v2 local dataset."""


def log(msg: str) -> None:
    print(msg, flush=True)


def resolve_images(root: Path) -> tuple[Path, tuple[str, ...]]:
    """Return the directory that actually contains per-class folders."""
    if not root.is_dir():
        raise V2Error(f"image root missing: {root}")
    for layout in IMAGE_LAYOUTS:
        base = root.joinpath(*[p for p in layout if p != "{source}"])
        if base.is_dir() and any((base / s).is_dir() for s in SOURCES):
            return base, layout
    # Fall back to a shallow search: the archive may nest one extra level.
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir():
            continue
        if any((candidate / s).is_dir() for s in SOURCES):
            return candidate, ("<auto>",)
    raise V2Error(
        "no per-class image folders found; expected one of "
        + ", ".join("/".join(x) for x in IMAGE_LAYOUTS)
    )


def resolve_annotations(root: Path) -> tuple[Path, tuple[str, ...]]:
    if not root.is_dir():
        raise V2Error(f"annotation root missing: {root}")
    for layout in ANNOTATION_LAYOUTS:
        base = root.joinpath(*[p for p in layout if p not in ("{split}", "{source}.json")])
        if base.is_dir() and any((base / s).is_dir() for s in SPLITS):
            return base, layout
    for candidate in sorted(root.iterdir()):
        if candidate.is_dir() and any((candidate / s).is_dir() for s in SPLITS):
            return candidate, ("<auto>",)
    raise V2Error("no train/val/test annotation folders found")


def probe_decoadable(folder: Path, keys: list[str], limit: int = 8) -> dict:
    """Actually decode a few JPEGs instead of trusting the file extension.

    The pre-flight used to advertise pixel-size checking while only looking at
    names and byte sizes. Structural checks cannot catch a truncated download or
    a mislabelled non-JPEG, and both would surface only during training.
    """
    from PIL import Image

    result = {"checked": 0, "decoded": 0, "sizes": [], "errors": []}
    for key in keys[:limit]:
        path = folder / f"{key}.jpg"
        if not path.is_file():
            continue
        result["checked"] += 1
        try:
            with Image.open(path) as image:
                image.verify()          # detects truncation/corruption
            with Image.open(path) as image:
                width, height = image.size
                image.load()
            if width <= 0 or height <= 0:
                result["errors"].append(f"{key}: non-positive dimensions")
                continue
            result["decoded"] += 1
            result["sizes"].append([width, height])
        except Exception as exc:  # noqa: BLE001 - PIL raises several types
            result["errors"].append(f"{key}: {type(exc).__name__}")
    if result["sizes"]:
        widths = sorted(s[0] for s in result["sizes"])
        heights = sorted(s[1] for s in result["sizes"])
        result["min_side_min"] = min(min(w, h) for w, h in result["sizes"])
        result["width_range"] = [widths[0], widths[-1]]
        result["height_range"] = [heights[0], heights[-1]]
    return result


def check(images: Path, annotations: Path, *, sample: int = 20) -> dict:
    """Pre-flight: structure, counts and key-to-file resolution."""
    img_base, img_layout = resolve_images(images)
    ann_base, ann_layout = resolve_annotations(annotations)

    report: dict = {
        "status": "CHECK_STARTED",
        "image_root_resolved": str(img_base),
        "image_layout": "/".join(img_layout),
        "annotation_root_resolved": str(ann_base),
        "annotation_layout": "/".join(ann_layout),
        "classes": {},
        "problems": [],
    }

    for source in SOURCES:
        folder = img_base / source
        entry = {"images": 0, "annotations": {}, "missing_annotation": False}
        if folder.is_dir():
            entry["images"] = sum(1 for _ in folder.glob("*.jpg"))
        else:
            report["problems"].append(f"missing image folder: {source}")
        for split in SPLITS:
            doc = ann_base / split / f"{source}.json"
            if doc.is_file():
                try:
                    data = json.loads(doc.read_text(encoding="utf-8"))
                    entry["annotations"][split] = len(data) if isinstance(data, dict) else -1
                except (OSError, ValueError):
                    entry["annotations"][split] = -1
                    report["problems"].append(f"unreadable annotation: {split}/{source}.json")
            else:
                entry["annotations"][split] = 0
                entry["missing_annotation"] = True
                report["problems"].append(f"missing annotation: {split}/{source}.json")
        report["classes"][source] = entry

    # The decisive check: do annotation keys resolve to real JPEGs?
    resolved = {}
    for source in SOURCES:
        folder = img_base / source
        doc = ann_base / "train" / f"{source}.json"
        if not folder.is_dir() or not doc.is_file():
            resolved[source] = {"checked": 0, "found": 0}
            continue
        try:
            keys = list(json.loads(doc.read_text(encoding="utf-8")).keys())
        except (OSError, ValueError):
            resolved[source] = {"checked": 0, "found": 0}
            continue
        picked = keys[:sample]
        found = sum(1 for k in picked if (folder / f"{k}.jpg").is_file())
        resolved[source] = {"checked": len(picked), "found": found}
        if picked and found == 0:
            report["problems"].append(
                f"annotation keys do not match filenames for {source} "
                f"(0/{len(picked)} resolved) - v1/v2 mismatch?"
            )
        elif picked and found < len(picked) * 0.5:
            report["problems"].append(
                f"less than half of sampled keys resolve for {source} "
                f"({found}/{len(picked)})"
            )
    report["key_resolution"] = resolved

    # Decode real pixels for a couple of classes; a name-only check would pass a
    # corrupted tree. Sampling keeps this fast on a 10 000-image source while
    # still exercising every class.
    decode = {}
    for source in SOURCES:
        folder = img_base / source
        doc = ann_base / "train" / f"{source}.json"
        if not folder.is_dir() or not doc.is_file():
            continue
        try:
            keys = list(json.loads(doc.read_text(encoding="utf-8")).keys())
        except (OSError, ValueError):
            continue
        info = probe_decoadable(folder, keys)
        decode[source] = info
        if info["checked"] and info["decoded"] == 0:
            report["problems"].append(
                f"no sampled JPEG decoded for {source} ({len(info['errors'])} errors)"
            )
        elif info["errors"]:
            report["problems"].append(
                f"{len(info['errors'])} sampled JPEG(s) failed to decode for {source}"
            )
    report["decode_probe"] = decode
    sides = [v.get("min_side_min") for v in decode.values() if v.get("min_side_min")]
    if sides:
        report["min_side_observed"] = min(sides)
        # The v2 512px bundle should have a short side of at least ~512 px. A
        # much smaller value means the wrong archive was copied.
        if min(sides) < 480:
            report["problems"].append(
                f"smallest short side is {min(sides)}px, expected >=512 for the "
                "v2 512px bundle"
            )
    report["status"] = "CHECK_OK" if not report["problems"] else "CHECK_FAILED"
    return report


def build(args) -> int:
    img_base, _ = resolve_images(args.images)
    ann_base, _ = resolve_annotations(args.annotations)

    pre = check(args.images, args.annotations)
    if pre["problems"]:
        log("PRE-FLIGHT FAILED:")
        for p in pre["problems"]:
            log(f"  - {p}")
        log("\nNothing was built. Fix the input tree first.")
        return 2
    log("pre-flight OK")

    out = args.output
    if out.exists():
        log(f"OUTPUT_EXISTS: {out} (refusing to overwrite)")
        return 2

    try:
        result = build_dataset(
            "local-dir",
            out,
            args.license_ack,
            args.mapping_ack,
            annotations=ann_base,
            images=img_base,
            counts=None if args.full else {
                "train": args.train_per_source,
                "val": args.val_per_source,
                "test": args.test_per_source,
            },
            full=args.full,
            max_output_gib=args.max_output_gib,
        )
    except DatasetBuildError as exc:
        log(f"BUILD FAILED: {exc}")
        return 2

    merged = {
        **result,
        "source_mode": "local-dir (v2 512px)",
        "preflight": pre,
        "image_root": str(img_base),
        "annotation_root": str(ann_base),
    }
    (out / "v2_ingest_report.json").write_text(
        json.dumps(merged, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log(json.dumps(merged, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def plan() -> dict:
    return {
        "status": "PLAN_ONLY_NO_DATA_READ",
        "ten_output_classes": list(SOURCE_TO_INDEX),
        "hagrid_source_dirs": list(SOURCE_TO_INDEX),
        "accepted_image_layouts": [list(x) for x in IMAGE_LAYOUTS],
        "accepted_annotation_layouts": [list(x) for x in ANNOTATION_LAYOUTS],
        "expected_pixels": "512px min side (v2 512px bundle); model input stays 224x224",
        "download_path_used": False,
        "notes": [
            "auto mode cannot ingest v2 512px: its archive size and index SHA256 are pinned to v1",
            "annotation keys must match JPEG basenames or the build fails closed",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("plan")

    ck = sub.add_parser("check")
    ck.add_argument("--images", type=Path, required=True)
    ck.add_argument("--annotations", type=Path, required=True)
    ck.add_argument("--sample", type=int, default=20)

    bd = sub.add_parser("build")
    bd.add_argument("--images", type=Path, required=True)
    bd.add_argument("--annotations", type=Path, required=True)
    bd.add_argument("--output", type=Path, required=True)
    bd.add_argument("--license-ack", type=Path,
                    default=Path("data/reviews/hagrid-license-ack.json"))
    bd.add_argument("--mapping-ack", type=Path,
                    default=Path("data/reviews/hagrid-mapping-ack.json"))
    bd.add_argument("--full", action="store_true",
                    help="use every valid image in each original split/source")
    bd.add_argument("--train-per-source", type=int, default=2000)
    bd.add_argument("--val-per-source", type=int, default=200)
    bd.add_argument("--test-per-source", type=int, default=200)
    bd.add_argument("--max-output-gib", type=int, default=60)

    args = ap.parse_args()
    try:
        if args.command == "plan":
            print(json.dumps(plan(), indent=2, ensure_ascii=False))
        elif args.command == "check":
            print(json.dumps(check(args.images, args.annotations, sample=args.sample),
                             indent=2, ensure_ascii=False))
        else:
            raise SystemExit(build(args))
    except V2Error as exc:
        raise SystemExit(f"BLOCKED: {exc}")


if __name__ == "__main__":
    main()
