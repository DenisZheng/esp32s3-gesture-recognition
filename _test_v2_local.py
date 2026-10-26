"""Synthetic test for scripts/p1_v2_local.py - no real HaGRID data touched.

Builds a tiny fake tree in two layouts (flat and nested) and checks that the
pre-flight accepts the good one and rejects the mismatched-key one.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.p1_v2_local import check, resolve_annotations, resolve_images, V2Error
from scripts.data_contract import SOURCE_TO_INDEX

SOURCES = tuple(SOURCE_TO_INDEX)
SPLITS = ("train", "val", "test")
FAILS = []


def ok(cond, label):
    print(("PASS  " if cond else "FAIL  ") + label)
    if not cond:
        FAILS.append(label)


def make_tree(root: Path, *, nested: bool, mismatch: bool, per_class: int = 3):
    img_root = root / "imgs" / ("hagrid_dataset" if nested else "")
    ann_root = root / "anns" / ("hagrid_annotations" if nested else "")
    for src in SOURCES:
        folder = img_root / src
        folder.mkdir(parents=True, exist_ok=True)
        keys = []
        for i in range(per_class):
            key = f"{src}_img_{i}"
            keys.append(key)
            (folder / f"{key}.jpg").write_bytes(b"\xff\xd8\xff" + key.encode() + b"\xff\xd9")
        for split in SPLITS:
            d = ann_root / split
            d.mkdir(parents=True, exist_ok=True)
            # `mismatch` writes keys that do not exist on disk.
            doc = {
                (f"nonexistent_{i}" if mismatch else keys[i]): {
                    "user_id": f"u_{src}_{split}_{i}",
                    "bboxes": [[0.1, 0.2, 0.3, 0.4]],
                    "labels": [src],
                }
                for i in range(per_class)
            }
            (d / f"{src}.json").write_text(json.dumps(doc), encoding="utf-8")
    return img_root.parent if nested else root / "imgs", ann_root.parent if nested else root / "anns"


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    # 1. flat layout, matching keys -> should pass
    imgs, anns = make_tree(tmp / "flat", nested=False, mismatch=False)
    r = check(imgs, anns, sample=3)
    ok(r["status"] == "CHECK_OK", f"flat layout accepted ({r['status']})")
    ok(r["image_layout"] == "{source}", f"flat image layout detected: {r['image_layout']}")
    ok(all(v["images"] > 0 for v in r["classes"].values()), "all classes have images")
    ok(all(v["found"] == v["checked"] and v["checked"] > 0
           for v in r["key_resolution"].values()), "all sampled annotation keys resolve")

    # 2. nested layout -> resolution should still work
    imgs2, anns2 = make_tree(tmp / "nested", nested=True, mismatch=False)
    base_i, layout_i = resolve_images(imgs2)
    base_a, layout_a = resolve_annotations(anns2)
    ok(base_i.name == "hagrid_dataset", f"nested image base = {base_i.name}")
    ok(base_a.name == "hagrid_annotations", f"nested annotation base = {base_a.name}")
    r2 = check(imgs2, anns2, sample=3)
    ok(r2["status"] == "CHECK_OK", f"nested layout accepted ({r2['status']})")

    # 3. mismatched keys -> must fail
    imgs3, anns3 = make_tree(tmp / "bad", nested=False, mismatch=True)
    r3 = check(imgs3, anns3, sample=3)
    ok(r3["status"] == "CHECK_FAILED", f"mismatched keys rejected ({r3['status']})")
    ok(any("do not match filenames" in p for p in r3["problems"]),
       "mismatch reported explicitly")

    # 4. missing class dir -> must fail
    imgs4, anns4 = make_tree(tmp / "missing", nested=False, mismatch=False)
    import shutil
    shutil.rmtree(imgs4 / "peace")
    r4 = check(imgs4, anns4, sample=3)
    ok(r4["status"] == "CHECK_FAILED", "missing class dir rejected")

    # 5. bad root -> explicit error
    try:
        resolve_images(tmp / "does_not_exist")
        ok(False, "missing root should raise")
    except V2Error:
        ok(True, "missing root raises V2Error")

print()
if FAILS:
    print(f"{len(FAILS)} CHECK(S) FAILED")
    sys.exit(1)
print("ALL V2 INGEST CHECKS PASSED")
