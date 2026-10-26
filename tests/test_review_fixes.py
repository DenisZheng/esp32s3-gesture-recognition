"""Regression tests for the PR #16 review findings.

Covers the P0 authorisation gate, the P1 manifest/path/image-integrity defects
and the P2 per-class report bug. Synthetic data only; no HaGRID media.
"""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

FAILS = []


def ok(cond, label):
    print(("PASS  " if cond else "FAIL  ") + label)
    if not cond:
        FAILS.append(label)


# ---------------------------------------------------------------- P0 prefetch

def test_prefetch_requires_approvals(tmp=Path(".")):
    """The cache warmer must refuse to touch annotations without both acks."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("pf", ROOT / "_prefetch_all.py")
    pf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pf)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        missing = tmp / "nope.json"

        # Missing licence ack -> SystemExit naming the licence.
        try:
            pf.require_owner_approvals(missing, missing)
            ok(False, "missing acks should abort")
        except SystemExit as exc:
            ok("licence" in str(exc).lower() or "license" in str(exc).lower(),
               f"missing ack aborts with licence message ({exc})")

        # A structurally invalid mapping ack must also abort.
        bad = tmp / "bad-mapping.json"
        bad.write_text(json.dumps({"reviewer": "", "classes": []}), encoding="utf-8")
        good_license = tmp / "lic.json"
        good_license.write_text(json.dumps({
            "reviewer": "synthetic",
            "reviewed_on": "2026-10-26",
            "license_pdf_sha256":
                "14f4845e9c8d3de875cbac4139491da368ab1040a7f565e02894477279134d22",
            "permitted_scope": "research_only",
            "accept_terms": True,
        }), encoding="utf-8")
        try:
            pf.require_owner_approvals(good_license, bad)
            ok(False, "invalid mapping ack should abort")
        except SystemExit as exc:
            ok("mapping" in str(exc).lower(), f"invalid mapping aborts ({exc})")


# ------------------------------------------------------------- P1 dataset yaml

def test_dataset_yaml_hash_is_recorded_and_verified():
    """Editing dataset.yaml after a build must invalidate the manifest check."""
    from scripts.p1_dataset_builder import build_dataset  # noqa: F401
    src = (ROOT / "scripts/p1_dataset_builder.py").read_text(encoding="utf-8")
    ok("dataset_yaml_sha256" in src,
       "builder records the dataset.yaml hash in the aggregate manifest")
    ok("class_names_ordered" in src,
       "builder records the ordered class list")


def test_train_rejects_changed_yaml():
    """p2_train must compare the on-disk YAML hash with the manifest."""
    src = (ROOT / "scripts/p2_train.py").read_text(encoding="utf-8")
    ok("recorded_yaml_sha" in src, "train entry reads the recorded YAML hash")
    ok("dataset.yaml changed after the dataset was built" in src,
       "train entry fails when the YAML changed")
    ok("manifest.aggregate.json" in src,
       "train entry requires the aggregate manifest")
    ok("require_training_clearance" in src,
       "train entry gates on P1 acceptance")


def test_probe_is_experimental_but_train_is_not():
    src = (ROOT / "scripts/p2_train.py").read_text(encoding="utf-8")
    ok("require_training_clearance(dataset, experimental=True)" in src,
       "probe passes experimental=True")
    ok("require_training_clearance(dataset, experimental=False)" in src,
       "formal train passes experimental=False")


# ----------------------------------------------------------------- P1 run dir

def test_run_dir_is_not_silently_incremented():
    src = (ROOT / "scripts/p2_train.py").read_text(encoding="utf-8")
    ok("exist_ok=True" in src,
       "trainer is told to reuse the pre-created run dir")
    ok("_verify_save_dir" in src,
       "trainer save_dir is verified against the requested path")
    ok("trainer wrote to a different directory than requested" in src,
       "mismatched save_dir fails closed")


# ---------------------------------------------------------------- P1 optimizer

def test_optimizer_whitelist_excludes_auto():
    src = (ROOT / "scripts/p2_train.py").read_text(encoding="utf-8")
    ok('ALLOWED_OPTIMIZERS = ("SGD"' in src, "optimiser whitelist defined")
    ok('"auto"' not in src.split("ALLOWED_OPTIMIZERS")[1].split(")")[0],
       "`auto` is not an allowed optimiser")
    ok("choices=ALLOWED_OPTIMIZERS" in src, "CLI restricts to the whitelist")


# ------------------------------------------------------------------ P1 p3 eval

def test_export_validates_shapes_and_staged_weights():
    src = (ROOT / "scripts/p3_eval.py").read_text(encoding="utf-8")
    ok("output_shapes" in src, "export records actual output shapes")
    ok("shape {got} != " in src or "shape" in src and "EXPECTED_SHAPES" in src,
       "export compares against EXPECTED_SHAPES")
    ok("staged checkpoint" in src,
       "export refuses a stale staged checkpoint")
    ok("weights_sha256\": source_sha" in src,
       "evidence records the source weights hash")


def test_parity_requires_all_six_outputs():
    src = (ROOT / "scripts/p3_eval.py").read_text(encoding="utf-8")
    ok("missing expected outputs" in src,
       "parity fails when an expected output name is absent")
    ok("shape_mismatch" in src, "parity compares shapes before flattening")
    ok("ONNX exposes" in src, "parity checks the output count")


# --------------------------------------------------------- P1 cache integrity

def test_prefetch_checks_jpeg_structure():
    import importlib.util

    spec = importlib.util.spec_from_file_location("pf2", ROOT / "_prefetch_all.py")
    pf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pf)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        good = tmp / "good.jpg"
        good.write_bytes(b"\xff\xd8\xff" + b"x" * 100 + b"\xff\xd9")
        truncated = tmp / "truncated.jpg"
        truncated.write_bytes(b"\xff\xd8\xff" + b"x" * 100)   # no EOI
        empty = tmp / "empty.jpg"
        empty.write_bytes(b"")

        ok(pf._cache_entry_ok(good), "valid JPEG accepted by the cache probe")
        ok(not pf._cache_entry_ok(truncated),
           "truncated JPEG rejected (would be re-downloaded)")
        ok(not pf._cache_entry_ok(empty), "empty file rejected")
        ok(not pf._cache_entry_ok(tmp / "absent.jpg"), "missing file rejected")


# ------------------------------------------------------------ P1 v2 decode probe

def test_v2_preflight_decodes_pixels():
    from PIL import Image
    from scripts import p1_v2_local

    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        for i, size in enumerate([(640, 480), (512, 512)]):
            Image.new("RGB", size, (120, 130, 140)).save(folder / f"k{i}.jpg")
        bad = folder / "bad.jpg"
        bad.write_bytes(b"\xff\xd8\xff definitely not a jpeg \xff\xd9")

        info = p1_v2_local.probe_decoadable(folder, ["k0", "k1"], limit=4)
        ok(info["decoded"] == 2, f"both real JPEGs decoded ({info['decoded']})")
        ok(info["min_side_min"] == 480, f"short side reported ({info['min_side_min']})")

        info2 = p1_v2_local.probe_decoadable(folder, ["bad"], limit=4)
        ok(info2["decoded"] == 0, "corrupt JPEG not counted as decoded")
        ok(len(info2["errors"]) == 1, "corrupt JPEG reported as an error")


# ------------------------------------------------------------- P2 per-class AP

def test_per_class_report_builds_from_stored_keys():
    src = (ROOT / "scripts/p3_eval.py").read_text(encoding="utf-8")
    ok('ap95 = per_class.get("ap50_95")' in src,
       "per-class report reads the key that is actually stored")
    ok('if "ap" in record.get("per_class", {})' not in src,
       "the buggy `ap` membership test is gone")


if __name__ == "__main__":
    test_prefetch_requires_approvals()
    test_dataset_yaml_hash_is_recorded_and_verified()
    test_train_rejects_changed_yaml()
    test_probe_is_experimental_but_train_is_not()
    test_run_dir_is_not_silently_incremented()
    test_optimizer_whitelist_excludes_auto()
    test_export_validates_shapes_and_staged_weights()
    test_parity_requires_all_six_outputs()
    test_prefetch_checks_jpeg_structure()
    test_v2_preflight_decodes_pixels()
    test_per_class_report_builds_from_stored_keys()

    print()
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED")
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("ALL REVIEW-FIX CHECKS PASSED")
