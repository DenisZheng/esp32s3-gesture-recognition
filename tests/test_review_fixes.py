"""Regression tests for the PR #16 review findings.

Covers the P0 authorisation gate, the P1 manifest/path/image-integrity defects
and the P2 per-class report bug.

Stdlib ``unittest`` only: the contract CI job installs no third-party test
runner, and adding one just for these checks would widen the supply-chain
surface of the gate. Synthetic data only; no HaGRID media is read.
"""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrefetchAuthorisationTests(unittest.TestCase):
    """P0-1: warming the cache must not bypass the licence gate."""

    @classmethod
    def setUpClass(cls):
        cls.pf = _load_module("_pf_under_test", ROOT / "_prefetch_all.py")

    def test_missing_acks_abort_before_any_data_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "absent.json"
            with self.assertRaises(SystemExit) as ctx:
                self.pf.require_owner_approvals(missing, missing)
            self.assertIn("licen", str(ctx.exception).lower())

    def test_invalid_mapping_ack_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            lic = tmp / "lic.json"
            lic.write_text(json.dumps({
                "reviewer": "synthetic",
                "reviewed_on": "2026-10-26",
                "license_pdf_sha256":
                    "14f4845e9c8d3de875cbac4139491da368ab1040a7f565e02894477279134d22",
                "permitted_scope": "research_only",
                "accept_terms": True,
            }), encoding="utf-8")
            bad = tmp / "map.json"
            bad.write_text(json.dumps({"reviewer": "", "classes": []}), encoding="utf-8")
            with self.assertRaises(SystemExit) as ctx:
                self.pf.require_owner_approvals(lic, bad)
            self.assertIn("mapping", str(ctx.exception).lower())


class DatasetManifestTests(unittest.TestCase):
    """P1: dataset.yaml must be inside the integrity contract."""

    def test_builder_records_yaml_hash_and_class_order(self):
        src = (ROOT / "scripts/p1_dataset_builder.py").read_text(encoding="utf-8")
        self.assertIn("dataset_yaml_sha256", src)
        self.assertIn("class_names_ordered", src)

    def test_train_entry_verifies_yaml_and_manifest(self):
        src = (ROOT / "scripts/p2_train.py").read_text(encoding="utf-8")
        self.assertIn("recorded_yaml_sha", src)
        self.assertIn("dataset.yaml changed after the dataset was built", src)
        self.assertIn("manifest.aggregate.json", src)
        self.assertIn("require_training_clearance", src)

    def test_probe_experimental_train_strict(self):
        src = (ROOT / "scripts/p2_train.py").read_text(encoding="utf-8")
        self.assertIn("require_training_clearance(dataset, experimental=True)", src)
        self.assertIn("require_training_clearance(dataset, experimental=False)", src)


class RunDirectoryTests(unittest.TestCase):
    """P1: Ultralytics must not fork the run directory behind our back."""

    def test_run_dir_reuse_and_save_dir_verification(self):
        src = (ROOT / "scripts/p2_train.py").read_text(encoding="utf-8")
        self.assertIn("exist_ok=True", src)
        self.assertIn("_verify_save_dir", src)
        self.assertIn("trainer wrote to a different directory than requested", src)


class OptimizerWhitelistTests(unittest.TestCase):
    """P1: the optimiser that froze ESPDet weights must not be selectable."""

    def test_auto_is_not_allowed(self):
        src = (ROOT / "scripts/p2_train.py").read_text(encoding="utf-8")
        self.assertIn('ALLOWED_OPTIMIZERS = ("SGD"', src)
        after = src.split("ALLOWED_OPTIMIZERS")[1].split(")")[0]
        self.assertNotIn('"auto"', after)
        self.assertIn("choices=ALLOWED_OPTIMIZERS", src)


class OnnxExportTests(unittest.TestCase):
    """P1: export and parity must not accept a half-verified graph."""

    def test_export_validates_shapes_and_weights(self):
        src = (ROOT / "scripts/p3_eval.py").read_text(encoding="utf-8")
        self.assertIn("output_shapes", src)
        self.assertIn("EXPECTED_SHAPES", src)
        self.assertIn("staged checkpoint", src)
        self.assertIn('"weights_sha256": source_sha', src)

    def test_parity_requires_every_output(self):
        src = (ROOT / "scripts/p3_eval.py").read_text(encoding="utf-8")
        self.assertIn("missing expected outputs", src)
        self.assertIn("shape_mismatch", src)
        self.assertIn("ONNX exposes", src)


class CacheIntegrityTests(unittest.TestCase):
    """P1: a truncated cache file must not count as downloaded."""

    @classmethod
    def setUpClass(cls):
        cls.pf = _load_module("_pf_cache_test", ROOT / "_prefetch_all.py")

    def test_jpeg_structure_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            good = tmp / "good.jpg"
            good.write_bytes(b"\xff\xd8\xff" + b"x" * 100 + b"\xff\xd9")
            truncated = tmp / "truncated.jpg"
            truncated.write_bytes(b"\xff\xd8\xff" + b"x" * 100)
            empty = tmp / "empty.jpg"
            empty.write_bytes(b"")

            self.assertTrue(self.pf._cache_entry_ok(good))
            self.assertFalse(self.pf._cache_entry_ok(truncated))
            self.assertFalse(self.pf._cache_entry_ok(empty))
            self.assertFalse(self.pf._cache_entry_ok(tmp / "absent.jpg"))


class V2DecodeTests(unittest.TestCase):
    """P1: the v2 pre-flight claims to decode pixels, so it must.

    Pillow is not installed in the contract CI job (it only needs stdlib), so the
    execute-the-decoder test is skipped there while a static check still guards
    that the decoder is wired into the pre-flight.
    """

    def test_preflight_wires_in_a_real_decoder(self):
        src = (ROOT / "scripts/p1_v2_local.py").read_text(encoding="utf-8")
        self.assertIn("def probe_decoadable", src)
        self.assertIn("image.verify()", src)
        self.assertIn("image.load()", src)
        self.assertIn("decode_probe", src)
        # The short-side expectation must actually be enforced, not just logged.
        self.assertIn("min_side_observed", src)
        self.assertIn("expected >=512", src)

    @unittest.skipUnless(
        importlib.util.find_spec("PIL") is not None,
        "Pillow not installed in this environment (contract CI has no optional deps)",
    )
    def test_real_jpegs_decode_and_corrupt_ones_fail(self):
        from PIL import Image
        from scripts import p1_v2_local

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for i, size in enumerate([(640, 480), (512, 512)]):
                Image.new("RGB", size, (120, 130, 140)).save(folder / f"k{i}.jpg")
            (folder / "bad.jpg").write_bytes(b"\xff\xd8\xff not a jpeg \xff\xd9")

            info = p1_v2_local.probe_decoadable(folder, ["k0", "k1"], limit=4)
            self.assertEqual(info["decoded"], 2)
            self.assertEqual(info["min_side_min"], 480)

            bad = p1_v2_local.probe_decoadable(folder, ["bad"], limit=4)
            self.assertEqual(bad["decoded"], 0)
            self.assertEqual(len(bad["errors"]), 1)


class PerClassReportTests(unittest.TestCase):
    """P2: the report read a key that was never stored, so it was always null."""

    def test_report_reads_stored_key(self):
        src = (ROOT / "scripts/p3_eval.py").read_text(encoding="utf-8")
        self.assertIn('ap95 = per_class.get("ap50_95")', src)
        self.assertNotIn('if "ap" in record.get("per_class", {})', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
