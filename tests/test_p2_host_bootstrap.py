"""Offline P2-00 checks. Fake host data only; never install packages or train."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import p2_host_doctor as host
from scripts import p2_graph_smoke as graph


class HostDoctorTests(unittest.TestCase):
    def setUp(self):
        self.sha = "a" * 40
        self.packages = {
            "torch": "2.2.0+cu121",
            "torchvision": "0.17.0+cu121",
            "numpy": "1.24.4",
            "ultralytics": "8.3.250",
        }
        self.gpu = {
            "cuda_available": True, "mps_available": False,
            "cuda_runtime_build": "12.1", "vram_gib": 8,
            "error": None,
        }

    def review(self, **changes):
        params = {
            "package_versions": self.packages,
            "python_version": (3, 10),
            "upstream_head": self.sha,
            "upstream_expected": self.sha,
            "gpu": self.gpu,
            "detect_supports_reg_max": True,
            "require_cuda": True,
        }
        params.update(changes)
        return host.evaluate(**params)

    def test_plan_offline_and_not_a_training_claim(self):
        a = host.offline_plan()
        self.assertEqual(a["classes"], 10)
        self.assertEqual(a["input_hw"], [224, 224])
        self.assertFalse(a["downloads_or_training_started"])
        self.assertEqual(a["p1_real_data_and_license_gate"], "PENDING")

    def test_gpu_candidate_host_can_attempt_synthetic_model_smoke(self):
        result = self.review()
        self.assertTrue(result["host_prerequisites_ready"])
        self.assertEqual(result["status"], "HOST_PREREQUISITES_READY_FOR_SYNTHETIC_SMOKE")
        self.assertFalse(result["real_training_authorized"])
        self.assertEqual(result["model_output_classes"], 10)

    def test_cpu_only_can_run_smoke_but_not_cuda_training(self):
        cpu = dict(self.gpu, cuda_available=False, vram_gib=None)
        with_cuda = self.review(gpu=cpu)
        self.assertFalse(with_cuda["host_prerequisites_ready"])
        self.assertTrue(any("CUDA" in line for line in with_cuda["blockers"]))
        without = self.review(gpu=cpu, require_cuda=False)
        self.assertTrue(without["host_prerequisites_ready"])
        self.assertTrue(any("CPU graph smoke" in line for line in without["advisories"]))

    def test_wrong_python_minor_fails(self):
        self.assertFalse(self.review(python_version=(3, 12))["host_prerequisites_ready"])

    def test_missing_packages_and_wrong_checkout_fail(self):
        pkg = dict(self.packages, torch=None)
        status = self.review(package_versions=pkg, upstream_head=None)
        self.assertFalse(status["host_prerequisites_ready"])
        self.assertTrue(any("torch" in s for s in status["blockers"]))
        self.assertFalse(status["upstream_clone_matches_pin"])

    def test_unsupported_ultralytics_constructor_fails(self):
        report = self.review(detect_supports_reg_max=False)
        self.assertTrue(any("reg_max" in s for s in report["blockers"]))

    def test_old_ultralytics_version_fails_minimum(self):
        pkg = dict(self.packages, ultralytics="8.2.99")
        self.assertFalse(self.review(package_versions=pkg)["host_prerequisites_ready"])

    def test_vram_warning_not_forced_failure(self):
        report = self.review(gpu=dict(self.gpu, vram_gib=4.0))
        self.assertTrue(report["host_prerequisites_ready"])
        self.assertTrue(any("batch" in s for s in report["advisories"]))

    def test_upstream_package_candidate_mismatch_is_not_fake_pass(self):
        pkg = dict(self.packages, numpy="2.0.0")
        report = self.review(package_versions=pkg)
        self.assertTrue(any("numpy" in s for s in report["advisories"]))
        self.assertTrue(report["host_prerequisites_ready"])
        # Still MUST run actual graph compatibility instead of assuming it.
        self.assertEqual(report["status"], "HOST_PREREQUISITES_READY_FOR_SYNTHETIC_SMOKE")

    def test_package_versions_parser_accepts_cuda_suffix(self):
        self.assertEqual(host._version_tuple("8.3.250+cu121"), (8, 3, 250))
        self.assertIsNone(host._version_tuple("junk"))

    def test_source_sha_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "upstreams.lock.json").write_text(json.dumps({
                "dependencies": [{"id": "esp-detection", "commit": "bad"}]
            }))
            with self.assertRaises(ValueError):
                host._expected_source_sha(root)


class SyntheticGraphContractTests(unittest.TestCase):
    def test_plan_does_not_run_training(self):
        out = graph.offline_plan()
        self.assertEqual(out["num_classes"], 10)
        self.assertEqual(out["input_shape"], [1, 3, 224, 224])
        self.assertFalse(out["real_dataset_accessed"])
        self.assertFalse(out["p2_training_started"])

    def test_three_expected_ten_class_feature_maps(self):
        fake = [
            SimpleNamespace(shape=(1, 14, 28, 28)),
            SimpleNamespace(shape=(1, 14, 14, 14)),
            SimpleNamespace(shape=(1, 14, 7, 7)),
        ]
        self.assertEqual(graph.check_shapes(fake), [list(x.shape) for x in fake])

    def test_reject_single_class_and_eleven_class_graph(self):
        for channels in (5, 15):
            fake = [SimpleNamespace(shape=(1, channels, *hw)) for hw in graph.STAGE_SIZES]
            with self.assertRaises(graph.GraphSmokeError):
                graph.check_shapes(fake)

    def test_graph_rejects_wrong_number_of_stages(self):
        with self.assertRaises(graph.GraphSmokeError):
            graph.check_shapes([SimpleNamespace(shape=(1, 14, 28, 28))])
        with self.assertRaises(graph.GraphSmokeError):
            graph.check_shapes({})

    def test_graph_source_sha_fail_closed_without_pytorch(self):
        with tempfile.TemporaryDirectory() as d:
            with patch("scripts.p2_graph_smoke._expected_source_sha", return_value="a"*40):
                with patch("scripts.p2_graph_smoke._git_head", return_value=None):
                    with self.assertRaisesRegex(graph.GraphSmokeError, "pinned SHA"):
                        graph.validate_upstream(Path(d))

    def test_pinned_yaml_render_without_model_import(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / "cfg/models/espdet_pico.yaml"
            path.parent.mkdir(parents=True)
            path.write_text(
                "nc: 1\nbackbone: []\nhead:\n - [[0, 1, 2], 1, ESPDetect, [nc]]\n"
            )
            with patch("scripts.p2_graph_smoke._expected_source_sha", return_value="a"*40):
                with patch("scripts.p2_graph_smoke._git_head", return_value="a"*40):
                    cfg, sha = graph.validate_upstream(root)
            self.assertEqual(sha, "a"*40)
            self.assertIn("nc: 10", cfg)
            self.assertIn("ESPDetect, [nc]", cfg)


if __name__ == "__main__":
    unittest.main()
