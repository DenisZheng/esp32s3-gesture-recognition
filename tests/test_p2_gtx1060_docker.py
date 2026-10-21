"""Offline safeguards for GTX1060 P2-00 container bootstrap.

Never build Docker images, run a GPU, access HaGRID, install PyTorch or train
inside GitHub's mandatory lightweight test workflow.
"""
from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

from scripts import p2_cuda_probe

ROOT = Path(__file__).resolve().parents[1]


class PascalHostProbeTests(unittest.TestCase):
    def setUp(self):
        self.ready = {
            "cuda_available": True,
            "torch_cuda_build": "12.1",
            "device_name": "SYNTHETIC GTX 1060",
            "device_capability": (6, 1),
            "vram_gib": 5.94,
        }

    def test_plan_has_no_gpu_or_media_dependencies(self):
        report = p2_cuda_probe.offline_plan()
        self.assertEqual(report["status"], "P2_00_GPU_PROBE_PLAN_ONLY")
        self.assertTrue(report["no_real_training"])
        self.assertTrue(report["no_media_or_checkpoint"])

    def test_pascal_sm61_metadata_allows_real_kernel_probe(self):
        report = p2_cuda_probe.validate_metadata(self.ready, expect_sm_61=True)
        self.assertTrue(report["host_cuda_metadata_ready"])
        self.assertEqual(report["compute_capability"], [6, 1])
        self.assertFalse(report["real_model_training_pass"])

    def test_pytorch_cuda_build_mismatch_fails_closed(self):
        fail = dict(self.ready, torch_cuda_build="13.0")
        report = p2_cuda_probe.validate_metadata(fail, expect_sm_61=True)
        self.assertFalse(report["host_cuda_metadata_ready"])
        self.assertTrue(any("12.1" in x for x in report["blockers"]))

    def test_cuda_unavailable_blocks_probe(self):
        fail = dict(self.ready, cuda_available=False)
        report = p2_cuda_probe.validate_metadata(fail, expect_sm_61=True)
        self.assertFalse(report["host_cuda_metadata_ready"])

    def test_active_gpu_wrong_capability_is_not_our_gtx1060(self):
        fail = dict(self.ready, device_capability=(8, 6))
        report = p2_cuda_probe.validate_metadata(fail, expect_sm_61=True)
        self.assertFalse(report["host_cuda_metadata_ready"])
        self.assertTrue(any("sm_61" in x for x in report["blockers"]))

    def test_other_gpu_is_accepted_if_pascal_requirement_is_disabled(self):
        alt = dict(self.ready, device_capability=(8, 6))
        report = p2_cuda_probe.validate_metadata(alt, expect_sm_61=False)
        self.assertTrue(report["host_cuda_metadata_ready"])

    def test_gtx1060_3gb_is_not_mistaken_for_6gb(self):
        small = dict(self.ready, vram_gib=2.92)
        report = p2_cuda_probe.validate_metadata(small, expect_sm_61=True)
        self.assertFalse(report["host_cuda_metadata_ready"])

    def test_missing_gpu_capability_fails(self):
        bad = dict(self.ready, device_capability=None)
        self.assertFalse(
            p2_cuda_probe.validate_metadata(bad, expect_sm_61=True)[
                "host_cuda_metadata_ready"
            ]
        )


class DockerHostScaffoldingTests(unittest.TestCase):
    def test_shell_script_parses_using_actual_bash(self):
        bash = shutil.which("bash")
        if not bash:
            self.skipTest("Bash not installed on this offline test host")
        path = ROOT / "scripts/p2_docker_host.sh"
        result = subprocess.run(
            [bash, "-n", str(path)], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_no_training_or_data_download_action(self):
        s = (ROOT / "scripts/p2_docker_host.sh").read_text(encoding="utf-8")
        for x in (
            'build|source|gpu|doctor|smoke|shell',
            'python scripts/fetch_upstreams.py --id esp-detection',
            'python -m scripts.p2_host_doctor check --strict --require-cuda',
            'python -m scripts.p2_graph_smoke run --device cuda',
            'python -m scripts.p2_cuda_probe run --expect-sm-61',
        ):
            self.assertIn(x, s)
        self.assertNotIn('train)', s)
        self.assertIn("--user", s)
        self.assertIn("--shm-size=2g", s)
        self.assertIn("--gpus all", s)

    def test_docker_image_separate_from_scrfd_and_cuda13(self):
        s = (ROOT / "docker/Dockerfile.p2-espdet-cu121").read_text(encoding="utf-8")
        self.assertIn("FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime@sha256:", s)
        self.assertIn("torch.version.cuda=='12.1'", s)
        self.assertIn('"ultralytics==8.4.0"', s)
        self.assertIn('"numpy==1.24.4"', s)
        self.assertIn("COPY requirements/p2-candidate-constraints.txt", s)
        self.assertNotIn("mmcv", s.lower())
        self.assertNotIn("COPY . ", s)
        self.assertNotIn("FROM scrfd-", s)

    def test_context_excludes_everything_except_public_constraints(self):
        contents = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        active = [x.strip() for x in contents.splitlines() if x.strip() and not x.lstrip().startswith("#")]
        self.assertEqual(active[0], "*")
        self.assertEqual(active, ["*", "!requirements/", "!requirements/p2-candidate-constraints.txt"])
        self.assertIn("!requirements/p2-candidate-constraints.txt", contents)
        self.assertNotIn("!data/", contents)
        self.assertNotIn("!external/", contents)

    def test_shell_mount_persists_workspace_not_training_artifact_in_image(self):
        s = (ROOT / "scripts/p2_docker_host.sh").read_text(encoding="utf-8")
        self.assertIn('$ROOT_DIR:/workspace/repo', s)
        self.assertIn('--env HOME=/tmp', s)
        self.assertIn('--rm', s)
        self.assertNotIn("scrfd-debug-new", s)


if __name__ == "__main__":
    unittest.main()
