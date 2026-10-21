#!/usr/bin/env python3
"""Offline-first synthetic ESPDet-Pico nc=10 graph/backward compatibility smoke.

No licensed HaGRID data, pretrained weights, model training, checkpoint output,
remote download, or user identity. This cannot authorize P2 after P1 blockade.

  python3 -m scripts.p2_graph_smoke plan
  python3 -m scripts.p2_graph_smoke run --device cpu
  python3 -m scripts.p2_graph_smoke run --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import sys
import tempfile
from pathlib import Path

from scripts.p2_host_doctor import ROOT, _expected_source_sha, _git_head
from scripts.prepare_model_config import render

STAGE_SIZES = ((28, 28), (14, 14), (7, 7))


class GraphSmokeError(ValueError):
    """Pinned model compatibility or synthetic gradient failed."""


def offline_plan():
    return {
        "status": "P2_00_SYNTHETIC_GRAPH_PLAN_ONLY",
        "model": "ESPDet-Pico",
        "input_shape": [1, 3, 224, 224],
        "num_classes": 10,
        "expected_training_stage_shapes": [[1, 14, *hw] for hw in STAGE_SIZES],
        "pretrained_weights_loaded": False,
        "real_dataset_accessed": False,
        "model_saved": False,
        "p2_training_started": False,
    }


def validate_upstream(upstream: Path, root: Path = ROOT) -> tuple[str, str]:
    expected = _expected_source_sha(root)
    if _git_head(upstream) != expected:
        raise GraphSmokeError("ESP-Detection clone is missing or not at its pinned SHA")
    yaml_path = upstream / "cfg/models/espdet_pico.yaml"
    if not yaml_path.is_file() or yaml_path.is_symlink():
        raise GraphSmokeError("ESPDet source YAML missing")
    return render(yaml_path.read_text(encoding="utf-8")), expected


def check_shapes(outputs) -> list[list[int]]:
    """Pure stage contract, independently unit tested without PyTorch."""
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 3:
        raise GraphSmokeError("Expected three ESPDet training feature maps")
    shapes = []
    for i, (tensor, hw) in enumerate(zip(outputs, STAGE_SIZES)):
        shape = getattr(tensor, "shape", None)
        if shape is None or list(shape) != [1, 14, *hw]:
            raise GraphSmokeError(f"Stage {i} dimensions differ from nc=10 contract")
        shapes.append(list(shape))
    return shapes


def inspect_training_dictionary(predictions) -> list:
    """Ultralytics 8.4.x Detect.train() gives boxes/scores/feats, not a list.

    This is a shape/schema check only; actual custom ESPDetect head outputs
    still need a runtime check against each feature tensor in run().
    """
    if not isinstance(predictions, dict):
        raise GraphSmokeError("Expected Ultralytics 8.4 training output dict")
    boxes, scores, feats = (
        predictions.get("boxes"), predictions.get("scores"), predictions.get("feats")
    )
    total_cells = sum(h * w for h, w in STAGE_SIZES)
    if (
        getattr(boxes, "shape", None) is None
        or list(boxes.shape) != [1, 4, total_cells]
        or getattr(scores, "shape", None) is None
        or list(scores.shape) != [1, 10, total_cells]
        or not isinstance(feats, (list, tuple))
        or len(feats) != 3
    ):
        raise GraphSmokeError("Ultralytics boxes/scores/feats violate ten-class contract")
    for i, (tensor, hw) in enumerate(zip(feats, STAGE_SIZES)):
        shape = getattr(tensor, "shape", None)
        if shape is None or len(shape) != 4 or list(shape)[0] != 1 or list(shape)[2:] != list(hw):
            raise GraphSmokeError(f"Training feature level {i} has invalid resolution")
    return list(feats)


def run(upstream: Path, *, device: str = "cpu", seed: int = 20261021):
    if device not in ("cpu", "cuda") or not 0 <= seed < 2 ** 31:
        raise GraphSmokeError("Unsupported device or seed")
    generated, commit = validate_upstream(upstream)
    try:
        torch = importlib.import_module("torch")
        head = importlib.import_module("ultralytics.nn.modules.head")
        tasks = importlib.import_module("ultralytics.nn.tasks")
        ultralytics = importlib.import_module("ultralytics")
    except (ImportError, OSError, RuntimeError) as exc:
        raise GraphSmokeError("Install compatible Torch, Torchvision and Ultralytics") from exc
    if "reg_max" not in inspect.signature(head.Detect.__init__).parameters:
        raise GraphSmokeError("Installed Ultralytics Detect constructor lacks reg_max")
    if device == "cuda" and not torch.cuda.is_available():
        raise GraphSmokeError("CUDA requested but torch.cuda.is_available() is false")
    torch.manual_seed(seed)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    old_path = list(sys.path)
    sys.path.insert(0, str(upstream.resolve()))
    try:
        from nn.esp_tasks import custom_parse_model
        with tempfile.TemporaryDirectory(prefix="p2-nc10-synthetic-") as d:
            yaml_file = Path(d) / "espdet_10.yaml"
            yaml_file.write_text(generated, encoding="utf-8")
            previous = tasks.parse_model
            try:
                tasks.parse_model = custom_parse_model
                model = ultralytics.YOLO(str(yaml_file))
            finally:
                tasks.parse_model = previous
            net = model.model
            if int(net.yaml.get("nc", -1)) != 10:
                raise GraphSmokeError("Expected ten-class scratch network")
            net = net.to(device).train()
            optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
            optimizer.zero_grad(set_to_none=True)
            noise = torch.randn(1, 3, 224, 224, device=device)
            predictions = net(noise)
            if not isinstance(predictions, dict):
                raise GraphSmokeError("Expected Ultralytics 8.4 training output dictionary")
            feats = inspect_training_dictionary(predictions)
            last = net.model[-1]
            if type(last).__name__ != "ESPDetect":
                raise GraphSmokeError("Last model layer is not the ESPDet custom head")
            # Re-run only the two small *custom* per-stage heads on their
            # in-memory feature tensors. The original upstream's export method
            # produces precisely these 4-box/10-class channel groups.
            stages = [
                torch.cat((last.cv2[i](feat), last.cv3[i](feat)), dim=1)
                for i, feat in enumerate(feats)
            ]
            shapes = check_shapes(stages)
            synthetic_objective = sum(v.square().mean() for v in stages)
            if not bool(torch.isfinite(synthetic_objective)):
                raise GraphSmokeError("Nonfinite synthetic objective")
            synthetic_objective.backward()
            grads = [p.grad for p in net.parameters() if p.grad is not None]
            if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
                raise GraphSmokeError("Missing/nonfinite synthetic gradient")
            custom_cls_grad = [
                p.grad for p in last.cv3.parameters()
                if p.grad is not None and bool(torch.isfinite(p.grad).all())
            ]
            if not custom_cls_grad:
                raise GraphSmokeError("Custom ten-class ESPDetect classification head has no gradient")
            optimizer.step()
            return {
                "status": "SYNTHETIC_GRAPH_BACKPROP_ONLY_PASS",
                "model": "ESPDet-Pico",
                "num_classes": 10,
                "input_hw": [224, 224],
                "stage_shapes": shapes,
                "pinned_upstream_sha": commit,
                "config_sha256": hashlib.sha256(generated.encode("utf-8")).hexdigest(),
                "torch_version": importlib.metadata.version("torch"),
                "ultralytics_version": importlib.metadata.version("ultralytics"),
                "gradient_tensors": len(grads),
                "custom_ten_class_head_gradients": len(custom_cls_grad),
                "device": device,
                "real_dataset_accessed": False,
                "pretrained_weights_loaded": False,
                "checkpoint_saved": False,
                "actual_training_completed": False,
            }
    except GraphSmokeError:
        raise
    except (ImportError, OSError, RuntimeError, AttributeError, ValueError, TypeError) as exc:
        raise GraphSmokeError(
            "Model graph incompatible with installed packages: " + type(exc).__name__
        ) from exc
    finally:
        sys.path[:] = old_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run"))
    parser.add_argument("--upstream", type=Path, default=ROOT / "external/esp-detection")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    try:
        result = offline_plan() if args.mode == "plan" else run(
            args.upstream, device=args.device, seed=args.seed
        )
    except GraphSmokeError as exc:
        parser.exit(2, f"BLOCKED: {exc}\n")
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
