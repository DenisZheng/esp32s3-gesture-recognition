# Acceptance protocol: evidence before stage progression

## Result taxonomy

- **DONE**: operation actually executed and artifacts/results verified, with dates, source and model SHA256, test command, platform, human reviewer as applicable.
- **FAIL**: operation executed, assertion/scenario failed; attach diagnosis and an issue; do not convert to BLOCKED to hide failures.
- **BLOCKED**: operation cannot start/finish because missing remote repo, authorization, license, dataset, device, toolchain or environment; list concrete missing prerequisite.
- **PENDING**: planned and not executed. No implied PASS.

## Acceptance gates

| Phase | Required evidence and PASS condition | Explicit failure or block |
|---|---|---|
| P0 | GitHub repo **private**, initial commits reachable; upstream full SHAs; ten class JSON + YAML identical; static compatibility audit; privacy + license register; local unit tests **and GitHub Actions run** PASS; main protected via PR + CI check | Current **BLOCKED** (no remote repo/Actions/ruleset); host code readiness alone not P0 PASS |
| P1 | 10-source mapping signed including `two_up/two` and `palm/five`; real small licensed sample YOLO conversion; 100% valid numeric labels; 0 cross-split `user_id` overlaps; per-hand completeness; negative and non-command samples audited; conversion reproducible | Empty/unknown hands or mixed `user_id` sets => FAIL; missing license/data access => BLOCKED |
| P2 | Scratch training proves no initialized pretrained checkpoints; `best.pt` hash + full train config/environment/seed; train/val loss, mAP50, mAP50-95, recall, confusion, per-class and multi-hand analysis on held-out data | No actual training or test run => PENDING; false claims of model quality => FAIL |
| P3 | ONNX checker, one input, six typed outputs, **10 score channels on all 3 strides**; FP32-vs-ONNX parity on fixed images; ESP-PPQ `esp32s3` W8A8 sym PoT; quantized class/box parity, mAP drop quantified, provenance SHA256 and full color/letterbox calibration parity | `.espdl` export alone = NOT PASS; `NC=1` = FAIL; unquantized model or non-S3 target = FAIL |
| P4 | IDF 5.4.4 / ESP-DL v3.3.14 on **real CoreS3**; own `.espdl` SHA matches P3; first fixed-image numeric parity, then continuous live camera demonstration of all 10 classes, no-hand scenes, other gestures, 2 hands, overlap/occlusion; p50/p95 latency, FPS, memory and PSRAM observed; test log signed | Firmware compiles alone != device PASS; board unavailable = BLOCKED; official weights substituted = FAIL |
| P5 | Apples-to-apples one-model vs official two-model camera comparison; quality/latency/memory/size and false triggers; risks/licensing next-step recommendation; reproducible report | Missing P4 evidence => BLOCKED |

## Input-resolution contract

P2/P3/P4 baseline: `ESPDet-Pico`, **single model**, `nc=10`, input **224x224x3** with whole-frame letterbox. **128x128 is the official ROI classifier size, not our one-model image size.** Resolution changes require a versioned training+export+calibration+firmware agreement and separate experiment evidence; see [INPUT_RESOLUTION.md](INPUT_RESOLUTION.md).

## Quantitative thresholds policy

P0 does **not** invent empirical success rates. At P1 completion, write a versioned acceptance addendum with PoC minimum mAP50/mAP50-95 and per-class recall, no-hand false-trigger metric, and P4 target FPS/p95 memory limit derived from observed baseline/sample difficulty. Freeze thresholds **before** running the final held-out test. Do not retroactively adjust thresholds to a result. Always report numbers even when failing.

For numerical parity, record both intermediate tensor absolute/relative error statistics and post-NMS matched detections (class match, bounding-box IoU, score deviation and counts) before accepting tolerances. Threshold choice must account for FP32 vs W8A8 quantization and pixel rounding. Fixed-image test fixture SHA and exact byte-to-float preprocessing are required.

## Device test matrix and log fields

All ten mapped classes (minimum multiple camera presentations each), idle `no_gesture`, `no_hand`/empty room, non-target classes, second hand in scene, simultaneous different classes, touching/overlapping hands, partial occlusion, distance/scale and variable light; orientation/mirror checks. Record board ID (non-secret), sensor, input resolution, firmware commit, IDF and ESP-DL versions, model SHA256, FPS and p50/p95 latency, free/min heap/PSRAM, crash/restarts, detection class/score/box and reviewer result. All image files and raw video stay local.

Template log: `experiments/templates/experiment.json`. A CI run only validates code/contracts, never substitutes for physical inspection.
