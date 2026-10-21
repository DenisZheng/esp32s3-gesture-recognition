# Acceptance protocol: evidence before stage progression

## Result taxonomy

- **DONE**: operation actually executed and artifacts/results verified, with dates, source and model SHA256, test command, platform, human reviewer as applicable.
- **FAIL**: operation executed, assertion/scenario failed; attach diagnosis and an issue; do not convert to BLOCKED to hide failures.
- **BLOCKED**: operation cannot start/finish because missing remote repo, authorization, license, dataset, device, toolchain or environment; list concrete missing prerequisite.
- **PENDING**: planned and not executed. No implied PASS.

## Acceptance gates

| Phase | Required evidence and PASS condition | Explicit failure or block |
|---|---|---|
| P0 | Owner-approved **public** independent repo, initial commits reachable; upstream SHAs; class config; source audit; privacy/license inventory; GitHub Actions PASS; main protected via PR + p0-contract | GitHub infrastructure VERIFIED. Custom PDF source **reviewed**; owner/license acceptance and ten-class visual signoff still pending; research gate PARTIAL, not P0 full PASS |
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


## P1 hard-negative acceptance addendum

Do **not** accept a data release based merely on a list of ten HaGRID source folders. The 24 other raw categories include command-lookalikes (e.g. `peace` / `stop`) and double-hand composites. A negative class-7 detection must have a **verified hand bbox** and a currently signed source-specific review; unknown or ambiguous hands must cause entire-image rejection. Source-policy digest in private signoff must match committed JSON. Legitimate hand-free `no_hand` samples must have evidence of zero visible hands. The official `two_up -> two` mapping is still provisional and must be adjudicated against licensed `peace` variants before final P1 PASS. These gates cannot be substituted by synthetic CI.


### P1-03 licensed visual evidence

A local HTML contact sheet can support mapping signoff only when licensed real train images, sufficient class/alias comparisons, verified per-hand boxes and human-reviewed signatures exist. The new `scripts/p1_visual_review.py` refuses test images, binds image integrity and produces pending evidence forms, but synthetic CI does not constitute semantic, legal, dataset or device PASS. See [P1_VISUAL_REVIEW.md](P1_VISUAL_REVIEW.md).


### P1-04 ZIP64 image acquisition evidence

Index-only remote smoke tests establish that a 42.73 GB archive supports ranged metadata reads, but **do not prove real JPEG extraction or subject rights**. The local-only extractor must verify immutable/unchanged archive byte range, ZIP64 offset and filenames, CRC32, JPEG signature, SHA256 and research-license approval, and preserve its ignored manifest. A fully decoded/visually checked sample with reviewer decisions is required before P1 PASS. See [P1_IMAGE_ZIP64.md](P1_IMAGE_ZIP64.md).


**P1-04 metadata baseline update:** Four official source archives (two_up, peace, peace_inverted, two_up_inverted) now have verified byte sizes and central directory SHA256 pins in code, and full ZIP64 central index parsing succeeded in GitHub Actions without real image payloads. Authentic image extraction and licensed human labeling are still outstanding.


### P1 local four-way review workflow

`python3 -m scripts.p1_review_workflow plan` is offline-only. `prepare` requires authentic local owner license acceptance before *any* real image/annotation download and refuses public CI environments; it publishes an ignored local workspace only after all four train-only classes + HTML review packet are present. `verify` checks locally signed evidence integrity but NEVER substitutes for an actual human gesture judgment or final ten-class P1 signoff. Source tests use synthetic media only. [Operator guide](P1_ONE_CLICK_REVIEW.md).


**P1 archive index selection regression:** An official annotations.zip contains large unrelated JSON entries. Metadata parsing must tolerate oversized **unselected** classes while rejecting requested files above the decode/download cap and honoring aggregate byte budget. This was verified on metadata-only official four-class selection (4 train members totaling 54,388,405 compressed bytes); real authorizations/data review remain pending.


### P2-00 local-host bootstrap is not real training

`p2_host_doctor` checks Python, packages, source SHA, `reg_max` and optional CUDA. `p2_graph_smoke` constructs ESPDet-Pico from a generated ten-class YAML and computes a single **random-noise mathematical gradient** without pretrained weights, human images, genuine target detection loss, saved checkpoint, mAP or board run. Its success is a host compatibility prerequisite only, not P1 license approval or P2 model training PASS. See [P2_HOST_SETUP.md](P2_HOST_SETUP.md).


**Source-only CPU evidence:** [Actions run](https://github.com/DenisZheng/esp32s3-gesture-recognition/actions/runs/39523052733) completed a real ten-class ESPDet-Pico synthetic forward/backward (not just offline unit tests). Exact Torch/Ultralytics CPU versions and stage shapes are recorded in `upstreams.lock.json`. **No real detection loss, dataset, GPU CUDA or CoreS3 metrics may be inferred.**


### P2-00 separate GTX1060 Docker GPU gate

A prior SCRFD PyTorch1.8/CUDA11.1 image is NOT an ESPDet-Pico-compatible runtime. The separate [GTX1060 Linux Docker workflow](P2_GTX1060_DOCKER.md) must pass a real build and all of `p2_docker_host.sh gpu` (sm_61/6GiB approximate, CUDA12.1 matmul + cuDNN backward), `source` (pinned source), `doctor` (Torch/Ultralytics/CUDA) and `smoke` (random 224px nc10 CUDA graph/backprop) on the owner's GPU. Offline Bash syntax tests are NOT a substitute. Save non-sensitive CUDA evidence locally, do not include HaGRID photos or weights in GitHub. **P1 human gates still block real training.**


### P1-05 automatic/manual full ten-source dataset candidate

The unified builder requires actual private owner research-use approval and all-ten source/gesture mapping review before reading any HaGRID JPEG or original person annotation. Local-dir, local-zip and automatic pinned ZIP Range inputs must yield identical strict per-hand nc10 YOLO labels and separated train/val/test subjects; unknown/unapproved second hands and compound boxes reject the full frame. Candidate output requires requested quotas in every source/split, SHA256 integrity evidence and no unreviewed `no_hand` negatives. Synthetic CI does not prove actual image decode or rights. Final stage still requires private human inspection/QA and **does not unlock training automatically**. [Instructions](P1_DUAL_SOURCE_DATASET.md).
