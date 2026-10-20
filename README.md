# ESP32-S3 Gesture Detection (ESPDet-Pico + HaGRID)

**Research PoC | 10-class, single-model hand localization and gesture recognition | M5Stack CoreS3**

> **Project status: P0 BLOCKED (remote GitHub repository/CI not yet established).** This local repository contains audited technical design, source pins, data-policy safeguards, offline contract tests, and CI configuration. It **does not** contain a trained model, data, a firmware image, or real-device results.

[中文说明](README.zh-CN.md) | [Implementation plan](docs/PLAN.md) | [Technical architecture](docs/ARCHITECTURE.md) | [Upstream compatibility](docs/UPSTREAM_COMPATIBILITY.md) | [Licenses](docs/THIRD_PARTY.md)

## Scope

One *self-trained* ESPDet-Pico detection network (not the Espressif two-model hand-detection + classification pipeline), at 224 x 224, with **10 outputs**: hand bounding box + gesture category + confidence. Target: **ESP32-S3**, 8 MB PSRAM, **ESP-IDF 5.4.4**, **ESP-DL v3.3.13**, and **ESP-PPQ W8A8 / symmetric power-of-two**.

**Immutable class index contract:**

| ID | Label | HaGRID candidate | Mapping status |
|---:|---|---|---|
| 0 | one | one | Pending visual sample audit |
| 1 | two | two_up | **Pending visual confirmation** |
| 2 | three | three | Pending visual sample audit |
| 3 | four | four | Pending visual sample audit |
| 4 | five | palm | **Pending visual confirmation** |
| 5 | like | like | Pending visual sample audit |
| 6 | ok | ok | Pending visual sample audit |
| 7 | no_gesture | no_gesture | Must have hand bbox |
| 8 | call | call | Pending visual sample audit |
| 9 | dislike | dislike | Pending visual sample audit |

`no_hand` is **not** a class. A hand-free image is a negative sample with zero detections. Full multi-hand annotation and disjoint `user_id` groups across splits are mandatory. See [DATA_POLICY](docs/DATA_POLICY.md).

## Pipeline

```text
HaGRIDv2 (licensed images + per-hand annotations)
    -> audited, subject-disjoint YOLO 10-class detection data
    -> ESPDet-Pico training from random initialization (PyTorch, best.pt)
    -> one-input, six-output ONNX
    -> ESP-PPQ PTQ (esp32s3, W8A8, symmetric PoT)
    -> own .espdl artifact
    -> ESP-DL v3.3.13 / IDF 5.4.4
    -> M5Stack CoreS3 fixed-image parity, THEN live camera
```

## Getting started (P0 only, no dataset required)

```bash
python3 scripts/check_contract.py
python3 scripts/check_repo_policy.py
python3 -m unittest discover -s tests -v
```

CI performs the same **offline** P0 checks on pushes and PRs. CI is **not** evidence of model accuracy, quantization parity, device boot, or camera inference. Do not interpret a green CI badge as successful deployment.

When the repository exists on GitHub, create feature branches and PRs against protected `main`. See [GitHub setup](docs/GITHUB_SETUP.md) for the one-time repository and ruleset steps. Upstream repositories are referenced and pinned in [upstreams.lock.json](upstreams.lock.json), **not vendored or copied**.

## Non-goals and release policy

- No official pretrained `.espdl` as a substitute for self-trained weights.
- No `no_hand` 11th output; no reduction of the 10 classes.
- No full HaGRID download during P0 or at P1 kickoff.
- No models/firmware/datasets/personal images in Git or CI logs/artifacts.
- No claim of commercial readiness; upstream AGPL and custom HaGRID license require a separate clearance review.

## Stage gates

| Stage | Goal | Current status |
|---|---|---|
| P0 | Repository, compatibility audit, plan, CI | **BLOCKED**: remote repo + CI + branch ruleset; documents and local validation implemented |
| P1 | Audited HaGRID sample conversion | PENDING |
| P2 | Reproducible scratch training / FP32 eval | PENDING |
| P3 | ONNX parity / W8A8 ESP-PPQ eval | PENDING |
| P4 | CoreS3 fixed image + live camera | PENDING |
| P5 | Compare one-model vs official two-model pipeline | PENDING |

See [Acceptance](docs/ACCEPTANCE.md) for exact evidence definitions.
