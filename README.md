# ESP32-S3 Hand Gesture Recognition (ESPDet-Pico + HaGRID)

**Research PoC | Single-model ten-class hand gesture recognition with localization | M5Stack CoreS3**

> **Project status: GitHub P0 infrastructure COMPLETE; HaGRID custom PDF source has been read and pinned, but owner/legal usage acceptance and 10-class visual signoff remain pending. P1 selective metadata tools are synthetic-tested; no private data or real training is claimed.** The repository contains audited technical design, source pins, data-policy safeguards, offline contract tests, and CI configuration. It **does not** contain a trained model, data, a firmware image, or real-device results.

[中文说明](README.zh-CN.md) | [Implementation plan](docs/PLAN.md) | [Technical architecture](docs/ARCHITECTURE.md) | [Upstream compatibility](docs/UPSTREAM_COMPATIBILITY.md) | [CoreS3 board baseline](docs/BOARD_BASELINE.md) | [Input size decision](docs/INPUT_RESOLUTION.md) | [Licenses](docs/THIRD_PARTY.md) | [P1 data tooling](docs/P1_DATA_TOOLING.md) | [Custom dataset license review](docs/HAGRID_LICENSE_AUDIT.md)

## Scope

One *self-trained* ESPDet-Pico **gesture-recognition network built on a detection architecture** (not the Espressif two-model detection + classification pipeline), at **224 x 224**, with **10 gesture classes** and per-hand bounding boxes + confidences. Target: **ESP32-S3**, 8 MB PSRAM, **ESP-IDF 5.4.4**, **ESP-DL v3.3.14**, and **ESP-PPQ W8A8 / symmetric power-of-two**.

**Why 224x224 instead of 128x128?** In the official two-model system, ESPDet-Pico first locates hands in full camera frames at **224x224**, and MobileNetV2 then classifies **already-cropped hands at 128x128**. Our single ESPDet-Pico sees the full frame and must both locate and identify gestures, so the **224x224 detector baseline** is the relevant starting point. Lower-resolution tests (160x160 / 192x192) may be evaluated later, but not quietly substituted for the P2 baseline. See [input-resolution decision](docs/INPUT_RESOLUTION.md).

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
    -> ESP-DL v3.3.14 / IDF 5.4.4
    -> M5Stack CoreS3 fixed-image parity, THEN live camera
```

## Getting started (P0 only, no dataset required)

```bash
python3 scripts/check_contract.py
python3 scripts/check_repo_policy.py
python3 -m unittest discover -s tests -v
```

CI performs the same **offline** P0 checks on pushes and PRs. CI is **not** evidence of model accuracy, quantization parity, device boot, or camera inference. Do not interpret a green CI badge as successful deployment.

GitHub repo and `Protect main` ruleset are active; use feature branches and PRs (mandatory `p0-contract` job). See [GitHub setup](docs/GITHUB_SETUP.md). For synthetic-tested P1 data tooling see [P1_DATA_TOOLING](docs/P1_DATA_TOOLING.md). Upstream repositories are referenced and pinned in [upstreams.lock.json](upstreams.lock.json), **not vendored or copied**.

## Non-goals and release policy

- No official pretrained `.espdl` as a substitute for self-trained weights.
- No `no_hand` 11th output; no reduction of the 10 classes.
- No full HaGRID download during P0 or at P1 kickoff.
- No models/firmware/datasets/personal images in Git or CI logs/artifacts.
- No claim of commercial readiness; upstream AGPL and custom HaGRID license require a separate clearance review.

## Stage gates

| Stage | Goal | Current status |
|---|---|---|
| P0 | Repository, compatibility audit, plan, CI | GitHub/CI/ruleset **DONE**; license review and mapping visual signoff pending |
| P1 | Audited HaGRID sample conversion | **PARTIAL**: offline metadata/converter scripts + synthetic CI; real licensed data and mapping pending |
| P2 | Reproducible scratch training / FP32 eval | PENDING |
| P3 | ONNX parity / W8A8 ESP-PPQ eval | PENDING |
| P4 | CoreS3 fixed image + live camera | PENDING |
| P5 | Compare one-model vs official two-model pipeline | PENDING |

See [Acceptance](docs/ACCEPTANCE.md) for exact evidence definitions.
