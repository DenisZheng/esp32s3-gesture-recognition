# Input resolution decision - single-model ten-class gesture recognition

Decision ID: `P0-IMG-001` | 2026-10-20 | **224 x 224 x 3 baseline** | Reasoning DONE; hardware/accuracy validation PENDING.

## Goal first: gesture recognition, not merely hand detection

This project is **hand gesture recognition** on the M5Stack CoreS3. We train one ESPDet-Pico object-detection network end to end with `nc=10` so that each hand prediction returns **(bbox, gesture class, confidence)**. The bounding box is an intermediate localization result exposed to the application; there is **no second classifier, no cropped-hand MobileNetV2 inference, and no pretrained official `.espdl` substituted**.

## Why official models have two image sizes

| Official ESP-DL v3.3.14 component | Model | Input | Data seen by model | Role |
|---|---|---|---|---|
| [hand_detect/README.md](https://github.com/espressif/esp-dl/blob/v3.3.14/models/hand_detect/README.md) | ESPDet-Pico | **224 x 224 x 3** | Whole camera scene, letterboxed | Find hand bounding boxes |
| [hand_gesture_recognition/README.md](https://github.com/espressif/esp-dl/blob/v3.3.14/models/hand_gesture_recognition/README.md) | MobileNetV2-0.5 | **128 x 128 x 3** | **Already cropped** hand ROI | Classify a hand |
| **Our new model** | ESPDet-Pico, `nc=10` | **224 x 224 x 3** | Whole camera scene, letterboxed | **Recognize gesture + locate each hand simultaneously** |

The official [hand gesture recognition example](https://github.com/espressif/esp-dl/blob/v3.3.14/examples/hand_gesture_recognition/README.md) uses both models. Source [hand_gesture_recognition.cpp](https://github.com/espressif/esp-dl/blob/v3.3.14/models/hand_gesture_recognition/hand_gesture_recognition.cpp) calls classification on *each detection crop* (`run_crop`); `128` is **not** a demonstrated resolution for full-scene detection with the ESPDet-Pico 10-class head.

## Selection and measurable hypotheses

- **P2/P3/P4 default**: one 224x224 input, original RGB frame via deterministic letterbox with pad 114; class count 10. Train/validation/calibration/firmware agree on crop, color, padding, normalization and bbox inversion. Always record exact preprocessed tensor shape and hash.
- **Why 224**: preserves more hand/finger detail when multiple hands appear, small/distant fingers exist, or hand occupies only part of frame. It follows the reference detector input for the full scene while retaining shared features for detection and classification. This is an *engineering hypothesis*, not verified superiority for our dataset or device.
- **128 risk**: reducing whole-frame 224 to 128 gives only `128^2/224^2 = 0.3265` as many pixels; fingers shrink, anchor grids become coarser, and multi-hand/overlap recall may degrade. The smaller total pixel count is not a proven linear model speed improvement. The official 128 classifier avoids this issue by first cropping the hand using a separate detector, which our project explicitly forbids.
- **Optional P5 studies**: 160x160, 192x192, 224x224 (and 128x128 only as a *separately trained detector ablation* if warranted). All must keep `nc=10`, one model, independent hand boxes and equal evaluation data. Measure per-class mAP/recall, no-hand false positives, small-hand recall, dual-hand correctness, p50/p95 inference/end-to-end latency, heap+PSRAM and wall power if available. Never replace the P2 baseline silently.
- **No unsupported throughput claim**: official v3.3.14 model documentation reports S3 hand detection model time **123.6ms** at 224 and MobileNetV2 classification model time **115.4ms** at 128 on *different model tasks*. These do not predict the speed of the newly trained 10-class model. The final latency must come from real CoreS3 measurements.

## Gate

This *architecture decision* is DONE (source inspection). Ten-class FP32 model quality, quantized compatibility and actual device latency remain PENDING. A request to change the input resolution must update `configs/classes.json` (single source of truth for one-model and 224x224 image contract), experiment configuration, model/export shape assertions, calibration images and firmware preprocess at the same time via a reviewable PR with quality/latency evidence.
