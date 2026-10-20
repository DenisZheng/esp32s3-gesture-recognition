# Architecture and interface contracts

## System boundary

The PoC produces **one trainable object detector**. Its ten class logits express nine commands and a visible-hand rejection class (`no_gesture`); the tenth channel is NOT a second model. On no-hand frames, detector should return zero accepted boxes rather than an eleventh `no_hand` class. Firmware only performs image conversion, inference, postprocessing and optional temporal debouncing.

```text
                 Offline training/PC                                      ESP32-S3 runtime
HaGRIDv2 per-hand annotations                                    CoreS3 camera -> RGB frame
    | user-id split / bbox audit / semantics gate                         |
    v                                                                       v
YOLO 10-class detection dataset                                  deterministic RGB565/RGB888
    |                                                                       |
    v                                                                       v
ESPDet-Pico YAML + nc=10 -> scratch PyTorch FP32                  letterbox 224x224 / pad=114
    |                             |                                          | RGB [0,1] normalization
    | FP32 validation             v                                          v
    |                        best.pt -> 6-output ONNX -> ESP-PPQ -> own .espdl
    |                                                  W8A8 / esp32s3     |
    |                                                                      v
    |                                        six output tensors -> candidate decode
    |                                                                      |
    +--------------- PC raw output / box and score parity <---------------+
                                                                         |
                                             NMS + one-label/hand -> boxes + score + class
```

## Raw model interface, to verify in runtime

- `ESPDet(nc=10)` uses three YOLO strides 8,16,32; reference model defines reg_max=1, 4 raw box values, and `nc` raw scores per feature cell.
- For 224x224 input, expected nominal grids are 28x28, 14x14, 7x7. Under *PyTorch NCHW* export convention: `box{i}=(1,4,H_i,W_i)`, `score{i}=(1,10,H_i,W_i)`; six output names ordered `box0,score0,box1,score1,box2,score2` (check ONNX graph in P3 rather than assume).
- ESP-DL inference/postprocessor uses NHWC-like `score->shape[1],shape[2],shape[3]` with class loop `for c in C`; test converted `.espdl` actual tensor layout and scales. `box` is class-independent per location, and scores are logits thresholded by inverse sigmoid before conversion to probabilities.
- Each accepted observation is `(class_id 0..9, confidence, x1,y1,x2,y2)` in original image coordinates. Apply inverse letterbox and clamp coordinates. A camera orientation flip requires annotation/evaluation alignment.
- Across frame: no accepted detections means `no_hand` *state*, not model label; `no_gesture` is positive hand box with ID=7; multiple hands imply multiple results, never classify the entire image from one folder label.

## Official model compatibility findings

- ESP-Detection `cfg/models/espdet_pico.yaml` defaults `nc=1`; its `nn/modules/esp_head.py` inherits Ultralytics Detect and its `cv3` branch produces `nc` score channels. Thus graph-level multi-class is plausible, **not yet measured**.
- ESP-Detection `deploy/export.py` explicitly exports 6 raw outputs; `deploy/quantize.py` can target `esp32s3`; `deploy/eval_quantized_model.py` has hard-coded `NC=1`, so do not use its output as a ten-class correctness certificate.
- ESP-DL v3.3.13 `ESPDetPostProcessor::parse_stage` iterates score channels and records `category=c`, so **multi-class candidates are parsed**.
- ESP-DL v3.3.13 `DetectPostprocessor::nms` compares IoU without comparing categories, i.e. **class-agnostic suppression**. This can intentionally arbitrate conflicting labels for one hand, but may suppress two overlapping hands with distinct gestures. Need controlled tests of same/different classes, adjacent versus occluded hands and score ties; optionally write a small, reviewed adapter overriding choice of NMS policy rather than patching upstream in-place.
- C++ default `m_top_k=10` in the ESPDet template caps reported detections; determine board perf/memory vs candidate count; do not assume 10 is adequate for every two-hand test when duplicates exist before NMS.
- Upstream C++ template invokes RGB565 big-endian `ImagePreprocessor` (S3), enables letterbox pad=114, while upstream Python calibration `ToTensor()->Resize()` does not use letterbox. Align in P3 and build fixed-image pixel tensor parity.

## Run-time memory and latency

Memory on 8MB PSRAM is shared by camera frame buffers, model arena, scratch tensors and display stack; flash partition also needs a realistic model/storage budget. Single-class cat reference on S3 reportedly 126.2ms at 224, which is *not* a ten-class target or a CoreS3 FPS guarantee. Measure separately: capture, colorspace conversion, preprocessing, model run, postprocessing, display and total camera-to-result p50/p95; also measure PSRAM high-water and free heap during multi-hand frames.

## P3/P4 acceptance parity harness design

1. Offline fixture: licensed or synthetic local image with known RGB pixel data, SHA256, explicit orientation, camera format and frame size (fixture image itself not tracked if license/private).
2. Export PC preprocessed float tensor and raw ONNX stages (metadata and hashed files), compare with model input/output after quantization; recorded tolerance must acknowledge integer scales and rounding.
3. Decode all stage candidates with class IDs, not just argmax for the whole image; test before/after NMS and the top-k limit. Compare box IoU, confidence difference, category, and number of hands.
4. Run the same fixed pixels on board and then switch to live video only after parity gate passes.

Sources: exact files, line audit and commit links in [UPSTREAM_COMPATIBILITY.md](UPSTREAM_COMPATIBILITY.md).
