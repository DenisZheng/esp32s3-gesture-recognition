# Upstream compatibility audit — pinned code, 2026-10-20

| Source | Pinned revision | Relevant code / observation | Conclusion |
|---|---|---|---|
| ESP-Detection | `28de5902a357d99dd9a5db0f52905255641c5a78` | [`cfg/models/espdet_pico.yaml`](https://github.com/espressif/esp-detection/blob/28de5902a357d99dd9a5db0f52905255641c5a78/cfg/models/espdet_pico.yaml) `nc: 1`; [`nn/modules/esp_head.py`](https://github.com/espressif/esp-detection/blob/28de5902a357d99dd9a5db0f52905255641c5a78/nn/modules/esp_head.py) `ESPDetect(nc)` `cv3` | Ten-class head **structurally supported**; no runtime PASS |
| ESP-Detection | same | [`deploy/export.py`](https://github.com/espressif/esp-detection/blob/28de5902a357d99dd9a5db0f52905255641c5a78/deploy/export.py) exports box0 score0 box1 score1 box2 score2, ONNX opset 13 | Need ten-class output shape test |
| ESP-Detection | same | [`train.py`](https://github.com/espressif/esp-detection/blob/28de5902a357d99dd9a5db0f52905255641c5a78/train.py) `YOLO('cfg/models/espdet_pico.yaml')` if `pretrained_path` absent, but defaults include CPU, 1200 epochs, batch=128 | Scratch path exists; override resource settings; test exact Ultralytics version |
| ESP-Detection | same | [`deploy/eval_quantized_model.py`](https://github.com/espressif/esp-detection/blob/28de5902a357d99dd9a5db0f52905255641c5a78/deploy/eval_quantized_model.py) `NC=1` in PPQ inference, fixed-size reshape and simulated `Detect(nc=NC)` | **Ten-class PPQ evaluation code NOT compatible as-is** |
| ESP-Detection | same | [`deploy/quantize.py`](https://github.com/espressif/esp-detection/blob/28de5902a357d99dd9a5db0f52905255641c5a78/deploy/quantize.py) `target`, 8-bit; calibration ToTensor then Resize (no letterbox) | Quantization export API present; input preprocessing parity open |
| ESP-Detection | same | [`deploy/espdet_model_template/espdet_detect.cpp`](https://github.com/espressif/esp-detection/blob/28de5902a357d99dd9a5db0f52905255641c5a78/deploy/espdet_model_template/espdet_detect.cpp) pads letterbox=114, RGB565 on S3; `top_k=10` | Template needs camera integration and ten-label mapping |
| ESP-DL | `v3.3.13` => `6d3c4da1087d35916858dd0f9b3e9d6a85186e11` | [`dl_detect_espdet_postprocessor.cpp`](https://github.com/espressif/esp-dl/blob/v3.3.13/esp-dl/vision/detect/dl_detect_espdet_postprocessor.cpp) loops `c < score->shape[3]` and writes `category=c` | Multiclass score candidates parsed |
| ESP-DL | same | [`dl_detect_postprocessor.cpp`](https://github.com/espressif/esp-dl/blob/v3.3.13/esp-dl/vision/detect/dl_detect_postprocessor.cpp) IoU suppression has **no `category` equality guard** | NMS class-agnostic; overlapping different-gesture hands risk suppression; adapt/test |
| ESP-DL | same | [`esp-dl/idf_component.yml`](https://github.com/espressif/esp-dl/blob/v3.3.13/esp-dl/idf_component.yml) includes ESP32-S3 and IDF >= 5.3 | IDF 5.4.4 matches declared range, **not tested on CoreS3** |
| ESP-PPQ | `e8881cb9ea5ba18d314fe84c9ca2ab23a27da6dd` | [`esp-ppq`](https://github.com/espressif/esp-ppq) and ESP-DL quantization tutorial | W8A8 sym PoT supported in docs; ten-class runtime unverified |
| HaGRID | code `Hagrid_v2-1M` => `cade4dd4ba162832ef2b62b92d89d4d7a2dc2b75` | [`converters/hagrid_to_yolo.py`](https://github.com/hukenovs/hagrid/blob/Hagrid_v2-1M/converters/hagrid_to_yolo.py) can use `united_bbox` for joint gestures; `labels` and `user_id` in annotations | **Cannot trust converter blindly** for independent-hand classes/splits |
| Official gesture example | ESP-DL tag v3.3.13 | [`hand_gesture_recognition/README.md`](https://github.com/espressif/esp-dl/blob/v3.3.13/examples/hand_gesture_recognition/README.md) explicitly lists exact 10 classes + no_hand state | Names confirmed; `two_up->two`, `palm->five` semantic **visual** audit PENDING |

## Reproducible source pins

Use `upstreams.lock.json` as the exact source list. The Git commit of an upstream code repository does **not** pin downloaded HaGRID image/annotation archives; the selected dataset version, URL, license and SHA256 must be independently recorded in each local manifest. ESP-IDF v5.4.4 annotated tag dereferences to commit `296b6eab9445fd720e71aecab961e2d3fbca9944`.

## Open technical questions / proof gates

- Can this exact ESPDetect commit + selected Ultralytics pin instantiate, train 1 step and export 10-class output on target host? Status PENDING.
- Do 6 output tensors survive PPQ quantization for `esp32s3` with `nc=10`? Status PENDING.
- Can multi-hand NMS avoid merging two overlapping physical hands while still returning one category per hand? Status PENDING.
- Is S3 RGB565 big-endian correct for **CoreS3's actual camera capture path**? Status PENDING.
- Can the unified model fit flash / PSRAM / latency on CoreS3? Status PENDING.
- What exact HaGRID modification obligations apply to research, weights and distribution? Status BLOCKED pending direct `license/en_us.pdf` legal review.

Do not substitute upstream README FPS or a green host CI for the above evidence.
