# Risk register — reviewed 2026-10-20

Scale: severity H/M/L; evidence state DONE = static inspection only unless marked otherwise. No hardware accuracy/performance has been measured.

| ID | Severity | Finding and evidence | Mitigation and owner stage | State |
|---|---|---|---|---|
| R01 | H | ESP-Detection quantized validation code hardcodes `NC=1` | P3 adapter must use 10, assert score channels and compare raw/decoded outputs | DONE static finding; validation PENDING |
| R02 | H | ESP-DL v3.3.14 generic NMS is **category-agnostic** | P4 evaluate two overlapped different gestures; choose class-aware policy or one-hand arbitration; demonstrate no duplicates/suppression errors | DONE static finding; test PENDING |
| R03 | H | HaGRID `united_bbox` may combine two hands; folder name not equivalent to labels for all hands | P1 use `bboxes` and per-hand `labels`; reject ambiguity and audit multi-hand | DONE static finding; actual data PENDING |
| R04 | H | Non-target gestures ignored in training become unlabeled foreground; `no_gesture` undersampled | Map reviewed per-hand noncommands to `no_gesture` or reject; sample natural hand/other pose and nohand controls | PENDING |
| R05 | H | User leakage train vs val/test after class filtering/relabelling | P1 disjoint `user_id`, duplicate media hashes, hold-out fixed before training | PENDING |
| R06 | H | HaGRID uses a **custom non-CC license** (PDF page-1 footnote), with attribution/share-alike requirements; model commercial/portrait/privacy rights unclear | Full five-page PDF source reviewed + SHA256 pinned; require local owner acceptance before using data and counsel before commercial shipping | SOURCE AUDIT DONE; OWNER SIGNOFF / COMMERCIAL CLEARANCE BLOCKED |
| R07 | H | AGPL training/modified-code obligations not resolved by private repo | Source inventory, modifications/notice tracking, counsel/commercial license decision | BLOCKED for commercial |
| R08 | H | Host calibration Resize differs from board letterbox=114 and RGB565 preprocessing | P3 exact RGB, size, pad, normalized tensor numeric check using fixed pixels | PENDING |
| R09 | H | 8MB PSRAM + display/camera frames + 10-class inference can OOM or miss FPS target | P4 app memory map, top-k, p50/p95 and PSRAM high water; build minimal vision app first | PENDING |
| R10 | M | Small/distant fingers disappear at 224x224; nohand false positives | Per-class/scale/occlusion metrics, controlled augment and resolution ablations | PENDING |
| R11 | M | CoreS3 sensor formats/orientation/mirror and ESP-DL RGB565 format differ | P4 pixel parity per actual camera sensor, not on computer-only images | PENDING |
| R12 | M | Upstream `ultralytics>=8.3.112` and ESP-PPQ git unpinned can drift; `ESPDetect` changed signatures | `upstreams.lock.json`; first smoke-test exact Python env then pin dependency versions in tested lock | PENDING |
| R13 | M | GitHub connector cannot create repos or write rulesets | Owner created Public repo, authorized connector and enabled Protect main; verified via API | MITIGATED; remote/CI/ruleset DONE |
| R14 | M | Firmware build or .espdl export mistaken for model verification | Strict evidence ladder: PyTorch -> ONNX -> quantized -> board fixed image -> live camera | Controlled by acceptance plan |
| R15 | M | Model weights/data caches leak through Git, Actions artifacts or logs | `.gitignore` + tracked-file policy script + minimal no-data CI; manual review for screenshots | Guard implemented; GitHub CI PASS; continued manual data hygiene required |
| R16 | M | Off-target model-size/CPU claims inferred from single-class cat 126.2ms | Measure ten-class CoreS3 model; report net latency and end-to-end camera FPS | PENDING |
| R17 | M | P1 source semantic mismatch `two_up->two` and `palm->five` | Before conversion signoff official gesture picture vs independent HaGRID samples | PENDING |
| R18 | M | HaGRID git commit does not identify dataset image/annotation archives | Local SHA256 dataset manifests, clear dataset release and download URLs | PENDING |
| R20 | M | Mistaking official 128x128 ROI classifier size for full-frame one-model input | Freeze 224x224 ESPDet-Pico baseline; only change size after multi-hand, small-hand recall and CoreS3 tests (see `INPUT_RESOLUTION.md`) | DONE architecture decision; P2/P4 performance PENDING |
| R19 | H | CoreS3 GC0308 source camera interface and BSP/esp_video driver not built against IDF 5.4.4 | P4 pin tested BSP, validate capture/format byte order, shared I2C power/reset and PSRAM pressure | PENDING |
