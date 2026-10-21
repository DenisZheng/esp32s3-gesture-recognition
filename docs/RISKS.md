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
| R17 | H | ESP-DL `two` is V-shape, HaGRID `two_up` is more parallel; `peace` may align better. Palm/stop and three variants also confusable | Quarantine `peace`, `stop` etc from negative remap, compare licensed real samples, sign approved mapping with full evidence | FIGURE AUDIT DONE; REAL CLASS SIGNOFF PENDING |
| R18 | M | HaGRID git commit does not identify dataset image/annotation archives | Local SHA256 dataset manifests, clear dataset release and download URLs | PENDING |
| R20 | M | Mistaking official 128x128 ROI classifier size for full-frame one-model input | Freeze 224x224 ESPDet-Pico baseline; only change size after multi-hand, small-hand recall and CoreS3 tests (see `INPUT_RESOLUTION.md`) | DONE architecture decision; P2/P4 performance PENDING |
| R19 | H | CoreS3 GC0308 source camera interface and BSP/esp_video driver not built against IDF 5.4.4 | P4 pin tested BSP, validate capture/format byte order, shared I2C power/reset and PSRAM pressure | PENDING |

| R21 | H | Blindly using all remaining 24 HaGRID sources as class-7 negatives poisons near-matching true commands and mislabels two-hand compounds | Explicit 10/7/10/7 source policy, signed local hard-negative reviews, compound/lookalike quarantine, fail-closed entire-image rejection | CODE + SYNTHETIC TESTS DONE; REAL EVIDENCE PENDING |

| R22 | H | HaGRID JPEG ZIPs are 40+ GB per source; ZIP64 offsets, changing upstream bytes, ID leakage and unlicensed media downloads | Range-only inspector and local gate; all four live ZIP64 index smoke PASS, archive sizes/index SHA256 pinned, 23 synthetic parser/security tests; CRC/member SHA256; real JPEG and license QA pending | ZIP64 METADATA VERIFIED; REAL IMAGES PENDING |

| R23 | H | Fragmented P1 data commands could leave incomplete local sample directories, while older annotation extractor had no explicit CI download refusal | New transactional four-source local workflow, independent annotation+JPEG CI download bans, signed-license check before I/O, and offline integration regression tests | SOURCE + SYNTHETIC CI DONE; HUMAN AUTHORIZATION AND REAL IMAGES PENDING |

| R24 | H | Pinned ESP-Detection `requirements.txt` accepts Ultralytics v8.3.x while ESPDetect calls parent `Detect(..., reg_max=1)`, absent in 8.3.112 and 8.3.250 but present in v8.4.0; Torch CUDA versions may also vary by GPU | Select source-audited Ultralytics 8.4.0 *candidate*, require exact source SHA/constructor checks and real synthetic CPU/GPU backprop before freezing dependency lock; no P1 bypass | VERSION MISMATCH VERIFIED; COMPLETE RUNTIME / OWNER GPU NOT YET ACCEPTED |


**R24 update (2026-10-21):** The Ultralytics 8.4.0 candidate with pinned Torch 2.2.0 **passed an actual CPU synthetic forward/backward** for ESPDet-Pico `nc=10` with custom classification gradients. CUDA GPU on owner's machine and real data/loss remain open; do not call R24 fully closed. [CPU evidence](https://github.com/DenisZheng/esp32s3-gesture-recognition/actions/runs/39523052733).

| R25 | M | Reusing the SCRFD PyTorch1.8/CUDA11.1/MMCV Docker image for ESPDet-Pico could yield silent Torch/Ultralytics incompatibility or overwrite old checkpoints; CUDA13 excludes Pascal sm_61 | Separate pinned PyTorch2.2/CUDA12.1 Linux/amd64 Docker image, minimal context, 2GiB shm, host-owned mounts, sm61 real-kernel GPU + nc10 graph smokes; explicit no-train helper | DOCKER SOURCE/OFFLINE CI DONE; OWNER CUDA PASS PENDING |

| R26 | H | Manually downloaded ZIP may contain traversal/symlink/encrypted entries, and source `no_gesture` is a special 493.9MB ZIP32 root-JPEG archive at a different official URL | Restricted ZIP reads, exact canonical stems, no extractall, official HTTP index byte-length/SHA256 pins for ten sources, per-member size/CRC, cached resume, SHA256 labels and fail-closed atomic dataset output | METADATA + SYNTHETIC CODE PASS; REAL LICENSED JPG DOWNLOAD/DECODE PENDING |
| R27 | H | Inadvertently treating a 10-folder dataset as fully labeled when second hands, gesture aliases or subject leakage exist | Signed license + all-ten mapping, strict whole-frame rejection, original subject-disjoint audit, class-7 visible-hand box, no automatic no_hand, private human QA mandatory | SOURCE GUARDS IMPLEMENTED; REAL DATA QA PENDING |
