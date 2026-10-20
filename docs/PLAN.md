# Master implementation plan and evidence ledger

Date: 2026-10-20. **Goal: single-model hand gesture recognition with localization.** Project: `DenisZheng/esp32s3-gesture-recognition`. Baseline: CoreS3 / ESP32-S3 / 8MB PSRAM / IDF 5.4.4 / ESP-DL v3.3.14. Stage gate is sequential: do not enter the next phase until the previous stage has a signed, evidence-backed PASS. An outcome is `DONE` only with artifact, command, version and result. `PENDING` is neither PASS nor FAIL.

**Current P0 outcome: GITHUB INFRASTRUCTURE DONE; RESEARCH GATE PARTIAL.** Public repository, real GitHub commits, green Actions and active main Ruleset `Protect main` (PR + `p0-contract`) were verified via GitHub API on 2026-10-20. The owner's Public-visibility choice supersedes the original Private-repo suggestion. HaGRID custom-license PDF, class visual signoff and real samples are still pending. P1 metadata/converter tooling can proceed independently without datasets; P2 training cannot begin until licensed P1 acceptance.

## P0 — Repository and feasibility (current)

| ID | Task | Exit evidence | Status |
|---|---|---|---|
| P0-01 | Create the owner's **Public** independent GitHub repo, no fork | GitHub API repository metadata: public, owned by DenisZheng | DONE |
| P0-02 | Verify upstream commits, v3.3.14 ESP-DL, IDF v5.4.4, ESP-PPQ, HaGRID | `upstreams.lock.json` full SHA and source links | DONE (source metadata only) |
| P0-03 | Validate `ESPDetect(nc=10)` structural interface and six raw outputs | Read `esp_head.py`, model YAML, export script | DONE (source audit); runtime test PENDING |
| P0-04 | Inspect ESP-DL NMS and multi-hand behavior | Read pinned C++ sources, document class-agnostic NMS | DONE (static audit); runtime test PENDING |
| P0-05 | Audit upstream single-class assumptions | Quant validator `NC=1`, cat-only templates, device labeling | DONE (static audit) |
| P0-06 | Define 10 exact classes + ignored other-gesture policy + leakage contract | Class JSON, data policy and unit tests | DONE (contract); actual HaGRID audit PENDING |
| P0-07 | Design repo / evidence / gitignore / CI | Source tree, workflow, PR template; GitHub Actions passing | DONE |
| P0-08 | Source/license inventory | `THIRD_PARTY.md` preliminary inventory | DONE (inventory); license clearance BLOCKED |
| P0-09 | Initial remote commits, review flow and protected `main` | GitHub SHA, successful Actions, ruleset id 25458714 Active; PR, CI, linear history | DONE |
| P0-10 | Resolve gesture reference shapes `two_up/two`, `palm/five` | Comparison sheet with official gesture figure and independent samples | PENDING (before P1 conversion signoff) |
| P0-11 | Correct application naming, choose single-model input resolution, update latest stable ESP-DL lock | [INPUT_RESOLUTION](INPUT_RESOLUTION.md), `README`, `upstreams.lock.json` | DONE (source/config review); hardware PENDING |

**P0 remaining exit gates:** Data license rights from actual custom PDF and visual confirmation of ten HaGRID-to-official classes. Public remote, main commits, green GitHub Actions and protected PR + mandatory check are **VERIFIED**. P1 **tooling** implementation uses synthetic tests only and does not constitute licensed data acceptance. Do not declare P0 fully PASS or enter P2 until the outstanding human evidence gates are signed.

The frozen start-of-P0 latest stable ESP-DL release is v3.3.14, with a full Git SHA. Before upgrading again, file a compatibility-review PR against the updated upstream release. Training and image-shape selection remain experimental; [224x224 rationale](INPUT_RESOLUTION.md) is reviewed at P0, not yet validated on device.

## P1 — HaGRIDv2 minimal dataset PoC (TOOLING PARTIAL; real data BLOCKED)

1. **P1-01 License/provenance:** read `license/en_us.pdf` and record permitted research/redistribution/model usage; keep private annotation and subject data; record HaGRIDv2 `Hagrid_v2-1M` source branch and annotation archive version/hash, never assume git SHA identifies downloaded dataset bytes.
2. **P1-02 Counts/metadata:** download *annotation metadata only if access permits*; inspect split JSON schema, `user_id`, per-hand `bboxes` and `labels`, `united_bbox`/`united_label`. Make counts by source class, distinct user_id, multi-hand cases, `no_gesture`, unreadable media. Generate metadata-only manifest (local, ignored).
3. **P1-03 Visual class signoff:** compare official ESP-DL `examples/hand_gesture_recognition/gestures.png` with licensed HaGRID images for all 10 classes; specifically verify `two_up -> two` and `palm -> five`; exclude any semantically mismatched source rather than silently re-labeling.
4. **P1-04 Controlled micro-sample:** choose small *licensed* images by user_id after license gate; never download the full 1.5TB; prefer selected images and preserve original annotations; if archive access is coarse, document download cost and propose approved sampling.
5. **P1-05 Complete per-hand boxes:** use HaGRID `hagrid_to_yolo.py` only as a reference. Review whether `--mode gestures` applies `united_bbox` to joint/two-hand labels; implement audited converter that emits one YOLO row per **physical hand**, with coherent unknown-gesture policy. Never trust source folder names as labels for all hands.
6. **P1-06 Split policy:** train/val/test `user_id` sets pairwise disjoint. If canonical splits fail this after filtering/merge, rebuild with deterministic subject-level seed, preserve held-out test and document differences. No subject ID in public logs or issues.
7. **P1-07 Validate:** each YOLO row exactly `cls cx cy w h`, normalized to [0,1], strictly positive box within image bounds, id 0-9; zero-line label only for truly audited hand-free images. Review random stratified annotated montages privately, manually sign off multiple hands, natural poses, occlusions and ignored other gestures.
8. **P1-08 Manifest/evidence:** counts per split/class/subjects, duplicate-file SHA256 checks (within accessible subset), source license, sample selection method, conversion hash, rejects and QA log. Only share aggregate anonymized metrics.

**P1 tooling (PR):** `scripts/hagrid_p1.py`, `tests/test_hagrid_p1.py` and [P1 data runbook](P1_DATA_TOOLING.md) implement offline synthetic-tested metadata audit + guarded micro-sampling. No real HaGRID bytes were acquired. **P1 PASS remains blocked** by exact license PDF, signed visual mapping, real data and private visual QA. See runbook for restrictions and evidence. No mAP required yet.

## P2 — Scratch PyTorch training (PENDING)

- **P2-01** Verify pinned ESP-Detection and test `nc=10` graph construction / gradient step with resolved Ultralytics+Torch versions. Train from *random initialization*, no secret/implicit pretrained checkpoints. Record source snapshots and `pip freeze`.
- **P2-02** Configure `224x224`, random seed, optimizer, schedule, epochs, augmentation, batch, data split, resize/letterbox. Validate that any transfer/rect config is intentional. Create one reproducible command and experiment ID.
- **P2-03** Train scratch baseline, produce `best.pt` **locally only**. Persist model hash, epoch/loss and config hash; do not treat a pilot run as the baseline.
- **P2-04** FP32 held-out evaluation: mAP50, mAP50-95, Precision, Recall, per-class AP/recall, 10x10 confusion (plus background handling), false positives on `no_hand`, multiple hands, overlaps and subtle poses; commit aggregate metrics only.
- **P2-05** Compare at least two seeds if capacity permits; log nondeterministic CUDA flags and environment, not claim bit-identical GPU results.

**P2 PASS:** traceable scratch training with `best.pt` SHA256 and end-to-end FP32 validation/plots; class/box quality reported. Quantization and board deployment remain PENDING.

## P3 — ONNX + ESP-PPQ W8A8 (PENDING)

- **P3-01** Export single `images` input and 6 deterministic `box{i}/score{i}` outputs using pinned export code; validate ONNX checker, shapes, dynamic=false, opset=13, `nc=10`.
- **P3-02** Fixed-image PyTorch vs ONNX: align preprocessing and head output tensors, decode boxes, compare class identities and matched boxes with documented numerical tolerances and representative edge cases.
- **P3-03** Build calibration set from *training users only* that includes ten classes, no-hand, multiple hands, clutter, exposure extremes. Match camera RGB/BGR, RGB565 handling, letterbox pad 114, spatial order, normalization and size exactly. Note upstream `CaliDataset` uses ToTensor+Resize, while MCU model template uses letterbox: adapt and validate, do not copy blindly.
- **P3-04** ESP-PPQ PTQ `target='esp32s3'`, `W8A8`, symmetric PoT, original source SHA. Check calibration steps vs actual number of batches, 8-bit operator policy / unsupported operators and model size.
- **P3-05** Replace hardcoded `NC=1` in *our local validation adapter*, not in upstream source. Quantized raw `scores` must have C=10 at each stage and 6 output names must survive.
- **P3-06** Compare PyTorch/ONNX/quantized `score`, `bbox`, class-aware or class-agnostic NMS behavior, mAP50/mAP50-95 degradation; investigate sensitive layers if regression is unacceptable.

**P3 PASS:** OWN `best.pt -> ONNX -> own .espdl` recorded by hashes and config, fixed-image parity and quantitative accuracy report. Successful export alone is NOT PASS.

## P4 — Real CoreS3 (PENDING; real hardware mandatory)

- **P4-01** Implement minimal IDF v5.4.4 app / M5Stack CoreS3 config (GC0308 sensor; BSP/esp_video pin after smoke test) with locked ESP-DL v3.3.14; camera and PSRAM initialization, image format, resolution/stride, rotation/mirror documented. App consumes generated `.espdl`, not official weights.
- **P4-02** Implement ten-name lookup and maximum box count; decode 3 stages, compare options for class-agnostic vs class-aware NMS (and one-class-per-hand arbitration). Test two hands same/different gestures and overlaps; preserve access to per-hand confidence.
- **P4-03** Fixed-image parity FIRST, compare preprocessed tensor checksums and decoded boxes/labels/scores against PC, define tolerances for quantization and pixel coordinate rounding.
- **P4-04** Real live camera stream, at least all 10 gestures, `no_hand`, idle natural postures, other gestures, two hands, overlapping hands, occlusion, multiple scales, lighting. Measure p50/p95 full pipeline latency, model-only latency, FPS, internal heap/PSRAM peak/free and 30-minute crash-free capture session; raw images stay private.
- **P4-05** Negative-trigger rate, false-positive/false-negative examples, reproducible serial logs with no personal data, demo evidence under controlled access. Compare actual metrics against P2/P3 and record board/firmware hashes.

**P4 PASS:** real board camera tests with verified self-trained model identity, no untested assumption that a built firmware is a working vision system.

## P5 — Analysis and optimization (PENDING)

- Same board, camera, dataset and scenarios: compare own one-model pipeline with official detect+classify two-model baseline without claiming their weights as ours.
- Compare real inference latency, camera-to-result p50/p95, FPS, model/flash/heap footprint, multi-hand behavior, class confusion and UX false trigger rate.
- Experiment with input resolution, sample strategy, NMS and quant-sensitive-head options one change at a time; publish reproducible aggregate report and commercial/legal stop/go recommendation.

## Project conventions

- Prefix branches `feat/pN-task-id`, `fix/` or `docs/`. One task per PR; never merge red CI.
- `main`: require PR, passing `p0-contract`, prevent force push/deletion, resolve discussions; do not allow workflow to override P2-P4 human evidence.
- Every experimental artifact has `experiment_id`, upstream SHAs, code SHA, subject-safe data manifest SHA, seed, tool versions, exact command, metrics, `best.pt`/ONNX/ESP-DL SHA256, license statement, and hardware hash when relevant.
- Status semantics: DONE = independently verifiable evidence; FAIL = attempted and failed with diagnosis; BLOCKED = missing data/device/permission; PENDING = not attempted.
