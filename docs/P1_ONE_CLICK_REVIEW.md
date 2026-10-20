# P1-03 / P1-04 — One-command private four-way gesture review

**Delivery state: code implemented + offline synthetic integration tests; authorized real HaGRID images have NOT been acquired.**

This is the *local handoff* for comparing ESP-DL's official `two` pose with HaGRIDv2 `two_up`, `peace`, `peace_inverted` and `two_up_inverted`. It integrates tools that previously required six or more separate commands:

- Official annotation ZIP: byte-ranged, **train-only** JSON selection
- Four official ZIP64 JPEG archives: exact-index-hash-locked, bounded image extraction
- Train subject-distinct sampling, **3–20 photos per source** (default 6)
- HTML gallery showing **each original physical-hand bounding box and class**
- SHA256-bound private human review template and structural verification

The ESP32-S3 **single ESPDet-Pico model remains 224×224, exactly 10 output classes**. None of this is training, model export, quantization or hardware validation. **The name of the source folder is NOT proof of the output gesture semantics.**

## 0. Preview everything OFFLINE (no license record or network needed)

From the repository root:

    python3 -m scripts.p1_review_workflow plan

This reports the four official source categories, pinned archive sizes and default transfer budgets. `plan` **does not acquire annotations, thumbnails, JPGs, subject IDs, or model weights**. It works in CI.

## 1. First obtain authorized research-use signoff

The dataset has a custom **non-Creative-Commons** license, reviewed in [HAGRID_LICENSE_AUDIT.md](HAGRID_LICENSE_AUDIT.md). It is NOT standard CC BY-SA. The original official PDF SHA256 is pinned in code.

An authorized human reviewer/owner **must actually read and accept applicable terms for the proposed research use** and keep a locally filled JSON record at:

    data/reviews/hagrid-license-ack.json

Start from [docs/templates/p1-license-ack.example.json](templates/p1-license-ack.example.json), but do **not** simply change `accept_terms` to true without reviewing actual rights (including attribution, share-alike/modified data, privacy/portrait rights). The model-weights and commercial rights questions remain open. No assistant, Codex, template, or CI job can approve on behalf of a human.

Signed reviewer information, original annotation JSON, JPGs and derived HTML **must stay in ignored `data/` or another private folder**, not in this Public GitHub repo, GitHub Issues, or CI artifacts.

## 2. Prepare the whole number-two comparison review, on a local machine ONLY

After real owner permission:

    python3 -m scripts.p1_review_workflow prepare \
      --license-ack data/reviews/hagrid-license-ack.json \
      --output data/reviews/p1-two-comparison-v1 \
      --per-source 6 \
      --annotation-budget-mib 80 \
      --jpg-budget-mib 80 \
      --seed p1-two-review-v1

The default annotation byte budget is **80 MiB TOTAL** for four *train-only* class JSONs. The JPEG byte budget is **80 MiB PER CLASS**, including metadata overhead. Inspect specific costs using `scripts.hagrid_zip_range inspect` first if source sizes have changed. The IMG source archives are **over 40 GB each**, but are *never downloaded in full*. The reader obtains only the 64KiB ZIP64 footer, ~4MB central index and actual selected JPEG byte ranges.

**Fail-closed and transactional:** no workspace directory is published until all four sources, their photo hashes, private annotation JSONs and the gallery have been prepared. If any subtask fails, the temporary staging directory is removed; simply choose a fresh output directory for a retry. It will not overwrite prior signed evidence.

### Resulting local layout

    data/reviews/p1-two-comparison-v1/
      annotations/train/{two_up,peace,peace_inverted,two_up_inverted}.json
      images/{two_up,peace,peace_inverted,two_up_inverted}/<original_key>.jpg
      images/<source>/_provenance.private.json
      packet/index.html
      packet/images/<source>-001.jpg ...
      packet/packet.manifest.json
      packet/review.template.json
      workflow.private.json

The original image keys, normalized box values, file hashes and private subject data reside **only in the ignored local workspace**. The aggregate command output contains no raw user_id or annotation demographic attributes.

Open this HTML file locally:

    data/reviews/p1-two-comparison-v1/packet/index.html

The official ESP-DL \`two\` appears to be a separated V, while HaGRID \`two_up\` may be closer/parallel; \`peace\` may be closer to the V pose. This observation is **not** an approved mapping. Do **not** assign \`peace\` to class-7 \`no_gesture\` until licensed image review resolves the ambiguity.

## 3. The reviewer judges images; software verifies only the evidence structure

After personally checking the original images and per-hand boxes, **copy** `packet/review.template.json` into another ignored local path (e.g. `data/reviews/p1-two-v1-signed.json`). Complete real:

- `reviewer`, `reviewed_on`
- `reviewed_packet_ids` for each source (at least three actual photos for positive primary mapping)
- `official_reference_checked=true` after verifying the ESP-DL official gestures illustration
- `comparative_notes` explaining finger separation, handedness, orientation, occlusion and potential model command confusion
- `decision` as `match`, `mismatch`, or `uncertain` for **each** of the four source labels

**Meaning of the choices:** for the primary \`two_up\` source, `match` = matches the existing tentative \`two\` output. For contrast \`peace\` etc, `match` = also matches the *official \`two\` command*, i.e. **potential conflict**; it does not approve use as a negative.

Then, **locally**:

    python3 -m scripts.p1_review_workflow verify \
      --workspace data/reviews/p1-two-comparison-v1 \
      --review data/reviews/p1-two-v1-signed.json

Verification binds decisions to the SHA256 of the packet manifest, the individual JPEG hashes and the claimed original output names. It reports unresolved source mappings and possible \`two\` alias conflicts. **It does not validate what the hand looks like, establish rights, digitally sign identity, or approve all ten categories.**

### Stage-gate boundary

- A structurally valid **four-source** signed packet is still **not P1 PASS**.
- The other nine official output classes, variant contrasts, hand-free negatives, subject-disjoint train/val/test splits, privacy/portrait clearance and full JPEG/label QA remain to be accepted locally.
- If the actual pictures favor `peace` rather than `two_up` for output \`two\`, open an explicit mapping-change PR, retain output ID 1 and regenerate labels. Never auto-relabel \`peace\` as \`no_gesture\`.
- **P2 training is not started** until P1 has the evidence-backed, user-approved final PASS.

## Safety guarantees validated with synthetic CI

- `plan` requires neither a license file nor any media access.
- `prepare` requires a *real* locally signed license acknowledgement and refuses execution if `CI=true` or `GITHUB_ACTIONS=true`.
- Standalone annotation *and* image extractors independently refuse real data acquisition in CI; metadata-only index inspections may still run.
- Byte budgets, exact official ZIP64 SHA256 baseline, train-only record selection, per-hand labels and distinct subjects are enforced by the underlying source readers.
- Directory rollback on missing classes, budget errors, missing license, malformed output, duplicate/symlink inputs, wrong reviewer evidence, or a failed subcommand.
- The synthetic integration tests deliberately include fictional IDs and fake JPEG-like bytes. **They are not genuine licensed photos or visual evaluation.**


### Actual four-source annotation ZIP directory cost (metadata-only check)

On 2026-10-20 a public-metadata-only GitHub Actions run examined the official annotation ZIP's index for `train/{two_up,peace,peace_inverted,two_up_inverted}.json`:

- **4** JSON members, **54,388,405 compressed bytes** (~51.9 MiB)
- **323,899,528 uncompressed bytes** (~308.9 MiB)
- Default **80 MiB total annotation network budget** is above the exact compressed bytes, but you need additional local disk/RAM capacity to parse the JSON.
- [Metadata-only Actions verification](https://github.com/DenisZheng/esp32s3-gesture-recognition/actions/runs/39429491488)

These figures describe ZIP member metadata and do **not** mean we obtained person images/annotation records, accepted image rights, or completed source mapping.
