# P1 data preparation — local-only, fail-closed micro-sample

**Status (2026-10-20): TOOLING IMPLEMENTED / SYNTHETIC TESTS ONLY; P1 NOT PASSED.**

## Verified HaGRIDv2 annotation facts

- Source branch: [HaGRIDv2 `Hagrid_v2-1M`](https://github.com/hukenovs/hagrid/tree/Hagrid_v2-1M), pinned in `upstreams.lock.json`.
- Official dataset README reports **1,086,158 images**, ~**1.5 TB**, with train/val/test split by `user_id`. **Do not download the full dataset for this PoC.**
- The separately hosted `annotations.zip` is not part of the Git repository; its **actual byte SHA256 must be recorded locally**. A Git SHA does not pin annotation-archive bytes.
- Each split has `<source>.json`, a JSON object indexed by filename stem. Each record may contain normalized `bboxes`, parallel `labels`, `user_id`, `united_bbox`, `united_label`, `hand_landmarks` and sensitive `meta`.
- `united_bbox` can describe a **two-hand combined gesture**, not a physical hand. This tool rejects any record with a nonempty union box/label and never converts it as a single-hand detection.
- A hand with `no_gesture` is still labeled with an actual box. Empty YOLO labels are **not generated** by this P1 micro-sample tool; negative `no_hand` images need separate proof that **no visible hand was omitted**.

## Research/license stop gate

The dataset README says **"variant of CC BY-SA 4.0"**, pointing to the custom [`license/en_us.pdf`](https://github.com/hukenovs/hagrid/blob/Hagrid_v2-1M/license/en_us.pdf). The official five-page custom PDF **was completely extracted and source-reviewed on 2026-10-20**, but a signed owner/legal authorization for data use is still PENDING. This unsigned authorization is a **BLOCKER to real dataset acquisition/conversion**, and a public GitHub repository does not waive license or privacy duties.

The tool will run `audit` over **user-provided local metadata** without fetching anything. Actual `convert` **requires two local files**, both outside Git:
- `data/reviews/hagrid-license-ack.json`: a *real* downloaded PDF SHA256, review author/date, confirmed research-use scope and express acceptance of exact terms. This is an attestation, **not independent legal clearance**.
- `data/reviews/hagrid-mapping-ack.json`: visual review of **all 10** output mappings against [ESP-DL v3.3.14 gesture reference](https://github.com/espressif/esp-dl/blob/v3.3.14/examples/hand_gesture_recognition/gestures.png) and licensed HaGRID samples, especially `two_up -> two` and `palm -> five`. Each row requires an approved, dated, signed sample review. No automatic approval is provided in Git. The verified PDF SHA256 is 14f4845e9c8d3de875cbac4139491da368ab1040a7f565e02894477279134d22; the local review must reference the actual PDF. See HAGRID_LICENSE_AUDIT.md.

See `docs/templates/` for **unsigned illustrative templates**, deliberately invalid until completed from actual evidence. Do **not** commit signed review files.

## Expected local paths

```text
data/raw/hagrid_annotations/{train,val,test}/{one,two_up,three,four,palm,like,ok,no_gesture,call,dislike}.json
data/raw/hagrid_dataset/{source}/{image_key}.jpg
data/reviews/hagrid-license-ack.json         # manually signed / ignored by git
data/reviews/hagrid-mapping-ack.json         # manually signed / ignored by git
data/processed/hagrid10_micro/               # derived JPEG and YOLO labels, ignored
```

The tool uses the *record's per-hand* `labels`, never assumes that every hand has the source JSON's category. Unknown or compound gestures, missing boxes, invalid coordinates and unreviewed negatives are rejected. Such rejects are included in aggregate counts, without printing identities. It can now map approved additional single-hand sources to `no_gesture` **only after independent, signed, current policy-SHA-bound manual review**; ambiguous lookalikes and compound two-hand sources remain forbidden. See [HARD_NEGATIVES.md](HARD_NEGATIVES.md).

## Run after the applicable rights/visual gates

**1. Metadata-only audit** (no images downloaded or copied; usable with a permitted annotation subset):

```bash
python3 -m scripts.hagrid_p1 audit \
  --annotations data/raw/hagrid_annotations
```

Only to audit one class first, add `--sources one`. Audits still require train/val/test JSON for each selected class. It reports subject totals and split overlaps without exposing individual `user_id`.

**2. Strict controlled micro-conversion** (requires signed reviews and local JPEG files):

```bash
python3 -m scripts.hagrid_p1 convert \
  --annotations data/raw/hagrid_annotations \
  --images data/raw/hagrid_dataset \
  --output data/processed/hagrid10_micro \
  --license-ack data/reviews/hagrid-license-ack.json \
  --mapping-ack data/reviews/hagrid-mapping-ack.json \
  --limit-per-source 5
```

- Max **5 distinct subjects per source/split** by default, capped at 100. Selection is deterministic by sorted image stem, not random.
- Output: `images/{split}/{source}/{stem}.jpg`, `labels/{split}/{source}/{stem}.txt`, `manifest.aggregate.json`. The source-category subfolder prevents identical stems in different source JSONs colliding.
- The converter rejects subject overlap across splits before creating output, rejects duplicate selected image SHA256, checks JPEG signature, copies to a temporary directory and renames into place only if successful.
- Existing output paths are never overwritten. Run a new, versioned output path for each experiment.
- Local report contains aggregate split/class counts, source JSON SHA256 and total subjects; **no raw subject IDs, raw sample metadata, biometric demographics or images**.
- Real JPEG decoding, unlabeled visible hands, sampling bias, fairness, all-class coverage, class-semantic mapping and mixed-hand QA **require human/visual review**. `P1 PASS` is not produced automatically.

## P1 acceptance remaining

1. Obtain and review the exact custom PDF, determine permitted research/training/model/redistribution uses; license BLOCKED until signed.
2. Obtain a small *authorized* annotation archive and selective JPGs; record archive URLs, SHA256, dataset version, selected categories and disk cost.
3. Compare ten classes visually; sign mapping and reject mismatching variants; confirm sample ID mapping offline, never via public images.
4. Run metadata audit and sample conversion on *real* HaGRIDv2 data, record rejects, verify no subject overlap and test duplication.
5. Decode/inspect JPEG images and visually QA multi-hand boxes, empty negatives, ambiguous poses, orientation and occlusions. Freeze held-out subjects and split policy.
6. Establish P1 evidence + acceptance thresholds *before* P2 experiments.

Until these are verified, **P1 is PARTIAL**, and **P2 training must not start**.


## Bounded metadata ZIP reader — added in P1-01 follow-up

Official annotation ZIP is 719041314 bytes, but HTTP 206 Range works. A last-64KiB directory request identified exactly 30 ten-class train/val/test JSON files totaling 171848608 compressed bytes. Neither the full archive nor any raw annotations were downloaded for this public audit.

**Safe remote ZIP directory inspection (no raw metadata fetched):**

    python3 -m scripts.hagrid_zip_range inspect --sources no_gesture
    python3 -m scripts.hagrid_zip_range inspect

**After signed research-license approval, on your own workstation ONLY** (default 5 MiB transfer cap; approximately 1.4 MB for three no_gesture JSONs):

    python3 -m scripts.hagrid_zip_range extract \
      --sources no_gesture \
      --output data/raw/hagrid_annotations \
      --license-ack data/reviews/hagrid-license-ack.json \
      --max-transfer-mib 5

Then audit these real locally stored annotations:

    python3 -m scripts.hagrid_p1 audit \
      --annotations data/raw/hagrid_annotations \
      --sources no_gesture

For all ten JSON source classes, choose a distinct **new** output directory, use all default sources, and opt in to a sufficient download cap (e.g. --max-transfer-mib 180). Never silently overwrite; the archive byte size and member CRC32 are checked. Each extracted entry gets a local SHA256 in _provenance.aggregate.json. The archive-wide SHA256 is **unknown**, as only selected entries are downloaded. This is not equivalent to hashing the entire archive.

The extracted JSON still contains personal/subject-related attributes. Never commit, attach as PR artifacts, or run on public GitHub-hosted CI. A valid PDF acknowledgment does **not** waive the separate ten-gesture visual mapping approval needed before moving on to image conversion. The new script's CI tests use synthetic in-memory ZIPs only.


## Optional additional hard negatives — never automatic

See [HARD_NEGATIVES.md](HARD_NEGATIVES.md), the [34-source registry](../configs/hagrid_v2_source_policy.json), and the deliberately invalid [local signature template](templates/p1-hard-negative-ack.example.json). The bounded ZIP inspector can now list all official source categories, but **conversion** requires the separate `--hard-negative-sources` and `--hard-negative-ack` flags. Only the seven reviewable single-hand candidates may ever be mapped to output class 7 in this policy revision. The `two_up`/`peace` mapping discrepancy must be resolved with licensed examples before the real ten-class data gate can pass. No real hard-negative review is claimed.


## P1-03 visual mapping comparison packet

Run the local-only, license-gated [P1_VISUAL_REVIEW.md](P1_VISUAL_REVIEW.md) procedure to generate a **train-split** HTML comparison sheet with per-hand bounding boxes, multi-hand flags and a hashed private human review form. No real photos were acquired or approved in CI.


## Private ZIP64 JPEG selection (new P1-04 tooling)

[docs/P1_IMAGE_ZIP64.md](P1_IMAGE_ZIP64.md) explains how to retrieve **only 3–20 licensed source JPEGs per class** by HTTP Range from 40+GB ZIP64 archives, using the four-way `two_up`, `peace`, `peace_inverted`, `two_up_inverted` comparison. No full image ZIP was downloaded in the public CI probe, and no actual JPEG has been acquired by this project yet. `scripts/hagrid_image_zip64.py` requires the real owner-signed HaGRID license record, refuses actual CI downloads, checks ZIP64 offsets and CRC32, and writes only ignored local output.


**Measured four-class train metadata cost (2026-10-20):** For `two_up/peace/peace_inverted/two_up_inverted`, the official ZIP index lists **4 files**, **54,388,405 compressed** and **323,899,528 uncompressed bytes**. The 80 MiB one-command default cap is sufficient for compressed transfer. This is directory-only inspection; personal data remains unacquired. See [one-command local review](P1_ONE_CLICK_REVIEW.md).


### P1-05 bulk pilot import, manual ZIP and auto resumed Range mode

The old `hagrid_p1.convert` was deliberately limited to a 1–100-image **micro** tool and `hagrid_image_zip64.extract` to 3–20 JPEGs for reviews. For a *full ten-source* authorized pilot (default 500 train/50 val/50 test **per source**) use the separate [P1-05 dual-source dataset builder](P1_DUAL_SOURCE_DATASET.md), which accepts manually downloaded archives/directories or resumes selected official ZIP members into a private cache. The builder **still requires** both real license and all-ten mapping approval, and never claims P1 PASS or downloads photos in CI.
