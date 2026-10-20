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

The dataset README says **"variant of CC BY-SA 4.0"**, pointing to the custom [`license/en_us.pdf`](https://github.com/hukenovs/hagrid/blob/Hagrid_v2-1M/license/en_us.pdf). The custom PDF **has not yet been completely read and approved in this project**. This is a **BLOCKER to real dataset acquisition/conversion**, and a public GitHub repository does not waive license or privacy duties.

The tool will run `audit` over **user-provided local metadata** without fetching anything. Actual `convert` **requires two local files**, both outside Git:
- `data/reviews/hagrid-license-ack.json`: a *real* downloaded PDF SHA256, review author/date, confirmed research-use scope and express acceptance of exact terms. This is an attestation, **not independent legal clearance**.
- `data/reviews/hagrid-mapping-ack.json`: visual review of **all 10** output mappings against [ESP-DL v3.3.14 gesture reference](https://github.com/espressif/esp-dl/blob/v3.3.14/examples/hand_gesture_recognition/gestures.png) and licensed HaGRID samples, especially `two_up -> two` and `palm -> five`. Each row requires an approved, dated, signed sample review. No automatic approval is provided in Git.

See `docs/templates/` for **unsigned illustrative templates**, deliberately invalid until completed from actual evidence. Do **not** commit signed review files.

## Expected local paths

```text
data/raw/hagrid_annotations/{train,val,test}/{one,two_up,three,four,palm,like,ok,no_gesture,call,dislike}.json
data/raw/hagrid_dataset/{source}/{image_key}.jpg
data/reviews/hagrid-license-ack.json         # manually signed / ignored by git
data/reviews/hagrid-mapping-ack.json         # manually signed / ignored by git
data/processed/hagrid10_micro/               # derived JPEG and YOLO labels, ignored
```

The tool uses the *record's per-hand* `labels`, never assumes that every hand has the source JSON's category. Unknown or compound gestures, missing boxes, invalid coordinates and unreviewed negatives are rejected. Such rejects are included in aggregate counts, without printing identities. It does not yet remap other non-command HaGRID gestures to `no_gesture`; those images are excluded pending case-specific signoff.

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
