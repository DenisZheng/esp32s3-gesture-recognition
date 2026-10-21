# HaGRIDv2 data, identity, labeling and conversion policy

## Canonical label mapping

The only model classes are, in order: `0 one`, `1 two`, `2 three`, `3 four`, `4 five`, `5 like`, `6 ok`, `7 no_gesture`, `8 call`, `9 dislike`. Source candidates: `one`, `two_up`, `three`, `four`, `palm`, `like`, `ok`, `no_gesture`, `call`, `dislike`. `two_up -> two` and `palm -> five` are **not approved** until the official ESP-DL gesture illustration (`examples/hand_gesture_recognition/gestures.png`) is compared to *licensed* HaGRID samples. Review also `three` vs `three2/three3` and potential angle/mirror ambiguity; freeze approval per class before full conversion.

## Distinguish no_hand, no_gesture, other gesture

- `no_gesture` (ID 7) means **visible hand with a true bounding box**, e.g. benign posture or non-command. The model must learn to emit a positive class-7 detection, not silence.
- `no_hand` means **no visible hands**. It has no category ID. Only manually/audit-confirmed hand-free frames may have empty YOLO label files.
- If a hand is doing a HaGRID gesture outside the nine commands (`fist`, `rock`, etc.), it must not disappear from truth labels. Policy: map its **per-hand box** to `no_gesture` only if it is semantically a non-command and subject to deterministic provenance and representative manual review; otherwise mark **REJECT/REVIEW** and exclude the entire image. Do not silently ignore known hands as background.
- Some HaGRID poses represent **two-hand compound gestures**, with `united_bbox`. Those must **not** be treated as one ordinary single hand. Use per-hand `bboxes`, `labels` and review; ambiguous labels => REJECT/REVIEW. The nine command mappings must apply per physical hand.
- A frame might contain a target gesture and a non-command second hand: include both boxes and separate class IDs. Do not infer all hands in the frame from directory `target` or sample filename.
- If raw annotations lack a box for any visible hand or a box is corrupt/ambiguous, reject or manually relabel the frame under verified rights; **never** use it as a hand-free negative by mistake.

## Subject-level splits and leakage

HaGRID lists original train/val/test subject grouping by `user_id`. Still independently check that **after subset selection and relabeling** subject ID sets remain disjoint. When combining all original categories or introducing negatives, one subject cannot appear in multiple splits. Dataset/hash deduplication checks source image duplication *in addition* to user ID. Preserve original `user_id` only in ignored private manifests; only publish aggregate counts, salted hashes if necessary (with confidential salt), and status.

Suggested deterministic reassignment, only if needed: stable SHA256 of `{split_seed}:{user_id}` used to group subjects, with a documented train/val/test rule; do not upload the seed with raw user IDs to GitHub. Never randomly split images without subject grouping. Keep validation and test held out from calibration and augmentation fit.

## Conversion details

Official HaGRID annotations contain `bboxes` as normalized top-left `x, y, width, height` and `labels` per hand. For valid single-hand boxes, YOLO row is `class_id (x+w/2) (y+h/2) w h`, all in `[0,1]`; require `w,h > 0` and `x+w <=1`, `y+h <=1` (small epsilon for floating point). Raw `united_bbox` is not a substitute for per-hand data. Output structure `images/{train,val,test}/...`, `labels/{train,val,test}/...`, with `configs/hagrid10.yaml` pointing to an ignored root.

## Acquisition and quality gates

1. No full 1.5 TB download. First audit annotation metadata and license, then request a small class-balanced licensed subset including multi-hand and hand-free images. Estimate disk/network cost before any archive fetch.
2. Keep all images, private JSON and local inference traces under ignored `data/` directory. No training weights, sample images, faces, user IDs or firmware binaries in GitHub/CI artifacts/logs.
3. Validate category IDs 0..9, finite bbox values, correct normalization, nonzero area, image-label one-to-one, recognized labels, duplicate filenames and subjects, no missing visible hand; record rejected/uncertain cases. Audit category distribution and rare negatives.
4. Visual review in private local environment: at least a stratified sample from each 10 classes, multiple hands, natural idle, unrelated gestures, occlusion, resize/crop effects and image flips. Sign off shape mappings explicitly.
5. Every dataset subset must have a local manifest with source dataset version, terms, acquisition date, (URL, checksum) of any archive, selection method, split seed or hash, reject reasons, individual sample hashes, counts and anonymized aggregate audit. Do not publicly expose user IDs.

## Converter decision

Reference [`hagrid_to_yolo.py`](https://github.com/hukenovs/hagrid/blob/master/converters/hagrid_to_yolo.py), but treat it as **unverified for this use**. It maps IDs by the order of its `targets` list, and in gestures mode can choose `united_bbox`/`united_label` for compound gestures. Therefore a project-specific auditable conversion / split-validator is a P1 deliverable. P0 only provides the label contract and offline synthetic tests.


## Additional noncommand hands — signed, not inferred

The pinned [34-source policy](../configs/hagrid_v2_source_policy.json) classifies the 24 extra sources as **7 single-hand review candidates + 10 visually confusable quarantined sources + 7 compound quarantined sources**. No extra source is automatically a `no_gesture`. Selected single-hand candidates such as `fist` may be mapped to index 7 with a **real hand bbox**, only with signed local source-policy-hash-bound review; an unapproved second hand forces rejecting the **entire image**, not quietly dropping its bbox. `peace`, `three2`, `stop` and compounds cannot enter this mapping under the current policy. Empty `no_hand` negatives require separately audited truly hand-free images. See [HARD_NEGATIVES.md](HARD_NEGATIVES.md).


### P1-05 source modes and source-identity safeguards

Use [P1_DUAL_SOURCE_DATASET.md](P1_DUAL_SOURCE_DATASET.md). The same ten canonical source mapping and strict per-physical-hand normalization feed local ZIP, extracted-directory or auto exact-member Range downloads. An approved dataset candidate requires all train/val/test JSONs and enforces `user_id` disjointness across **all included original annotation records**, even rows excluded for ambiguous per-hand labels. No automatic image-wise random split, no unreviewed `peace` negative, no `no_hand` class or synthetic hand-free frames. The auto cache is private, resumable by ZIP CRC/size, with independent final image/label SHA256s. **Dataset construction is not visual, legal or model-training acceptance.**
