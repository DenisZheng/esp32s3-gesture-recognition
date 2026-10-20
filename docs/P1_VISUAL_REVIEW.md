# P1-03 — Private visual mapping review workflow

**2026-10-20 / TOOLING IMPLEMENTED; HUMAN IMAGE EVIDENCE PENDING.**

Purpose: resolve the provisional ten-class mappings for a **single ESPDet-Pico model** with input **224x224** and exactly **10 output classes**, not a second MobileNetV2 model. Actual HaGRID images and original annotations remain private and local.

## Essential number-two question

The official [ESP-DL v3.3.14 gestures](https://github.com/espressif/esp-dl/blob/v3.3.14/examples/hand_gesture_recognition/gestures.png) depict `two` as a separated V. In [HaGRIDv2's gesture reference](https://github.com/hukenovs/hagrid/blob/Hagrid_v2-1M/images/gestures.png), `two_up` fingers appear closer while `peace` looks more V-like. **This is a figure-level hypothesis, NOT an approved mapping.** Never use `peace` as class-7 `no_gesture` while the correct number-two pose is unresolved.

| Model output | Current candidate | Must also compare |
|---|---|---|
| `two` | `two_up` | `peace`, `peace_inverted`, `two_up_inverted` |
| `five` | `palm` | `stop`, `stop_inverted` |
| `three` | `three` | `three2`, `three3`, `three_gun` |
| `one` | `one` | `point`, `mute` |

**Only train-split images may be used to decide mappings**; never preview held-out test-set photos to design mapping rules.

## 0. Required permission

The **real user/owner or authorized reviewer** must personally accept the actual [custom, non-CC HaGRID license terms](HAGRID_LICENSE_AUDIT.md) for research. Create ignored local `data/reviews/hagrid-license-ack.json` using `docs/templates/p1-license-ack.example.json`. This is *not* automatic approval or commercial/biometric clearance. Neither ChatGPT nor source CI has executed/accepted this legal gate. Do **not** commit completed reviews.

## 1. Obtain only selected permitted annotation JSON files

Read only the remote ZIP directory (no human data is fetched):

    python3 -m scripts.hagrid_zip_range inspect --splits train \
      --sources two_up peace peace_inverted two_up_inverted

On your **own workstation only, after valid license acknowledgment**, extract the specific train JSON files:

    python3 -m scripts.hagrid_zip_range extract \
      --splits train \
      --sources two_up peace peace_inverted two_up_inverted \
      --output data/raw/p1_two_annotations \
      --license-ack data/reviews/hagrid-license-ack.json \
      --max-transfer-mib 80

This extracts only compressed JSON members, NOT JPGs. The original HaGRID images are in **large per-gesture ZIP64 archives (e.g. two_up.zip = 42.73 GB)**. Rather than downloading any full class ZIP, the new [P1_IMAGE_ZIP64.md](P1_IMAGE_ZIP64.md) procedure can retrieve a few authorized source JPEGs by exact HTTP Range, using just ~4 MB of metadata index per class. **Real image download is blocked until actual owner license acceptance**, and only the two_up remote index has been smoke-tested.

## 2. Stage a few separately permitted local images

Place source-named original JPEGs under:

    data/raw/hagrid_dataset/two_up/<image_key>.jpg
    data/raw/hagrid_dataset/peace/<image_key>.jpg
    data/raw/hagrid_dataset/peace_inverted/<image_key>.jpg
    data/raw/hagrid_dataset/two_up_inverted/<image_key>.jpg

The `<image_key>` stems must match the annotation JSON record keys. Do not rename JPGs before running the sampling tool. No user IDs or HaGRID raw JSON/image bytes may enter the Public GitHub repository or Actions artifacts.

## 3. Produce the PRIVATE HTML review packet

    python3 -m scripts.p1_visual_review packet \
      --annotations data/raw/p1_two_annotations \
      --images data/raw/hagrid_dataset \
      --license-ack data/reviews/hagrid-license-ack.json \
      --output data/reviews/packets/two-v1 \
      --sources two_up peace peace_inverted two_up_inverted \
      --per-source 6 \
      --seed p1-two-review-v1

Open `data/reviews/packets/two-v1/index.html` locally. It shows the original **per-hand bounding boxes, actual raw labels, multi-hand/compound cautions, and official reference links** in adjacent image cards.

The packet folder also contains `images/` (small local JPG copies), `packet.manifest.json` (image/file hashes, original per-hand boxes, aggregate selection statistics), and `review.template.json` (intentionally invalid unsigned local form). All are **PRIVATE**. The generated filename never includes the original image key or `user_id`, and metadata fields such as demographics are not serialized. JPEG signature is a sanity check, not successful full-image decode or visual QA.

The tool first filters annotations to JPEG files **actually staged locally**, so it can pick 3–20 JPGs even when the original JSON describes tens of thousands of images not present on disk. It then samples distinct local people per class where possible. For a positive final mapping match, at least **three** actual images of that source must be reviewed. It **refuses zero-image categories** and never reads val/test. Output is transactional and must be a new path for every review run.

## 4. Human comparison, record and verify

Copy `review.template.json` to a separate private file, e.g. `data/reviews/two-v1-signed.json`. Set the real `reviewer`, `reviewed_on`, image `reviewed_packet_ids` (like `two_up:001`), `official_reference_checked=true`, and specific `comparative_notes`.

Set `decision` to `match`, `mismatch`, or `uncertain`:
- For a **primary** source (`two_up`), `match` means the real image visibly matches the **proposed ESP-DL command** (`two`).
- For a **contrast** (`peace`), `match` means it looks like the same **official command** under comparison — this is a **potential alias conflict**, not evidence it is safe to label as `no_gesture`.
- `mismatch` or `uncertain` on primary output means the current mapping needs further attention. An ambiguous lookalike is also reported as unresolved.

Validate locally:

    python3 -m scripts.p1_visual_review verify \
      --packet data/reviews/packets/two-v1 \
      --review data/reviews/two-v1-signed.json

Checks include signed reviewer/date, required sample IDs, manifest SHA256, copied-image SHA256, unchanged class indices, written rationale, official reference check, and required `two` contrasts. A structural PASS **does not constitute semantic, licensed-data or full ten-class P1 PASS**. These checks only establish that a human decision was recorded against matching evidence.

## 5. Finish all ten mappings, not just number two

Repeat with the other three contrast groups above and the six remaining primary sources (`four, like, ok, no_gesture, call, dislike`), on **licensed train images**. Inspect actual hand count, angle/mirror, partial occlusion, background, fingertip separations, left/right hands and pose diversity. A `no_gesture` example must have a real hand bbox; true `no_hand` negatives must be reviewed separately as images with **zero visible hands**.

If real evidence indicates `peace` is better than `two_up`, open a deliberate compatibility-reviewed mapping PR and regenerate all affected labels; **do not silently change `configs/classes.json`** or misclassify `peace` as a negative.

## Outstanding stage gates

- Issue #2: local owner/research-license acceptance (**missing**).
- Issue #3: real evidence and **signed decisions for all ten classes** (**missing**).
- Issue #4: authorized sample JPG availability, train-only visual QA, held-out data separation, verified empty-room negatives (**missing**).
- P2 training, ONNX quantization and CoreS3 real-camera proof are **NOT started**. A green CI tests only synthetic fixtures.


### Optional selective ZIP64 JPEG acquisition (after owner permission)

Once the corresponding `train/<source>.json` annotation records have been fetched locally, follow the four `inspect` and `extract` commands in [P1_IMAGE_ZIP64.md](P1_IMAGE_ZIP64.md) to stage 3–20 photos per class without downloading four 40+ GB archives. The packet builder still generates private HTML/image hashes locally and never uploads images to GitHub.
