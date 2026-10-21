# P1-05 — Ten-class HaGRIDv2 dataset preparation: auto or manual (Linux)

**Code implemented / synthetic CI only. Real licensed data, owner signoff and P1 PASS are still PENDING.**

This is a **dataset-builder**, not a training launcher. It produces a ten-class **single ESPDet-Pico / 224×224** Ultralytics/YOLO folder with one bounding box **per physical hand**. Its existing canonical source/output mapping is fixed: `one → one`, `two_up → two` (PROVISIONAL), `three → three`, `four → four`, `palm → five` (PROVISIONAL), `like → like`, `ok → ok`, `no_gesture → no_gesture`, `call → call`, `dislike → dislike`. **If real visual review establishes `peace` as the correct number-two source, stop and change the mapping in a reviewed PR BEFORE running the builder.** Do not mark `peace` as class-7 negatives.

## Three private input modes, one fixed conversion and QA contract

| Mode | Data you provide | What P1-05 does |
|---|---|---|
| `auto` | Authorized official source + local private cache | HTTP Range reads only the requested 30 train/val/test JSONs and selected JPEG members from official pinned archives; CRC-checked resumable cache, no full 40GB-class ZIP |
| `local-zip` | Official `annotations.zip` and folder of **ten per-class ZIP files** | Opens ZIP32/ZIP64 locally, reads only selected JPEG and JSON entries, checks central names/CRC/size/symlinks, never extracts all |
| `local-dir` | Already extracted annotation and image folders | Reads existing JPEGs and JSONs, checks hands, subject splits, hashes and counts |

**There is no automatic HaGRID license acceptance and no automatic change to the 10-class source map.** All modes require signed **private** acknowledgments for (1) reviewed research-use rights and (2) actual real-image comparisons for **all ten** model classes:

    data/reviews/hagrid-license-ack.json
    data/reviews/hagrid-mapping-ack.json

Use [unsigned license template](templates/p1-license-ack.example.json) and [unsigned mapping template](templates/p1-mapping-ack.example.json) only after reading the actual source terms and comparing permitted samples. Merely copying example fields is NOT genuine permission. Custom HaGRID research terms are **not commercial/model-weight privacy clearance**; [Issue #2](https://github.com/DenisZheng/esp32s3-gesture-recognition/issues/2) remains open until real human signoff.

Private data must remain in ignored `data/` or another private machine location. Public GitHub CI explicitly blocks **all three** real-data build modes. Synthetic tests use invented JPEG-like bytes and invented subject IDs only.

## Important source discovery: no_gesture is special

The official HaGRIDv2-1M [dataset README](https://github.com/hukenovs/hagrid/blob/Hagrid_v2-1M/README.md) points the first nine command gesture ZIPs to large `/datasets/hagrid/hagrid_dataset_new_554800/hagrid_dataset/*.zip` files, while **`no_gesture.zip`** has a *different* URL: `/datasets/hagrid_v2/hagrid_v2_zip/no_gesture.zip`.

- Those 12 original/comparison archives are **ZIP64**, typically ~39–44GB each. The `no_gesture` archive is **ZIP32**, exactly **517,860,551 bytes**, with **2,164 usable root JPEGs** and two nested metadata entries.
- Real public **index-only** GitHub Actions checked the original official sizes and SHA256 of central directory bytes for all **ten model sources** (no personal JPEG fetched). Source byte lengths, exact URLs and hashes are pinned in [hagrid_image_zip64.py](../scripts/hagrid_image_zip64.py).
- An upstream archive that changes its size/index digest **fails closed**. The SHA256 of a *full* 40GB archive remains unknown because we intentionally do not download it; selected JPGs have their own CRC/size and final SHA256 checks.

Metadata-only inspection of the original annotations ZIP established total selected **ten-class train+val+test** JSONs at **171,848,608 compressed bytes** (~163.9 MiB) / **1,022,736,786 uncompressed bytes** (~975.4 MiB). The builder reads one JSON at a time; your **32GB system RAM** helps, but set aside adequate storage for cache, selected JPEGs, finished dataset and checkpoints. It does not download the original ~1.5TB dataset.

## Step 0 — plan (always offline)

    python3 -m scripts.p1_dataset_builder plan

Defaults: **500 train + 50 validation + 50 test images per original source**, for **6,000 selected images total across the ten source folders**. This is a *pilot recommendation*, not a guarantee of accuracy. For the initial authorized end-to-end test, use `--train-per-source 10 --val-per-source 2 --test-per-source 2`. If the official source lacks enough valid, distinct image hashes, the build **fails** rather than silently producing a deficient dataset.

The train/val/test splits are the **official HaGRID subject-level splits**, independently rechecked using *all* original per-source `user_id` records for the chosen ten classes. Images are never randomly reassigned between splits. Source files/frames with unknown or unlabeled second hands, compound joint boxes, invalid normalized coordinates, or labels outside the ten approved categories are **rejected as whole frames**. An image with a real `no_gesture` hand keeps a **class 7 bounding box**; the builder never fabricates label-less `no_hand` images.

## Step 1A — fully automatic selective download (AFTER human approvals)

Run on your private Linux workstation, not public CI:

    python3 -m scripts.p1_dataset_builder build \
      --mode auto \
      --cache data/cache/hagrid10-p1-05 \
      --output data/processed/hagrid10-pilot-v1 \
      --license-ack data/reviews/hagrid-license-ack.json \
      --mapping-ack data/reviews/hagrid-mapping-ack.json \
      --train-per-source 500 --val-per-source 50 --test-per-source 50 \
      --max-transfer-gib 12 --max-output-gib 12

It reads **only selected ZIP members** using 206/Content-Range, source index SHA256, ZIP CRC32 and bounded decompression. Each successfully fetched private JPEG and JSON is stored in `--cache`; if the network interrupts, rerun the **same** command and cache, and verified existing files will be reused. The finished output directory is **published atomically only after all ten sources and all three splits pass**. A failure removes incomplete output staging but leaves the private verified cache for later retries.

Transient 429/5xx/transport errors receive bounded byte-range retries; no retry is permitted for HTTP 200 ignoring Range or unauthorized 403. Network cap is **per invocation**, not an unlimited background downloader. You can raise explicit budgets if actual image sizes demand it. Never bypass the central-index check because the source silently changed.

## Step 1B — manually downloaded original ZIPs

Prepare PRIVATE input paths without extracting 40GB ZIPs:

    data/manual/annotations.zip
    data/manual/hagrid_zips/one.zip
    data/manual/hagrid_zips/two_up.zip
    data/manual/hagrid_zips/three.zip
    data/manual/hagrid_zips/four.zip
    data/manual/hagrid_zips/palm.zip
    data/manual/hagrid_zips/like.zip
    data/manual/hagrid_zips/ok.zip
    data/manual/hagrid_zips/no_gesture.zip
    data/manual/hagrid_zips/call.zip
    data/manual/hagrid_zips/dislike.zip

These must correspond to original HaGRIDv2 source annotations/JPEGs, not model classes renamed to `two` or `five`. The `no_gesture` ZIP uses a different upstream URL and ZIP32 root-JPEG layout; both layouts are handled. Never unzip archive paths blindly.

    python3 -m scripts.p1_dataset_builder build \
      --mode local-zip \
      --annotations data/manual/annotations.zip \
      --images data/manual/hagrid_zips \
      --output data/processed/hagrid10-pilot-v1 \
      --license-ack data/reviews/hagrid-license-ack.json \
      --mapping-ack data/reviews/hagrid-mapping-ack.json \
      --train-per-source 500 --val-per-source 50 --test-per-source 50

This uses Python's random-access ZIP index; it reads **only selected JPEG contents**, verifies CRC32 and bounded file sizes and refuses traversal/encrypted/symlink entries. You need adequate disk to *store* manually downloaded archives, but no extra full unzip is required.

## Step 1C — already extracted local folders

Expected relative layout:

    data/manual/hagrid_annotations/train/one.json
    data/manual/hagrid_annotations/val/one.json
    data/manual/hagrid_annotations/test/one.json
    data/manual/hagrid_dataset/one/<original_key>.jpg
    data/manual/hagrid_dataset/no_gesture/<original_key>.jpg

Repeat 30 JSON files and the ten image subfolders. Keep the **original stem** to match the JSON keys, and do not mix HaGRIDv1/v2 archives.

    python3 -m scripts.p1_dataset_builder build \
      --mode local-dir \
      --annotations data/manual/hagrid_annotations \
      --images data/manual/hagrid_dataset \
      --output data/processed/hagrid10-pilot-v1 \
      --license-ack data/reviews/hagrid-license-ack.json \
      --mapping-ack data/reviews/hagrid-mapping-ack.json \
      --train-per-source 500 --val-per-source 50 --test-per-source 50

## Step 2 — check the *candidate* dataset

    python3 -m scripts.p1_dataset_builder verify \
      --output data/processed/hagrid10-pilot-v1

Output:

    data/processed/hagrid10-pilot-v1/
      dataset.yaml
      images/{train,val,test}/{original_source}/{original_key}.jpg
      labels/{train,val,test}/{original_source}/{original_key}.txt
      manifest.aggregate.json
      manifest.private.json

`dataset.yaml` defines `nc=10` and the project’s exact class-index order. YOLO labels are `class cx cy w h` on normalized per-physical-hand boxes, and mirror the image split/source relative directory. The **private** manifest includes original image keys and per-file image/label SHA256s; **aggregate-only** reports contain no original subject IDs or per-image keys. The separate `verify` command recalculates file hashes, checks per-split counts, and detects changed labels or JPEGs.

**This is NOT approval to run the real train.py**: opening actual JPEGs, decoding with a real image library, inspecting ambiguous poses and missing hands, checking `no_hand` from a separately audited source, reviewing rights, and final subject/identity/quality signoff are still **P1 mandatory gates**. Full detection-loss/mAP/GTX1060 GPU testing belongs to P2 only after all P1 signed PASS. This builder never runs any training or model download.

## Caveats and operational limits

- **Auto source ZIP index proof is real; actual licensed image download is NOT run in CI.** Range behavior for thousands of JPEGs, real image decode quality, source server availability, and full 6,000-image success must be measured on the authorized Linux host.
- Manual mode assumes authentic *original* HaGRIDv2 ZIP files or extracted JPEG/JSON matching keys. Do not mix annotations from a different dataset release.
- The auto per-image resume cache checks official ZIP CRC32 and source size before reuse and the final dataset hashes every included image/label. A corrupt cache fails until the operator removes/reacquires it; no silent bypass.
- Disk usage can reach many gigabytes. `--max-transfer-gib` and `--max-output-gib` fail closed, rather than spilling disk. Start with a **small authorized subset**, record download time, rejected counts, photo decode and GPU RAM.
- This first P1-05 slice intentionally covers only the **ten canonical source candidates**. Optional seven reviewed hard-negative sources, standalone hand-free negatives, and alternative `peace`→`two` mapping are separate human-reviewed dataset revisions, never auto-inferred.


### Version identity: verified against the official HaGRIDv2 annotation link

The ZIP fetched by existing `scripts/hagrid_zip_range.py` is the **official HaGRIDv2** `/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip`, observed as **719,041,314 bytes / 121 ZIP members**. A distinct old HaGRID `/datasets/hagrid/hagrid_dataset_new_554800/annotations.zip` is only **34,971,848 bytes / 57 members** and **must not be silently substituted**. This comparison was done using public HEAD + 64KiB footer metadata only. The builder's `auto` uses the already-correct official v2 URL; manual ZIP mode rejects archives missing any of the ten source files across all three splits.
