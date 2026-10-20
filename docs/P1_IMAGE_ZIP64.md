# P1-04 — Selective HaGRIDv2 JPEG extraction from 40+GB ZIP64 archives

**Implemented: source-only ZIP64 reader, real public-index smoke check, synthetic image tests.**
**NOT DONE: real JPEG extraction, licensed sample review, P1 full PASS.**

## Why this exists

The official HaGRIDv2 dataset offers large class ZIPs. An actual **metadata-only** Actions probe on 2026-10-20 confirmed that the HaGRID archive for `two_up` is **42,733,450,409 bytes** (~39.8 GiB), supports HTTP `206 Partial Content`, and contains a ZIP64 central directory occupying **3,921,893 bytes**. It lists **30,689 entries**: **30,688 JPGs** and one directory. **27,610 entries use ZIP64 local-header offsets**.

Verified evidence (no image content downloaded):
- [ZIP64 source-archive probe](https://github.com/DenisZheng/esp32s3-gesture-recognition/actions/runs/39397602371)
- [Actual `two_up` parser smoke test](https://github.com/DenisZheng/esp32s3-gesture-recognition/actions/runs/39419164125)
- [Remaining three source index smoke tests](https://github.com/DenisZheng/esp32s3-gesture-recognition/actions/runs/39420065430)

### Verified metadata: all four number-two comparison archives

| Source | Archive bytes | JPEG entries | ZIP64 index bytes |
|---|---:|---:|---:|
| `two_up` | 42,733,450,409 | 30,688 | 3,921,893 |
| `peace` | 43,368,944,020 | 31,801 | 4,032,564 |
| `peace_inverted` | 41,355,311,911 | 29,849 | 4,052,438 |
| `two_up_inverted` | 41,724,228,996 | 29,991 | 4,101,812 |

All four public ZIP64 indexes were downloaded **only as metadata**; no raw person images. Their SHA256 values and exact archive sizes are pinned in `scripts/hagrid_image_zip64.py`. If an upstream archive changes, `inspect` and `extract` refuse it until a new reviewed baseline PR is prepared.



We can download a **64-KiB ZIP64 footer**, at most an **8-MiB ZIP directory**, and then *only the compressed byte ranges for the exact image keys selected from train annotations*. There is no need to download the 40+ GB file simply to inspect a few photos.

Source image URLs are from [HaGRIDv2 official dataset README](https://github.com/hukenovs/hagrid/tree/Hagrid_v2-1M#dataset). The initial implementation pins **four sources for the number-two ambiguity**:
`two_up`, `peace`, `peace_inverted`, `two_up_inverted`.
All **four source ZIP64 indexes have now passed live metadata-only GitHub Actions**. Their archive byte lengths **and central-directory SHA256** are pinned in `scripts/hagrid_image_zip64.py`; the extractor refuses any remote index with a different digest or length. No real image payload was requested.

The workflow stays on ESPDet-Pico, **one 224x224 detector with exactly 10 model classes**. Nothing here changes output names/indices or downloads pretrained weights.

## Gate 0 — Owner/license and privacy

This is a **local research activity**, not automatically authorized by passing GitHub CI. Review the verified **custom non-CC HaGRID license** in [HAGRID_LICENSE_AUDIT.md](HAGRID_LICENSE_AUDIT.md), including privacy and portrait rights. The owner/authorized reviewer must sign an actual research-scope local acceptance as `data/reviews/hagrid-license-ack.json` using the unsigned template at `docs/templates/p1-license-ack.example.json`.

- `inspect` reads only ZIP source names/index statistics; it does **not** request personal JPG data, and does not require a signed license record.
- `extract` verifies the **exact original license PDF SHA256** (already pinned), a real reviewer, date, scope, and acceptance **before any image network request**. A local research signoff does NOT confer commercial model/weights rights.
- Real-image extraction is **refused when running on GitHub Actions or a CI environment**. Synthetic tests inject a fake local transport and do not touch the internet.
- Never commit `data/raw`, `data/reviews`, extracted JPGs, raw JSON, any reviewer identities, or raw-media manifests into this **Public** repository. Source URLs/directory counts are safe to include in PRs, not people data.

## Gate 1 — Locally obtain a small train annotation subset

The preceding [P1_DATA_TOOLING.md](P1_DATA_TOOLING.md) has the bounded metadata ZIP reader. First inspect compressed costs:

    python3 -m scripts.hagrid_zip_range inspect --splits train \
      --sources two_up peace peace_inverted two_up_inverted

Only after actual research permission is confirmed:

    python3 -m scripts.hagrid_zip_range extract \
      --splits train \
      --sources two_up peace peace_inverted two_up_inverted \
      --output data/raw/p1_two_annotations \
      --license-ack data/reviews/hagrid-license-ack.json \
      --max-transfer-mib 80

Increase the explicit byte cap only after inspecting the archive costs. Do not automatically fetch full train/val/test metadata or all 34 sources. The four extracted `annotations/train/<source>.json` files include subject IDs and sensitive metadata and **must remain local**.

## Gate 2 — Live **index-only** compatibility check

This command fetches no photo bytes, only ZIP64 metadata:

    python3 -m scripts.hagrid_image_zip64 inspect --source two_up
    python3 -m scripts.hagrid_image_zip64 inspect --source peace
    python3 -m scripts.hagrid_image_zip64 inspect --source peace_inverted
    python3 -m scripts.hagrid_image_zip64 inspect --source two_up_inverted

On the known-good `two_up` source, expected source index counts are ~30,688 JPG entries, ZIP64 30,689 entries and a 3,921,893-byte central directory. The inspector never prints original image filenames/user IDs. It verifies the source HEAD size (including hard-pinned `two_up` length), then requires correctly bounded HTTP `206` responses for footer and central directory. **HTTP 200 on a Range request immediately aborts**, before reading the bulk response.

## Gate 3 — Download **only selected local-review JPGs**

For each source, separately run (must have local permit + corresponding train JSON):

    python3 -m scripts.hagrid_image_zip64 extract \
      --source two_up \
      --annotations data/raw/p1_two_annotations \
      --output data/raw/hagrid_dataset \
      --license-ack data/reviews/hagrid-license-ack.json \
      --per-source 6 \
      --seed p1-visual-review-v1 \
      --max-transfer-mib 80

Repeat with `--source peace`, `peace_inverted`, `two_up_inverted`. The program:
1. Reads the ZIP64 footer (up to 64 KiB) + class central index (up to 8 MiB). Does not stream the whole archive.
2. Matches JPG stems to **licensed train** annotation keys only, validates per-hand boxes/labels and selects candidate images containing that source gesture.
3. Deterministically chooses **3–20 distinct local subjects** by hashed sampling seed. Never reads the test/validation annotations or logs raw user IDs / source image keys.
4. Computes a conservative total HTTP byte budget before issuing any JPG range requests; defaults to 80 MiB, with hard member-size caps.
5. Requests only required ZIP local headers + exact selected compressed JPEG byte spans. Verifies central directory name, mode, ZIP64 offsets/sizes, CRC32, JPG SOI/EOI marker, SHA256 and duplicate images.
6. Writes transactionally into ignored local `data/raw/hagrid_dataset/<source>/<original_key>.jpg`. Never overwrites a previously extracted source folder; failures leave no partial final folder.
7. Places **private** `_provenance.private.json` under each source folder, recording selected media SHA256, ZIP central-directory SHA256, source annotation file SHA256 and remote archive size. The archive-wide SHA256 is **unknown**, since we deliberately do not fetch the entire ZIP.

Note: JPEG signature/CRC proves transfer integrity, **not JPEG full decode, labels or gesture semantics**. That requires opening the private review packet below.

## Gate 4 — Review real gestures visually

After staged licensed JPGs exist for all four groups:

    python3 -m scripts.p1_visual_review packet \
      --annotations data/raw/p1_two_annotations \
      --images data/raw/hagrid_dataset \
      --license-ack data/reviews/hagrid-license-ack.json \
      --output data/reviews/packets/two-v1 \
      --sources two_up peace peace_inverted two_up_inverted \
      --per-source 6

Open the private file `data/reviews/packets/two-v1/index.html`, verify boxes and fingers, then record decisions in a separate **private** signed JSON following `review.template.json`. Details and verification command: [P1_VISUAL_REVIEW.md](P1_VISUAL_REVIEW.md).

**Crucial:** The `two_up -> two` mapping is still provisional. HaGRID `peace` may better match the ESP-DL V sign. Any such evidence must trigger a source-mapping review PR, **not** label `peace` as a negative by default.

## Risks and limitations

- **All four remote ZIP64 indexes have been smoke-tested** and hashes/lengths locked. Individual JPEG payload extraction and visual correctness are **still NOT tested on real images**.
- Source archives are large and have no convenient full-archive SHA256. **All four** observed byte lengths and central-index SHA256 digests are hard-pinned. Individual images are checked by member CRC32 and SHA256 on download; unexpected upstream changes block extraction until a reviewed baseline update. This is still not equivalent to hashing entire ZIP archives.
- Since each ZIP entry is independently DEFLATE-compressed, exact range extraction is viable **only with correct ZIP64 metadata and CRC**. This is proven on synthetic fixtures; actual image extraction is **NOT** yet proven.
- Human subject rights, true train/validation/test subject isolation, full JPEG decoding, image quality, label completeness, pose ambiguity, occlusion, command confusion, and person demographics require local human verification before P1 PASS.
- These tools do **not** train, export, quantize or run a model. P2, P3 and CoreS3 P4 are still PENDING.
