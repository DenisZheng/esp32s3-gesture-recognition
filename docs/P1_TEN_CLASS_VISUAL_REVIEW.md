# P1-06: complete local ten-gesture visual mapping campaign

**State: tooling implemented; human research-license signoff and real image QA are STILL PENDING.**

The exact five-page English custom HaGRIDv2 PDF was already inspected and its official SHA256 recorded in [HAGRID_LICENSE_AUDIT.md](HAGRID_LICENSE_AUDIT.md). It is NOT an ordinary CC BY-SA license; source-text audit is NOT the owner's acceptance and does not settle portrait rights, model-weight share-alike or commercial licensing.

P1-05 now supports automatic/manual training dataset preparation but rightly requires all-ten signed visual mappings *first*. P1-06 makes that prerequisite feasible: licensed **train-only** small JPEG packets, one group at a time, followed by human match/mismatch/uncertain decisions linked to actual private JPEG hashes. The code never chooses a gesture on behalf of a person, never fetches test-set images to decide training mappings and never trains ESPDet-Pico.

## 1. Offline plan, no network/PII

    git pull --ff-only
    python3 -m scripts.p1_ten_review plan

| Review group | Primary output classes | Sources (primary + lookalike) |
|---|---|---|
| one | one | one, point, mute |
| two | two | two_up, peace, peace_inverted, two_up_inverted |
| three | three | three, three2, three3, three_gun |
| five | five | palm, stop, stop_inverted |
| remaining | four, like, ok, no_gesture, call, dislike | four, like, ok, no_gesture, call, dislike |

20 unique source folders, exactly ten official output classes. Contrast sources are **review-only**, not automatically `no_gesture` negatives. In particular `two_up → two` and `palm → five` are NOT approved by the tool.

All 20 official image ZIP archive sizes and SHA256 indexes have been checked from public archive metadata ONLY, without fetching JPEGs. Seven new variant ZIP indexes were live verified in [GitHub Actions source-only probe](https://github.com/DenisZheng/esp32s3-gesture-recognition/actions/runs/39534592247). Special root-JPEG archives: `point`, `three3`, `three_gun` and `no_gesture`. ZIP32 `no_gesture` has 2,164 root photos; the others use ZIP64 as discovered.

## 2. Actual human research rights acceptance (local only)

Read the [five-page English HaGRID license and audit](HAGRID_LICENSE_AUDIT.md) and get your authorized reviewer to complete the deliberately INVALID [license template](templates/p1-license-ack.example.json) under ignored private `data/reviews/hagrid-license-ack.json`.

The approval must be a real human permission decision, not one invented by tests or AI. Do not commit the signed license, metadata, images, review packets or real user identifiers to this Public repo. P1-06 `prepare` refuses real-person downloads in public GitHub CI, and refuses before any network traffic unless the local research license record validates.

## 3. First group: prioritize the high-risk V sign (`two`)

After authorization on the owner's Linux workstation, run:

    python3 -m scripts.p1_ten_review prepare \
      --group two \
      --output data/reviews/p1-06/two-v1 \
      --license-ack data/reviews/hagrid-license-ack.json \
      --per-source 6 \
      --annotation-budget-mib 120 \
      --jpg-budget-mib 80

This fetches only original **train** annotations and 3–20 representative local JPEGs per source, selected from distinct subjects where possible, using official pinned ZIP indexes and HTTP Range. No 40GB full-class ZIP download, full dataset, val/test images, or model weights. Each review group is written atomically into a new ignored PRIVATE folder. If any acquisition or packet step fails, the incomplete output is deleted.

Open this file **locally**:

    data/reviews/p1-06/two-v1/packet/index.html

Compare its actual per-hand boxes and poses against both official illustrations:
- [ESP-DL v3.3.14 official gestures](https://github.com/espressif/esp-dl/blob/v3.3.14/examples/hand_gesture_recognition/gestures.png)
- [HaGRIDv2 gesture catalog](https://github.com/hukenovs/hagrid/blob/Hagrid_v2-1M/images/gestures.png)

HaGRID `two_up` shows closer fingers in the reference chart; `peace` may be more similar to ESP-DL's V. This is **an unverified hypothesis**. If actual licensed photo review agrees, do NOT approve the old mapping; file a deliberate policy/config PR and re-build affected labels. ZIP CRC and JPEG signature alone do not establish that the image is decodable, that the box is correct, or that every visible hand is labeled.

## 4. Record and check genuine human decisions

Never alter packet images or the manifest. Copy its explicitly invalid review template to a new private file:

    cp data/reviews/p1-06/two-v1/packet/review.template.json \
       data/reviews/p1-06/two-reviewed.json

Enter the real reviewer and review date, `decision` for every primary and contrast source, actual `reviewed_packet_ids` for three or more real photos per accepted primary, `official_reference_checked=true`, and written `comparative_notes` (minimum 12 characters).

Decision semantics:
- `match` on a primary = human judges this exact source to represent the proposed official command.
- `match` on a contrast (such as `peace`) = possible **alias conflict** with the official command; NEVER permission to label it a negative.
- `mismatch` = human judges that source not to resemble the official command.
- `uncertain` = needs more evidence, so no mapping approval.

Then run:

    python3 -m scripts.p1_ten_review verify \
      --group two \
      --workspace data/reviews/p1-06/two-v1 \
      --review data/reviews/p1-06/two-reviewed.json

It verifies all original JPEG SHA256s, packet hash, sample IDs and review structure. It is NOT an automated image-semantic or real identity/signature verifier.

## 5. Repeat four other private groups

Run `prepare` again with the following group + new output pair. Each produces a separate HTML packet and deliberately invalid review form:

    --group one       --output data/reviews/p1-06/one-v1
    --group three     --output data/reviews/p1-06/three-v1
    --group five      --output data/reviews/p1-06/five-v1
    --group remaining --output data/reviews/p1-06/remaining-v1

Keep the same `--license-ack data/reviews/hagrid-license-ack.json`. Open the packet HTML in each folder, compare all real poses, copy its template to `<group>-reviewed.json`, and call `verify --group <group> --workspace ... --review ...`.

Default is **6 JPEGs × 20 sources = 120 photos** *if the owner authorizes and all official sources are available*, never 120 photos downloaded by public CI. Different group failures do not remove earlier complete groups.

## 6. Consolidate all five signed group reviews into an INVALID unsigned draft

After genuinely reviewing and verifying each group:

    python3 -m scripts.p1_ten_review draft \
      --evidence one data/reviews/p1-06/one-v1 data/reviews/p1-06/one-reviewed.json \
      --evidence two data/reviews/p1-06/two-v1 data/reviews/p1-06/two-reviewed.json \
      --evidence three data/reviews/p1-06/three-v1 data/reviews/p1-06/three-reviewed.json \
      --evidence five data/reviews/p1-06/five-v1 data/reviews/p1-06/five-reviewed.json \
      --evidence remaining data/reviews/p1-06/remaining-v1 data/reviews/p1-06/remaining-reviewed.json \
      --output data/reviews/p1-06/campaign-v1

This rechecks each private packet image hash and each review's evidence linkage. It writes:
- `campaign.evidence.private.json`: five group review/packet hash digests, aggregate counts and any unresolved source names, no subject IDs
- `hagrid-mapping-ack.UNSIGNED.json`: all ten provisional source/output rows, with `approved:false`, `official_reference_checked:false`, blank final reviewer/date

These fields are deliberately **INVALID** as P1-05 inputs even if structural review is complete. An uncertain primary or matched/uncertain lookalike yields `REVIEW_CONFLICTS_REQUIRE_ADJUDICATION`. **A real authorized person** must resolve those conflicts and sign a separate private final mapping acknowledgment; the script NEVER supplies that signature.

## 7. Remaining stage gates

P1-05 real training dataset can be prepared after the **actual local research license acceptance and all-ten human visual mapping signoff**, but still requires private image decoding/label coverage QA and split audits; no automatic legal, commercial or identity rights are granted. GTX1060 CUDA smoke (#12), true YOLO detection-loss training and board inference remain separate.

All GitHub tests use wholly synthetic fake JPEG-like bytes/fictional subjects. A green CI proves that commands and safeguards work, **NOT** that real photos were downloaded/reviewed.
