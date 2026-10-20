# P1-05 — Ten-class gesture recognition with reviewed hard negatives

**Status: policy and synthetic tests implemented; real HaGRID images and owner signoffs still PENDING.**

## Frozen architecture and original 34-class taxonomy

HaGRIDv2-1M [official constants.py](https://github.com/hukenovs/hagrid/blob/Hagrid_v2-1M/constants.py) lists **33 gesture source labels plus no_gesture = 34 raw source categories**. The ESP32-S3 project's **single ESPDet-Pico** remains `224x224`, `nc=10`; no MobileNetV2 or additional model/class is introduced.

Our nine commands are `one, two, three, four, five, like, ok, call, dislike`. Output index 7 `no_gesture` is **a real visible hand with a bbox**, not an empty background sample. `no_hand` is a separately verified hand-free negative with **no YOLO row**.

The pinned 34-class registry [configs/hagrid_v2_source_policy.json](../configs/hagrid_v2_source_policy.json) partitions sources:

| Group | Count | Names | Conversion |
|---|---:|---|---|
| Primary model source labels | 10 | `one, two_up, three, four, palm, like, ok, no_gesture, call, dislike` | All ten real-sample visual mappings must be signed |
| Candidate *single-hand* noncommands | 7 | `fist, rock, grabbing, grip, little_finger, middle_finger, thumb_index` | May be converted to class 7 **only after separate per-class signed visual approval** |
| **Lookalikes held in quarantine** | 10 | `peace, peace_inverted, two_up_inverted, three2, three3, three_gun, stop, stop_inverted, point, mute` | **Blocked** until a versioned policy review and real-sample adjudication |
| **Compound two-hand categories held in quarantine** | 7 | `holy, timeout, xsign, hand_heart, hand_heart2, take_picture, thumb_index2` | **Blocked**; group/union boxes are not physical-hand boxes |

These are 34 source *types*, not counts of accepted training images. Every image may have multiple hands with different per-hand labels. There is **no automatic 24-class-to-no_gesture mapping**.

## P0-10 critical finding: number-two source is provisional

The [ESP-DL v3.3.14 official gesture figure](https://github.com/espressif/esp-dl/blob/v3.3.14/examples/hand_gesture_recognition/gestures.png) (Git blob `64c4e7c856780d50ecfe17f62f94d491362c48d8`) shows `two` with a separated **V**. The [HaGRIDv2 figure](https://github.com/hukenovs/hagrid/blob/Hagrid_v2-1M/images/gestures.png) (Git blob `c766c4eec61481326235c0dc8d464d5c6ff9d9b1`) shows `two_up` with closer, almost parallel fingers, while `peace` seems more like the official V.

**This is evidence of a potential mapping mismatch, not a completed or signed real-data review.** Keep the source `two_up -> two` provisional in `configs/classes.json` pending licensed examples. NEVER train `peace` as `no_gesture` in the current policy. Review `two_up_inverted` and `peace_inverted`, the `three` variants, and `stop` versus `palm` before selecting any confusing hard negatives. A revised mapping requires a deliberate policy/version PR and regenerated labels; no stealth index changes.

## Review requirements: three distinct gates

1. **HaGRID custom license**: owner's signed research-only acceptance of the *verified* non-CC custom PDF and its SHA256, stored only in ignored `data/reviews/hagrid-license-ack.json`. This is NOT commercial/legal clearance.
2. **Ten target mappings**: privately signed licensed-image visual check of each mapping, especially `two_up`/`peace` and `palm`/`stop`; stored in `data/reviews/hagrid-mapping-ack.json`.
3. **Optional hard negatives**: a separately signed, private file `data/reviews/p1-hard-negative-ack.json` that records reviewer, real review date, **SHA256 of the exact committed source policy JSON**, and the exact approved subset of the seven reviewable sources. Each entry needs valid image review, checked physical-hand boxes, explicit approval and positive sample count. See deliberately *unsigned* [example](templates/p1-hard-negative-ack.example.json).

If a policy file changes, its SHA256 changes and all old negative approvals become **invalid**. No method accepts `peace` or compound classes as class-7 negatives in this policy version, even with a forged approval entry.

## Read-only data inspection, then local-only approved conversion

To see compressed metadata sizes (no subject records downloaded), use:

    python3 -m scripts.hagrid_zip_range inspect --sources fist peace

After user license acceptance, on a **local private machine only**, selectively extract *annotation metadata*, not the full archive:

    python3 -m scripts.hagrid_zip_range extract \
      --sources fist \
      --output data/raw/hagrid_negatives_fist \
      --license-ack data/reviews/hagrid-license-ack.json \
      --max-transfer-mib 64

Metadata audits may inspect any known 34 sources. Until the negative review is signed, candidate hands are **rejected, not silently used**:

    python3 -m scripts.hagrid_p1 audit \
      --annotations data/raw/hagrid_negatives_fist --sources fist

After all approvals, when original ten JSONs + optional selected negative JSONs and licensed JPEGs are present under **combined** local roots:

    python3 -m scripts.hagrid_p1 convert \
      --annotations data/raw/combined_annotations \
      --images data/raw/hagrid_dataset \
      --sources one two_up three four palm like ok no_gesture call dislike fist \
      --license-ack data/reviews/hagrid-license-ack.json \
      --mapping-ack data/reviews/hagrid-mapping-ack.json \
      --hard-negative-sources fist \
      --hard-negative-ack data/reviews/p1-hard-negative-ack.json \
      --output data/processed/hagrid10_micro_reviewed_fist \
      --limit-per-source 5

**Not yet executed**: no real HaGRID data or private identity fields were used in this PR. The converter rejects unknown/ambiguous second hands by rejecting the **whole frame**, not by silently omitting the unwanted bbox. A reviewed additional `fist` hand gets bbox + class 7, while an adjacent valid `like` hand retains class 5. No `no_hand` negative is automatically created. The model still has ten output classes.

## P1 acceptance still blocked by human evidence

Owner/research license acceptance (Issue #2), ten licensed-sample gesture mapping adjudication (Issue #3), small local subset + visual multi-hand QA + separated subjects + real negatives (Issue #4) are **PENDING**. Synthetic-only Actions PASS validates source safeguards, not real model accuracy, input image coverage or CoreS3 performance.
