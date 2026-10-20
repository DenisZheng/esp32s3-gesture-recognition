# Class semantics — review required before P1 PASS

Canonical source of truth: `configs/classes.json`; generated/training YAML must agree exactly.

| ID | Output | HaGRID candidate | Visual confirmation against Espressif illustration | Notes |
|---:|---|---|---|---|
| 0 | one | one | PENDING | Inspect left/right index finger and flips |
| 1 | two | two_up | **PENDING** | Compare official "two" sign, avoid `peace`/`two_up_inverted` conflation |
| 2 | three | three | PENDING | Explicitly compare `three2`, `three3`, `three_gun` signs |
| 3 | four | four | PENDING | Different orientations/handedness |
| 4 | five | palm | **PENDING** | Check official open-five vs HaGRID palm and stop |
| 5 | like | like | PENDING | Compare `dislike`, other thumbs poses |
| 6 | ok | ok | PENDING | Index-thumb circle visibility at 224 |
| 7 | no_gesture | no_gesture | PENDING | Per-hand box and other non-command conversion rule |
| 8 | call | call | PENDING | Compare official "call" orientation |
| 9 | dislike | dislike | PENDING | Contrast thumb-down vs folded hand |

Official reference: https://github.com/espressif/esp-dl/blob/v3.3.14/examples/hand_gesture_recognition/gestures.png

HaGRID source: https://github.com/hukenovs/hagrid/blob/Hagrid_v2-1M/images/gestures.png

**Decision gate**: for each class, record sample IDs only in private local QA, reviewer and review date, accepted / rejected visual variants, mapping rationale and any confusion notes. Never change class indices without an explicit versioned compatibility break. In case of mismatch, refine allowed source selection or gather separately licensed data; do not silently change target ten classes.


## Figure-to-figure comparison: 2026-10-20 (NOT a signed sample review)

The ESP-DL v3.3.14 official gesture illustration (Git blob `64c4e7c856780d50ecfe17f62f94d491362c48d8`) shows `two` with separated index/middle fingers (V). The HaGRIDv2 official illustrated catalog (Git blob `c766c4eec61481326235c0dc8d464d5c6ff9d9b1`) shows `two_up` with much closer fingers; HaGRID `peace` appears more similar to ESP-DL's V. Therefore **`two_up → two` is an elevated-risk provisional mapping, NOT approved**. Do not train `peace` as class-7 negative; keep it and its variants in quarantine.

`palm → five` appears similar at figure-level, but `stop` variants may be confusable; licensed real samples, rotations, mirrors, multiple hands, distance and occlusion must still be reviewed. Full 34-source policy: [HARD_NEGATIVES.md](HARD_NEGATIVES.md).
