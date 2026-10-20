# Data boundary

All raw images, annotations, subject IDs, private samples, intermediate YOLO files, and calibration images live **only under ignored `data/` paths**. Never publish individual HaGRID images or personal identifiers in commits, issues, CI artifacts, or screenshots without rights/privacy review.

P1 sample layout after licensed acquisition (not created here):

```text
data/raw/hagrid_v2/       # local licensed subset only
 data/processed/hagrid10/images/{train,val,test}
 data/processed/hagrid10/labels/{train,val,test}
 data/manifests/          # user_id group split, provenance, SHA256, all private
```

Only hand-free, fully audited images may have empty `.txt` labels. Uncertain/missing hand labels => reject or manual review, NOT treat as no_hand.


P1 audit/guarded micro-conversion: [docs/P1_DATA_TOOLING.md](../docs/P1_DATA_TOOLING.md). No automatic downloads; signed license and ten visual mappings are required before local media conversion. Source annotation layout is `data/raw/hagrid_annotations/{train,val,test}/{source}.json`, and source images are `data/raw/hagrid_dataset/{source}/{stem}.jpg`. Only aggregate manifests may be cited publicly.
