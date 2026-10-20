# Reproducibility and experiment artifact registry

All training/quantization/firmware binary outputs (and any media) must be stored outside Git under ignored `runs/` / `artifacts/` / `data/`. Store only a **redacted metadata-only** manifest in a PR when an experiment is completed. Use `templates/experiment.json` as a starting schema; no metric set to zero if unmeasured; use `null` and status `PENDING`.

Experiment ID format: `YYYYMMDD-stage-shortslug-seed` (e.g. `20261020-p2-scratch-s42`). Must include config SHA256, git commit, 5 upstream pins, dataset manifest SHA256, exact command and package environment, split/grouping seed, GPU/board details, measured metrics, all model artifact hashes and pass criteria.

Never attach original user IDs, individual images/video, training weights, data archives or firmware to PRs/Actions. If a review requires visual evidence, use a separately authorized private local review channel.
