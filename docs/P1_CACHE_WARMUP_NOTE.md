# P1 cache warm-up operations note

**Status:** operating note for the owner's private RTX 4080 host. It records
field experience only; it is not a P1 PASS and grants no data licence.

## Why the dataset build needs a separate warm-up step

`p1_dataset_builder build` is deliberately all-or-nothing: it stages the whole
ten-class set and refuses to publish if any class misses its per-split quota.
That is the right safety property for a training input, but it makes `build` a
poor download tool — a failure in the last class discards the staged output and
the run looks like total loss even though the private cache is intact.

`_prefetch_all.py` therefore fills `data/cache/` only. It never writes a
dataset, never converts, never trains. Once the cache is complete, `build` runs
from cache in minutes and can be repeated cheaply.

## Measured behaviour on the owner host (2026-10-22/23)

| Observation | Value | Consequence |
|---|---|---|
| Single-thread range reads | ~6 img/min | Unusable for a 6000-image build |
| 8 threads | ~37 img/min | First workable setting |
| 16 threads | ~29 img/min on `call`/`dislike` | Source-dependent, not universal |
| `call` source alone, 8 threads | 11.7 img/min | Slowest measured source |
| Pinned ZIP index read (36–40 GiB archive) | 14–33 s | Not the bottleneck; do not optimise it |
| Cached after interrupted runs | 7 957 JPEG / 11 GiB | Resume works; nothing was lost |

**Rule of thumb:** size the estimate from the slowest *remaining* source, not
from the fastest observed average. Estimating 6000 images from a 37 img/min
average on the fast classes under-predicted the tail by roughly 3x.

## Ordering matters more than parallelism

`SOURCES` order in the builder puts `call` and `dislike` last. A full build
started in that order spent hours on fast classes and had not downloaded a
single `call` image when it was stopped. `_prefetch_all.py` sorts pending work
by descending gap so a slow source is discovered in the first minutes instead
of the last hour.

The same mistake appears in the prefetch window: sizing one window with
`max(counts[s] ...)` for every split made `val`/`test` fetch as much as `train`.
Measured waste was about 5x on the 500/50/50 attempt. The window must be per
split.

## Transport faults are not integrity failures

The HaGRID object store closes long-lived connections
(`http.client.RemoteDisconnected`). The original retry predicate only accepted
`URLError` and a set of HTTP status codes, so one dropped connection aborted an
hours-long run with the misleading message "Private source failed
integrity/size/permission checks".

The retry predicate now treats `HTTPException`, `ConnectionError`,
`TimeoutError`, `socket.timeout` and `URLError` as transient, with exponential
backoff and jitter, while **safety** failures still fail immediately and are
never retried:

* HTTP 200 instead of 206 (server ignored Range → would mean a bulk download)
* changed `Content-Range`, `ETag` or `Last-Modified`
* short/oversized payload, CRC32 mismatch, bad JPEG signature
* unpinned central-directory SHA256 or archive size

A byte-budget stop is a `BudgetError` and must abort the build; a per-image
transport failure may be skipped so the rest of the run continues.

## Windows-specific test failures are expected

The repository's synthetic test suite has 29 pre-existing failures on a Windows
workstation (GBK default decoding in `read_text()`, `os.fchmod` absent,
symlink creation requiring privileges). The correct acceptance criterion for a
change made from Windows is **no new failures against that baseline**, not a
green suite. Establish the baseline before editing:

```powershell
python -m pytest tests/ -q 2>&1 | Select-String "^FAILED" | Sort-Object
```

## Pre-flight for a long unattended run

1. `python3 _prefetch_all.py --report-only` — prints pending counts per class
   without downloading. Confirm the totals are plausible before committing a
   weekend to it.
2. `python3 _selfcheck.py` — offline static checks on the warmer's control flow.
3. Start under `tmux`, never a bare SSH command; the host reboots (and this
   dual-boot machine may come back in Windows instead of Linux).
4. Watch `runs/weekend-<quota>.log`. Exit codes: `0` cache complete, `1` report
   only, `2` stopped after N passes, `3` stalled with no progress.

## Do not

* Do not `rm -rf data/cache` to "start clean". The cache is CRC-checked and
  resumable; deleting it costs hours.
* Do not delete a staged `.p1-05-dataset-*` directory while a build is running.
* Do not treat `nvidia-smi` alone as GPU readiness — the driver/library mismatch
  on this host required a reboot before any container could see the GPU.
* Do not reboot this dual-boot machine casually; it boots Windows by default and
  someone must select Linux at the console.
