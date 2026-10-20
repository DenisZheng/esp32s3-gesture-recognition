#!/usr/bin/env python3
"""Fetch source-only upstream exact commit pins on explicit local invocation.

No dataset download. Never run during P0 CI. Requires network, git and license review.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(args, cwd=cwd or ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", action="append", required=True, help="Repeat id per wanted source; excluding full HaGRID image dataset")
    args = parser.parse_args()
    deps = {d["id"]: d for d in json.loads((ROOT / "upstreams.lock.json").read_text()) ["dependencies"]}
    for key in args.id:
        if key not in deps:
            raise SystemExit(f"Unknown source id: {key}")
        dep = deps[key]
        dst = ROOT / "external" / key
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            print(f"Fetching {key} git *source code* into ignored external/", flush=True)
            run(["git", "clone", "--filter=blob:none", "--no-checkout", dep["url"], str(dst)])
        run(["git", "fetch", "origin", dep["commit"]], cwd=dst)
        run(["git", "checkout", "--detach", dep["commit"]], cwd=dst)
        got = run(["git", "rev-parse", "HEAD"], cwd=dst)
        if got != dep["commit"]:
            raise RuntimeError(f"Source revision mismatch for {key}")
        print(f"Pinned {key} @ {got}")


if __name__ == "__main__":
    main()
