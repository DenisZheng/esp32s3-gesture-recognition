#!/usr/bin/env python3
"""P2 helper: generate an ignored 10-class YAML from a pinned upstream file.

Does not vendor upstream code in this repository. Generated source still retains
upstream copyright/license and must comply with AGPL obligations.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def render(src: str) -> str:
    new, count = re.subn(r"^nc:\s*1\s*(?:#.*)?$", "nc: 10 # ten gesture detection classes", src, count=1, flags=re.M)
    if count != 1:
        raise ValueError("Expected exactly one original 'nc: 1' declaration")
    if "ESPDetect, [nc]" not in new:
        raise ValueError("Not the expected upstream ESPDet-Pico architecture")
    return new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-yaml", required=True, type=Path)
    parser.add_argument("--output", default="runs/config/espdet_pico_10.yaml", type=Path)
    args = parser.parse_args()
    src = args.upstream_yaml.read_text(encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(src), encoding="utf-8")
    print(f"Generated ignored experimental config: {args.output}")


if __name__ == "__main__":
    main()
