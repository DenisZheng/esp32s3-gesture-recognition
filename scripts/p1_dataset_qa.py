#!/usr/bin/env python3
"""P1 human dataset QA: build a review packet, then sign off per-sample.

The dataset builder publishes every candidate with ``P1_pass: false`` and
``human_visual_decode_and_legal_qa_status: PENDING``. Until this module existed,
nothing in the repository could legitimately change those fields, so the P0-2
training gate in ``p2_train`` could never be satisfied — the gate blocked
training but offered no way through.

What this tool does
-------------------
``prepare``  Build a private HTML packet of randomly sampled, per-class images
             with their boxes drawn on, bound to the dataset's own SHA256
             manifest. Nothing is approved here.
``verify``   Validate a filled-in review form against the packet: every sampled
             image must carry an explicit decision, counts must meet the declared
             minimum, and every image hash must still match. On success it writes
             a separate ``qa-ack`` file the builder/trainer can consume.
``status``   Show which classes still lack signatures.

Design constraints
------------------
* The reviewer must decide on *every* sampled image; a blank form never verifies.
* Image bytes are hashed before and after, so a swap is detected.
* The ack is a separate private file under ``data/`` — never committed.
* This tool records an inspection. It cannot judge whether a gesture was
  correctly labelled; only the human can.

Usage
-----
    python3 -m scripts.p1_dataset_qa prepare \
        --dataset data/processed/hagrid10-v2-512 \
        --output data/reviews/qa-v1 --per-class 24

    python3 -m scripts.p1_dataset_qa verify \
        --dataset data/processed/hagrid10-v2-512 \
        --workspace data/reviews/qa-v1 \
        --review data/reviews/qa-v1/review.json

    python3 -m scripts.p1_dataset_qa status --dataset data/processed/hagrid10-v2-512
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import shutil
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Dataset manifests retain the original HaGRID source directory names. The
# model mapping is peace->two and palm->five, but QA must bind to the source
# directories that actually exist in images/<split>/.
CLASS_NAMES = (
    "one", "peace", "three", "four", "palm", "like", "ok", "no_gesture", "call", "dislike",
)
SCHEMA = 1
MIN_PER_CLASS = 5
MAX_PER_CLASS = 60
ACK_NAME = "qa-ack.json"
ACCEPTANCE_NAME = "p1-acceptance.json"


class QAError(ValueError):
    """Refuse to produce an unverifiable QA record."""


def log(msg: str) -> None:
    print(msg, flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _private_dir(path: Path, *, data_root: Path | None = None) -> Path:
    """Insist that private QA material lives under the ignored data/ tree."""
    resolved = path.resolve()
    root = (data_root or (ROOT / "data")).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise QAError(
            f"{path} must live under a private data/ directory; QA material "
            "includes person images and reviewer identity"
        )
    return resolved


def _load_dataset(dataset: Path) -> tuple[dict, dict]:
    if not dataset.is_dir():
        raise QAError(f"dataset directory missing: {dataset}")
    agg_path = dataset / "manifest.aggregate.json"
    priv_path = dataset / "manifest.private.json"
    for path in (agg_path, priv_path):
        if not path.is_file() or path.is_symlink():
            raise QAError(f"missing or symlinked manifest: {path.name}")
    try:
        aggregate = json.loads(agg_path.read_text(encoding="utf-8"))
        private = json.loads(priv_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QAError("dataset manifests are unreadable") from exc
    if aggregate.get("model_classes") != 10:
        raise QAError("dataset is not a ten-class dataset")
    if not isinstance(private.get("sample_files"), list) or not private["sample_files"]:
        raise QAError("private sample manifest is empty")
    return aggregate, private


def _sample(rows: list[dict], per_class: int, seed: str) -> list[dict]:
    """Deterministically pick ``per_class`` rows per class, spread across splits."""
    by_class: dict[str, list[dict]] = {}
    for row in rows:
        image = row.get("image", "")
        parts = Path(image).parts
        if len(parts) < 3 or parts[0] != "images":
            continue
        source = parts[2]
        by_class.setdefault(source, []).append(row)

    picked: list[dict] = []
    for source in sorted(by_class):
        candidates = sorted(by_class[source], key=lambda r: r["image"])
        rng = random.Random(f"{seed}\0{source}")
        chosen = candidates if len(candidates) <= per_class else rng.sample(
            candidates, per_class
        )
        picked.extend(sorted(chosen, key=lambda r: r["image"]))
    return picked


def prepare(dataset: Path, output: Path, per_class: int, seed: str,
            *, data_root: Path | None = None) -> dict:
    """Create a private contact-sheet packet. Approves nothing."""
    if not MIN_PER_CLASS <= per_class <= MAX_PER_CLASS:
        raise QAError(f"per-class sample must be {MIN_PER_CLASS}..{MAX_PER_CLASS}")
    output = _private_dir(output, data_root=data_root)
    if output.exists():
        raise QAError(f"refusing to overwrite existing workspace: {output}")
    aggregate, private = _load_dataset(dataset)

    rows = _sample(private["sample_files"], per_class, seed)
    if not rows:
        raise QAError("no class-labelled samples found in the private manifest")

    counts: dict[str, int] = {}
    for row in rows:
        source = Path(row["image"]).parts[2]
        counts[source] = counts.get(source, 0) + 1
    missing = [name for name in CLASS_NAMES if counts.get(name, 0) == 0]
    if missing:
        raise QAError(f"dataset has no samples for classes: {missing}")

    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    stage = Path(tempfile.mkdtemp(prefix=".qa-packet-", dir=output.parent))
    try:
        images = stage / "images"
        images.mkdir(mode=0o700)
        samples = []
        for index, row in enumerate(rows):
            rel = Path(row["image"])
            src = dataset / rel
            if not src.is_file():
                raise QAError(f"manifest references missing image: {rel}")
            actual = _sha256(src)
            if actual != row.get("sha256"):
                raise QAError(
                    f"image bytes changed since the dataset was built: {rel}"
                )
            label_path = dataset / row["label"]
            if not label_path.is_file():
                raise QAError(f"manifest references missing label: {row['label']}")
            if _sha256(label_path) != row.get("label_sha256"):
                raise QAError(f"label changed since the dataset was built: {row['label']}")

            visible = f"{index:04d}.jpg"
            shutil.copyfile(src, images / visible)
            samples.append({
                "packet_id": visible,
                "class": rel.parts[2],
                "split": rel.parts[1],
                "image_sha256": actual,
                "label_sha256": row.get("label_sha256"),
                "label_lines": label_path.read_text(encoding="utf-8").strip().splitlines(),
            })

        manifest = {
            "schema_version": SCHEMA,
            "dataset": str(dataset.resolve()),
            "dataset_manifest_sha256": _sha256(dataset / "manifest.aggregate.json"),
            "dataset_p1_status": aggregate.get("status"),
            "per_class_requested": per_class,
            "counts_by_class": counts,
            "samples": samples,
            "status": "PACKET_GENERATED_NOT_APPROVED",
        }
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        (stage / "packet.manifest.json").write_bytes(manifest_bytes)
        (stage / "index.html").write_text(_render_html(manifest), encoding="utf-8")

        form = {
            "schema_version": SCHEMA,
            "reviewer": "",
            "reviewed_on": "",
            "packet_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "decisions": [
                {
                    "packet_id": s["packet_id"],
                    "class": s["class"],
                    "decision": "",
                    "notes": "",
                }
                for s in samples
            ],
        }
        (stage / "review.json").write_text(
            json.dumps(form, indent=2) + "\n", encoding="utf-8"
        )

        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    return {
        "status": "PACKET_GENERATED_NOT_APPROVED",
        "workspace": str(output),
        "review_page": str(output / "index.html"),
        "review_form": str(output / "review.json"),
        "samples": len(samples),
        "counts_by_class": counts,
        "approves_anything": False,
    }


def verify(dataset: Path, workspace: Path, review: Path,
           *, data_root: Path | None = None) -> dict:
    """Bind human decisions to unchanged sample bytes, then emit the ack."""
    workspace = _private_dir(workspace, data_root=data_root)
    review = review.resolve()
    manifest_path = workspace / "packet.manifest.json"
    if not manifest_path.is_file():
        raise QAError("packet manifest missing; run `prepare` first")
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        form = json.loads(review.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QAError("packet manifest or review form is unreadable") from exc

    if form.get("packet_manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest():
        raise QAError("review references a different packet manifest")

    # The packet is bound to one dataset. Without this check a signed packet could
    # be replayed against any other dataset, because the review only references a
    # manifest hash that both sides happen to share.
    current_manifest_sha = _sha256(dataset / "manifest.aggregate.json")
    if manifest.get("dataset_manifest_sha256") != current_manifest_sha:
        raise QAError(
            "this packet was built for a different dataset manifest; the dataset "
            "changed or a packet from another dataset was supplied"
        )
    if Path(manifest.get("dataset", "")).resolve() != dataset.resolve():
        raise QAError("packet was created for a different dataset directory")

    reviewer = form.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise QAError("reviewer name is required")
    try:
        date.fromisoformat(str(form.get("reviewed_on")))
    except ValueError as exc:
        raise QAError("reviewed_on must be an ISO date") from exc

    bulk_attestation = form.get("bulk_attestation")
    if bulk_attestation is not None:
        if (
            not isinstance(bulk_attestation, dict)
            or bulk_attestation.get("confirmed") is not True
            or bulk_attestation.get("statement")
            != "I reviewed every sampled image and all are OK"
        ):
            raise QAError("bulk attestation is malformed or not explicitly confirmed")

    samples = {s["packet_id"]: s for s in manifest.get("samples", [])}
    decisions = form.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != len(samples):
        raise QAError("decision count does not match the sampled images")

    seen: set[str] = set()
    per_class_ok: dict[str, int] = {}
    problems: list[str] = []
    for row in decisions:
        if not isinstance(row, dict):
            raise QAError("malformed decision row")
        pid = row.get("packet_id")
        if pid not in samples:
            raise QAError(f"decision references unknown packet id: {pid!r}")
        if pid in seen:
            raise QAError(f"duplicate decision for packet id: {pid}")
        seen.add(pid)
        choice = row.get("decision")
        if choice not in ("ok", "bad_gesture", "bad_box", "bad_image"):
            raise QAError(
                f"packet {pid}: decision must be one of ok/bad_gesture/bad_box/"
                f"bad_image, got {choice!r}"
            )
        if choice == "ok":
            source = samples[pid]["class"]
            per_class_ok[source] = per_class_ok.get(source, 0) + 1
        else:
            problems.append(f"{pid}:{choice}")
    if seen != set(samples):
        raise QAError("not every sampled image has a decision")
    if bulk_attestation is not None and problems:
        raise QAError("bulk all-OK attestation cannot contain rejected samples")

    # Re-verify bytes so a swapped image cannot ride on an old signature.
    images = workspace / "images"
    for pid, sample in samples.items():
        path = images / pid
        if not path.is_file():
            raise QAError(f"packet image missing: {pid}")
        if _sha256(path) != sample["image_sha256"]:
            raise QAError(f"packet image changed after the packet was created: {pid}")

    # Guard against a rubber-stamp: every class needs a minimum of clean samples.
    minimum = max(MIN_PER_CLASS, int(manifest["per_class_requested"]) // 2)
    for name in CLASS_NAMES:
        approved = per_class_ok.get(name, 0)
        if approved < minimum:
            raise QAError(
                f"class {name}: only {approved} sample(s) marked ok, need at least "
                f"{minimum}; a blank or rushed review cannot pass"
            )

    total = len(samples)
    ok_count = sum(per_class_ok.values())
    rate = ok_count / total if total else 0.0
    if rate < 0.9:
        raise QAError(
            f"only {rate:.1%} of sampled images were marked ok; investigate the "
            f"dataset instead of signing it off ({len(problems)} problems)"
        )

    ack = {
        "schema_version": SCHEMA,
        "dataset": str(dataset.resolve()),
        "dataset_manifest_sha256": _sha256(dataset / "manifest.aggregate.json"),
        "packet_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "reviewer": reviewer.strip(),
        "reviewed_on": form["reviewed_on"],
        "samples_reviewed": total,
        "samples_ok": ok_count,
        "ok_rate": round(rate, 4),
        "per_class_ok": per_class_ok,
        "rejected_samples": problems,
        "human_visual_decode_reviewed": True,
        "bulk_attestation": bulk_attestation,
        "note": (
            "Records that a human inspected these samples. It does not assert that "
            "the dataset is legally cleared for commercial use."
        ),
    }
    ack_path = workspace / ACK_NAME
    if ack_path.exists():
        raise QAError(f"ack already exists: {ack_path}")
    ack_path.write_text(
        json.dumps(ack, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.chmod(ack_path, 0o600)
    return {
        "status": "QA_ACKNOWLEDGED_NOT_P1_PASS",
        "ack": str(ack_path),
        "samples_reviewed": total,
        "samples_ok": ok_count,
        "ok_rate": round(rate, 4),
        "per_class_ok": per_class_ok,
        "rejected": len(problems),
        "note": "Run `apply` to fold this ack into the dataset manifest.",
    }


def apply_ack(dataset: Path, ack: Path, *, data_root: Path | None = None) -> dict:
    """Fold a verified ack into the dataset manifest.

    Only the *human visual* half of P1 is recorded here. ``P1_pass`` additionally
    needs the legal/licence clearance, so this deliberately sets
    ``human_visual_decode_and_legal_qa_status`` while leaving a separate legal
    field for a future step rather than overstating acceptance.
    """
    dataset = dataset.resolve()
    private_ack = _private_dir(ack.parent, data_root=data_root) / ack.name
    if not private_ack.is_file():
        raise QAError(f"ack file missing: {private_ack}")
    ack_doc = json.loads(private_ack.read_text(encoding="utf-8"))

    agg_path = dataset / "manifest.aggregate.json"
    priv_path = dataset / "manifest.private.json"
    aggregate = json.loads(agg_path.read_text(encoding="utf-8"))
    private = json.loads(priv_path.read_text(encoding="utf-8"))

    current = _sha256(agg_path)
    if ack_doc.get("dataset_manifest_sha256") != current:
        raise QAError(
            "ack was produced against a different dataset manifest; re-run `verify`"
        )
    if Path(ack_doc.get("dataset", "")).resolve() != dataset.resolve():
        raise QAError("ack belongs to a different dataset directory")
    if ack_doc.get("samples_reviewed", 0) < 10:
        raise QAError("ack covers too few samples")
    if ack_doc.get("ok_rate", 0) < 0.9:
        raise QAError("ack ok-rate is below the acceptance threshold")

    aggregate["human_visual_decode_and_legal_qa_status"] = "PASS"
    aggregate["human_visual_qa"] = {
        "reviewer": ack_doc["reviewer"],
        "reviewed_on": ack_doc["reviewed_on"],
        "samples_reviewed": ack_doc["samples_reviewed"],
        "samples_ok": ack_doc["samples_ok"],
        "ok_rate": ack_doc["ok_rate"],
        "per_class_ok": ack_doc["per_class_ok"],
    }
    # P1 also requires the licence/legal position to be settled. Keep that
    # explicit instead of flipping P1_pass on a visual review alone.
    aggregate["P1_pass"] = False
    aggregate["legal_clearance_status"] = aggregate.get(
        "legal_clearance_status", "UNRESOLVED_COMMERCIAL_USE"
    )

    # The aggregate hash is embedded in the private manifest, so both must be
    # rewritten together or `verify_output` rejects the dataset.
    aggregate_bytes = (
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    tmp = agg_path.with_suffix(".json.tmp")
    tmp.write_bytes(aggregate_bytes)
    private["aggregate_sha256"] = hashlib.sha256(
        json.dumps(aggregate, sort_keys=True).encode("utf-8")
    ).hexdigest()
    tmp.replace(agg_path)
    priv_path.write_text(
        json.dumps(private, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    acceptance_path = private_ack.parent / ACCEPTANCE_NAME
    if not acceptance_path.exists():
        acceptance = {
            "schema_version": SCHEMA,
            "dataset_manifest_sha256": _sha256(agg_path),
            "reviewer": "",
            "accepted_on": "",
            "scope": "research_only_training",
            "accept_research_only_terms": False,
            "commercial_use": False,
        }
        acceptance_path.write_text(
            json.dumps(acceptance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(acceptance_path, 0o600)
    return {
        "status": "HUMAN_VISUAL_QA_RECORDED",
        "dataset": str(dataset),
        "human_visual_decode_and_legal_qa_status": "PASS",
        "P1_pass": False,
        "acceptance_form": str(acceptance_path),
        "remaining_for_P1": [
            "owner must explicitly accept research-only training scope",
            "commercial use and redistribution remain prohibited/unresolved",
        ],
    }


def accept_research_only(dataset: Path, acceptance: Path, license_ack: Path,
                         *, data_root: Path | None = None) -> dict:
    """Record explicit owner acceptance for research-only training."""
    from scripts.hagrid_p1 import _approval

    dataset = dataset.resolve()
    acceptance = _private_dir(acceptance, data_root=data_root)
    if not acceptance.is_file():
        raise QAError(f"acceptance form missing: {acceptance}")
    agg_path = dataset / "manifest.aggregate.json"
    priv_path = dataset / "manifest.private.json"
    aggregate = json.loads(agg_path.read_text(encoding="utf-8"))
    private = json.loads(priv_path.read_text(encoding="utf-8"))
    form = json.loads(acceptance.read_text(encoding="utf-8"))
    current = _sha256(agg_path)
    if form.get("dataset_manifest_sha256") != current:
        raise QAError("acceptance form references a different dataset manifest")
    if aggregate.get("human_visual_decode_and_legal_qa_status") not in ("PASS", "APPROVED"):
        raise QAError("visual QA must be applied before formal training acceptance")
    reviewer = form.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise QAError("acceptance reviewer is required")
    try:
        date.fromisoformat(str(form.get("accepted_on")))
    except ValueError as exc:
        raise QAError("accepted_on must be an ISO date") from exc
    if form.get("scope") != "research_only_training":
        raise QAError("only research_only_training scope is permitted")
    if form.get("accept_research_only_terms") is not True:
        raise QAError("research-only terms must be explicitly accepted")
    if form.get("commercial_use") is not False:
        raise QAError("commercial_use must remain false")
    _approval(license_ack, "license")

    aggregate["P1_pass"] = True
    aggregate["legal_clearance_status"] = "RESEARCH_ONLY_TRAINING_ONLY"
    aggregate["p1_formal_acceptance"] = {
        "reviewer": reviewer.strip(),
        "accepted_on": form["accepted_on"],
        "scope": "research_only_training",
        "commercial_use": False,
    }
    agg_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    private["aggregate_sha256"] = hashlib.sha256(
        json.dumps(aggregate, sort_keys=True).encode("utf-8")
    ).hexdigest()
    priv_path.write_text(
        json.dumps(private, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "status": "P1_ACCEPTED_FOR_RESEARCH_ONLY_TRAINING",
        "dataset": str(dataset),
        "P1_pass": True,
        "legal_clearance_status": "RESEARCH_ONLY_TRAINING_ONLY",
        "commercial_use": False,
    }


def status(dataset: Path) -> dict:
    aggregate, _ = _load_dataset(dataset)
    return {
        "dataset": str(dataset),
        "status": aggregate.get("status"),
        "P1_pass": aggregate.get("P1_pass"),
        "human_visual_decode_and_legal_qa_status": aggregate.get(
            "human_visual_decode_and_legal_qa_status"
        ),
        "human_visual_qa": aggregate.get("human_visual_qa"),
        "legal_clearance_status": aggregate.get("legal_clearance_status"),
    }


def _render_html(manifest: dict) -> str:
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>P1 dataset QA packet</title>",
        "<style>body{font-family:sans-serif;margin:16px}"
        ".c{display:inline-block;width:220px;margin:6px;vertical-align:top}"
        "img{width:220px;height:auto;border:1px solid #ccc}"
        "code{font-size:11px}</style>",
        "<h1>P1 dataset QA packet</h1>",
        "<p>Every sample must receive a decision in <code>review.json</code>. "
        "Decision values: <code>ok</code>, <code>bad_gesture</code>, "
        "<code>bad_box</code>, <code>bad_image</code>.</p>",
        f"<p>Samples: {len(manifest['samples'])} — "
        f"{html.escape(json.dumps(manifest['counts_by_class'], sort_keys=True))}</p>",
    ]
    current = None
    for sample in manifest["samples"]:
        if sample["class"] != current:
            if current is not None:
                parts.append("</div>")
            current = sample["class"]
            parts.append(f"<h2>{html.escape(current)}</h2><div>")
        boxes = "; ".join(sample["label_lines"])
        parts.append(
            "<div class='c'>"
            f"<img src='images/{html.escape(sample['packet_id'])}' loading='lazy'>"
            f"<div><code>{html.escape(sample['packet_id'])}</code></div>"
            f"<div><code>{html.escape(sample['split'])}</code></div>"
            f"<div><code>{html.escape(boxes)}</code></div>"
            "</div>"
        )
    if current is not None:
        parts.append("</div>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-class", type=int, default=24)
    p.add_argument("--seed", default="p1-qa-v1")

    v = sub.add_parser("verify")
    v.add_argument("--dataset", type=Path, required=True)
    v.add_argument("--workspace", type=Path, required=True)
    v.add_argument("--review", type=Path, required=True)

    a = sub.add_parser("apply")
    a.add_argument("--dataset", type=Path, required=True)
    a.add_argument("--ack", type=Path, required=True)

    f = sub.add_parser("accept")
    f.add_argument("--dataset", type=Path, required=True)
    f.add_argument("--acceptance", type=Path, required=True)
    f.add_argument("--license-ack", type=Path, required=True)

    s = sub.add_parser("status")
    s.add_argument("--dataset", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.dataset, args.output, args.per_class, args.seed)
        elif args.command == "verify":
            result = verify(args.dataset, args.workspace, args.review)
        elif args.command == "apply":
            result = apply_ack(args.dataset, args.ack)
        elif args.command == "accept":
            result = accept_research_only(
                args.dataset, args.acceptance, args.license_ack
            )
        else:
            result = status(args.dataset)
    except QAError as exc:
        raise SystemExit(f"BLOCKED: {exc}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
