"""
gec_probe.data.ingest_bea2019
-----------------------------
Read W&I+LOCNESS .m2 files and emit one normalized JSONL row per sentence.

M2 format reminder
------------------
S <tokenized source sentence>
A start end|||TYPE|||correction|||REQUIRED|||-NONE-|||annotator_id
...
(blank line)

We keep annotator 0 as the canonical reference (BEA-2019 convention).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator
from collections import defaultdict

from gec_probe.config.settings import (
    RAW_DIR, NORMALIZED_DIR, SPLIT_FILES,
)


def _doc_id_to_cefr(doc_id: str) -> str:
    """W&I files are named A/B/C/N depending on CEFR level."""
    head = doc_id[:1].upper()
    return head if head in {"A", "B", "C", "N"} else "X"


def _apply_edits(tokens: list, edits: list) -> str:
    """Apply ERRANT edits (annotator 0) to produce a reference correction."""
    # Edits are token-index spans. Apply in reverse so indices stay valid.
    new_tokens = list(tokens)
    for e in sorted(edits, key=lambda x: -x["start"]):
        if e["type"] == "noop":
            continue
        replacement = e["correction"].split() if e["correction"] else []
        new_tokens[e["start"]:e["end"]] = replacement
    return " ".join(new_tokens)


def parse_m2(path: Path) -> Iterator[dict]:
    """Yield one record per sentence in the m2 file."""
    block: list = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "":
                if block:
                    yield _block_to_record(block, path.name)
                    block = []
            else:
                block.append(line)
        if block:
            yield _block_to_record(block, path.name)


def _block_to_record(block: list, source_file: str) -> dict:
    src_line = block[0]
    assert src_line.startswith("S "), f"Malformed block: {src_line[:40]}"
    tokens = src_line[2:].split()
    edits = []
    edits_by_annotator = defaultdict(list)
    for ln in block[1:]:
        if not ln.startswith("A "):
            continue
        # A start end|||TYPE|||correction|||REQUIRED|||-NONE-|||annotator
        body = ln[2:]
        span_part, rest = body.split("|||", 1)
        start_s, end_s = span_part.split()
        parts = rest.split("|||")
        etype = parts[0]
        correction = parts[1]
        annotator = parts[-1]
        edits_by_annotator[annotator].append({
            "start": int(start_s),
            "end":   int(end_s),
            "type":  etype,
            "correction": correction,
        })

    # Canonical = annotator 0; keep others as alt references.
    canonical = edits_by_annotator.get("0", [])
    ref = _apply_edits(tokens, canonical) if canonical else " ".join(tokens)
    alt_refs = []
    for ann_id, ann_edits in edits_by_annotator.items():
        if ann_id == "0":
            continue
        alt_refs.append(_apply_edits(tokens, ann_edits))

    error_types = sorted({e["type"] for e in canonical if e["type"] != "noop"})

    return {
        "dataset": "BEA2019_WI_LOCNESS",
        "record_type": "normalized_ingestion_row",
        "source_file": source_file,
        "source_text": " ".join(tokens),
        "reference_correction": ref,
        "alt_references": alt_refs,
        "errant_error_types": error_types,
        "n_edits_canonical": len([e for e in canonical if e["type"] != "noop"]),
        "n_tokens": len(tokens),
    }


def ingest_split(split_name: str) -> Path:
    """Ingest a single split into JSONL. Returns output path."""
    m2_path = RAW_DIR / SPLIT_FILES[split_name]
    if not m2_path.exists():
        raise FileNotFoundError(
            f"Expected M2 file at {m2_path}. "
            f"Download BEA-2019 W&I+LOCNESS into {RAW_DIR} first."
        )
    out_path = NORMALIZED_DIR / f"bea2019_{split_name}.jsonl"
    n = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for i, rec in enumerate(parse_m2(m2_path)):
            rec["sentence_id"] = f"BEA_{split_name.upper()}_{i:06d}"
            rec["cefr_level"] = _doc_id_to_cefr(rec["source_file"])
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"[ingest] {split_name}: {n} records -> {out_path}")
    return out_path


if __name__ == "__main__":
    for split in ("train", "dev"):
        ingest_split(split)
