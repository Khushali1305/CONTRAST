"""
gec_probe.data.stratified_sample
--------------------------------
Sample TARGET_BASE_ITEMS sentences from the normalized BEA-2019 ingestion,
stratified across (CEFR level, primary error type, length quartile, density).
"""
from __future__ import annotations
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List

from gec_probe.config.settings import (
    NORMALIZED_DIR, PROBES_DIR, TARGET_BASE_ITEMS,
    CEFR_LEVELS, ERROR_TYPE_BUCKETS, LENGTH_QUARTILES,
)

RANDOM_SEED = 20260101


def _load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def _primary_error_bucket(types: List[str]) -> str:
    """Pick first matching bucket; otherwise R:OTHER."""
    for t in types:
        if t in ERROR_TYPE_BUCKETS:
            return t
    return "R:OTHER"


def _density(rec: dict) -> str:
    n = rec["n_edits_canonical"]
    if n == 0:
        return "zero"
    if n == 1:
        return "single"
    if n <= 3:
        return "low"
    return "high"


def _length_quartile_thresholds(records: List[dict]) -> List[int]:
    lengths = sorted(r["n_tokens"] for r in records)
    qs = []
    for q in range(1, LENGTH_QUARTILES):
        qs.append(lengths[int(len(lengths) * q / LENGTH_QUARTILES)])
    return qs


def _length_bucket(n_tokens: int, thresholds: List[int]) -> int:
    for i, t in enumerate(thresholds):
        if n_tokens <= t:
            return i
    return len(thresholds)


def build_probecore_base(
    splits: List[str] = ("train", "dev"),
    target: int = TARGET_BASE_ITEMS,
) -> Path:
    """Build the stratified base set; emit JSONL."""
    rng = random.Random(RANDOM_SEED)

    records = []
    for split in splits:
        records.extend(_load_jsonl(NORMALIZED_DIR / f"bea2019_{split}.jsonl"))

    # Keep only sentences with >=1 canonical edit (otherwise no error to probe).
    records = [r for r in records if r["n_edits_canonical"] >= 1]

    # Strata
    thresholds = _length_quartile_thresholds(records)
    buckets: dict = defaultdict(list)
    for r in records:
        key = (
            r["cefr_level"] if r["cefr_level"] in CEFR_LEVELS else "X",
            _primary_error_bucket(r["errant_error_types"]),
            _length_bucket(r["n_tokens"], thresholds),
            _density(r),
        )
        buckets[key].append(r)

    # Proportional allocation with floor=1 per nonempty bucket.
    total = sum(len(v) for v in buckets.values())
    sampled: List[dict] = []
    for key, items in buckets.items():
        share = max(1, round(target * len(items) / total))
        rng.shuffle(items)
        sampled.extend(items[:share])

    # Trim/extend to exact target.
    rng.shuffle(sampled)
    if len(sampled) > target:
        sampled = sampled[:target]
    elif len(sampled) < target:
        # back-fill from the pool of unused records, error-type stratified
        used_ids = {r["sentence_id"] for r in sampled}
        leftover = [r for r in records if r["sentence_id"] not in used_ids]
        rng.shuffle(leftover)
        sampled.extend(leftover[: target - len(sampled)])

    # Attach preferred_probe_family (mirrors SRS schema convention).
    for r in sampled:
        primary = _primary_error_bucket(r["errant_error_types"])
        if primary.startswith("R:WO") or primary.startswith("R:ORTH"):
            r["preferred_probe_family"] = "invariance"
        elif primary.startswith(("R:VERB", "R:NOUN", "R:MORPH")):
            r["preferred_probe_family"] = "directional"
        else:
            r["preferred_probe_family"] = "overcorrection"

    out = PROBES_DIR / "gec_probecore_base.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in sampled:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[strat] {len(sampled)} items -> {out}")
    return out


if __name__ == "__main__":
    build_probecore_base()
