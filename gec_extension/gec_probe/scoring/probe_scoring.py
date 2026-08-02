"""
gec_probe.scoring.probe_scoring
-------------------------------
Score each (model, probe) pair and aggregate per family with item-clustered
bootstrap CIs. Sentinel-aware: invalid probes (excluded by rule filter or
judges) are NEVER scored as 0; they are dropped from the denominator.

Per-family scoring rules
------------------------
INVARIANCE
    Pass if model's corrections on src and on probe are SEMANTICALLY
    ALIGNED. Operationally: edit_signature(src->src_out) and
    edit_signature(probe->probe_out) should reflect corrections of the
    SAME error type set. We compute set equality of corrected error types.

DIRECTIONAL
    Pass if the DELTA between model's corrections (probe_out vs src_out)
    reflects the planned delta (added/removed/swapped types).

OVERCORRECTION
    Pass if the model's edits to the probe stay WITHIN the protected span.
    Any edits outside the protected span are counted as overcorrection.
"""
from __future__ import annotations
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple

from gec_probe.config.settings import (
    INFERENCE_DIR, SCORES_DIR, CORRECTOR_MODELS, SENTINEL_INVALID,
    JUDGE_DIR, PROBES_DIR,
)
from gec_probe.probes.errant_utils import (
    error_types, spans_outside,
)


# ------------------------------------------------------------------
# Per-pair scoring
# ------------------------------------------------------------------
def score_invariance(src: str, src_out: str,
                     probe_src: str, probe_out: str) -> int:
    types_src = set(error_types(src, src_out))
    types_probe = set(error_types(probe_src, probe_out))
    # NOTE on interpretation: if both edit sets are empty (model did nothing
    # on both inputs), this returns 1. That is "aligned inaction", which is
    # technically invariance but masks the case where the model is simply
    # silent. For RQ2 reporting, consider also computing an "active_invariance"
    # variant that excludes cases where len(types_src)==0==len(types_probe).
    if not types_src and not types_probe:
        return 1   # both left unchanged is also alignment
    return int(types_src == types_probe)


def score_directional(src: str, src_out: str,
                      probe_src: str, probe_out: str,
                      expected_added: list, expected_removed: list) -> int:
    types_src = set(error_types(src, src_out))
    types_probe = set(error_types(probe_src, probe_out))

    actual_added = types_probe - types_src
    actual_removed = types_src - types_probe

    add_ok = set(expected_added or []).issubset(actual_added)
    remove_ok = set(expected_removed or []).issubset(actual_removed)
    return int(add_ok and remove_ok)


def score_overcorrection(probe_src: str, probe_out: str,
                         protected_span: Tuple[int, int]) -> int:
    """Pass if no edits land outside the protected span."""
    outside = spans_outside(probe_src, probe_out, protected_span)
    return int(len(outside) == 0)


# ------------------------------------------------------------------
# Output loading
# ------------------------------------------------------------------
def _load_outputs(model_id: str) -> dict:
    """Returns {probe_id_or_src_key: output_text}."""
    out_dir = INFERENCE_DIR / model_id
    out = {}
    if not out_dir.exists():
        return out
    for f in out_dir.glob("*.json"):
        try:
            payload = json.load(f.open())
            out[payload["probe_id"]] = payload
        except Exception:
            continue
    return out


def _load_jsonl(p: Path) -> List[dict]:
    return [json.loads(l) for l in p.open() if l.strip()]


# ------------------------------------------------------------------
# Bootstrap CI (item-clustered)
# ------------------------------------------------------------------
def item_clustered_bootstrap(
    pairs: List[Tuple[str, int]], n_iter: int = 2000, seed: int = 7,
) -> Tuple[float, float, float]:
    """
    pairs: list of (item_id, 0/1). Returns (mean, lo_95, hi_95).
    """
    if not pairs:
        return (float("nan"), float("nan"), float("nan"))
    by_item: dict = defaultdict(list)
    for iid, s in pairs:
        by_item[iid].append(s)
    items = list(by_item.keys())
    rng = random.Random(seed)
    means = []
    for _ in range(n_iter):
        sample = [rng.choice(items) for _ in items]
        total = 0
        count = 0
        for it in sample:
            for s in by_item[it]:
                total += s
                count += 1
        means.append(total / count if count else 0.0)
    means.sort()
    mean = sum(s for _, s in pairs) / len(pairs)
    lo = means[int(0.025 * n_iter)]
    hi = means[int(0.975 * n_iter)]
    return mean, lo, hi


# ------------------------------------------------------------------
# Main scoring loop
# ------------------------------------------------------------------
def score_all() -> Path:
    base_items = {r["sentence_id"]: r for r in _load_jsonl(
        PROBES_DIR / "gec_probecore_base.jsonl")}
    accepted = {r["probe_id"]: r for r in _load_jsonl(
        JUDGE_DIR / "gec_judge_decisions.jsonl") if r.get("accepted")}
    probe_pool = {}
    for fname in ("gec_probes_invariance.jsonl",
                  "gec_probes_directional.jsonl",
                  "gec_probes_overcorrection.jsonl"):
        for p in _load_jsonl(PROBES_DIR / fname):
            probe_pool[p["probe_id"]] = p

    rows = []
    for m in CORRECTOR_MODELS:
        outputs = _load_outputs(m["id"])
        per_family: dict = defaultdict(list)

        for pid, jrec in accepted.items():
            probe = probe_pool.get(pid)
            if not probe:
                continue
            base = base_items.get(probe["base_item_id"])
            if not base:
                continue
            src_key = f"{probe['base_item_id']}__src"

            src_out_pl = outputs.get(src_key)
            probe_out_pl = outputs.get(pid)
            if not (src_out_pl and probe_out_pl):
                # Sentinel: missing output is NOT scored as 0.
                per_family[probe["family"]].append(
                    (probe["base_item_id"], SENTINEL_INVALID))
                continue

            src_out = src_out_pl["output"]
            probe_out = probe_out_pl["output"]

            fam = probe["family"]
            if fam == "invariance":
                s = score_invariance(
                    base["source_text"], src_out,
                    probe["probe_source"], probe_out,
                )
            elif fam == "directional":
                s = score_directional(
                    base["source_text"], src_out,
                    probe["probe_source"], probe_out,
                    probe.get("delta_added_types", []),
                    probe.get("delta_removed_types", []),
                )
            elif fam == "overcorrection":
                s = score_overcorrection(
                    probe["probe_source"], probe_out,
                    tuple(probe["protected_span"]),
                )
            else:
                continue
            per_family[fam].append((probe["base_item_id"], s))

        # Aggregate, dropping sentinel values from numerator/denominator
        row = {"model": m["id"], "family": m["family"]}
        global_pairs = []
        for fam, pairs in per_family.items():
            scored = [(i, s) for (i, s) in pairs if s != SENTINEL_INVALID]
            mean, lo, hi = item_clustered_bootstrap(scored)
            row[f"{fam}_reliability"] = mean
            row[f"{fam}_ci_lo"] = lo
            row[f"{fam}_ci_hi"] = hi
            row[f"{fam}_passed"] = sum(s for _, s in scored)
            row[f"{fam}_total"] = len(scored)
            row[f"{fam}_excluded"] = len(pairs) - len(scored)
            global_pairs.extend(scored)
        gm, glo, ghi = item_clustered_bootstrap(global_pairs)
        row["global_reliability"] = gm
        row["global_ci_lo"] = glo
        row["global_ci_hi"] = ghi
        rows.append(row)

    out_path = SCORES_DIR / "rq1_reliability_gec.jsonl"
    with out_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[score] wrote {len(rows)} model rows -> {out_path}")
    return out_path


if __name__ == "__main__":
    score_all()
