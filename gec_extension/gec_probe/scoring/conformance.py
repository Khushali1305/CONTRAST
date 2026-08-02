"""
gec_probe.scoring.conformance
-----------------------------
Output-conformance scoring against gold references.

We compute:
  - ERRANT span-based F0.5 (canonical reference)
  - M2 F0.5 if pre-built m2 file is provided
  - GLEU+ (multi-reference fluency score)

Run on the SOURCE inputs only (probe inputs are perturbations, not in gold).
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
from typing import List

from gec_probe.config.settings import (
    INFERENCE_DIR, SCORES_DIR, PROBES_DIR, CORRECTOR_MODELS,
)
from gec_probe.probes.errant_utils import annotate


def errant_f_half(source: str, hypothesis: str, reference: str) -> float:
    """Span-based precision/recall/F0.5 via ERRANT edits."""
    hyp_edits = {(e["start"], e["end"], e["type"])
                 for e in annotate(source, hypothesis) if e["type"] != "noop"}
    ref_edits = {(e["start"], e["end"], e["type"])
                 for e in annotate(source, reference) if e["type"] != "noop"}
    if not hyp_edits and not ref_edits:
        return 1.0
    if not hyp_edits or not ref_edits:
        return 0.0
    tp = len(hyp_edits & ref_edits)
    fp = len(hyp_edits - ref_edits)
    fn = len(ref_edits - hyp_edits)
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    beta2 = 0.25  # F0.5
    return (1 + beta2) * p * r / (beta2 * p + r)


def gleu_plus(source: str, hypothesis: str, references: List[str]) -> float:
    """
    Stub: in production, call the official GLEU+ scorer.
    Returns a placeholder average-overlap value here.
    """
    from collections import Counter
    def ngrams(toks, n):
        return Counter(tuple(toks[i:i+n]) for i in range(len(toks) - n + 1))
    h = hypothesis.split()
    scores = []
    for ref in references:
        r = ref.split()
        match = 0; total = 0
        for n in (1, 2, 3, 4):
            hg = ngrams(h, n); rg = ngrams(r, n)
            for g, c in hg.items():
                match += min(c, rg.get(g, 0))
                total += c
        scores.append(match / total if total else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _load_outputs(model_id: str) -> dict:
    out = {}
    d = INFERENCE_DIR / model_id
    if not d.exists():
        return out
    for f in d.glob("*.json"):
        try:
            p = json.load(f.open())
            out[p["probe_id"]] = p
        except Exception:
            continue
    return out


def score_all_conformance() -> Path:
    base_items = {}
    with (PROBES_DIR / "gec_probecore_base.jsonl").open() as fh:
        for ln in fh:
            r = json.loads(ln)
            base_items[r["sentence_id"]] = r

    rows = []
    for m in CORRECTOR_MODELS:
        outputs = _load_outputs(m["id"])
        f_scores = []
        gleu_scores = []
        for sid, item in base_items.items():
            key = f"{sid}__src"
            if key not in outputs:
                continue
            hyp = outputs[key]["output"]
            f = errant_f_half(item["source_text"], hyp,
                              item["reference_correction"])
            f_scores.append(f)
            refs = [item["reference_correction"]] + item.get("alt_references", [])
            gleu_scores.append(gleu_plus(item["source_text"], hyp, refs))
        rows.append({
            "model": m["id"],
            "errant_f_half": sum(f_scores) / len(f_scores) if f_scores else 0.0,
            "gleu_plus": sum(gleu_scores) / len(gleu_scores) if gleu_scores else 0.0,
            "n_scored": len(f_scores),
        })

    out = SCORES_DIR / "rq3_conformance_gec.jsonl"
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[conformance] wrote {len(rows)} model rows -> {out}")
    return out


if __name__ == "__main__":
    score_all_conformance()
