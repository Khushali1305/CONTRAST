# Cross-Domain Analysis — RQ3 Proxy Validation

**RQ3** — do probe-based reliability scores align with expert/gold
judgments better than ordinary evaluation metrics (template conformance,
reference similarity)?

The methodology's intended design (see the paper's methodology section)
calls for a dedicated expert-anchor study: a small sample rated by human
experts on structural conformance, meaning preservation, modal
preservation, critical-detail preservation, and overall acceptability.
**That expert-annotation study has not yet been run.** What exists in this
folder instead is a **proxy** analysis that substitutes the Großer
benchmark's own clean-reference rewrites for expert judgment, since Großer
already ships expert-constructed `(free_text, EARS_template)` pairs.

This distinction matters and should not be blurred: the 219 Großer
reference rewrites answer *"does this rewrite match a template a domain
expert wrote for a different purpose?"*, not *"would an expert rate this
specific model output as reliable?"* — this is a real validation gap, not
just a caveat.

## What's in this folder

| File | Role |
|---|---|
| `rq3_proxy_gold_alignment.py` | Original proxy analysis: computes gold-alignment scores directly from model outputs against Großer references (uses SequenceMatcher-based meaning preservation) |
| `step5_cross_domain.py` | Cross-domain comparison utility: loads `reliability_summary.json` from the SRS and Simplification results directories and produces a side-by-side comparison table/figure input |
| `results/rq3_proxy_*.csv/.txt` | Output of `rq3_proxy_gold_alignment.py` |
| `results/rq3_corrected_*.csv/.txt` | A corrected re-run: uses the **established, already-validated** R(f) and per-family scores from `srs_probecore/data/week4_model_outputs.json` rather than recomputing them, and restricts the "ordinary evaluation" baseline O(f) to structural conformance only (no SequenceMatcher-based meaning preservation, which was found to be an unreliable proxy for meaning). Prefer these files over the `rq3_proxy_*` ones — they are the corrected/final version. |

## Running it

```bash
cd cross_domain_analysis

# Proxy gold-alignment analysis (no LLM calls, pure structural metrics)
python rq3_proxy_gold_alignment.py
#   reads:  ../srs_probecore/data/SRS-ProbeCore-v5-frozen.json
#           ../srs_probecore/data/week4_model_outputs.json
#   writes: results/rq3_corrected_gold_alignment.csv
#           results/rq3_corrected_correlations.txt
#           results/rq3_corrected_per_family.csv
#           results/rq3_corrected_component_correlations.csv

# Cross-domain (SRS vs. Simplification) reliability comparison
python step5_cross_domain.py \
    --simp-dir ../simplification_transfer/results/final \
    --srs-dir  ../srs_probecore/data \
    --output-dir results
```

## Honest result summary

On the currently available proxy data:

```
Spearman(R(f), gold_combined) = −0.24, p = 0.57   (non-significant)
```

This proxy hypothesis is **not supported**. Rather than treating this as
a failure of the framework, RQ3 has been reframed around a more specific
and better-supported finding: **the three probe families capture
complementary, sometimes orthogonal quality dimensions**, rather than one
converging on a single "reliability" scalar that the proxy gold score also
measures:

- Directional sensitivity (DS) strongly **predicts** content preservation
  (ρ = +0.93)
- Shortcut resistance (SR) **inversely predicts** content preservation
  (ρ = −0.86)

In other words, a model that reliably reacts to meaning-bearing edits and
a model that reliably resists shortcut cues are pulling in different
directions with respect to raw content-preservation proxies — which is
itself informative about what "reliability" means for a reviser, even
though it means the single-scalar RQ3 hypothesis doesn't hold up against
this particular proxy.

## Known caveat in the underlying data

There is a known inconsistency between item-clustered R-values (as
reported in the SRS domain's main results table) and micro-averaged R
(as computed in some of the per-family diagnostic breakdowns) — both are
internally consistent, but they answer subtly different questions (does
every item count once, or does every probe count once). If you recompute
any of these correlations yourself, be explicit about which R you used.
