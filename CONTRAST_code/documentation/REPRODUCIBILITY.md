# Reproducibility record

## Frozen object

The primary reproducibility object is one canonical base input and one canonical probe input rewritten by the same system. Joins use `canonical_pair_id`, `canonical_item_id`, and `model_id`; the non-unique legacy `original_item_id` is never a join key.

The released `pair_scores.csv` has one row for every expected canonical probe-model pair. Source eligibility is fixed before model outputs are inspected. A technical pair is available only when both output records are present, have status `ok`, and contain nonempty rewrites.

## Scoring

- Modal extraction uses the first explicit occurrence of `shall`, `must`, `should`, `will`, `may`, or `can` in complete-text order.
- Invariance requires both output endpoints to match the base source's first modal class.
- Directional scoring is strict; equality fails. Exact response requires both output endpoints to match their corresponding source modal classes.
- The collateral proxy requires non-focal token-set F1 at least 0.80, shared-anchor recall at least 0.50, exact shared-number consistency, and exact shared-negation consistency.
- Addition uptake at 70% is descriptive and is never interpreted as correctness or harm.

## Aggregation

Scores are averaged within canonical item and then across items. Confidence intervals use nonparametric percentile bootstrapping over canonical items, stratified by source dataset, with 10,000 replicates and seed `20260731`.

## Missingness

Four Qwen output records are absent for one repaired duplicate-ID item: one base and three probe outputs. These create one unavailable pair in each analysis family. Missing pairs are excluded from rewrite-dependent denominators and reported through coverage fields; no text or score is imputed. Present but unparsable rewrites remain scored failures for the affected focal component.

## Deliberate exclusions

The release excludes raw corpora, upstream text, model rewrites, API traces, raw judge outputs, partial results, failed logs, legacy snapshots, and stale cross-domain aggregates. This keeps the blind package unambiguous and avoids redistributing assets whose terms do not permit or clearly authorize redistribution.

