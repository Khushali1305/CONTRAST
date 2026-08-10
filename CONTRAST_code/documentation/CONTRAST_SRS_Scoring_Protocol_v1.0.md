# CONTRAST SRS Frozen Scoring Protocol

**Contract:** `CONTRAST-SRS-CR-v1.0.0`  
**Status:** Frozen on 2026-07-31  
**Contract SHA-256:** `fd05049ff79f20ec974daea815a4110bca84419e8bcc55d64d7d9dd7a21388f5`

## What this contract measures

The analysis unit is a base source, its controlled variant, and the two complete
rewrites produced by the same system. The primary question is whether the focal
source change receives the response required by the communicated SRS norm while
non-focal material remains stable.

## Non-negotiable analysis rules

1. Source eligibility is fixed before inspecting model outputs.
2. A technically missing cached inference is excluded and reported; it is never
   imputed or counted as a model error.
3. A present rewrite with no extractable explicit modal remains in the
   denominator and fails the focal modal component.
4. The primary modal scale is `{may, can} < {should, will} <
   {shall, must}`; the first explicit modal is the frozen focal extractor.
5. Invariance contains 1286
   source probes. Directional response contains
   762 source
   probes after source-only exclusions. Addition uptake contains
   891 source probes.
6. Complete response is the exact focal response conjoined with the automatic
   collateral-preservation proxy.
7. The collateral proxy requires non-focal content-set F1 at least 0.80,
   shared-source-anchor recall at least 0.50, exact cross-output number
   consistency, and exact cross-output negation consistency.
8. Endpoint passing means both outputs score 1.0 under `EARS-Struct-v1`. This is
   structural form, not semantic correctness.
9. Scores are averaged within canonical item and then across items. Confidence
   intervals use 10,000 source-dataset-stratified item bootstrap samples with
   seed 20260731.
10. There is no global reliability score, winner tier, or shortcut-resistance
    score.

## Family interpretations

- **Invariance:** report relational modal stability, source-aligned exact focal
  response, collateral proxy, and their conjunctions.
- **Directional:** equality fails. Report strict directional response,
  source-aligned exact endpoint response, collateral proxy, and their
  conjunctions.
- **Addition:** report incremental uptake descriptively. Do not label retention
  harmful or removal correct.

## Locked exclusions

Directional probes exclude 86 same-class modal substitutions, 23 non-modal
scope interventions without an executable scope scorer, and 20 probes whose
declared focal modal is not the first source modal. These are recorded, not
discarded silently.

## Claim boundaries

The collateral measure remains an automatic proxy until item-level human
validation is available. The FLAN-T5 systems used a shorter prompt, so tables
must display prompt condition and must not be presented as a controlled model
ranking. Simplification, GEC, and cross-setting correlation require separate
contracts and are not silently governed by this one.
