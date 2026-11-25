# CONTRAST derived-data supplement

Data supplement for the CONTRAST paper by Khushali Mandalia, Sourish Dasgupta, and Saurabh Tiwari (Dhirubhai Ambani University). Version 1.1.0 (2026-11-25).

This archive contains redistributable derived scores, aggregate tables, missingness records, and provenance documentation. It contains no upstream requirement, learner, or simplification text; no model rewrite; no persistent learner identifier; no name, email, location, author metadata, credential, raw judge response, or partial run.

## SRS files

| File | Role |
|---|---|
| `srs/contract.json` | Machine-readable frozen contract `CONTRAST-SRS-CR-v1.0.0`. |
| `srs/contract_v1_1.json` | Eligibility amendment `CONTRAST-SRS-CR-v1.1.0` used in the paper (excludes 166 direct *will*→*shall* pairs). |
| `srs/primary_results_v1_1.csv`, `srs/hidden_failures_v1_1.csv` | Contract v1.1.0 tables reported in the paper. |
| `srs/pair_scores.csv` | 24,544 expected canonical probe-model pairs; 24,541 are technically available. |
| `srs/primary_results.csv` | Contract v1.0.0 item-macro primary results and 95% bootstrap intervals. |
| `srs/hidden_failures.csv` | Contract v1.0.0 endpoint-conditioned focal and complete-response failures. |
| `srs/eligibility.csv` | Source-side inclusion and exclusion counts. |
| `srs/rules.csv`, `srs/metrics.csv` | Human-readable frozen decisions and metric definitions. |
| `srs/sensitivity.csv`, `srs/dataset_operations.csv` | Threshold, extractor, dataset, and operation breakdowns. |
| `srs/coverage.csv`, `srs/qwen_missing.csv` | Explicit output coverage and the four non-imputed Qwen output gaps. |
| `srs/qa.csv` | Sixteen frozen contract checks; all pass. |

`pair_scores.csv` uses collision-safe canonical identifiers. It contains no `requirement_text`, `input_text`, `rewrite`, or legacy document text column. Addition fields are descriptive. The collateral columns are automatic proxies pending item-level human validation.

## Text simplification

`simplification/shared_invariance_summary.csv` reports actual output relations on 371 validated shared probes. `simplification/addition_response_summary.csv` reports descriptive qualifier-retention sensitivity. These files do not define or support a simplification leaderboard. Semantic-probe and directional-readability designs that do not isolate the intended norm are not included as confirmatory results.

## GEC

`gec/bounded_endpoint_overcorrection_summary.csv` and `gec/overcorrection_by_operation.csv` retain only a bounded diagnostic illustration. The endpoint column is named `archived_endpoint_span_type_f0_5_proxy` because the archived implementation matched edit spans/types without replacement text; it is not official ERRANT F0.5. The former `GECToR` label is replaced by `Identity / no-edit baseline` wherever that path appears. No placeholder GLEU value is included.

## Integrity

`manifest.json` records the SHA-256 digest, byte size, and CSV data-row count of every included file other than the manifest itself. The software archive verifies these values before recomputation.

## Licence and upstream assets

Original derived tables and metadata are released under CC BY 4.0. This licence does not apply to upstream datasets, which are not included and remain governed by their own terms. See `assets/THIRD_PARTY_ASSETS.md` and `assets/DATA_STATEMENT.md`.

