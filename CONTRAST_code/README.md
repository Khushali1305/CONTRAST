# CONTRAST: When Acceptable Text Revisions are Unreliable

Anonymous ARR software supplement, version 1.0.0 (2026-08-04).

This archive contains executable scoring, aggregation, validation, and tests. It intentionally contains no paper source, author metadata, raw corpora, model outputs, API credentials, failed logs, partial runs, legacy snapshots, or reviewer-process text. The separately submitted `CONTRAST_data.zip` contains only redistributable derived tables and metadata.

## Reproduce the frozen SRS tables

Extract `CONTRAST_code.zip` and `CONTRAST_data.zip` next to each other, then run:

```bash
cd CONTRAST_code
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
./evaluation/run_all.sh ../CONTRAST_data
```

The command runs unit tests, verifies every data hash and row count, recomputes the SRS primary and hidden-failure tables with 10,000 dataset-stratified item bootstraps, and checks them against the frozen reference tables.

The frozen analysis contract is `CONTRAST-SRS-CR-v1.0.0` (2026-07-31):

- 1,286 source-eligible invariance probes.
- 762 source-eligible cross-class, first-locus directional probes.
- 891 descriptive addition probes.
- 24,544 expected probe-model pairs and 24,541 technically available pairs.
- Three unavailable Qwen2.5-32B pairs, created by four absent output records for one repaired duplicate-ID item. They remain excluded, reported, and never imputed.
- Item-macro aggregation with 10,000 source-dataset-stratified bootstrap replicates and seed `20260731`.

Directional equality is a failure. A complete response requires the exact focal response and the frozen automatic collateral-preservation proxy. Addition uptake is descriptive; it is not labelled shortcut resistance, harmful copying, or correctness. `EARS-Struct-v1` is a structural endpoint comparator, not semantic correctness or full EARS validity.

## Scope of the other domains

The data archive includes clean aggregate summaries for text simplification and a bounded GEC illustration. Text-simplification shared invariance is computed from actual output relations and is not a leaderboard. The GEC endpoint value is explicitly labelled an archived span/type proxy, not official ERRANT F0.5. No global reliability composite or causal cross-domain norm-shift claim is produced by this package.

## External assets

Upstream datasets must be obtained from their official sources under their own access and licence terms. `acquisition/check_external_assets.py` checks user-supplied local paths but never downloads or copies an asset. The separate data archive documents the source, intended use, licence status, and redistribution decision for each asset.

## Environment and security

- Python 3.11 or newer.
- `numpy==2.3.5`; all other runtime modules are from the Python standard library.
- No external service or GPU is needed to verify the released tables.
- No credential file is used. If model generation is repeated independently, credentials must be supplied through session-scoped environment variables and must never be committed or archived.

## Generative-AI assistance

Generative-AI coding assistance was used for forensic package auditing, refactoring, documentation, and test generation. The authors inspected the resulting code, ran the included checks, and retain responsibility for the scientific claims and released artifacts. This scope should also be reported in the submission's Responsible NLP checklist and acknowledgements.

## Licence

Original software in this archive is released under the MIT License. Third-party datasets and tools retain their own terms and are not included or relicensed here.

