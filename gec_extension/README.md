# GEC Extension — Grammatical Error Correction

**Second transfer domain**, added as a post-submission generalization
extension to the CONTRAST paper. Mirrors the SRS and simplification
pipelines: same three probe families, same family-disjointness principle,
same deterministic-scorer-plus-judges-for-construction-only separation.

## Start here

This domain's own module (`gec_probe/`) already ships thorough,
self-contained documentation — read it in this order:

1. **`gec_probe/METHODOLOGY.md`** — design rationale, how the three probe
   families were adapted for GEC (`invariance`, `directional`, and
   `overcorrection` in place of SRS's `shortcut`), and what's different
   from the SRS pipeline.
2. **`gec_probe/SETUP.md`** — exact command sequence to run end-to-end,
   including the Lightning AI GPU walkthrough.
3. **`gec_probe/README.md`** — module architecture and file tree.

## What changed during reorganization

- **`data_artifacts/` was duplicated** in the original upload (once at
  `GEC/data_artifacts/` and once at `GEC/gec_probe/data_artifacts/`,
  byte-identical). This repo keeps a **single canonical copy** at
  `gec_extension/data_artifacts/` (sibling of `gec_probe/`, not nested
  inside it).
- **Path convention:** `gec_probe/config/settings.py` resolves
  `DATA_DIR = PROJECT_ROOT / "data_artifacts"`, where `PROJECT_ROOT`
  defaults to the current working directory and can be overridden with
  the `GEC_PROBE_ROOT` environment variable. To match the consolidated
  layout, either:
  ```bash
  cd gec_extension && python gec_probe/run_pipeline.py ...
  # or, from anywhere:
  export GEC_PROBE_ROOT=/absolute/path/to/CONTRAST/gec_extension
  ```
- **`.sh` scripts moved** into `gec_probe/scripts/` (`run_lightning.sh`,
  `run_vllm_for_one_model.sh`) — this is where `gec_probe/README.md`'s own
  architecture diagram already expected them; they were previously sitting
  directly under `gec_probe/` in the original upload.

Nothing else in `gec_probe/` was restructured — its internal module layout
(`data/`, `probes/`, `rule_filters/`, `judges/`, `inference/`, `scoring/`,
`analysis/`) was already clean and is kept exactly as authored.

## Known limitations (stated plainly, per this project's own documentation)

- The **directional probe family collapsed**: 146/147 probes are a single
  sub-operation (`remove_error`), rather than a balanced mix of
  add/remove/swap operations. Treat directional results as narrower
  evidence than the SRS/simplification directional results.
- **`gector_baseline` is an identity passthrough**, not an actual GECToR
  model — it must never be presented as GECToR-equivalent performance in
  any downstream writeup.
- **No RQ3 equivalent** exists for GEC (no expert/gold-alignment
  validation set) — reliability claims here rest on the deterministic
  scorer and the two-judge probe validation alone, with single-dataset
  (BEA-2019) coverage.
- Overcorrection probes required deduplication of 146 duplicate
  `probe_id`s during construction — already fixed in the frozen
  `data_artifacts/probes/` output, noted here for audit transparency.
- The one finding this domain robustly supports is a **rank inversion for
  Llama-3.1-70B**: best corpus-level F0.5, worst overcorrection
  reliability. Broader claims beyond this specific finding are not
  supported by the current data and should be scoped narrowly in any
  paper text that cites this domain.

## Folder Reference

```
gec_extension/
├── gec_probe/              # pipeline code — see its own README/SETUP/METHODOLOGY
│   └── scripts/            # run_lightning.sh, run_vllm_for_one_model.sh
└── data_artifacts/         # single canonical copy: raw/, normalized/, probes/, judge/, inference/, scores/
```
