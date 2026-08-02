# GEC CONTRAST — Methodology

This document describes the design of the GEC third-domain extension of the
CONTRAST framework. It mirrors the SRS pipeline where principles transfer,
adapts where the domain differs, and explicitly tightens two methodological
commitments (family disjointness, synthetic-error disclosure) that were soft
in the SRS pipeline.

## 1. What CONTRAST is and how this fits

CONTRAST evaluates norm-governed text revision by treating reliability as a
*behavioral relation between paired outputs*, not as a single-output quality
score. For each base item, we construct counterfactual probe inputs related
to it by controlled transformations, then check whether the model's outputs
on the original and the probe satisfy the expected relation.

The framework has three probe families: **invariance**, **directional**, and
**shortcut** (in SRS) / **overcorrection** (in GEC — see §3.3). For each
model, family-level reliability scores aggregate into a global reliability
score. Comparison against ordinary metrics (template conformance, reference
proximity) is what answers RQ1.

The GEC extension keeps the framework intact and tests whether the same
behavioral-relation logic generalizes to a third norm-governed revision
task: correcting grammatical errors in learner text under a minimal-edit
norm.

## 2. Dataset choice

We use only **BEA-2019 W&I+LOCNESS** as the base corpus. Rationale:

- **Natural errors at the base layer.** Unlike the seven datasets in the
  existing SRS + simplification pipeline, BEA-2019 contains real learner
  text with naturally-occurring errors. This eliminates the synthetic-error
  reviewer attack surface at the base layer; we still inject errors at the
  *probe* layer for directional ADD/SWAP and we disclose this explicitly
  (see §3.4).
- **ERRANT type-level annotation.** Every edit is tagged with an ERRANT
  type (`R:VERB:SVA`, `M:DET`, etc.), enabling stratified sampling and
  deterministic rule-based filtering of probes.
- **Multi-CEFR-level coverage.** A/B/C/N proficiency levels let us stratify
  the base set across difficulty.
- **Strong gold references.** Multi-annotator references give a better
  RQ3 anchor than Großer's reference rewrites alone.

We do **not** use JFLEG, CoNLL-2014, or MultiGEC-2025. JFLEG anchors to a
fluency norm rather than minimal-edit; mixing the two norms in one analysis
contaminates the answer to "what is the model's reliability." CoNLL-2014 is
held in reserve as a sanity baseline. MultiGEC-2025 is multilingual and out
of scope.

### Stratification axes

The ProbeCore base set (target 900 items) is stratified across:

1. CEFR level (A / B / C / N)
2. Primary ERRANT error bucket (SVA, DET, PREP, MORPH, ORTH, WO, VERB:TENSE, ...)
3. Sentence-length quartile
4. Error density (zero / single / low / high)

Each non-empty bucket gets at least one item; remaining capacity is
allocated proportionally. The output is frozen as
`gec_probecore_base.jsonl`.

## 3. Probe families

### 3.1 Invariance

**Goal:** test whether the corrector is stable under paraphrase that
preserves the same error.

**Operation:** the generator rewrites the source sentence so that the
**wording changes** but the **ERRANT error type set is preserved**. The
probe reference is a clean correction of the paraphrased sentence.

**Expected relation:** the corrector's edits on the source and on the
paraphrase should target the same error types. A stable corrector
produces semantically aligned corrections regardless of surface form.

**Validity criteria (rule filter):**
- ERRANT(probe_src → probe_ref) error-type set = ERRANT(src → ref) set
- Token Jaccard(src, probe_src) ≤ 0.85 (surface must differ)
- Length ratio 0.6–1.6
- Two-judge LLM confirms preservation of error and naturalness

### 3.2 Directional

**Goal:** test whether the corrector reacts in the right direction to a
controlled error-set delta.

**Three sub-operations:**

- **ADD** — inject one new error of a planned ERRANT type into a position
  adjacent to the original error.
- **REMOVE** — partially correct the source, leaving one of multiple
  original errors uncorrected. Only valid when the base item has ≥ 2 errors.
- **SWAP** — replace the original error with a different ERRANT error type
  in the same position.

**Expected relation:** the corrector's output edit set should reflect the
planned delta — additional corrections for ADD, fewer for REMOVE, changed
type for SWAP.

**Synthetic-error disclosure:** ADD and SWAP construct sentences that did
not originally contain those errors. Each probe records `synthetic: true`
and `injection_method: typed_llm_injection`. REMOVE applies real gold
edits and records `synthetic: false`. This is the one reviewer-facing
synthetic-error vector; the schema makes the exact fraction auditable.

### 3.3 Overcorrection (replaces SRS "shortcut")

**Goal:** test whether the corrector edits *outside the original error
span* when an extraneous-but-grammatical cue is inserted.

**Why "overcorrection" and not "shortcut":** in SRS, a shortcut cue
("temporarily") tempts a model to hallucinate behavior; in GEC, the
analogous failure is editing text that has nothing wrong with it. A
shortcut probe in the SRS sense would be uninformative because GEC
correctors are reasonably good at "do nothing to grammatical text" — the
informative failure mode is *over-editing* when the surrounding text
becomes stylistically unusual.

**Operation:** insert a grammatical lexical cue (register, discourse, or
topicalization category) OUTSIDE the original error span. The protected
span is recorded with token indices.

**Expected relation:** the corrector should fix the original error and
leave the inserted cue untouched. Any edits outside the protected span
count as overcorrection.

**Scoring:** rule-based via ERRANT — `spans_outside(probe_src, probe_out,
protected_span)` returns the edits that fall outside the protected span;
the probe passes if and only if that list is empty.

### 3.4 Why family-disjoint generators matter (and what we changed)

The SRS pipeline used Llama-3.3-70B as the single probe generator while
Llama-3.1-8B and Llama-3.1-70B were in the corrector pool. This is a
soft same-family bias: probes written by a Llama model carry phrasing
priors that other Llama models may handle more naturally, inflating
in-family corrector scores. The SRS pipeline's post-freeze semantic-probe
layer tightened the *judge* side of this (switching to Qwen3-32B and
GPT-OSS-20B as judges) but did not tighten the *generator* side.

The GEC pipeline tightens both. The three roles — generator, judge,
corrector — must come from three pairwise-disjoint model families:

| Role | Model | Family |
|---|---|---|
| Probe generator | DeepSeek V4 Flash | `deepseek` |
| Judge 1 | Claude Haiku 4.5 | `anthropic` |
| Judge 2 | Mistral Large 3 | `mistral` |
| Correctors | (8 models — see §5) | `flan_t5`, `llama`, `qwen`, `gemma`, `openai`, `rule` |

This is enforced by `assert_generator_disjoint_from_judges_and_correctors()`
at import time. If the config is ever edited to violate the principle, the
judge module fails to load.

**Note on Groq-hosted DeepSeek.** Groq hosts `deepseek-r1-distill-llama-70b`
— a Llama 70B model distilled from DeepSeek R1's outputs. Its architecture,
parameters, and family lineage are Llama. Using this model as generator
would re-introduce the same-family bias against Llama correctors. The
pipeline uses the **native DeepSeek API** at `https://api.deepseek.com/v1`
to access the true `deepseek` family.

## 4. Validation pipeline

### 4.1 Deterministic rule filter

For each generated probe, a family-specific rule filter runs first.
Outputs `pass` / `borderline` / `fail` and a structured detail record.
Probes that `fail` the rule filter are excluded with sentinel value −1
— they are **never** scored as 0.

### 4.2 Two-judge LLM validation

Probes that pass the rule filter are sent to two independent LLM judges
(Claude Haiku 4.5 and Mistral Large 3). Each judge returns:

```json
{"equivalent": true|false, "confidence": "high|medium|low", "failures": [...]}
```

Final verdict:
- `agree_equivalent` — both judges accept → probe is added to ProbeCore
- `agree_not_equivalent` — both reject → excluded
- `disagree` — sent to a manual-review queue (matches SRS practice of
  sampling ~130 disagreements per pipeline)
- `error` — at least one judge failed to return parseable JSON → excluded

### 4.3 Probes are validated, model outputs are not

We use LLM judges **only** during probe filtering, never to score
evaluated corrector outputs. This matches the SRS practice and the
methodology's stated principle in `methodology.md`: "Probe validation
uses LLM judges only before evaluation, not for scoring evaluated
outputs. This reduces judge-induced circularity."

## 5. Corrector pool

For cross-domain comparability with the SRS pipeline, we use the same
eight correctors:

1. **GECToR baseline** (rule-based, CPU)
2. **FLAN-T5-Large** (770M, local)
3. **FLAN-T5-XL** (3B, local)
4. **Llama-3.1-8B Instruct** (Groq API)
5. **Llama-3.1-70B Instruct** (Groq API)
6. **Qwen2.5-32B Instruct** (local, AWQ-INT4)
7. **Gemma-3-27B Instruct** (local)
8. **GPT-4.1** (OpenAI API)

Cross-domain claim: if a corrector's relative ranking holds across SRS,
simplification, and GEC, that supports the framework's task-agnostic
reliability story.

## 6. Scoring

### 6.1 Probe-pair scoring

| Family | Pass criterion |
|---|---|
| Invariance | ERRANT(src → src_out) error-type set == ERRANT(probe_src → probe_out) error-type set |
| Directional | actual edit-set delta covers planned `delta_added_types` and `delta_removed_types` |
| Overcorrection | `spans_outside(probe_src, probe_out, protected_span)` is empty |

### 6.2 Aggregation

For each (model, family), the pass rate is computed with sentinel-aware
aggregation: invalid probes and missing corrector outputs are dropped from
both numerator and denominator. Item-clustered bootstrap CIs (2000 iter)
follow the SRS practice.

### 6.3 Conformance (RQ3 anchor)

In addition to probe-pair scoring, we compute:
- **ERRANT span-based F0.5** against gold references (canonical metric)
- **GLEU+** (multi-reference fluency)

These play the role that `gold_combined` / `gold_structural` play in the
SRS RQ3 analysis.

## 7. Cross-domain analysis

The interesting science is the three-domain triangle. The SRS pipeline
reported a strong negative Spearman ρ between SRS shortcut reliability and
simplification shortcut reliability (ρ = −0.86, p = 0.01), reflecting the
copy-vs-rewrite tension. If GEC overcorrection reliability clusters with
one side of that axis — concretely, if rewrite-aggressive correctors
score low on both simplification shortcut and GEC overcorrection — that's
a clean unifying story.

`analysis/cross_domain_rho.py` computes all three pairwise Spearman ρ
values plus permutation p-values across the eight correctors.

## 8. Limitations to disclose in the paper

1. **Synthetic error injection at the probe layer.** Directional ADD and
   SWAP synthesize new errors. The `synthetic: true` flag and
   `injection_method` field make the exact fraction auditable.

2. **RQ3 calibration is partial.** BEA-2019 gold references are
   corrections of natural errors, not expert judgments of model outputs
   under perturbation. A small fresh expert-annotation study on ~100
   paired probe outputs would fully close the gap; this pipeline does
   not include that study.

3. **Norm choice anchors to minimal-edit.** Switching to a fluency norm
   (JFLEG-style) would change the reference scoring and possibly the
   reliability ordering. The pipeline is set up for minimal-edit because
   that aligns with BEA-2019's annotation guidelines.

4. **Overcorrection family is a domain-specific adaptation.** It replaces
   the SRS shortcut family. Direct comparability of family-level scores
   across the three domains therefore holds for invariance and directional
   but not for the third family.
