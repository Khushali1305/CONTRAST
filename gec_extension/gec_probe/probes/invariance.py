"""
gec_probe.probes.invariance
---------------------------
Invariance probes for GEC.

Goal: rewrite the sentence so that wording changes but the SAME error
(same ERRANT type set) is still present in the same semantic position.

Why this is invariance for GEC:
    A reliable corrector should produce semantically aligned corrections
    across both inputs. If model corrections diverge meaningfully when the
    surface form changes (but the error is the same), the model is unstable
    to paraphrase.

We use an LLM paraphraser with a constrained prompt and ERRANT-based
post-validation. Invalid generations get SENTINEL_INVALID.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator

from gec_probe.config.settings import (
    PROBES_DIR, GENERATOR_MODELS, PROBES_PER_ITEM, SENTINEL_INVALID,
)
from gec_probe.probes.errant_utils import error_types, annotate
from gec_probe.inference.llm_client import call_llm

PROMPT_TEMPLATE = """You will rewrite a sentence that contains grammatical errors.

CRITICAL RULES:
1. {preserve_rule}
2. You may freely change wording, structure, and phrasing of every OTHER part
   of the sentence (synonyms, reordering, paraphrase).
3. Preserve the same overall meaning, register, and content words.
4. Do NOT add any new errors anywhere else in the sentence.
5. Output ONLY the rewritten sentence on a single line. No quotes, no commentary.

Original sentence with errors:
{src}

Reference correction (for your understanding of what the errors are):
{ref}

Rewritten sentence (errors preserved, wording changed):"""


def _error_spans(source_text: str, reference_correction: str) -> list[str]:
    """
    Extract the literal substring(s) of `source_text` that ERRANT identifies
    as erroneous (token-offset based, using whitespace tokenization to stay
    consistent with the rest of this module's token handling). Substitution/
    deletion-type edits have a non-empty span; pure insertion-type edits
    (a missing word) have start==end and yield no quotable span.
    """
    toks = source_text.split()
    spans: list[str] = []
    for e in annotate(source_text, reference_correction):
        if e["type"] == "noop":
            continue
        s, en = e["start"], e["end"]
        if s < en:
            span_text = " ".join(toks[s:en])
            if span_text and span_text not in spans:
                spans.append(span_text)
    return spans


def _build_prompt(source_text: str, reference_correction: str) -> str:
    spans = _error_spans(source_text, reference_correction)
    if spans:
        quoted = "; ".join(f'"{s}"' for s in spans)
        preserve_rule = (
            f"The exact text span(s) {quoted} from the original sentence "
            f"contain the grammatical error(s). They MUST appear in your "
            f"rewritten sentence completely unchanged, character-for-character. "
            f"Do NOT fix, remove, correct, or reword these spans in any way -- "
            f"copy them verbatim into your output."
        )
    else:
        # Insertion-type error (a word/punctuation mark is missing). There is
        # no erroneous span to quote -- the error IS the absence of something.
        preserve_rule = (
            "The error in this sentence is a MISSING word or punctuation mark "
            "(an omission, not a wrong word). Do NOT insert the missing "
            "word/punctuation anywhere in your rewrite -- the omission itself "
            "must be preserved exactly as in the original."
        )
    return PROMPT_TEMPLATE.format(preserve_rule=preserve_rule,
                                   src=source_text, ref=reference_correction)


def _is_valid_paraphrase(
    orig_src: str, orig_ref: str, probe_src: str,
    probe_ref_candidate: str,
) -> bool:
    """
    A valid invariance probe must:
      - have the same ERRANT error type set as the source
      - differ in wording from the source (token Jaccard < 0.85)
      - have a reasonable length ratio (0.6..1.6)
    """
    src_types = set(error_types(orig_src, orig_ref))
    probe_types = set(error_types(probe_src, probe_ref_candidate))
    if not src_types or src_types != probe_types:
        return False

    orig_toks = set(orig_src.lower().split())
    probe_toks = set(probe_src.lower().split())
    if not orig_toks or not probe_toks:
        return False
    jaccard = len(orig_toks & probe_toks) / len(orig_toks | probe_toks)
    if jaccard > 0.85:
        return False

    ratio = len(probe_src.split()) / max(1, len(orig_src.split()))
    if ratio < 0.6 or ratio > 1.6:
        return False
    return True


def _generate_reference_for_probe(probe_src: str, ref_template: str) -> str:
    """
    Use the same paraphraser to produce the corrected version of the probe.
    The corrected version should match the same edits as the original reference.
    """
    prompt = (
        "Correct ONLY the grammatical errors in this sentence; preserve "
        "all other wording. Output one sentence only.\n\n"
        f"Sentence: {probe_src}\n\nCorrected:"
    )
    return call_llm(GENERATOR_MODELS["paraphraser"]["id"], prompt).strip()


def generate_for_item(item: dict) -> Iterator[dict]:
    """Yield up to PROBES_PER_ITEM['invariance'] candidate invariance probes."""
    target_n = PROBES_PER_ITEM["invariance"]
    prompt = _build_prompt(item["source_text"], item["reference_correction"])
    attempts = 0
    accepted = 0
    while accepted < target_n and attempts < target_n * 3:
        attempts += 1
        probe_src = call_llm(GENERATOR_MODELS["paraphraser"]["id"], prompt).strip()
        if not probe_src or "\n" in probe_src:
            probe_src = probe_src.split("\n")[0].strip() if probe_src else ""
        if not probe_src:
            continue
        probe_ref = _generate_reference_for_probe(
            probe_src, item["reference_correction"]
        )
        valid = _is_valid_paraphrase(
            item["source_text"], item["reference_correction"],
            probe_src, probe_ref,
        )
        yield {
            "probe_id": f"{item['sentence_id']}_inv_{accepted if valid else SENTINEL_INVALID}_a{attempts}",
            "base_item_id": item["sentence_id"],
            "family": "invariance",
            "operation": "paraphrase_preserve_error",
            "probe_source": probe_src,
            "probe_reference": probe_ref,
            "valid": bool(valid),
            "expected_relation": "stable",
            "errant_source_types": sorted(error_types(item["source_text"], item["reference_correction"])),
            "errant_probe_types": sorted(error_types(probe_src, probe_ref)) if probe_ref else [],
        }
        if valid:
            accepted += 1


def run(base_path: Path = None) -> Path:
    base_path = base_path or (PROBES_DIR / "gec_probecore_base.jsonl")
    out_path = PROBES_DIR / "gec_probes_invariance.jsonl"
    n_total = n_valid = 0
    with base_path.open() as src_fh, out_path.open("w") as out_fh:
        for ln in src_fh:
            if not ln.strip():
                continue
            item = json.loads(ln)
            for probe in generate_for_item(item):
                out_fh.write(json.dumps(probe, ensure_ascii=False) + "\n")
                n_total += 1
                n_valid += int(probe["valid"])
    print(f"[inv] generated={n_total} valid={n_valid} -> {out_path}")
    return out_path


if __name__ == "__main__":
    run()