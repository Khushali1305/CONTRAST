"""
gec_probe.probes.directional
----------------------------
Directional probes for GEC.

Three sub-operations:
  - ADD:    inject a new error in a position adjacent to (not overlapping)
            the original error. We use a typed error bank seeded from
            real BEA-2019 errors; the inserted error type is recorded.
  - REMOVE: apply a partial correction that fixes one of multiple errors
            (only valid if base item has >= 2 errors).
  - SWAP:   replace the original error with a different ERRANT error type
            in the same position.

Expected relation: corrector output's edit set should reflect the delta
in the source's error set.

We treat all directional injections as SYNTHETIC and tag them as such; this
is the one place where reviewer-facing synthetic-error risk lives, and we
make it explicit in the schema (`synthetic: true`, `injection_method`).
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Iterator, Optional

from gec_probe.config.settings import PROBES_DIR, GENERATOR_MODELS, PROBES_PER_ITEM
from gec_probe.probes.errant_utils import error_types, annotate
from gec_probe.inference.llm_client import call_llm

RANDOM_SEED = 20260102

ADD_PROMPT = """Add ONE grammatical error of type "{etype}" to this sentence.
Keep all other wording identical. Output one sentence only, no commentary.

Sentence:
{src}

Sentence with one added {etype} error:"""

SWAP_PROMPT = """In the following sentence there is a grammatical error of
type "{old_type}". Rewrite the sentence so that the error becomes a "{new_type}"
error in the SAME position. Keep all other wording identical.
Output one sentence only.

Sentence:
{src}

Reference correction (so you can see where the error is):
{ref}

Rewritten sentence with {new_type} error instead of {old_type}:"""


# Error type pool for ADD/SWAP injections (typed bank, not arbitrary corruption).
_INJECTION_BANK = [
    "M:DET", "R:DET", "U:DET",
    "M:PREP", "R:PREP",
    "R:VERB:SVA", "R:VERB:TENSE", "R:VERB:FORM",
    "R:NOUN:NUM", "R:MORPH",
]


def _partial_correct(item: dict, keep_n_errors: int = 1) -> Optional[str]:
    """
    Construct a partial correction: apply only some of the canonical edits.
    For the REMOVE operation, we want a probe-source that still contains
    `keep_n_errors` original errors.

    We rebuild from tokens and apply a subset of edits.
    """
    src = item["source_text"]
    ref = item["reference_correction"]
    edits = [e for e in annotate(src, ref) if e["type"] != "noop"]
    if len(edits) <= keep_n_errors:
        return None
    # Apply (len - keep_n_errors) random edits, leaving keep_n_errors errors.
    rng = random.Random(hash(item["sentence_id"]) & 0xFFFFFFFF)
    rng.shuffle(edits)
    to_apply = edits[keep_n_errors:]
    tokens = src.split()
    for e in sorted(to_apply, key=lambda x: -x["start"]):
        replacement = e["correction"].split() if e["correction"] else []
        tokens[e["start"]:e["end"]] = replacement
    return " ".join(tokens)


_FULL_CORRECT_PROMPT = (
    "Correct ALL grammatical errors in the sentence. Preserve all other "
    "wording. Output one corrected sentence only.\n\nSentence: {src}\n\nCorrected:"
)


def _full_correct(text: str) -> str:
    """Generate a clean correction; used for synthetic-probe references."""
    out = call_llm(
        GENERATOR_MODELS["error_injector"]["id"],
        _FULL_CORRECT_PROMPT.format(src=text),
    ).strip().split("\n")[0]
    return out


def _op_add(item: dict, rng: random.Random) -> Optional[dict]:
    src_types = set(error_types(item["source_text"], item["reference_correction"]))
    candidates = [t for t in _INJECTION_BANK if t not in src_types]
    if not candidates:
        return None
    new_type = rng.choice(candidates)
    out = call_llm(
        GENERATOR_MODELS["error_injector"]["id"],
        ADD_PROMPT.format(etype=new_type, src=item["source_text"]),
    ).strip().split("\n")[0]
    if not out or out == item["source_text"]:
        return None
    # Probe reference must correct BOTH the original errors AND the injected
    # one. We get this by running a full correction on the augmented sentence.
    # The rule filter then validates that error_types(probe_src -> probe_ref)
    # equals src_types ∪ {new_type}.
    augmented_ref = _full_correct(out)
    if not augmented_ref or augmented_ref == out:
        return None
    return {
        "operation": "add_error",
        "probe_source": out,
        "probe_reference": augmented_ref,
        "synthetic": True,
        "injection_method": "typed_llm_injection",
        "delta_added_types": [new_type],
        "delta_removed_types": [],
        "expected_relation": "directional_more_corrections",
    }


def _op_remove(item: dict) -> Optional[dict]:
    src = item["source_text"]
    ref = item["reference_correction"]
    edits = [e for e in annotate(src, ref) if e["type"] != "noop"]
    if len(edits) < 2:
        return None
    probe_src = _partial_correct(item, keep_n_errors=1)
    if probe_src is None or probe_src == src:
        return None
    return {
        "operation": "remove_error",
        "probe_source": probe_src,
        "probe_reference": ref,
        "synthetic": False,         # we applied real gold edits
        "injection_method": "partial_gold_correction",
        "delta_added_types": [],
        "delta_removed_types": [
            e["type"] for e in edits[1:]    # types we removed
        ],
        "expected_relation": "directional_fewer_corrections",
    }


def _op_swap(item: dict, rng: random.Random) -> Optional[dict]:
    src_types = error_types(item["source_text"], item["reference_correction"])
    if not src_types:
        return None
    old_type = src_types[0]
    candidates = [t for t in _INJECTION_BANK if t != old_type]
    new_type = rng.choice(candidates)
    out = call_llm(
        GENERATOR_MODELS["error_injector"]["id"],
        SWAP_PROMPT.format(
            old_type=old_type, new_type=new_type,
            src=item["source_text"], ref=item["reference_correction"],
        ),
    ).strip().split("\n")[0]
    if not out or out == item["source_text"]:
        return None
    # Same reasoning as _op_add: regenerate a clean correction of the swapped
    # sentence so that ERRANT can recover the new error-type set.
    swapped_ref = _full_correct(out)
    if not swapped_ref or swapped_ref == out:
        return None
    return {
        "operation": "swap_error_type",
        "probe_source": out,
        "probe_reference": swapped_ref,
        "synthetic": True,
        "injection_method": "typed_llm_injection",
        "delta_added_types": [new_type],
        "delta_removed_types": [old_type],
        "expected_relation": "directional_type_change",
    }


def generate_for_item(item: dict) -> Iterator[dict]:
    rng = random.Random(RANDOM_SEED ^ (hash(item["sentence_id"]) & 0xFFFFFFFF))
    ops = [_op_add(item, rng), _op_remove(item), _op_swap(item, rng)]
    for i, op in enumerate(ops):
        if op is None:
            continue
        # Compute validity at probe-emit time (deep checks in rule filter).
        valid = (
            op["probe_source"] != item["source_text"]
            and len(op["probe_source"].split()) > 0
        )
        yield {
            "probe_id": f"{item['sentence_id']}_dir_{i}",
            "base_item_id": item["sentence_id"],
            "family": "directional",
            **op,
            "valid": bool(valid),
            "errant_source_types": sorted(
                error_types(item["source_text"], item["reference_correction"])
            ),
        }


def run(base_path: Path = None) -> Path:
    base_path = base_path or (PROBES_DIR / "gec_probecore_base.jsonl")
    out_path = PROBES_DIR / "gec_probes_directional.jsonl"
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
    print(f"[dir] generated={n_total} valid={n_valid} -> {out_path}")
    return out_path


if __name__ == "__main__":
    run()
