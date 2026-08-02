"""
gec_probe.probes.overcorrection
-------------------------------
The "overcorrection" family REPLACES the SRS-style "shortcut" family for GEC.

Why we replace it:
    In SRS, a "shortcut" cue (e.g., "temporarily") tempts the model to invent
    new behavior. In GEC, the analogous failure is OVERCORRECTION:
    the model edits OUTSIDE the original error span because an inserted
    stylistic cue triggers stylistic rewriting.

Operation:
    1. Locate the original error span via ERRANT.
    2. Insert a stylistic / discourse / topicalization cue OUTSIDE that span.
    3. Expected behavior: corrector fixes the original error and leaves
       the inserted cue untouched. ANY edits outside the original span
       (other than fixing the original error) count as overcorrection.

Scoring rule (in rule_filters/overcorrection_rule.py):
    valid := cue token verbatim in probe_source; error span unchanged;
             probe_source differs from source only by cue insertion.
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Iterator, Optional, Tuple

from gec_probe.config.settings import (
    PROBES_DIR, PROBES_PER_ITEM, OVERCORRECTION_CUES,
)
from gec_probe.probes.errant_utils import annotate, error_types

RANDOM_SEED = 20260103


def _original_error_span(item: dict) -> Optional[Tuple[int, int]]:
    """Return token-index span covering all canonical edits."""
    src = item["source_text"]
    ref = item["reference_correction"]
    edits = [e for e in annotate(src, ref) if e["type"] != "noop"]
    if not edits:
        return None
    return (min(e["start"] for e in edits), max(e["end"] for e in edits))


def _insert_cue_outside_span(
    tokens: list, span: Tuple[int, int], cue: str, prepend: bool,
) -> Tuple[list, int, int]:
    """
    Insert the cue OUTSIDE the protected span.
    Returns (new_tokens, new_span_start, new_span_end).
    """
    cue_tokens = cue.split()
    if prepend:
        new_tokens = cue_tokens + tokens
        offset = len(cue_tokens)
        return new_tokens, span[0] + offset, span[1] + offset
    # append at end
    new_tokens = tokens + cue_tokens
    return new_tokens, span[0], span[1]


def generate_for_item(item: dict) -> Iterator[dict]:
    span = _original_error_span(item)
    if span is None:
        return
    rng = random.Random(RANDOM_SEED ^ (hash(item["sentence_id"]) & 0xFFFFFFFF))

    cue_categories = list(OVERCORRECTION_CUES.keys())
    rng.shuffle(cue_categories)
    target_n = PROBES_PER_ITEM["overcorrection"]
    accepted = 0

    for cat in cue_categories:
        if accepted >= target_n:
            break
        cue = rng.choice(OVERCORRECTION_CUES[cat])
        # Topicalization cues prepend; others append.
        prepend = (cat == "topicalize")
        tokens = item["source_text"].split()
        new_tokens, ns, ne = _insert_cue_outside_span(tokens, span, cue, prepend)
        probe_src = " ".join(new_tokens)

        # Construct probe reference: apply the SAME canonical edits to the
        # new tokens; cue is preserved unchanged.
        edits = [e for e in annotate(item["source_text"],
                                     item["reference_correction"])
                 if e["type"] != "noop"]
        # Re-index edits if we prepended tokens.
        offset = len(cue.split()) if prepend else 0
        ref_tokens = list(new_tokens)
        for e in sorted(edits, key=lambda x: -x["start"]):
            s, en = e["start"] + offset, e["end"] + offset
            replacement = e["correction"].split() if e["correction"] else []
            ref_tokens[s:en] = replacement
        probe_ref = " ".join(ref_tokens)

        # Validity (post-checks; rule filter re-verifies):
        valid = (
            cue in probe_src
            and probe_src != item["source_text"]
            and probe_src.split()[ns:ne] == item["source_text"].split()[span[0]:span[1]]
        )

        yield {
            "probe_id": f"{item['sentence_id']}_oc_{accepted}",
            "base_item_id": item["sentence_id"],
            "family": "overcorrection",
            "operation": f"insert_{cat}_cue",
            "probe_source": probe_src,
            "probe_reference": probe_ref,
            "cue_token": cue,
            "cue_category": cat,
            "protected_span": [ns, ne],
            "valid": bool(valid),
            "expected_relation": "no_overcorrection",
            "errant_source_types": sorted(
                error_types(item["source_text"], item["reference_correction"])
            ),
        }
        if valid:
            accepted += 1


def run(base_path: Path = None) -> Path:
    base_path = base_path or (PROBES_DIR / "gec_probecore_base.jsonl")
    out_path = PROBES_DIR / "gec_probes_overcorrection.jsonl"
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
    print(f"[oc] generated={n_total} valid={n_valid} -> {out_path}")
    return out_path


if __name__ == "__main__":
    run()
