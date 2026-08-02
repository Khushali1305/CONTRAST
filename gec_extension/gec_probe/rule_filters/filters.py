"""
gec_probe.rule_filters.filters
------------------------------
Deterministic rule-based filters that run BEFORE the two-judge LLM
validation. Output schema mirrors `rule_filter_analysis.csv` from the
SRS pipeline (rule_verdict, lexical_pass, transform_verify_pass,
modal_negation_pass repurposed as type_negation_pass).

Verdicts:
    "pass"       : all checks pass; probe goes to judges with high prior
    "borderline" : minor issue; still goes to judges, flagged
    "fail"       : critical issue; probe is excluded (sentinel value -1)
"""
from __future__ import annotations
from typing import Tuple

from gec_probe.probes.errant_utils import (
    annotate, error_types, spans_outside,
)


# ------------------------------------------------------------------
# Invariance rule filter
# ------------------------------------------------------------------
def filter_invariance(orig_src: str, orig_ref: str,
                      probe_src: str, probe_ref: str) -> dict:
    src_types = set(error_types(orig_src, orig_ref))
    probe_types = set(error_types(probe_src, probe_ref))
    lex_pass = src_types == probe_types and len(src_types) > 0

    orig_toks = set(orig_src.lower().split())
    probe_toks = set(probe_src.lower().split())
    if orig_toks and probe_toks:
        jaccard = len(orig_toks & probe_toks) / len(orig_toks | probe_toks)
    else:
        jaccard = 1.0
    surface_pass = jaccard <= 0.85

    ratio = len(probe_src.split()) / max(1, len(orig_src.split()))
    length_pass = 0.6 <= ratio <= 1.6

    type_pass = lex_pass  # for invariance, type set must match
    if lex_pass and surface_pass and length_pass:
        verdict = "pass"
    elif lex_pass and (surface_pass or length_pass):
        verdict = "borderline"
    else:
        verdict = "fail"

    return {
        "rule_verdict": verdict,
        "lexical_pass": lex_pass,
        "lexical_recall": float(jaccard),
        "transform_verify_pass": surface_pass and length_pass,
        "transform_detail": f"jaccard={jaccard:.3f},len_ratio={ratio:.2f}",
        "type_negation_pass": type_pass,
        "source_error_types": sorted(src_types),
        "probe_error_types": sorted(probe_types),
    }


# ------------------------------------------------------------------
# Directional rule filter
# ------------------------------------------------------------------
def filter_directional(orig_src: str, orig_ref: str,
                       probe_src: str, probe_ref: str,
                       expected_added: list, expected_removed: list) -> dict:
    src_types = set(error_types(orig_src, orig_ref))
    probe_types = set(error_types(probe_src, probe_ref))

    actual_added = probe_types - src_types
    actual_removed = src_types - probe_types

    expected_added_set = set(expected_added or [])
    expected_removed_set = set(expected_removed or [])

    add_ok = expected_added_set.issubset(actual_added)
    remove_ok = expected_removed_set.issubset(actual_removed)
    type_pass = add_ok and remove_ok

    # Lexical: probe must not be identical to source
    lex_pass = probe_src != orig_src

    if type_pass and lex_pass:
        verdict = "pass"
    elif (add_ok or remove_ok) and lex_pass:
        verdict = "borderline"
    else:
        verdict = "fail"

    return {
        "rule_verdict": verdict,
        "lexical_pass": lex_pass,
        "lexical_recall": 1.0 if probe_src != orig_src else 0.0,
        "transform_verify_pass": type_pass,
        "transform_detail": (
            f"expected_add={sorted(expected_added_set)},"
            f"actual_add={sorted(actual_added)},"
            f"expected_rm={sorted(expected_removed_set)},"
            f"actual_rm={sorted(actual_removed)}"
        ),
        "type_negation_pass": type_pass,
        "source_error_types": sorted(src_types),
        "probe_error_types": sorted(probe_types),
    }


# ------------------------------------------------------------------
# Overcorrection rule filter
# ------------------------------------------------------------------
def filter_overcorrection(orig_src: str, orig_ref: str,
                          probe_src: str, probe_ref: str,
                          cue_token: str,
                          protected_span: Tuple[int, int]) -> dict:
    cue_present = cue_token in probe_src
    # Error types unchanged (cue insertion should not change the underlying error)
    src_types = set(error_types(orig_src, orig_ref))
    probe_types = set(error_types(probe_src, probe_ref))
    type_pass = src_types == probe_types

    # Probe must be strictly longer than source (we inserted a cue)
    length_pass = len(probe_src.split()) > len(orig_src.split())

    lex_pass = cue_present and length_pass

    if cue_present and type_pass and length_pass:
        verdict = "pass"
    elif cue_present and (type_pass or length_pass):
        verdict = "borderline"
    else:
        verdict = "fail"

    return {
        "rule_verdict": verdict,
        "lexical_pass": lex_pass,
        "lexical_recall": 1.0 if cue_present else 0.0,
        "transform_verify_pass": type_pass and length_pass,
        "transform_detail": f"cue={cue_token!r},span={protected_span}",
        "type_negation_pass": type_pass,
        "source_error_types": sorted(src_types),
        "probe_error_types": sorted(probe_types),
    }


def filter_probe(orig_item: dict, probe: dict) -> dict:
    """Dispatch by family."""
    fam = probe["family"]
    if fam == "invariance":
        out = filter_invariance(
            orig_item["source_text"], orig_item["reference_correction"],
            probe["probe_source"], probe["probe_reference"],
        )
    elif fam == "directional":
        out = filter_directional(
            orig_item["source_text"], orig_item["reference_correction"],
            probe["probe_source"], probe["probe_reference"],
            probe.get("delta_added_types", []),
            probe.get("delta_removed_types", []),
        )
    elif fam == "overcorrection":
        out = filter_overcorrection(
            orig_item["source_text"], orig_item["reference_correction"],
            probe["probe_source"], probe["probe_reference"],
            probe["cue_token"], tuple(probe["protected_span"]),
        )
    else:
        raise ValueError(f"Unknown probe family: {fam}")
    out["dataset"] = "BEA2019_WI_LOCNESS"
    out["probe_id"] = probe["probe_id"]
    out["item_id"] = probe["base_item_id"]
    out["transform_type"] = fam
    return out
