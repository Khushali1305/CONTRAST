"""
gec_probe.probes.errant_utils
-----------------------------
Wrappers around ERRANT for use in probe generation and rule filtering.

Install:
    pip install errant
    python -m spacy download en_core_web_sm
"""
from __future__ import annotations
from typing import List, Tuple
from functools import lru_cache

# ERRANT is loaded lazily so unit tests can stub it.
_annotator = None


def get_annotator():
    """Lazy load (annotator init is expensive)."""
    global _annotator
    if _annotator is None:
        import errant
        _annotator = errant.load("en")
    return _annotator


@lru_cache(maxsize=8192)
def _parse(text: str):
    ann = get_annotator()
    return ann.parse(text)


def annotate(source: str, target: str) -> List[dict]:
    """
    Return ERRANT edits transforming `source` -> `target`.
    Each edit: {start, end, type, correction}.
    """
    ann = get_annotator()
    src = _parse(source)
    tgt = _parse(target)
    edits = ann.annotate(src, tgt)
    out = []
    for e in edits:
        out.append({
            "start": e.o_start,
            "end": e.o_end,
            "type": e.type,
            "correction": e.c_str,
        })
    return out


def error_types(source: str, target: str) -> List[str]:
    return [e["type"] for e in annotate(source, target) if e["type"] != "noop"]


def edit_signature(source: str, target: str) -> Tuple[Tuple[int, int, str], ...]:
    """Hashable signature of an edit set; used to compare probe vs source edits."""
    return tuple(
        (e["start"], e["end"], e["type"])
        for e in annotate(source, target)
        if e["type"] != "noop"
    )


def has_same_error_type(
    src1: str, ref1: str, src2: str, ref2: str,
) -> bool:
    """Check that probe (src2->ref2) preserves source's (src1->ref1) error types."""
    return set(error_types(src1, ref1)) == set(error_types(src2, ref2))


def spans_outside(
    source: str, output: str, protected_span: Tuple[int, int],
) -> List[dict]:
    """
    Return edits in (source -> output) that fall outside the protected token span.
    Used for the overcorrection probe family: any edits outside the original
    error span count as overcorrection.
    """
    start, end = protected_span
    outside = []
    for e in annotate(source, output):
        if e["type"] == "noop":
            continue
        if e["end"] <= start or e["start"] >= end:
            outside.append(e)
    return outside
