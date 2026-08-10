"""Exact text-level scorer specified by CONTRAST-SRS-CR-v1.0.0.

The public ARR data archive contains pair-level scores rather than upstream
requirements or model rewrites. This module makes the frozen scoring logic
inspectable and testable on synthetic fixtures without redistributing source
corpora.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

from .contract import (
    ANCHOR_RECALL_THRESHOLD,
    CONTENT_F1_THRESHOLD,
    CONTENT_STOPWORDS,
    MODAL_CLASSES,
    NEGATION_TOKENS,
)

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
MODAL_RE = re.compile(r"\b(shall|must|should|will|may|can)\b")
EARS_MODAL_RE = re.compile(r"\b(shall|should|must|may|will)\b")
EARS_SUBJECT_RE = re.compile(r"\bthe\s+\w+(?:\s+\w+)?\s+(?:shall|should|must|may|will)\b")
EARS_TRIGGER_RE = re.compile(r"\b(?:when|if|while|where)\b")


def normalize_text(text: str | None) -> str:
    return unicodedata.normalize("NFKC", text or "").casefold()


def tokens(text: str | None) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def first_modal_token(text: str | None) -> str | None:
    match = MODAL_RE.search(normalize_text(text))
    return match.group(1) if match else None


def first_modal_class(text: str | None) -> int:
    modal = first_modal_token(text)
    return MODAL_CLASSES.get(modal, 0) if modal else 0


def highest_modal_class(text: str | None) -> int:
    values = [MODAL_CLASSES[match.group(1)] for match in MODAL_RE.finditer(normalize_text(text))]
    return max(values) if values else 0


def unique_modal_class(text: str | None) -> int:
    values = {MODAL_CLASSES[match.group(1)] for match in MODAL_RE.finditer(normalize_text(text))}
    return next(iter(values)) if len(values) == 1 else 0


def content_token_set(text: str | None, excluded: Iterable[str] = ()) -> set[str]:
    excluded_set = set(excluded)
    return {
        token
        for token in tokens(text)
        if len(token) > 1
        and token not in CONTENT_STOPWORDS
        and token not in MODAL_CLASSES
        and token not in excluded_set
    }


def set_f1(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    precision = overlap / len(left)
    recall = overlap / len(right)
    return 2.0 * precision * recall / (precision + recall) if overlap else 0.0


def number_set(text: str | None) -> set[str]:
    return set(NUMBER_RE.findall(normalize_text(text)))


def negation_set(text: str | None) -> set[str]:
    return set(tokens(text)) & NEGATION_TOKENS


def exact_set_consistency(
    base_source: str,
    probe_source: str,
    base_rewrite: str,
    probe_rewrite: str,
    extractor,
) -> bool:
    shared_source = extractor(base_source) & extractor(probe_source)
    base_output = extractor(base_rewrite)
    probe_output = extractor(probe_rewrite)
    return shared_source <= base_output and shared_source <= probe_output and base_output == probe_output


@dataclass(frozen=True)
class CollateralScores:
    content_f1: float
    shared_anchor_recall_min: float
    number_consistency_pass: int
    negation_consistency_pass: int
    collateral_primary_pass: int


def collateral_scores(
    base_source: str,
    probe_source: str,
    base_rewrite: str,
    probe_rewrite: str,
    *,
    distractor: str | None = None,
) -> CollateralScores:
    distractor_tokens = content_token_set(distractor) if distractor else set()
    base_output_tokens = content_token_set(base_rewrite, distractor_tokens)
    probe_output_tokens = content_token_set(probe_rewrite, distractor_tokens)
    content_f1 = set_f1(base_output_tokens, probe_output_tokens)

    base_source_tokens = content_token_set(base_source)
    probe_source_tokens = content_token_set(probe_source)
    shared_anchors = base_source_tokens & probe_source_tokens
    if shared_anchors:
        base_recall = len(shared_anchors & base_output_tokens) / len(shared_anchors)
        probe_recall = len(shared_anchors & probe_output_tokens) / len(shared_anchors)
        anchor_recall = min(base_recall, probe_recall)
    else:
        anchor_recall = 1.0

    number_pass = int(
        exact_set_consistency(base_source, probe_source, base_rewrite, probe_rewrite, number_set)
    )
    negation_pass = int(
        exact_set_consistency(base_source, probe_source, base_rewrite, probe_rewrite, negation_set)
    )
    collateral_pass = int(
        content_f1 >= CONTENT_F1_THRESHOLD
        and anchor_recall >= ANCHOR_RECALL_THRESHOLD
        and number_pass == 1
        and negation_pass == 1
    )
    return CollateralScores(content_f1, anchor_recall, number_pass, negation_pass, collateral_pass)


def ears_struct_v1(text: str | None) -> float:
    normalized = normalize_text(text).strip()
    if len(normalized) < 5:
        return 0.0
    score = 0.0
    score += 0.4 if EARS_MODAL_RE.search(normalized) else 0.0
    score += 0.3 if EARS_SUBJECT_RE.search(normalized) else 0.0
    score += 0.2 if normalized.endswith(".") else 0.0
    if EARS_TRIGGER_RE.search(normalized) or re.match(r"^the\s+\w+", normalized):
        score += 0.1
    return round(min(score, 1.0), 3)


def distractor_coverage(distractor: str | None, rewrite: str | None) -> float:
    distractor_tokens = content_token_set(distractor)
    if not distractor_tokens:
        return 1.0
    return len(distractor_tokens & content_token_set(rewrite)) / len(distractor_tokens)


def normalized_phrase_present(phrase: str | None, text: str | None) -> bool:
    normalized_phrase = normalize_text(phrase).strip()
    normalized_text = normalize_text(text)
    return bool(normalized_phrase) and normalized_phrase in normalized_text


def source_eligibility(record: dict[str, Any]) -> tuple[bool, str]:
    family = record.get("analysis_family") or record.get("probe_family")
    base_source = record.get("base_source", "")
    probe_source = record.get("probe_source", "")
    if family == "invariance":
        passed = record.get("expected_relation") == "stable" and first_modal_class(base_source) != 0
        return passed, "eligible" if passed else "invariance_source_rule_failed"
    if family == "directional":
        original_modal = normalize_text(record.get("original_modal", ""))
        new_modal = normalize_text(record.get("new_modal", ""))
        direction = record.get("direction")
        original_class = MODAL_CLASSES.get(original_modal)
        new_class = MODAL_CLASSES.get(new_modal)
        direction_ok = (
            (direction == "strengthen" and original_class is not None and new_class is not None and new_class > original_class)
            or (direction == "weaken" and original_class is not None and new_class is not None and new_class < original_class)
        )
        passed = all(
            [
                record.get("operation") == "modal_substitution",
                direction_ok,
                first_modal_token(base_source) == original_modal,
                first_modal_token(probe_source) == new_modal,
            ]
        )
        return passed, "eligible" if passed else "directional_source_rule_failed"
    if family in {"addition", "shortcut"}:
        distractor = record.get("distractor", "")
        passed = (
            record.get("operation") == "distractor_insertion"
            and bool(content_token_set(distractor))
            and normalize_text(distractor).strip() in normalize_text(probe_source)
            and normalize_text(distractor).strip() not in normalize_text(base_source)
        )
        return passed, "eligible" if passed else "addition_source_rule_failed"
    return False, "unsupported_family"


def score_pair(record: dict[str, Any]) -> dict[str, float | int | None]:
    family = record.get("analysis_family") or record.get("probe_family")
    if family == "shortcut":
        family = "addition"
    base_source = record.get("base_source", "")
    probe_source = record.get("probe_source", "")
    base_rewrite = record.get("base_rewrite", "")
    probe_rewrite = record.get("probe_rewrite", "")
    available = bool(base_rewrite.strip() and probe_rewrite.strip())
    result: dict[str, float | int | None] = {"technical_pair_available": int(available)}
    if not available:
        return result

    base_output_class = first_modal_class(base_rewrite)
    probe_output_class = first_modal_class(probe_rewrite)
    focal_relation = 0
    focal_exact = 0
    if family == "invariance":
        source_class = first_modal_class(base_source)
        focal_relation = int(
            base_output_class != 0
            and probe_output_class != 0
            and base_output_class == probe_output_class
        )
        focal_exact = int(
            source_class != 0
            and base_output_class == source_class
            and probe_output_class == source_class
        )
    elif family == "directional":
        direction = record.get("direction")
        focal_relation = int(
            base_output_class != 0
            and probe_output_class != 0
            and (
                (direction == "strengthen" and probe_output_class > base_output_class)
                or (direction == "weaken" and probe_output_class < base_output_class)
            )
        )
        original_class = MODAL_CLASSES.get(normalize_text(record.get("original_modal", "")))
        new_class = MODAL_CLASSES.get(normalize_text(record.get("new_modal", "")))
        focal_exact = int(
            original_class is not None
            and new_class is not None
            and base_output_class == original_class
            and probe_output_class == new_class
        )
    elif family != "addition":
        raise ValueError(f"unsupported analysis family: {family!r}")

    distractor = record.get("distractor") if family == "addition" else None
    collateral = collateral_scores(
        base_source,
        probe_source,
        base_rewrite,
        probe_rewrite,
        distractor=distractor,
    )
    result.update(
        {
            "base_output_first_modal_class": base_output_class,
            "probe_output_first_modal_class": probe_output_class,
            "both_endpoint_pass_100": int(
                ears_struct_v1(base_rewrite) == 1.0 and ears_struct_v1(probe_rewrite) == 1.0
            ),
            "output_nonfocal_content_f1": collateral.content_f1,
            "shared_anchor_recall_min": collateral.shared_anchor_recall_min,
            "number_consistency_pass": collateral.number_consistency_pass,
            "negation_consistency_pass": collateral.negation_consistency_pass,
            "collateral_primary_pass": collateral.collateral_primary_pass,
        }
    )
    if family in {"invariance", "directional"}:
        result.update(
            {
                "focal_relation_pass": focal_relation,
                "focal_exact_pass": focal_exact,
                "complete_relation_primary": focal_relation * collateral.collateral_primary_pass,
                "complete_exact_primary": focal_exact * collateral.collateral_primary_pass,
            }
        )
    else:
        base_coverage = distractor_coverage(distractor, base_rewrite)
        probe_coverage = distractor_coverage(distractor, probe_rewrite)
        uptake = int(probe_coverage >= 0.70 and probe_coverage > base_coverage)
        exact = int(
            normalized_phrase_present(distractor, probe_rewrite)
            and not normalized_phrase_present(distractor, base_rewrite)
        )
        result.update(
            {
                "distractor_base_coverage": base_coverage,
                "distractor_probe_coverage": probe_coverage,
                "addition_uptake_70": uptake,
                "addition_probe_only_exact_phrase": exact,
                "addition_uptake_with_collateral": uptake * collateral.collateral_primary_pass,
            }
        )
    return result
