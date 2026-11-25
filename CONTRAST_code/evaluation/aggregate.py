"""Recompute frozen SRS item-macro tables from pair-level scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .contract import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONTRACT_VERSIONS,
    DEFAULT_CONTRACT_VERSION,
    MODEL_METADATA,
    MODEL_ORDER,
    is_v11_excluded,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def apply_contract_version(rows: list[dict[str, str]], version: str) -> list[dict[str, str]]:
    """Mark v1.1.0 exclusions as source-ineligible; v1.0 returns the rows unchanged."""
    if version not in CONTRACT_VERSIONS:
        raise ValueError(f"unknown contract version: {version}")
    if version == "1.0":
        return rows
    return [
        {**row, "source_eligible": "0"} if is_v11_excluded(row) else row
        for row in rows
    ]


def numeric(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    return float(value) if value not in {"", None} else None


def eligible_available(rows: Iterable[dict[str, str]], family: str) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["analysis_family"] == family
        and row["source_eligible"] == "1"
        and row["technical_pair_available"] == "1"
    ]


def item_scores(rows: Iterable[dict[str, str]], metric: str) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = numeric(row, metric)
        if value is not None:
            grouped[(row["source_dataset"], row["canonical_item_id"])].append(value)
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def item_macro(rows: Iterable[dict[str, str]], metric: str) -> float:
    scores = item_scores(rows, metric)
    return float(np.mean(list(scores.values()))) if scores else float("nan")


def stratified_item_bootstrap(
    scores: dict[tuple[str, str], float],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    if not scores or replicates <= 0:
        return float("nan"), float("nan")
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for (dataset, _), value in scores.items():
        by_dataset[dataset].append(value)
    arrays = [np.asarray(by_dataset[name], dtype=float) for name in sorted(by_dataset)]
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=float)
    chunk_size = 500
    total_items = sum(array.size for array in arrays)
    for start in range(0, replicates, chunk_size):
        size = min(chunk_size, replicates - start)
        combined_sum = np.zeros(size, dtype=float)
        for array in arrays:
            indices = rng.integers(0, array.size, size=(size, array.size))
            combined_sum += array[indices].sum(axis=1)
        samples[start:start + size] = combined_sum / total_items
    low, high = np.quantile(samples, [0.025, 0.975], method="linear")
    return float(low), float(high)


def stable_seed(base_seed: int, *parts: str) -> int:
    material = "\x1f".join(parts).encode("utf-8")
    offset = int(hashlib.sha256(material).hexdigest()[:8], 16)
    return (base_seed + offset) % (2**32 - 1)


def family_counts(model_rows: list[dict[str, str]], family: str) -> tuple[int, int]:
    eligible = [row for row in model_rows if row["analysis_family"] == family and row["source_eligible"] == "1"]
    available = [row for row in eligible if row["technical_pair_available"] == "1"]
    return len(eligible), len(available)


def bootstrap_metric(
    rows: list[dict[str, str]],
    family: str,
    model_id: str,
    metric: str,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    return stratified_item_bootstrap(
        item_scores(rows, metric),
        replicates=replicates,
        seed=stable_seed(seed, family, model_id, metric),
    )


def compute_primary_results(
    rows: list[dict[str, str]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, object]]:
    output = []
    for model_id in MODEL_ORDER:
        model_rows = [row for row in rows if row["model_id"] == model_id]
        display, role, prompt = MODEL_METADATA[model_id]
        inv = eligible_available(model_rows, "invariance")
        directional = eligible_available(model_rows, "directional")
        addition = eligible_available(model_rows, "addition")
        inv_eligible, inv_available = family_counts(model_rows, "invariance")
        dir_eligible, dir_available = family_counts(model_rows, "directional")
        add_eligible, add_available = family_counts(model_rows, "addition")
        inv_ci = bootstrap_metric(
            inv, "invariance", model_id, "complete_exact_primary", replicates, seed
        )
        dir_ci = bootstrap_metric(
            directional, "directional", model_id, "complete_exact_primary", replicates, seed
        )
        output.append(
            {
                "model_id": model_id,
                "model_display_name": display,
                "model_role": role,
                "prompt_group": prompt,
                "inv_eligible_pairs": inv_eligible,
                "inv_available_pairs": inv_available,
                "inv_coverage": inv_available / inv_eligible,
                "inv_focal_relation_pass": item_macro(inv, "focal_relation_pass"),
                "inv_focal_exact_pass": item_macro(inv, "focal_exact_pass"),
                "inv_collateral_primary_pass": item_macro(inv, "collateral_primary_pass"),
                "inv_complete_exact_primary": item_macro(inv, "complete_exact_primary"),
                "inv_complete_exact_primary_ci_lo": inv_ci[0],
                "inv_complete_exact_primary_ci_hi": inv_ci[1],
                "dir_eligible_pairs": dir_eligible,
                "dir_available_pairs": dir_available,
                "dir_coverage": dir_available / dir_eligible,
                "dir_focal_relation_pass": item_macro(directional, "focal_relation_pass"),
                "dir_focal_exact_pass": item_macro(directional, "focal_exact_pass"),
                "dir_collateral_primary_pass": item_macro(directional, "collateral_primary_pass"),
                "dir_complete_exact_primary": item_macro(directional, "complete_exact_primary"),
                "dir_complete_exact_primary_ci_lo": dir_ci[0],
                "dir_complete_exact_primary_ci_hi": dir_ci[1],
                "add_eligible_pairs": add_eligible,
                "add_available_pairs": add_available,
                "add_coverage": add_available / add_eligible,
                "add_addition_uptake_70": item_macro(addition, "addition_uptake_70"),
                "add_collateral_primary_pass": item_macro(addition, "collateral_primary_pass"),
                "add_addition_uptake_with_collateral": item_macro(addition, "addition_uptake_with_collateral"),
            }
        )
    return output


def endpoint_subset(model_rows: list[dict[str, str]], family: str) -> list[dict[str, str]]:
    return [
        row
        for row in eligible_available(model_rows, family)
        if numeric(row, "both_endpoint_pass_100") == 1.0
    ]


def failure_item_scores(rows: list[dict[str, str]], pass_metric: str) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = numeric(row, pass_metric)
        if value is not None:
            grouped[(row["source_dataset"], row["canonical_item_id"])].append(1.0 - value)
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def failure_macro(rows: list[dict[str, str]], pass_metric: str) -> float:
    scores = failure_item_scores(rows, pass_metric)
    return float(np.mean(list(scores.values()))) if scores else float("nan")


def compute_hidden_failures(
    rows: list[dict[str, str]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, object]]:
    output = []
    for model_id in MODEL_ORDER:
        model_rows = [row for row in rows if row["model_id"] == model_id]
        display, role, prompt = MODEL_METADATA[model_id]
        inv = endpoint_subset(model_rows, "invariance")
        directional = endpoint_subset(model_rows, "directional")
        inv_ci = stratified_item_bootstrap(
            failure_item_scores(inv, "complete_exact_primary"),
            replicates=replicates,
            seed=stable_seed(seed, "invariance", model_id, "hidden_exact_complete_failure"),
        )
        dir_ci = stratified_item_bootstrap(
            failure_item_scores(directional, "complete_exact_primary"),
            replicates=replicates,
            seed=stable_seed(seed, "directional", model_id, "hidden_exact_complete_failure"),
        )
        output.append(
            {
                "model_id": model_id,
                "model_display_name": display,
                "model_role": role,
                "prompt_group": prompt,
                "inv_endpoint_passing_pairs": len(inv),
                "inv_endpoint_passing_items": len({row["canonical_item_id"] for row in inv}),
                "inv_hidden_relation_failure": failure_macro(inv, "focal_relation_pass"),
                "inv_hidden_exact_focal_failure": failure_macro(inv, "focal_exact_pass"),
                "inv_hidden_relation_complete_failure": failure_macro(inv, "complete_relation_primary"),
                "inv_hidden_exact_complete_failure": failure_macro(inv, "complete_exact_primary"),
                "inv_hidden_exact_complete_failure_ci_lo": inv_ci[0],
                "inv_hidden_exact_complete_failure_ci_hi": inv_ci[1],
                "dir_endpoint_passing_pairs": len(directional),
                "dir_endpoint_passing_items": len({row["canonical_item_id"] for row in directional}),
                "dir_hidden_relation_failure": failure_macro(directional, "focal_relation_pass"),
                "dir_hidden_exact_focal_failure": failure_macro(directional, "focal_exact_pass"),
                "dir_hidden_relation_complete_failure": failure_macro(directional, "complete_relation_primary"),
                "dir_hidden_exact_complete_failure": failure_macro(directional, "complete_exact_primary"),
                "dir_hidden_exact_complete_failure_ci_lo": dir_ci[0],
                "dir_hidden_exact_complete_failure_ci_hi": dir_ci[1],
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("recomputed"))
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--contract-version", choices=CONTRACT_VERSIONS, default=DEFAULT_CONTRACT_VERSION,
        help="1.1 (default) reproduces the paper; 1.0 reproduces the original frozen tables.",
    )
    args = parser.parse_args()
    rows = apply_contract_version(
        read_rows(args.data_dir / "srs/pair_scores.csv"), args.contract_version
    )
    write_csv(
        args.output_dir / "primary_results.csv",
        compute_primary_results(rows, replicates=args.bootstrap_replicates, seed=args.seed),
    )
    write_csv(
        args.output_dir / "hidden_failures.csv",
        compute_hidden_failures(rows, replicates=args.bootstrap_replicates, seed=args.seed),
    )
    print(f"Wrote contract v{args.contract_version} aggregates to {args.output_dir}")


if __name__ == "__main__":
    main()
