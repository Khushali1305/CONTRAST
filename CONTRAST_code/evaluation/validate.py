"""Validate package integrity, frozen counts, missingness, and table agreement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .contract import (
    CONTRACT_ID,
    CONTRACT_ID_V11,
    CONTRACT_VERSIONS,
    DEFAULT_CONTRACT_VERSION,
    EXPECTED_ELIGIBLE_PAIRS,
    EXPECTED_ELIGIBLE_PAIRS_V11,
    MODEL_ORDER,
    V11_EXCLUDED_PAIRS_PER_MODEL,
    is_v11_excluded,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_manifest(data_dir: Path) -> None:
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = data_dir / entry["path"]
        assert path.is_file(), f"missing manifest file: {entry['path']}"
        assert path.stat().st_size == entry["bytes"], f"size mismatch: {entry['path']}"
        assert sha256(path) == entry["sha256"], f"hash mismatch: {entry['path']}"
        if "data_rows" in entry:
            assert len(csv_rows(path)) == entry["data_rows"], f"row mismatch: {entry['path']}"


def verify_contract(data_dir: Path) -> None:
    contract = json.loads((data_dir / "srs/contract.json").read_text(encoding="utf-8"))
    assert contract["contract_id"] == CONTRACT_ID
    assert contract["status"] == "frozen"
    assert int(contract["aggregation.confidence_interval.replicates"]) == 10_000
    assert int(contract["aggregation.confidence_interval.seed"]) == 20_260_731


def verify_pair_scores(data_dir: Path) -> None:
    rows = csv_rows(data_dir / "srs/pair_scores.csv")
    assert len(rows) == 24_544
    ids = [row["canonical_pair_id"] for row in rows]
    assert len(ids) == len(set(ids)), "duplicate canonical_pair_id"
    forbidden = {"requirement_text", "input_text", "rewrite", "userid", "email"}
    assert not (forbidden & set(rows[0])), "raw or identifying columns are present"
    for model_id in MODEL_ORDER:
        model_rows = [row for row in rows if row["model_id"] == model_id]
        for family, expected in EXPECTED_ELIGIBLE_PAIRS.items():
            family_rows = [
                row for row in model_rows
                if row["analysis_family"] == family and row["source_eligible"] == "1"
            ]
            assert len(family_rows) == expected, (model_id, family, len(family_rows), expected)
            available = sum(row["technical_pair_available"] == "1" for row in family_rows)
            expected_available = expected - 1 if model_id == "qwen_32b" else expected
            assert available == expected_available, (model_id, family, available, expected_available)


def verify_v11_contract(data_dir: Path) -> None:
    contract = json.loads((data_dir / "srs/contract_v1_1.json").read_text(encoding="utf-8"))
    assert contract["contract_id"] == CONTRACT_ID_V11
    assert contract["amends"] == CONTRACT_ID
    rows = csv_rows(data_dir / "srs/pair_scores.csv")
    for model_id in MODEL_ORDER:
        model_rows = [row for row in rows if row["model_id"] == model_id]
        excluded = [
            row for row in model_rows
            if row["source_eligible"] == "1" and is_v11_excluded(row)
        ]
        assert len(excluded) == V11_EXCLUDED_PAIRS_PER_MODEL, (model_id, len(excluded))
        for family, expected in EXPECTED_ELIGIBLE_PAIRS_V11.items():
            kept = [
                row for row in model_rows
                if row["analysis_family"] == family
                and row["source_eligible"] == "1"
                and not is_v11_excluded(row)
            ]
            assert len(kept) == expected, (model_id, family, len(kept), expected)


def compare_numeric_tables(actual: Path, expected: Path, tolerance: float = 1e-12) -> None:
    actual_rows = csv_rows(actual)
    expected_rows = csv_rows(expected)
    assert len(actual_rows) == len(expected_rows)
    assert actual_rows[0].keys() == expected_rows[0].keys()
    for left, right in zip(actual_rows, expected_rows):
        assert left["model_id"] == right["model_id"]
        for field in left:
            if field in {"model_id", "model_display_name", "model_role", "prompt_group"}:
                assert left[field] == right[field]
                continue
            if left[field] == right[field]:
                continue
            assert abs(float(left[field]) - float(right[field])) <= tolerance, (
                left["model_id"], field, left[field], right[field]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--recomputed-dir", type=Path)
    parser.add_argument(
        "--contract-version", choices=CONTRACT_VERSIONS, default=DEFAULT_CONTRACT_VERSION
    )
    args = parser.parse_args()
    verify_manifest(args.data_dir)
    verify_contract(args.data_dir)
    verify_pair_scores(args.data_dir)
    verify_v11_contract(args.data_dir)
    if args.recomputed_dir:
        suffix = "_v1_1" if args.contract_version == "1.1" else ""
        compare_numeric_tables(
            args.recomputed_dir / "primary_results.csv",
            args.data_dir / f"srs/primary_results{suffix}.csv",
        )
        compare_numeric_tables(
            args.recomputed_dir / "hidden_failures.csv",
            args.data_dir / f"srs/hidden_failures{suffix}.csv",
        )
    print("All integrity, count, missingness, and table checks passed.")


if __name__ == "__main__":
    main()

