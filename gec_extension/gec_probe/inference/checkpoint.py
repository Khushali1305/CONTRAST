"""
gec_probe.inference.checkpoint
------------------------------
Checkpoint utility for long-running corrector runs.

CRITICAL: We verify ACTUAL OUTPUT EXISTENCE on disk, not just a flag.
Per known failure mode: checkpoint systems that mark crashed runs as
complete are a silent data-quality killer.

Schema:
    checkpoint.jsonl, one line per completed (model, probe_id) pair:
        {"model": "...", "probe_id": "...", "output_file": "..."}
    The output_file must exist and contain valid JSON.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Set, Tuple


def _output_filename(model_id: str, probe_id: str) -> str:
    return f"{model_id}/{probe_id}.json"


def load_completed(checkpoint_path: Path, base_dir: Path) -> Set[Tuple[str, str]]:
    """
    Return the set of (model, probe_id) pairs that are actually complete:
    a checkpoint entry exists AND the referenced file exists AND parses
    as JSON with the expected schema.
    """
    completed: Set[Tuple[str, str]] = set()
    if not checkpoint_path.exists():
        return completed

    salvageable_count = 0
    invalid_count = 0
    with checkpoint_path.open() as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                invalid_count += 1
                continue
            model = rec.get("model")
            probe_id = rec.get("probe_id")
            out_file = rec.get("output_file")
            if not (model and probe_id and out_file):
                invalid_count += 1
                continue
            full_path = base_dir / out_file
            if not full_path.exists():
                invalid_count += 1
                continue
            try:
                # Verify the file actually contains the expected payload
                with full_path.open() as f2:
                    payload = json.load(f2)
                if "output" not in payload:
                    invalid_count += 1
                    continue
            except Exception:
                invalid_count += 1
                continue
            completed.add((model, probe_id))
            salvageable_count += 1

    if invalid_count:
        print(f"[checkpoint] WARNING: {invalid_count} checkpoint entries "
              f"were INVALID (missing/corrupt file). Will re-run them. "
              f"({salvageable_count} are valid.)")
    return completed


def append_checkpoint(checkpoint_path: Path, model_id: str,
                      probe_id: str, output_payload: dict,
                      base_dir: Path) -> None:
    """Write the output then append the checkpoint entry. Order matters."""
    out_rel = _output_filename(model_id, probe_id)
    out_full = base_dir / out_rel
    out_full.parent.mkdir(parents=True, exist_ok=True)
    # Write output FIRST.
    with out_full.open("w") as fh:
        json.dump(output_payload, fh, ensure_ascii=False)
    # Then append checkpoint entry.
    with checkpoint_path.open("a") as fh:
        fh.write(json.dumps({
            "model": model_id,
            "probe_id": probe_id,
            "output_file": out_rel,
        }) + "\n")
