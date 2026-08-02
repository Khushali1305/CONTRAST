"""
gec_probe.run_pipeline
----------------------
Orchestrates the full GEC CONTRAST pipeline end-to-end.

Stage order (each writes JSONL; downstream stages read those JSONLs):
    1. ingest     -> data_artifacts/normalized/bea2019_{train,dev}.jsonl
    2. stratify   -> data_artifacts/probes/gec_probecore_base.jsonl    (FROZEN)
    3. probes     -> gec_probes_{invariance,directional,overcorrection}.jsonl
    4. rule_filter (applied during judge stage; written to a separate JSONL)
    5. judge      -> data_artifacts/judge/gec_judge_decisions.jsonl
    6. infer      -> data_artifacts/inference/<model>/<probe_id>.json
    7. score      -> data_artifacts/scores/rq1_reliability_gec.jsonl
                  -> data_artifacts/scores/rq3_conformance_gec.jsonl
    8. analyze    -> data_artifacts/scores/cross_domain_rho_triangle.csv

Each stage checks for the existence of its primary output before re-running;
to force a re-run, delete the output file or use --force STAGE.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from gec_probe.config.settings import (
    NORMALIZED_DIR, PROBES_DIR, JUDGE_DIR, INFERENCE_DIR, SCORES_DIR,
    SPLIT_FILES,
)


def _exists_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def stage_ingest(force: bool) -> None:
    out = NORMALIZED_DIR / "bea2019_dev.jsonl"
    if _exists_nonempty(out) and not force:
        print(f"[skip] ingest (exists: {out})")
        return
    from gec_probe.data import ingest_bea2019
    for split in ("train", "dev"):
        ingest_bea2019.ingest_split(split)


def stage_stratify(force: bool) -> None:
    out = PROBES_DIR / "gec_probecore_base.jsonl"
    if _exists_nonempty(out) and not force:
        print(f"[skip] stratify (exists: {out})")
        return
    from gec_probe.data import stratified_sample
    stratified_sample.build_probecore_base()


def _get_done_base_ids(out: Path) -> set:
    """Read already-processed base_item_ids from a partial output JSONL."""
    done = set()
    if not out.exists():
        return done
    with out.open(encoding="utf-8",errors="replace") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                done.add(json.loads(ln)["base_item_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _run_probe_with_resume(mod, name: str, force: bool) -> None:
    """
    Run a probe module with resume support.

    - force=True  : delete existing output and regenerate from scratch.
    - force=False : if output exists and is non-empty, read already-done
                    base_item_ids and skip them; append new probes to the
                    existing file so no work is lost.
    """
    import json as _json
    out = PROBES_DIR / f"gec_probes_{name}.jsonl"

    if force and out.exists():
        print(f"[force] deleting {out}")
        out.unlink()

    if not _exists_nonempty(out):
        # Clean run — delegate entirely to the module's own run().
        print(f"[run] probes:{name} (fresh)")
        mod.run()
        return

    # Partial file exists — resume from where we left off.
    done_ids = _get_done_base_ids(out)
    print(f"[resume] probes:{name} — {len(done_ids)} base items already done, "
          f"appending remaining to {out}")

    base_path = PROBES_DIR / "gec_probecore_base.jsonl"
    if not base_path.exists():
        raise FileNotFoundError(f"ProbeCore base not found: {base_path}")

    # Load all base items, filter to those not yet processed.
    remaining = []
    with base_path.open(encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            item = _json.loads(ln)
            if item["sentence_id"] not in done_ids:
                remaining.append(item)

    if not remaining:
        print(f"[skip] probes:{name} — all base items already processed")
        return

    print(f"[resume] probes:{name} — {len(remaining)} base items remaining")

    # Use the module's generate_for_item if available (invariance, directional,
    # overcorrection all expose it); fall back to full run() if not.
    if not hasattr(mod, "generate_for_item"):
        print(f"[warn] probes:{name} has no generate_for_item(); "
              f"cannot resume — running full mod.run() which will overwrite.")
        mod.run()
        return

    n_total = n_valid = 0
    with out.open("a",encoding="utf-8") as fh:          # APPEND, not overwrite
        for item in remaining:
            for probe in mod.generate_for_item(item):
                fh.write(_json.dumps(probe, ensure_ascii=False) + "\n")
                fh.flush()             # flush each probe so crashes don't lose work
                n_total += 1
                n_valid += int(probe.get("valid", False))

    print(f"[resume done] probes:{name} appended={n_total} valid={n_valid}")


def stage_probes(force: bool) -> None:
    from gec_probe.probes import invariance, directional, overcorrection
    for mod, name in [(invariance, "invariance"),
                      (directional, "directional"),
                      (overcorrection, "overcorrection")]:
        _run_probe_with_resume(mod, name, force)


def stage_judge(force: bool) -> None:
    out = JUDGE_DIR / "gec_judge_decisions.jsonl"
    if _exists_nonempty(out) and not force:
        print(f"[skip] judge (exists)")
        return
    from gec_probe.judges import two_judge_validator
    two_judge_validator.run(
        probes_paths=[
            PROBES_DIR / "gec_probes_invariance.jsonl",
            PROBES_DIR / "gec_probes_directional.jsonl",
            PROBES_DIR / "gec_probes_overcorrection.jsonl",
        ],
        base_items_path=PROBES_DIR / "gec_probecore_base.jsonl",
    )


def stage_infer(force: bool) -> None:
    # Inference always runs because checkpointing handles partial completion.
    from gec_probe.inference import run_correctors
    run_correctors.run(
        judge_decisions_path=JUDGE_DIR / "gec_judge_decisions.jsonl",
        probes_paths=[
            PROBES_DIR / "gec_probes_invariance.jsonl",
            PROBES_DIR / "gec_probes_directional.jsonl",
            PROBES_DIR / "gec_probes_overcorrection.jsonl",
        ],
        base_items_path=PROBES_DIR / "gec_probecore_base.jsonl",
    )


def stage_score(force: bool) -> None:
    from gec_probe.scoring import probe_scoring, conformance
    probe_scoring.score_all()
    conformance.score_all_conformance()


def stage_analyze(force: bool) -> None:
    from gec_probe.analysis import cross_domain_rho
    # Caller should pass real SRS/Simp paths; CLI default falls back gracefully.
    print("[analyze] run cross_domain_rho.run(srs, simp, gec) manually with "
          "your existing SRS/Simp tables.")


STAGES = {
    "ingest":   stage_ingest,
    "stratify": stage_stratify,
    "probes":   stage_probes,
    "judge":    stage_judge,
    "infer":    stage_infer,
    "score":    stage_score,
    "analyze":  stage_analyze,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stages", nargs="*", default=list(STAGES),
        help="Stage names to run (default: all).",
    )
    parser.add_argument(
        "--force", nargs="*", default=[],
        help="Stage names to force re-run.",
    )
    args = parser.parse_args()
    for s in args.stages:
        if s not in STAGES:
            raise SystemExit(f"Unknown stage: {s}. Valid: {list(STAGES)}")
        force = s in args.force
        print(f"\n=== STAGE: {s} (force={force}) ===")
        STAGES[s](force=force)


if __name__ == "__main__":
    main()