"""
gec_probe.judges.two_judge_validator
------------------------------------
Two-judge probe validation. Judges MUST come from model families that have
zero overlap with the corrector pool (enforced at import time via
assert_judge_corrector_disjoint()).

Schema mirrors `judge_decisions_detail.csv` and `peer_review_agreement.csv`
from the SRS pipeline.
"""
from __future__ import annotations
import json
from pathlib import Path

from gec_probe.config.settings import (
    JUDGE_MODELS, JUDGE_DIR,
    assert_judge_corrector_disjoint, assert_api_keys_present,
    assert_generator_disjoint_from_judges_and_correctors,
)
from gec_probe.inference.llm_client import call_llm

# Enforce family disjointness at import time
assert_judge_corrector_disjoint()
assert_generator_disjoint_from_judges_and_correctors()


JUDGE_PROMPT = """You are validating a counterfactual probe for grammatical
error correction evaluation.

ORIGINAL SENTENCE:
{orig_src}

PROBE SENTENCE:
{probe_src}

PROBE FAMILY: {family}
EXPECTED RELATION: {expected_relation}

Per family, the probe is VALID if:
  - invariance: probe rephrases the original, preserves the SAME grammatical
    errors, does not introduce new errors, and reads as natural English.
  - directional: probe modifies the error set in the stated direction
    (added/removed/swapped error types). The probe should otherwise be
    minimally changed from the original.
  - overcorrection: probe inserts the cue token "{cue_token}" outside
    the original error span; the original error is unchanged; no other
    edits are present.

Respond ONLY in this JSON format:
{{
  "equivalent": true | false,
  "confidence": "high" | "medium" | "low",
  "failures": ["..."]  // empty if equivalent=true
}}
"""


def _parse_judge_response(text: str) -> dict:
    """Permissive JSON parsing; falls back to error structure."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        # remove "json" tag if present
        text = text.split("\n", 1)[1] if "\n" in text else text
    try:
        obj = json.loads(text)
        return {
            "equivalent": bool(obj.get("equivalent")),
            "confidence": obj.get("confidence", "low"),
            "failures": obj.get("failures", []) or [],
            "parse_ok": True,
        }
    except Exception as e:
        return {
            "equivalent": None,
            "confidence": "low",
            "failures": [f"parse_error: {e}"],
            "parse_ok": False,
        }


def judge_probe(judge_cfg: dict, probe: dict, orig_item: dict) -> dict:
    prompt = JUDGE_PROMPT.format(
        orig_src=orig_item["source_text"],
        probe_src=probe["probe_source"],
        family=probe["family"],
        expected_relation=probe.get("expected_relation", ""),
        cue_token=probe.get("cue_token", ""),
    )
    raw = call_llm(judge_cfg["id"], prompt, api_key_env=judge_cfg["api_key_env"])
    parsed = _parse_judge_response(raw)
    return {
        "model": judge_cfg["id"],
        "family": judge_cfg["family"],
        "equivalent": parsed["equivalent"],
        "confidence": parsed["confidence"],
        "failures": parsed["failures"],
        "raw": raw,
    }


def validate_probe(probe: dict, orig_item: dict) -> dict:
    """Run both judges and assemble a final verdict + agreement record."""
    decisions = [judge_probe(j, probe, orig_item) for j in JUDGE_MODELS]
    j1, j2 = decisions
    if j1["equivalent"] is None or j2["equivalent"] is None:
        final = "error"
    elif j1["equivalent"] and j2["equivalent"]:
        final = "agree_equivalent"
    elif (not j1["equivalent"]) and (not j2["equivalent"]):
        final = "agree_not_equivalent"
    else:
        final = "disagree"

    accepted = final == "agree_equivalent"
    return {
        "dataset": "BEA2019_WI_LOCNESS",
        "probe_id": probe["probe_id"],
        "item_id": probe["base_item_id"],
        "transform_type": probe["family"],
        "judge_1_model": j1["model"], "judge_1_family": j1["family"],
        "judge_1_equivalent": j1["equivalent"],
        "judge_1_confidence": j1["confidence"],
        "judge_1_failures": j1["failures"],
        "judge_2_model": j2["model"], "judge_2_family": j2["family"],
        "judge_2_equivalent": j2["equivalent"],
        "judge_2_confidence": j2["confidence"],
        "judge_2_failures": j2["failures"],
        "final_verdict": final,
        "accepted": accepted,
    }


def run(probes_paths: list, base_items_path: Path) -> Path:
    assert_api_keys_present([j["host"] for j in JUDGE_MODELS])

    base = {}
    with base_items_path.open(encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            r = json.loads(ln)
            base[r["sentence_id"]] = r

    out_path = JUDGE_DIR / "gec_judge_decisions.jsonl"
    tmp_path = JUDGE_DIR / "gec_judge_decisions.jsonl.partial"

    # Resume support: if a previous run crashed (e.g. rate limit), the
    # partial file holds already-judged probe_ids. Load them so we don't
    # re-spend API quota re-judging probes that already succeeded.
    done_ids = set()
    if tmp_path.exists():
        kept_lines = []
        with tmp_path.open(encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except json.JSONDecodeError:
                    continue  # drop truncated trailing line from a crash
                done_ids.add(rec["probe_id"])
                kept_lines.append(ln)
        if done_ids:
            print(f"[judge] resuming: {len(done_ids)} probes already judged "
                  f"in a previous (interrupted) run")
        # Rewrite tmp_path with only the verified-parseable lines, so a
        # crash mid-write of the next record can't corrupt prior progress.
        with tmp_path.open("w", encoding="utf-8") as fh:
            for ln in kept_lines:
                fh.write(ln + "\n")

    n_accept = 0
    n_total = 0

    with tmp_path.open("a", encoding="utf-8") as out_fh:
        for pp in probes_paths:
            with Path(pp).open(encoding="utf-8", errors="replace") as fh:
                for ln in fh:
                    probe = json.loads(ln)
                    if not probe.get("valid", False):
                        continue
                    if probe["probe_id"] in done_ids:
                        continue
                    orig = base.get(probe["base_item_id"])
                    if orig is None:
                        continue
                    rec = validate_probe(probe, orig)
                    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_fh.flush()
                    done_ids.add(probe["probe_id"])

    # Recount totals from the now-complete partial file.
    with tmp_path.open(encoding="utf-8") as fh:
        for ln in fh:
            rec = json.loads(ln)
            n_total += 1
            n_accept += int(rec["accepted"])

    # Only now is the run actually complete -- atomically finalize so
    # `stage_judge`'s "[skip] judge (exists)" check can never mistake a
    # crashed/partial run for a finished one.
    tmp_path.replace(out_path)
    print(f"[judge] accepted={n_accept}/{n_total} -> {out_path}")
    return out_path