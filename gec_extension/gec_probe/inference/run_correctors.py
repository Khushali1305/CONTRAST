"""
gec_probe.inference.run_correctors
----------------------------------
Run every corrector model against every accepted probe pair (source + probe).
Uses robust checkpointing so a crash never silently marks a run complete.

Environment variable GEC_ACTIVE_LOCAL_MODEL (optional):
    When set, only the named local model AND all API-hosted models run on
    this pass. This lets the Lightning launch script serve one local model
    at a time via vLLM while still folding API-hosted corrector calls
    (GPT-4.1, Groq Llama, GECToR-CPU) into the same pass.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Iterable, List

from gec_probe.config.settings import (
    CORRECTOR_MODELS, INFERENCE_DIR, JUDGE_DIR, PROBES_DIR,
)
from gec_probe.inference.llm_client import call_llm
from gec_probe.inference.checkpoint import load_completed, append_checkpoint


def _active_models() -> List[dict]:
    """
    If GEC_ACTIVE_LOCAL_MODEL is set, keep:
      - the named local model, AND
      - all API-hosted models (openai, anthropic, mistral, groq).
    Otherwise return everything.
    """
    active = os.environ.get("GEC_ACTIVE_LOCAL_MODEL")
    if not active:
        return CORRECTOR_MODELS
    api_hosts = {"openai", "anthropic", "mistral", "groq"}
    kept = []
    for m in CORRECTOR_MODELS:
        if m["host"] in api_hosts:
            kept.append(m)
        elif m["id"] == active:
            kept.append(m)
    return kept


CORRECTOR_PROMPT = """Correct ONLY the grammatical errors in the sentence
below. Preserve all other wording, content, register, and style. Do not
paraphrase. Do not rewrite for clarity. Output ONE corrected sentence only,
with no quotes or commentary.

Sentence: {sentence}

Corrected sentence:"""


def _iter_accepted_probes(judge_decisions_path: Path) -> Iterable[dict]:
    with judge_decisions_path.open() as fh:
        for ln in fh:
            r = json.loads(ln)
            if r.get("accepted"):
                yield r


def _load_probe_text(probes_paths, probe_id: str) -> dict | None:
    for pp in probes_paths:
        with Path(pp).open() as fh:
            for ln in fh:
                p = json.loads(ln)
                if p["probe_id"] == probe_id:
                    return p
    return None


def _load_base_items(path: Path) -> dict:
    out = {}
    with path.open() as fh:
        for ln in fh:
            r = json.loads(ln)
            out[r["sentence_id"]] = r
    return out


def correct_one(model_id: str, sentence: str) -> str:
    return call_llm(model_id, CORRECTOR_PROMPT.format(sentence=sentence)
                    ).strip().split("\n")[0]


def run(judge_decisions_path: Path, probes_paths: list,
        base_items_path: Path) -> Path:
    base = _load_base_items(base_items_path)

    # Pre-load all probes once into memory for speed
    probes: dict = {}
    for pp in probes_paths:
        with Path(pp).open() as fh:
            for ln in fh:
                p = json.loads(ln)
                probes[p["probe_id"]] = p

    checkpoint_path = INFERENCE_DIR / "checkpoint.jsonl"
    completed = load_completed(checkpoint_path, INFERENCE_DIR)
    print(f"[infer] resuming with {len(completed)} completed outputs")

    accepted = list(_iter_accepted_probes(judge_decisions_path))
    models_to_run = _active_models()
    total_targets = len(accepted) * 2 * len(models_to_run)  # src + probe per model
    done = 0
    print(f"[infer] running {len(models_to_run)} models this pass: "
          f"{[m['id'] for m in models_to_run]}")

    for record in accepted:
        probe = probes.get(record["probe_id"])
        if probe is None:
            continue
        base_item = base.get(probe["base_item_id"])
        if base_item is None:
            continue

        for m in models_to_run:
            # SOURCE
            key_src = (m["id"], f"{probe['base_item_id']}__src")
            if key_src not in completed:
                try:
                    out = correct_one(m["id"], base_item["source_text"])
                    payload = {
                        "model": m["id"],
                        "probe_id": key_src[1],
                        "base_item_id": probe["base_item_id"],
                        "input_role": "source",
                        "input_text": base_item["source_text"],
                        "output": out,
                    }
                    append_checkpoint(checkpoint_path, m["id"],
                                      key_src[1], payload, INFERENCE_DIR)
                    completed.add(key_src)
                except Exception as e:
                    print(f"[infer] FAILED src {m['id']}/{probe['probe_id']}: {e}")

            # PROBE
            key_probe = (m["id"], probe["probe_id"])
            if key_probe not in completed:
                try:
                    out = correct_one(m["id"], probe["probe_source"])
                    payload = {
                        "model": m["id"],
                        "probe_id": probe["probe_id"],
                        "base_item_id": probe["base_item_id"],
                        "input_role": "probe",
                        "input_text": probe["probe_source"],
                        "output": out,
                    }
                    append_checkpoint(checkpoint_path, m["id"],
                                      probe["probe_id"], payload, INFERENCE_DIR)
                    completed.add(key_probe)
                except Exception as e:
                    print(f"[infer] FAILED probe {m['id']}/{probe['probe_id']}: {e}")

            done += 2
            if done % 200 == 0:
                print(f"[infer] {done}/{total_targets}")
    print(f"[infer] complete: {done}/{total_targets}")
    return INFERENCE_DIR


if __name__ == "__main__":
    run(
        judge_decisions_path=JUDGE_DIR / "gec_judge_decisions.jsonl",
        probes_paths=[
            PROBES_DIR / "gec_probes_invariance.jsonl",
            PROBES_DIR / "gec_probes_directional.jsonl",
            PROBES_DIR / "gec_probes_overcorrection.jsonl",
        ],
        base_items_path=PROBES_DIR / "gec_probecore_base.jsonl",
    )
