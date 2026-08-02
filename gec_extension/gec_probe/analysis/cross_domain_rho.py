"""
gec_probe.analysis.cross_domain_rho
-----------------------------------
Compute the three-domain Spearman triangle:
    SRS shortcut <-> Simplification shortcut    (known: rho=-0.86, p=0.01)
    SRS shortcut <-> GEC overcorrection         (new)
    Simp shortcut <-> GEC overcorrection        (new)

Inputs:
    - SRS RQ1 reliability table        (existing)
    - Simplification RQ1 table         (existing)
    - GEC RQ1 reliability table        (this pipeline)

Output:
    A small CSV showing per-pair Spearman rho, p-value, and the n.
"""
from __future__ import annotations
import json
import csv
from pathlib import Path

from gec_probe.config.settings import SCORES_DIR


def _load_reliability(path: Path, family_field: str) -> dict:
    """Return {model_id: family_reliability}."""
    out = {}
    with path.open() as fh:
        for ln in fh:
            r = json.loads(ln)
            mid = r.get("model") or r.get("model_id")
            v = r.get(family_field)
            if mid and v is not None:
                out[mid] = float(v)
    return out


def spearman(xs, ys):
    """Hand-rolled Spearman + permutation p-value (small n)."""
    import statistics
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan")

    def ranks(vals):
        s = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        for rank, idx in enumerate(s):
            r[idx] = rank + 1
        return r

    rx = ranks(xs); ry = ranks(ys)
    mx = sum(rx) / n; my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    rho = num / (dx * dy) if dx and dy else 0.0

    # Permutation p-value
    import random
    rng = random.Random(123)
    perms = 2000
    count = 0
    for _ in range(perms):
        ry_perm = ry[:]
        rng.shuffle(ry_perm)
        num2 = sum((rx[i] - mx) * (ry_perm[i] - my) for i in range(n))
        if abs(num2) >= abs(num):
            count += 1
    p = (count + 1) / (perms + 1)
    return rho, p


def run(srs_path: Path, simp_path: Path, gec_path: Path) -> Path:
    srs = _load_reliability(srs_path, "shortcut_reliability")
    simp = _load_reliability(simp_path, "shortcut_reliability")
    gec = _load_reliability(gec_path, "overcorrection_reliability")

    pairs = [
        ("SRS_shortcut", "Simp_shortcut", srs, simp),
        ("SRS_shortcut", "GEC_overcorr", srs, gec),
        ("Simp_shortcut", "GEC_overcorr", simp, gec),
    ]
    rows = []
    for a, b, da, db in pairs:
        common = sorted(set(da) & set(db))
        xs = [da[m] for m in common]
        ys = [db[m] for m in common]
        rho, p = spearman(xs, ys)
        rows.append({
            "pair": f"{a} <-> {b}",
            "n_models": len(common),
            "rho": rho,
            "p_value": p,
            "models": ";".join(common),
        })

    out = SCORES_DIR / "cross_domain_rho_triangle.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[rho] wrote {out}")
    return out


if __name__ == "__main__":
    # Adjust paths to wherever your existing SRS/simp tables live.
    run(
        srs_path=Path("../srs_pipeline/scores/rq1_reliability.jsonl"),
        simp_path=Path("../simp_pipeline/scores/rq1_reliability.jsonl"),
        gec_path=SCORES_DIR / "rq1_reliability_gec.jsonl",
    )
