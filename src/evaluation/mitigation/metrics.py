"""E7 / RQ4 mitigation metrics — NN-exit decomposition, McNemar, bootstrap.

Consumes the runner's `raw/tuples.jsonl` (one record per
(scenario_id, model, condition); on resume the LAST record for a key wins) and
emits the supplement-§132 table set. Stdlib only — exact McNemar (binomial
enumeration), Bonferroni / FDR-BH, and the scenario bootstrap are hand-rolled
(scipy is NOT in the venv; CLAUDE.md).

Pre-registered analysis rules (supplement §104-126)
---------------------------------------------------
NN-exit decomposition, only over (scenario,model) with **M0 tuple == NN**:
    Mk ∈ {YN, NY}  → frame-consistent  = CREDITED (real asymmetry correction)
    Mk == YY       → global yes-shift   = FLAGGED  (action-bias substitution,
                                          NOT counted as mitigation; RQ5 link)
    Mk == NN       → unchanged
Primary metric = credited_rate = Pr(M0=NN → Mk ∈ {YN,NY}).
Per-frame marginal yes-rate P(yes|A), P(yes|B) reported per condition to expose
a global-shift confound. naive ΔNN and credited ΔNN both reported.

McNemar: per (model, Mk≠M0) exact one-sided that NN *decreases* — discordant
b = #(M0=NN, Mk≠NN), c = #(M0≠NN, Mk=NN); under H0 b ~ Binom(b+c, ½), one-sided
p = P(X ≥ b). primary set {M2,M3,M3b,M4}×model → Bonferroni; M1 → FDR-BH.
A condition with credited < flagged(NN→YY) is net-harmful — flagged, never
counted as mitigation (§126 falsifiable commit).

Invoke:
    .venv/bin/python3 -m src.evaluation.mitigation.metrics \\
        --tuples outputs/experiments/E7_mitigation/raw/tuples.jsonl \\
        --labels data/panel_outputs/labels_openrouter_openai_gpt-4.1-mini.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from src.evaluation.mitigation.conditions import (
    PRIMARY_CONDITIONS, SECONDARY_CONDITIONS,
)
from src.analysis.fingerprint import load_labeled, leaning_vector, cosine
from src.data_construction.philosophy_panel.philosophies import PHILS

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TUPLES = ("YY", "YN", "NY", "NN")
FRAME_CONSISTENT = ("YN", "NY")


# ----------------------------------------------------------------- load ---
def load_tuples(path: Path):
    """-> by[(model,cond)][sid] = record (last write wins for resume)."""
    by: dict[tuple, dict] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by[(r["model"], r["condition"])][r["scenario_id"]] = r
    return by


def _yes_rate(records: dict, frame_key: str):
    vals = [r[frame_key] for r in records.values()
            if r.get(frame_key) in ("yes", "no")]
    return round(sum(v == "yes" for v in vals) / len(vals), 4) if vals else None


# -------------------------------------------------------------- binomial ---
def _binom_sf_ge(b: int, n: int) -> float:
    """P(X >= b), X ~ Binomial(n, 0.5). Exact. n is tiny (discordant pairs)."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(b, n + 1)) / (2 ** n)


def mcnemar_one_sided(m0: dict, mk: dict) -> dict:
    """One-sided exact McNemar that NN decreases M0→Mk."""
    sids = [s for s in m0 if s in mk
            and m0[s]["tuple"] in TUPLES and mk[s]["tuple"] in TUPLES]
    b = sum(1 for s in sids
            if m0[s]["tuple"] == "NN" and mk[s]["tuple"] != "NN")
    c = sum(1 for s in sids
            if m0[s]["tuple"] != "NN" and mk[s]["tuple"] == "NN")
    return {"n_paired": len(sids), "b_NN_to_other": b,
            "c_other_to_NN": c, "p_one_sided": _binom_sf_ge(b, b + c)}


# ----------------------------------------------------- NN-exit decomp ---
def nn_exit(m0: dict, mk: dict) -> dict:
    sids = [s for s in m0 if s in mk and m0[s]["tuple"] in TUPLES]
    nn0 = [s for s in sids if m0[s]["tuple"] == "NN"]
    cred = flag = unch = lost = 0
    for s in nn0:
        t = mk[s]["tuple"]
        if t in FRAME_CONSISTENT:
            cred += 1
        elif t == "YY":
            flag += 1
        elif t == "NN":
            unch += 1
    # net-harmful guard (§185): non-NN at M0 collapsing to NN under Mk
    non_nn0 = [s for s in sids if m0[s]["tuple"] != "NN"]
    for s in non_nn0:
        if mk[s]["tuple"] == "NN":
            lost += 1
    n = len(nn0)
    naive_nn_mk = sum(1 for s in sids if mk[s]["tuple"] == "NN")
    naive_nn_m0 = len(nn0)
    return {
        "n_M0_NN": n,
        "credited_NN_to_FC": cred,
        "flagged_NN_to_YY": flag,
        "unchanged_NN": unch,
        "control_nonNN_to_NN": lost,
        "credited_rate": round(cred / n, 4) if n else None,
        "flagged_rate": round(flag / n, 4) if n else None,
        "naive_dNN": round((naive_nn_m0 - naive_nn_mk) / len(sids), 4)
        if sids else None,
        "credited_dNN": round(cred / len(sids), 4) if sids else None,
        "_nn0": nn0,  # for bootstrap
    }


def bootstrap_credited(m0: dict, mk: dict, nn0: list, iters: int, seed: int):
    if not nn0:
        return (None, None)
    rng = random.Random(seed)
    vals = []
    for _ in range(iters):
        res = [rng.choice(nn0) for _ in nn0]
        c = sum(1 for s in res if mk[s]["tuple"] in FRAME_CONSISTENT)
        vals.append(c / len(nn0))
    vals.sort()
    return (round(vals[int(0.025 * len(vals))], 4),
            round(vals[min(len(vals) - 1, int(0.975 * len(vals)))], 4))


# ------------------------------------------------ multiple-test adjust ---
def bonferroni(pvals: list[float]) -> list[float]:
    n = len(pvals)
    return [min(1.0, p * n) for p in pvals]


def fdr_bh(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = n - rank + 1
        prev = min(prev, pvals[i] * n / k)
        adj[i] = round(prev, 6)
    return adj


# ----------------------------------------------------------------- main ---
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tuples", default=str(
        PROJECT_ROOT / "outputs" / "experiments" / "E7_mitigation"
        / "raw" / "tuples.jsonl"))
    p.add_argument("--labels", default=str(
        PROJECT_ROOT / "data" / "panel_outputs"
        / "labels_openrouter_openai_gpt-4.1-mini.jsonl"))
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=str(
        PROJECT_ROOT / "outputs" / "experiments" / "E7_mitigation"
        / "tables"))
    args = p.parse_args()

    by = load_tuples(Path(args.tuples))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = sorted({m for (m, _) in by})
    conds = sorted({c for (_, c) in by})
    if "M0" not in conds:
        raise SystemExit("no M0 records — McNemar pairing needs the baseline")

    # ---- per_scenario_tuples.csv ----
    rows = []
    all_sids = sorted({s for d in by.values() for s in d})
    for model in models:
        for sid in all_sids:
            row = {"model": model, "scenario_id": sid}
            present = False
            for c in conds:
                t = by.get((model, c), {}).get(sid, {}).get("tuple")
                row[c] = t or ""
                present = present or t is not None
            if present:
                rows.append(row)
    _write_csv(out_dir / "per_scenario_tuples.csv", rows)

    # ---- NN-exit + McNemar ----
    exit_rows, mc_primary, mc_secondary = [], [], []
    raw_p_primary, raw_p_secondary = [], []
    for model in models:
        m0 = by.get((model, "M0"), {})
        for c in conds:
            if c == "M0":
                continue
            mk = by.get((model, c), {})
            ex = nn_exit(m0, mk)
            mc = mcnemar_one_sided(m0, mk)
            lo, hi = bootstrap_credited(m0, mk, ex.pop("_nn0"),
                                        args.bootstrap, args.seed)
            net_harm = (ex["flagged_NN_to_YY"] or 0) > (
                ex["credited_NN_to_FC"] or 0)
            exit_rows.append({
                "model": model, "condition": c, **ex,
                "credited_dNN_lo": lo, "credited_dNN_hi": hi,
                "P_yes_A": _yes_rate(mk, "answer_A"),
                "P_yes_B": _yes_rate(mk, "answer_B"),
                "net_harmful": int(net_harm),
            })
            entry = {"model": model, "condition": c, **mc,
                     "credited_rate": ex["credited_rate"]}
            if c in PRIMARY_CONDITIONS:
                mc_primary.append(entry)
                raw_p_primary.append(mc["p_one_sided"])
            elif c in SECONDARY_CONDITIONS:
                mc_secondary.append(entry)
                raw_p_secondary.append(mc["p_one_sided"])

    for e, padj in zip(mc_primary, bonferroni(raw_p_primary)):
        e["p_adj"] = round(padj, 6)
        e["adjust"] = "bonferroni"
    for e, padj in zip(mc_secondary, fdr_bh(raw_p_secondary)):
        e["p_adj"] = padj
        e["adjust"] = "fdr_bh"

    _write_csv(out_dir / "nn_exit_decomposition.csv", exit_rows)
    _write_csv(out_dir / "mcnemar_primary.csv", mc_primary + mc_secondary)

    # ---- fingerprint shift (RQ3 profile: M0 base vs M4) ----
    labeled = load_labeled(Path(args.labels))
    fp_rows = []
    if "M4" in conds:
        for model in models:
            for cond in ("M0", "M4"):
                d = by.get((model, cond), {})
                esid = {s: r["tuple"] for s, r in d.items()}
                vec = leaning_vector(list(esid), esid, labeled)
                tot = sum(vec) or 1.0
                fp_rows.append({
                    "model": model, "condition": cond,
                    **{f"norm_{ph}": round(vec[i] / tot, 4)
                       for i, ph in enumerate(PHILS)},
                })
        # cosine(M0,M4) per model
        cos_rows = []
        for model in models:
            d0 = by.get((model, "M0"), {})
            d4 = by.get((model, "M4"), {})
            v0 = leaning_vector(list(d0),
                                {s: r["tuple"] for s, r in d0.items()},
                                labeled)
            v4 = leaning_vector(list(d4),
                                {s: r["tuple"] for s, r in d4.items()},
                                labeled)
            cos_rows.append({"model": model,
                             "cosine_M0_M4": _r(cosine(v0, v4))})
        _write_csv(out_dir / "fingerprint_shift.csv", fp_rows)
        _write_csv(out_dir / "fingerprint_shift_cosine.csv", cos_rows)

    # ---- console summary ----
    print(f"tuples      {Path(args.tuples)}")
    print(f"models      {len(models)}  conditions {conds}")
    print(f"out         {out_dir}")
    print()
    print(f"{'model':<30}{'cond':>5}{'M0_NN':>7}{'cred':>6}{'YY':>5}"
          f"{'cred_rate':>10}{'mcN p_adj':>11}{'flag':>6}")
    for e in mc_primary + mc_secondary:
        ex = next(x for x in exit_rows
                  if x["model"] == e["model"]
                  and x["condition"] == e["condition"])
        flag = "HARM" if ex["net_harmful"] else ""
        print(f"{e['model'].split('/')[-1]:<30}{e['condition']:>5}"
              f"{ex['n_M0_NN']:>7}{ex['credited_NN_to_FC']:>6}"
              f"{ex['flagged_NN_to_YY']:>5}"
              f"{str(ex['credited_rate']):>10}{str(e['p_adj']):>11}"
              f"{flag:>6}")
    print()
    print("Read (supplement §194 Verification): a PRIMARY condition with "
          "Bonferroni p_adj<0.05 AND credited_rate > the same model's M1 "
          "credited_rate satisfies H4 on that model; HARM = flagged(NN→YY) "
          "≥ credited, NOT counted as mitigation.")


def _r(x, nd=4):
    return round(x, nd) if isinstance(x, (int, float)) else ""


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
