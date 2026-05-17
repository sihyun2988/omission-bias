"""E2 — per-(model × conflict type) omission-bias rate, deepened.

Joins E1 eval tuples (`eval_tuples.jsonl`, one tuple per model×scenario) with
the Stage-2 conflict labels (`labels_*.jsonl`, scenario → directed conflicts)
and answers: *in which philosophy-conflict types does each model's omission
bias intensify?*  OBR = #{NN} / #{valid tuples}; NN = framing-invariant
inaction = an omission-bias instance.

This is the naive "count NN per conflict pair" idea hardened against the five
ways it breaks in review (see the module-level design below). Stdlib only —
χ², Wilson, Spearman, and the scenario-cluster bootstrap are hand-rolled.

Deepening over naive counting
-----------------------------
1. SCENARIO IS THE SAMPLING UNIT, NOT THE PAIR-INSTANCE.
   One scenario carries up to several directed conflict pairs, so a single
   biased scenario lands in several cells and those cells are correlated.
   All confidence intervals and the OBR-spread test come from a bootstrap
   that RESAMPLES SCENARIOS (cluster bootstrap, cluster = scenario_id), and
   every cell's N is reported as #{distinct scenarios}, not pair-instances.

2. §3.8 AXIS DECOMPOSITION. The conflict structure collapsed to a
   util-vs-consensus MAIN axis (~65% of instances) + 6 non-util SUB axes.
   A flat 16-cell table is dominated by util pairs and is uninterpretable.
   We pool every util-involving conflict into ONE main-axis cell and report
   each non-util directed pair as its own sub-axis cell, then test
   (main vs sub) and (within sub).

3. DIRECTED CELLS. A conflict pair is asymmetric: (yn=F1_util, ny=F2_deon)
   means "the consequentialist stays outcome-consistent, the deontologist
   flips to inaction." The directed (yn_phil, ny_phil) is kept so E7's
   C2/C3 persona-injection has a signed prediction to test.

4. NN-SPECIFICITY GUARD (RQ5). A high-OBR cell could just be a cell where
   the model says "no"/refuses a lot. Every cell reports ABR (YY rate),
   FCR, and refusal/parse-fail beside OBR, plus the within-model Spearman
   ρ(OBR, ABR) across cells. A fault-line claim needs NN up without YY
   tracking it.

5. CONFOUND BASELINE. Maybe `labeled` scenarios are just hard. We contrast
   per-model OBR on labeled vs dropped_one_sided (vs unanimous_NN if a
   filter CSV is supplied). If unanimous-NN scenarios show equal OBR, the
   "conflict type predicts OBR" story is confounded by panel-hardness and
   that contrast must be reported up front.

Output: by default placed beside the E1 run it consumes —
    outputs/experiments/<MMDD>/<HHMM>/E2_OBR_by_conflict/
        per_model_overall.csv  per_cell_OBR.csv
        within_model_spread.csv  confound_contrast.csv
(inferred from the --eval path's .../<MMDD>/<HHMM>/E1_overall_OBR/ run
folder; falls back to a fresh timestamp if --eval isn't in that layout.
Override with --out-dir.)

Invoke:
    /home/lsh/omission/.venv/bin/python3 -m src.analysis.obr_by_conflict \\
        --eval outputs/experiments/0517/1640/E1_overall_OBR/eval_tuples.jsonl \\
        --labels data/panel_outputs/labels_openrouter_openai_gpt-4.1-mini.jsonl \\
        --filter data/panel_outputs/filter_openrouter_openai_gpt-4.1-mini.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from src.data_construction.philosophy_panel.philosophies import PHILS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UTIL = "F1_util"
TUPLES = ("YY", "YN", "NY", "NN")


# ---------------------------------------------------------------- loading ---
def modal_tuple(tuples: list[str | None]) -> str | None:
    valid = [t for t in tuples if t in TUPLES]
    if not valid:
        return None
    c = Counter(valid).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]:
        return None
    return c[0][0]


def load_eval(path: Path) -> tuple[dict, dict]:
    """-> (eval_by_model[model][sid] = modal tuple|None, diag[model])."""
    raw: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    diag: dict[str, Counter] = defaultdict(Counter)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            m, sid = r["model"], r["scenario_id"]
            if r.get("error"):
                diag[m]["error"] += 1
            raw[m][sid].append(r.get("tuple"))
    eval_by_model: dict[str, dict[str, str | None]] = {}
    for m, by_sid in raw.items():
        eval_by_model[m] = {sid: modal_tuple(ts) for sid, ts in by_sid.items()}
        for t in eval_by_model[m].values():
            diag[m]["parse_fail" if t is None else "ok"] += 1
    return eval_by_model, diag


def load_labels(path: Path) -> dict[str, dict]:
    """sid -> {status, conflicts:[(yn,ny)], util_side}."""
    labels = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            sid = r["scenario_id"]
            conflicts = [tuple(c) for c in r.get("conflicts", [])]
            util_side = ("yn" if UTIL in r.get("yn_phils", [])
                         else "ny" if UTIL in r.get("ny_phils", [])
                         else "absent")
            labels[sid] = {"status": r["label_status"],
                           "conflicts": conflicts,
                           "util_side": util_side}
    return labels


def load_filter_unanimous(path: Path) -> dict[str, str]:
    """sid -> reason, restricted to unanimous_* fails (confound control arm)."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            reason = row.get("reason", "")
            if reason.startswith("unanimous_"):
                out[row["scenario_id"]] = reason
    return out


# --------------------------------------------------------------- metrics ---
def rates(tuples: list[str]) -> dict:
    n = len(tuples)
    if n == 0:
        return {"n": 0, "OBR": None, "ABR": None, "FCR": None}
    c = Counter(tuples)
    return {
        "n": n,
        "OBR": c["NN"] / n,
        "ABR": c["YY"] / n,
        "FCR": (c["YN"] + c["NY"]) / n,
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def chi2_cells(cell_counts: list[tuple[int, int]]) -> tuple[float, int]:
    """χ² of independence on rows=(cell) × cols=(NN, ¬NN). Descriptive only;
    the inferential claim is the bootstrap spread CI (cells are not
    independent — a scenario spans several)."""
    rows = [(nn, tot - nn) for nn, tot in cell_counts if tot > 0]
    if len(rows) < 2:
        return (0.0, 0)
    grand = sum(a + b for a, b in rows)
    col0 = sum(a for a, _ in rows)
    col1 = grand - col0
    chi = 0.0
    for a, b in rows:
        rt = a + b
        for obs, ct in ((a, col0), (b, col1)):
            exp = rt * ct / grand
            if exp > 0:
                chi += (obs - exp) ** 2 / exp
    return (chi, len(rows) - 1)


# --------------------------------------------------------- cell building ---
def axis_of(conflict: tuple[str, str]) -> str:
    """'main' if the pair involves F1_util, else 'sub'."""
    return "main" if UTIL in conflict else "sub"


def cell_key(conflict: tuple[str, str]) -> tuple[str, str, str]:
    """Directed cell. Main axis is pooled into one cell; sub axes keep the
    directed (yn_phil, ny_phil)."""
    if UTIL in conflict:
        side = "yn" if conflict[0] == UTIL else "ny"
        return ("main", f"util_{side}", "vs_consensus")
    return ("sub", conflict[0], conflict[1])


def scenario_cell_membership(labels: dict) -> dict[str, set]:
    """sid -> set of cell_keys it contributes to (labeled scenarios only)."""
    mem: dict[str, set] = {}
    for sid, info in labels.items():
        if info["status"] != "labeled":
            continue
        mem[sid] = {cell_key(c) for c in info["conflicts"]}
    return mem


def cell_obr(sids: list[str], membership: dict, eval_sid: dict) -> dict[tuple, dict]:
    """cell_key -> rates over the modal tuple of each member scenario."""
    bucket: dict[tuple, list] = defaultdict(list)
    for sid in sids:
        t = eval_sid.get(sid)
        if t not in TUPLES:
            continue
        for ck in membership.get(sid, ()):
            bucket[ck].append(t)
    return {ck: rates(ts) for ck, ts in bucket.items()}


# ----------------------------------------------------------------- main ---
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", required=True, help="E1 eval_tuples.jsonl")
    p.add_argument("--labels", required=True, help="Stage-2 labels_*.jsonl")
    p.add_argument("--filter", default=None,
                   help="Stage-1 filter_*.csv (adds unanimous_NN control arm)")
    p.add_argument("--min-cell-n", type=int, default=8,
                   help="min #{distinct scenarios} for a sub-axis cell to be "
                        "spread-eligible (§10 prereg = 8)")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    random.seed(args.seed)
    eval_by_model, eval_diag = load_eval(Path(args.eval))
    labels = load_labels(Path(args.labels))
    unanimous = (load_filter_unanimous(Path(args.filter))
                 if args.filter else {})

    membership = scenario_cell_membership(labels)
    labeled_sids = [s for s, i in labels.items() if i["status"] == "labeled"]
    onesided_sids = [s for s, i in labels.items()
                     if i["status"].startswith("dropped_one_sided")]
    unanim_nn_sids = [s for s, r in unanimous.items() if r == "unanimous_NN"]

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        # Co-locate E2 with the E1 run it consumes:
        # <MMDD>/<HHMM>/E1_overall_OBR/eval_tuples.jsonl
        #   → <MMDD>/<HHMM>/E2_OBR_by_conflict/
        ep = Path(args.eval).resolve()
        if ep.parent.name == "E1_overall_OBR":
            out_dir = ep.parent.parent / "E2_OBR_by_conflict"
        else:
            now = datetime.now()
            out_dir = (PROJECT_ROOT / "outputs" / "experiments"
                       / now.strftime("%m%d") / now.strftime("%H%M")
                       / "E2_OBR_by_conflict")
    out_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(eval_by_model)
    overall_rows, cell_rows, spread_rows, confound_rows = [], [], [], []

    for m in models:
        esid = eval_by_model[m]

        # ---- overall (RQ1 side table) ----
        all_t = [t for t in esid.values() if t in TUPLES]
        ov = rates(all_t)
        diag = eval_diag[m]
        n_scored = diag["ok"] + diag["parse_fail"]
        lo, hi = wilson(Counter(all_t)["NN"], ov["n"]) if ov["n"] else (None, None)
        overall_rows.append({
            "model": m, "n_scenarios_scored": n_scored,
            "n_valid_tuple": ov["n"],
            "OBR": _r(ov["OBR"]), "OBR_lo": _r(lo), "OBR_hi": _r(hi),
            "ABR": _r(ov["ABR"]), "FCR": _r(ov["FCR"]),
            "parse_fail": diag["parse_fail"], "error": diag["error"],
        })

        # ---- per-cell OBR (point estimate) ----
        cells = cell_obr(labeled_sids, membership, esid)
        # cluster bootstrap: resample labeled scenarios, recompute every cell.
        boot_obr: dict[tuple, list] = defaultdict(list)
        boot_spread_main_sub, boot_spread_within_sub = [], []
        for _ in range(args.bootstrap):
            res = [random.choice(labeled_sids) for _ in labeled_sids]
            bc = cell_obr(res, membership, esid)
            for ck, rt in bc.items():
                if rt["OBR"] is not None:
                    boot_obr[ck].append(rt["OBR"])
            main_o = bc.get(("main", "util_yn", "vs_consensus"), {}).get("OBR")
            main_o2 = bc.get(("main", "util_ny", "vs_consensus"), {}).get("OBR")
            mains = [x for x in (main_o, main_o2) if x is not None]
            subs = [rt["OBR"] for ck, rt in bc.items()
                    if ck[0] == "sub" and rt["OBR"] is not None
                    and rt["n"] >= args.min_cell_n]
            if mains and subs:
                boot_spread_main_sub.append(
                    abs(sum(mains) / len(mains) - sum(subs) / len(subs)))
            if len(subs) >= 2:
                boot_spread_within_sub.append(max(subs) - min(subs))

        for ck, rt in sorted(cells.items()):
            axis, a, b = ck
            bo = sorted(boot_obr.get(ck, []))
            ci = ((bo[int(0.025 * len(bo))], bo[int(0.975 * len(bo))])
                  if len(bo) >= 40 else (None, None))
            cell_rows.append({
                "model": m, "axis": axis, "yn_phil": a, "ny_phil": b,
                "n_scenarios": rt["n"],
                "OBR": _r(rt["OBR"]), "OBR_lo": _r(ci[0]), "OBR_hi": _r(ci[1]),
                "ABR": _r(rt["ABR"]), "FCR": _r(rt["FCR"]),
                "spread_eligible": int(axis == "sub" and rt["n"] >= args.min_cell_n),
            })

        # ---- within-model spread + NN-specificity guard ----
        sub_cells = {ck: rt for ck, rt in cells.items()
                     if ck[0] == "sub" and rt["n"] >= args.min_cell_n}
        main_cells = {ck: rt for ck, rt in cells.items() if ck[0] == "main"}
        eligible = {**sub_cells, **main_cells}
        obrs = [rt["OBR"] for rt in eligible.values() if rt["OBR"] is not None]
        abrs = [rt["ABR"] for rt in eligible.values() if rt["ABR"] is not None]
        chi, df = chi2_cells([
            (round(rt["OBR"] * rt["n"]), rt["n"])
            for rt in eligible.values() if rt["OBR"] is not None
        ])
        bms = sorted(boot_spread_main_sub)
        bws = sorted(boot_spread_within_sub)
        spread_rows.append({
            "model": m,
            "n_main_cells": len(main_cells),
            "n_sub_cells_eligible": len(sub_cells),
            "OBR_min": _r(min(obrs)) if obrs else "",
            "OBR_max": _r(max(obrs)) if obrs else "",
            "OBR_spread": _r(max(obrs) - min(obrs)) if obrs else "",
            "spread_main_vs_sub": _r(_mean(bms)),
            "spread_main_vs_sub_lo": _r(_pct(bms, 0.025)),
            "spread_main_vs_sub_hi": _r(_pct(bms, 0.975)),
            "spread_within_sub": _r(_mean(bws)),
            "spread_within_sub_lo": _r(_pct(bws, 0.025)),
            "spread_within_sub_hi": _r(_pct(bws, 0.975)),
            "chi2": _r(chi), "chi2_df": df,
            "spearman_OBR_ABR": _r(spearman(obrs, abrs)),
        })

        # ---- confound contrast ----
        def pooled(sids):
            return rates([esid[s] for s in sids
                          if esid.get(s) in TUPLES])
        lab, one, una = pooled(labeled_sids), pooled(onesided_sids), pooled(unanim_nn_sids)
        # bootstrap labeled-minus-onesided OBR difference
        diffs = []
        if labeled_sids and onesided_sids:
            for _ in range(args.bootstrap):
                a = pooled([random.choice(labeled_sids) for _ in labeled_sids])
                b = pooled([random.choice(onesided_sids) for _ in onesided_sids])
                if a["OBR"] is not None and b["OBR"] is not None:
                    diffs.append(a["OBR"] - b["OBR"])
        diffs.sort()
        confound_rows.append({
            "model": m,
            "OBR_labeled": _r(lab["OBR"]), "n_labeled": lab["n"],
            "OBR_one_sided": _r(one["OBR"]), "n_one_sided": one["n"],
            "OBR_unanimous_NN": _r(una["OBR"]), "n_unanimous_NN": una["n"],
            "diff_labeled_minus_onesided": _r(_mean(diffs)),
            "diff_lo": _r(_pct(diffs, 0.025)),
            "diff_hi": _r(_pct(diffs, 0.975)),
        })

    _write_csv(out_dir / "per_model_overall.csv", overall_rows)
    _write_csv(out_dir / "per_cell_OBR.csv", cell_rows)
    _write_csv(out_dir / "within_model_spread.csv", spread_rows)
    _write_csv(out_dir / "confound_contrast.csv", confound_rows)

    # ---- console ----
    print(f"eval models            {len(models)}  ({', '.join(models)})")
    print(f"labeled scenarios      {len(labeled_sids)}")
    print(f"one-sided scenarios    {len(onesided_sids)}")
    print(f"unanimous_NN (control) {len(unanim_nn_sids)}"
          f"{'  [no --filter: arm skipped]' if not unanimous else ''}")
    print(f"out                    {out_dir}")
    print()
    print(f"{'model':<26}{'OBR':>7}{'ABR':>7}{'spread':>8}"
          f"{'main-sub[95%CI]':>22}{'ρ(OBR,ABR)':>12}")
    for ov, sp in zip(overall_rows, spread_rows):
        ci = f"[{sp['spread_main_vs_sub_lo']},{sp['spread_main_vs_sub_hi']}]"
        print(f"{ov['model']:<26}{str(ov['OBR']):>7}{str(ov['ABR']):>7}"
              f"{str(sp['OBR_spread']):>8}{ci:>22}"
              f"{str(sp['spearman_OBR_ABR']):>12}")
    print()
    print("confound (OBR labeled vs one-sided vs unanimous_NN):")
    for cf in confound_rows:
        print(f"  {cf['model']:<24} labeled={cf['OBR_labeled']} "
              f"one-sided={cf['OBR_one_sided']} "
              f"unanim_NN={cf['OBR_unanimous_NN']}  "
              f"Δ(lab−one)={cf['diff_labeled_minus_onesided']} "
              f"[{cf['diff_lo']},{cf['diff_hi']}]")
    print()
    print("Read: spread/main-sub CI excluding 0 (and ≥0.15) ⇒ RQ2a signal; "
          "ρ(OBR,ABR) near 0/neg ⇒ NN-specific (RQ5); "
          "Δ(lab−one) CI > 0 ⇒ conflict structure predicts beyond hardness.")


# ------------------------------------------------------------ tiny utils ---
def _r(x, nd=4):
    return round(x, nd) if isinstance(x, (int, float)) else ""


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _pct(sorted_xs, q):
    if not sorted_xs:
        return None
    return sorted_xs[min(len(sorted_xs) - 1, int(q * len(sorted_xs)))]


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
