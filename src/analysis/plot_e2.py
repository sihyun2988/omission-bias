"""Render E2 figures from an obr_by_conflict.py output directory.

Decoupled from the analyzer ON PURPOSE: the CSVs stay the canonical result;
this module only *renders* them, so figures can be regenerated (recolour,
relabel, resize for camera-ready) without re-running any eval model.

Produces, into the same E2 dir:
  * obr_heatmap.png      — rows = eval models, cols = conflict cells
    (main: util_yn / util_ny vs consensus; sub: directed non-util pairs),
    colour = OBR, annotation = "OBR\\n(n)". Columns sorted by mean OBR
    descending (RESEARCH_PLAN_v4 §E2 / F3 — the paper's visual key).
  * rq1_overall_obr.png  — per-model overall OBR bar with Wilson 95% CI
    (RESEARCH_PLAN_v4 §E1).

Cells whose distinct-scenario n is below --min-cell-n are hatched and their
annotation parenthesised, so a low-power cell is never read as a flat result.

Invoke:
    /home/lsh/omission/.venv/bin/python3 -m src.analysis.plot_e2 \\
        --e2-dir outputs/experiments/0517/1640/E2_OBR_by_conflict
    /home/lsh/omission/.venv/bin/python3 -m src.analysis.plot_e2 \\
        --e1-dir outputs/experiments/0517/1640/E1_overall_OBR
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def col_label(axis: str, yn: str, ny: str) -> str:
    if axis == "main":
        return yn.replace("util_", "util·") + "\nvs consensus"
    return f"{yn}\n→{ny}"


def load_cells(e2_dir: Path):
    rows = list(csv.DictReader(open(e2_dir / "per_cell_OBR.csv")))
    models, cols = [], []
    obr, nsc = {}, {}
    for r in rows:
        m = r["model"]
        c = (r["axis"], r["yn_phil"], r["ny_phil"])
        if m not in models:
            models.append(m)
        if c not in cols:
            cols.append(c)
        obr[(m, c)] = _f(r["OBR"])
        nsc[(m, c)] = int(r["n_scenarios"]) if r["n_scenarios"] else 0
    return models, cols, obr, nsc


def mean_obr_per_col(cols, models, obr):
    out = {}
    for c in cols:
        vals = [obr[(m, c)] for m in models
                if obr.get((m, c)) is not None]
        out[c] = sum(vals) / len(vals) if vals else -1.0
    return out


def heatmap(e2_dir: Path, min_cell_n: int):
    models, cols, obr, nsc = load_cells(e2_dir)
    if not models or not cols:
        print("per_cell_OBR.csv empty — skipping heatmap")
        return
    # main axis first, then sub axes sorted by mean OBR desc
    mean_obr = mean_obr_per_col(cols, models, obr)
    cols = sorted(cols, key=lambda c: (c[0] != "main", -mean_obr[c]))

    fig, ax = plt.subplots(
        figsize=(max(6, 1.15 * len(cols) + 2), max(3, 0.6 * len(models) + 1.6)))
    grid = [[obr.get((m, c)) for c in cols] for m in models]
    plot = [[v if v is not None else float("nan") for v in row] for row in grid]
    im = ax.imshow(plot, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([col_label(*c) for c in cols], fontsize=7, rotation=0)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.split("/")[-1] for m in models], fontsize=8)

    n_main = sum(1 for c in cols if c[0] == "main")
    if 0 < n_main < len(cols):  # divider between main axis and sub axes
        ax.axvline(n_main - 0.5, color="black", lw=1.6)

    for i, m in enumerate(models):
        for j, c in enumerate(cols):
            v, n = obr.get((m, c)), nsc.get((m, c), 0)
            if v is None:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1,
                                       facecolor="lightgray"))
                ax.text(j, i, "·", ha="center", va="center", fontsize=8)
                continue
            low = n < min_cell_n
            if low:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                       hatch="///", edgecolor="gray", lw=0))
            txt = f"{v:.2f}\n({n})" if low else f"{v:.2f}\nn={n}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                    color="white" if v > 0.55 else "black")

    ax.set_title("E2 — OBR by philosophy-conflict cell  "
                 "(left of bar = util-vs-consensus main axis; "
                 "hatched = n < %d)" % min_cell_n, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="OBR (NN rate)")
    fig.tight_layout()
    out = e2_dir / "obr_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def rq1_bar(csv_path: Path, out_path: Path | None = None):
    """Per-model overall OBR bar + Wilson CI.

    Reads E1's per_model_summary.csv OR E2's per_model_overall.csv — both
    carry model / OBR / OBR_lo / OBR_hi / n_valid_tuple. Decoupled so the
    figure regenerates from CSV without re-running any eval model.
    """
    rows = list(csv.DictReader(open(csv_path)))
    rows = [r for r in rows if _f(r["OBR"]) is not None]
    if not rows:
        print(f"{csv_path.name} empty — skipping RQ1 bar")
        return
    if out_path is None:
        out_path = csv_path.parent / "rq1_overall_obr.png"
    rows.sort(key=lambda r: _f(r["OBR"]), reverse=True)
    labels = [r["model"].split("/")[-1] for r in rows]
    obr = [_f(r["OBR"]) for r in rows]
    lo = [max(0.0, o - (_f(r["OBR_lo"]) or o)) for o, r in zip(obr, rows)]
    hi = [max(0.0, (_f(r["OBR_hi"]) or o) - o) for o, r in zip(obr, rows)]

    fig, ax = plt.subplots(figsize=(max(4, 1.0 * len(rows) + 1.5), 3.4))
    x = range(len(rows))
    ax.bar(x, obr, yerr=[lo, hi], capsize=4, color="#4477aa",
           edgecolor="black", linewidth=.5)
    for i, (o, r) in enumerate(zip(obr, rows)):
        ax.text(i, o + (hi[i] or 0) + .02, f"{o:.2f}\nn={r['n_valid_tuple']}",
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Overall OBR (NN rate)")
    ax.set_ylim(0, min(1.0, max(o + h for o, h in zip(obr, hi)) + .12))
    ax.set_title("E1/RQ1 — per-model overall omission-bias rate "
                 "(Wilson 95% CI)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--e2-dir", default=None,
                   help="obr_by_conflict.py output dir → heatmap + RQ1 bar")
    p.add_argument("--e1-dir", default=None,
                   help="eval_model.py output dir → RQ1 bar only "
                        "(from per_model_summary.csv, no E2 join needed)")
    p.add_argument("--min-cell-n", type=int, default=8,
                   help="cells below this distinct-scenario n are hatched "
                        "(match obr_by_conflict --min-cell-n; prereg = 8)")
    args = p.parse_args()
    if not args.e2_dir and not args.e1_dir:
        raise SystemExit("pass --e1-dir and/or --e2-dir")
    if args.e1_dir:
        e1 = Path(args.e1_dir)
        csv_path = e1 / "per_model_summary.csv"
        if not csv_path.exists():
            raise SystemExit(f"no per_model_summary.csv in {e1}")
        rq1_bar(csv_path)
    if args.e2_dir:
        e2 = Path(args.e2_dir)
        if not (e2 / "per_cell_OBR.csv").exists():
            raise SystemExit(f"no per_cell_OBR.csv in {e2}")
        heatmap(e2, args.min_cell_n)
        rq1_bar(e2 / "per_model_overall.csv")


if __name__ == "__main__":
    main()
