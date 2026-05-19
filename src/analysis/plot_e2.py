"""Render E2 figures from an obr_by_conflict.py output directory.

Decoupled from the analyzer ON PURPOSE: the CSVs stay the canonical result;
this module only *renders* them, so figures can be regenerated (recolour,
relabel, resize for camera-ready) without re-running any eval model.

Figure ownership is split so the two run folders never duplicate a figure:
  * --e2-dir → obr_heatmap.png ONLY (RQ2a / §E2 — the paper's visual key):
    rows = eval models, cols = conflict cells (main: util_yn / util_ny vs
    consensus; sub: directed non-util pairs), colour = OBR, annotation =
    "OBR\\n(n)", columns sorted by mean OBR descending.
  * --e1-dir → rq1_overall_obr.png ONLY (RQ1 / §E1 — per-model overall OBR
    bar with Wilson 95% CI). eval_model.py auto-calls this on its own dir.

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


PHIL = {
    "F1_util": "Utilitarian",
    "F2_deon": "Deontological",
    "F3_virtue": "Virtue",
    "F4_care": "Care",
    "F5_contract": "Contractualist",
}


def _nice(p: str) -> str:
    return PHIL.get(p, p)


def col_label(pair: tuple[str, str]) -> str:
    """Undirected conflict label, e.g. 'Care\\nvs.\\nVirtue'."""
    a, b = pair
    return f"{_nice(a)}\nvs.\n{_nice(b)}"


def load_cells_undirected(e2_dir: Path):
    """Pool the directed per_cell rows into UNDIRECTED conflict columns.

    per_cell_OBR.csv keeps direction (yn_phil→ny_phil) for E7; the heatmap
    answers RQ2a ("which conflict TYPE intensifies the bias"), for which the
    unordered pair is the unit. We sum NN counts and scenario n across the
    two directed rows (X→Y and Y→X) per model, so n roughly doubles and the
    apparent "X vs Y / Y vs X duplicate columns" collapse into one.
    """
    rows = list(csv.DictReader(open(e2_dir / "per_cell_OBR.csv")))
    models, cols = [], []
    nn, nsc = defaultdict(int), defaultdict(int)
    for r in rows:
        m = r["model"]
        o = _f(r["OBR"])
        n = int(r["n_scenarios"]) if r["n_scenarios"] else 0
        if o is None or n == 0:
            continue
        pair = tuple(sorted((r["yn_phil"], r["ny_phil"])))
        if m not in models:
            models.append(m)
        if pair not in cols:
            cols.append(pair)
        nn[(m, pair)] += round(o * n)   # reconstruct NN successes
        nsc[(m, pair)] += n
    obr = {k: nn[k] / nsc[k] for k in nsc}
    return models, cols, obr, dict(nsc)


def heatmap(e2_dir: Path, min_cell_n: int, top_n: int = 8):
    models, cols, obr, nsc = load_cells_undirected(e2_dir)
    if not models or not cols:
        print("per_cell_OBR.csv empty — skipping heatmap")
        return

    # Column = undirected conflict type. Rank by mean OBR across models,
    # but ONLY among types with enough power (max per-model n ≥ min_cell_n)
    # so an n=6 fluke at OBR=1.0 cannot win a top slot. Keep the top N.
    def col_n(c):
        return max((nsc.get((m, c), 0) for m in models), default=0)

    def col_mean(c):
        vals = [obr[(m, c)] for m in models if (m, c) in obr]
        return sum(vals) / len(vals) if vals else -1.0

    eligible = [c for c in cols if col_n(c) >= min_cell_n]
    ranked = sorted(eligible, key=col_mean, reverse=True)
    dropped = len(cols) - len(eligible)
    cols = ranked[:top_n]
    if not cols:
        print("no conflict type meets min-cell-n — skipping heatmap")
        return

    # Rows: strongest-bias model on top. Prefer the true overall OBR from
    # per_model_overall.csv; fall back to mean over displayed cells.
    overall = {}
    pmo = e2_dir / "per_model_overall.csv"
    if pmo.exists():
        for r in csv.DictReader(open(pmo)):
            overall[r["model"]] = _f(r.get("OBR"))

    def model_rank(m):
        if overall.get(m) is not None:
            return overall[m]
        vs = [obr[(m, c)] for c in cols if (m, c) in obr]
        return sum(vs) / len(vs) if vs else -1.0

    models = sorted(models, key=model_rank, reverse=True)

    fig, ax = plt.subplots(
        figsize=(max(6, 1.25 * len(cols) + 2), max(3, 0.6 * len(models) + 1.8)))
    grid = [[obr.get((m, c)) for c in cols] for m in models]
    plot = [[v if v is not None else float("nan") for v in row] for row in grid]
    im = ax.imshow(plot, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([col_label(c) for c in cols], fontsize=7.5, rotation=0)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.split("/")[-1] for m in models], fontsize=8)

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

    ax.set_xlabel("Philosophy-conflict type "
                  f"(top {len(cols)} of {len(eligible)} by mean OBR; "
                  f"{dropped} types with n<{min_cell_n} excluded)",
                  fontsize=8, labelpad=8)
    ax.set_ylabel("Evaluation model", fontsize=9)
    ax.set_title("Omission-bias rate by philosophy-conflict type",
                 fontsize=10, pad=8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                 label="Omission-bias rate (NN)")
    fig.tight_layout()
    out = e2_dir / "obr_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def _wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - h) / d, (c + h) / d)


def rq1_bar(csv_path: Path, out_path: Path | None = None):
    """Per-model overall OBR + ABR grouped bars, both with Wilson 95% CI.

    Reads E1's per_model_summary.csv OR E2's per_model_overall.csv — both
    carry model / OBR / OBR_lo / OBR_hi / ABR / YY / n_valid_tuple. Decoupled
    so the figure regenerates from CSV without re-running any eval model.
    ABR (framing-invariant ACTION, #YY/n) is plotted beside OBR so the
    descriptive RQ1 panel shows both poles of the bias, not just inaction;
    its CI is reconstructed from the YY count + n_valid_tuple.
    """
    rows = list(csv.DictReader(open(csv_path)))
    rows = [r for r in rows if _f(r["OBR"]) is not None]
    if not rows:
        print(f"{csv_path.name} empty — skipping RQ1 bar")
        return
    if out_path is None:
        out_path = csv_path.parent / "rq1_overall_obr.png"
    rows.sort(key=lambda r: _f(r["OBR"]), reverse=True)
    labels = [f"{r['model'].split('/')[-1]}\n(n={r['n_valid_tuple']})"
              for r in rows]

    def _int(x):
        try:
            return int(float(x))
        except (TypeError, ValueError):
            return 0

    obr = [_f(r["OBR"]) or 0.0 for r in rows]
    o_lo = [max(0.0, o - (_f(r["OBR_lo"]) or o)) for o, r in zip(obr, rows)]
    o_hi = [max(0.0, (_f(r["OBR_hi"]) or o) - o) for o, r in zip(obr, rows)]
    abr, a_lo, a_hi = [], [], []
    for r in rows:
        a = _f(r["ABR"]) or 0.0
        n = _int(r.get("n_valid_tuple"))
        lo, hi = _wilson(_int(r.get("YY")), n)
        abr.append(a)
        a_lo.append(max(0.0, a - lo))
        a_hi.append(max(0.0, hi - a))

    fig, ax = plt.subplots(figsize=(max(4.5, 1.2 * len(rows) + 1.8), 3.6))
    x = list(range(len(rows)))
    w = 0.38
    bo = ax.bar([i - w / 2 for i in x], obr, w, yerr=[o_lo, o_hi], capsize=3,
                color="#4477aa", edgecolor="black", linewidth=.5,
                label="OBR — framing-invariant inaction (#NN/n)")
    ba = ax.bar([i + w / 2 for i in x], abr, w, yerr=[a_lo, a_hi], capsize=3,
                color="#cc6677", edgecolor="black", linewidth=.5,
                label="ABR — framing-invariant action (#YY/n)")
    for i, r in enumerate(rows):
        ax.text(i - w / 2, obr[i] + o_hi[i] + .02, f"{obr[i]:.2f}",
                ha="center", va="bottom", fontsize=6.5)
        ax.text(i + w / 2, abr[i] + a_hi[i] + .02, f"{abr[i]:.2f}",
                ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="center", fontsize=7.5)
    ax.set_ylabel("Rate")
    top = max([o + h for o, h in zip(obr, o_hi)]
              + [a + h for a, h in zip(abr, a_hi)])
    ax.set_ylim(0, min(1.0, top + .14))
    ax.set_title("E1/RQ1 — per-model overall omission vs action bias "
                 "(Wilson 95% CI)", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper right", framealpha=.9)
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
                   help="conflict types whose max per-model n is below this "
                        "are excluded from the heatmap; surviving cells with "
                        "n below it are hatched (match obr_by_conflict "
                        "--min-cell-n; prereg = 8)")
    p.add_argument("--top-n", type=int, default=8,
                   help="show only the top-N conflict types ranked by "
                        "mean OBR across models (default 8)")
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
        heatmap(e2, args.min_cell_n, args.top_n)
        # RQ1 bar is the E1 deliverable (per_model_summary.csv); it is NOT
        # re-emitted here. E2's per_model_overall.csv stays as a numeric
        # cross-check only — same tuples, so the figure would duplicate E1's.


if __name__ == "__main__":
    main()
