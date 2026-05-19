"""RQ4 mitigation figure — diverging credited(↑) vs harmful(↓).

Per (model, condition Mk≠M0), of the scenarios where M0=NN:
  • UP, green   = credited  = NN→{YN,NY} (genuine framing repair)
  • DOWN, red   = flagged   = NN→YY      (omission swapped for ACTION
                              bias — net-harmful, NOT mitigation)
The y-axis is a *signed* share of M0=NN: a tall green bar = real repair,
a deep red bar = the method made it worse in a different direction.
M1 (generic CoT) credited is drawn as a dotted reference per panel — a
philosophy/structured method only "wins" if it clears that line without
a large red foot. McNemar one-sided p (NN↓) and n annotate each bar;
parse/refusal coverage is a separate caption, never bar length.
Reuses metrics.py so the figure and the stats CSV agree.

    python -m src.analysis.plot_mitigation \\
        --tuples outputs/experiments/E7_mitigation/raw/tuples.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.mitigation.metrics import (
    load_tuples, nn_exit, mcnemar_one_sided,
)

COND_ORDER = ["M1", "M2", "M3", "M3b", "M4"]
G, R = "#2e7d32", "#c62828"


def _stars(p):
    if p is None:
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < .05 else "ns"


def _conds(cs):
    return [c for c in COND_ORDER if c in cs] + sorted(
        c for c in cs if c not in COND_ORDER and c != "M0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuples", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tp = Path(a.tuples)
    by = load_tuples(tp)
    models = sorted({m for m, _ in by})
    conds = _conds({c for _, c in by})
    n_scen = len({s for d in by.values() for s in d})

    fig, axes = plt.subplots(
        1, len(models), figsize=(max(4.4 * len(models), 6), 5),
        squeeze=False, sharey=True)
    axes = axes[0]

    for ax, model in zip(axes, models):
        m0 = by.get((model, "M0"), {})
        cred, flag, labels, caps, m1c = [], [], [], [], None
        for c in conds:
            ex = nn_exit(m0, by.get((model, c), {}))
            ex.pop("_nn0", None)
            n = ex["n_M0_NN"] or 0
            mc = mcnemar_one_sided(m0, by.get((model, c), {}))
            cr = (ex["credited_NN_to_FC"] / n) if n else 0
            fl = (ex["flagged_NN_to_YY"] / n) if n else 0
            unresolved = (n - ex["credited_NN_to_FC"] - ex["flagged_NN_to_YY"]
                          - ex["unchanged_NN"]) if n else 0
            if c == "M1":
                m1c = cr
            cred.append(cr)
            flag.append(-fl)
            labels.append(c)
            caps.append(f"{_stars(mc['p_one_sided'])}\nn={n}"
                        + (f"\n{unresolved} lost" if unresolved else ""))

        x = range(len(labels))
        ax.bar(x, cred, color=G, width=.6, label="credited NN→{YN,NY}")
        ax.bar(x, flag, color=R, width=.6, label="flagged NN→YY (harm)")
        ax.axhline(0, color="k", lw=.8)
        if m1c is not None:
            ax.axhline(m1c, ls=":", lw=1.1, color="#555")
            ax.text(len(labels) - .4, m1c, " M1", va="bottom",
                    ha="right", fontsize=7, color="#555")
        for i, (cr, fl, cap) in enumerate(zip(cred, flag, caps)):
            ax.text(i, cr + .03, f"{cr:.2f}", ha="center", fontsize=7.5)
            if fl < 0:
                ax.text(i, fl - .03, cap, ha="center", va="top", fontsize=6.5)
            else:
                ax.text(i, -.04, cap, ha="center", va="top", fontsize=6.5)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylim(-.7, 1.05)
        ax.set_title(model.split("/")[-1], fontsize=9)

    axes[0].set_ylabel("share of M0=NN  —  repair (↑) / harm (↓)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8,
               frameon=False, bbox_to_anchor=(.5, -.03))
    draft = "  — DRAFT (smoke)" if n_scen < 40 else ""
    fig.suptitle(f"RQ4 mitigation: genuine repair vs harmful substitution"
                 f"{draft}\n{tp}  ({n_scen} scenarios; "
                 f"* p<.05 ** p<.01 *** p<.001, one-sided McNemar NN↓)",
                 fontsize=8.5)
    fig.tight_layout(rect=(0, .04, 1, .92))
    out = Path(a.out) if a.out else tp.parent / "mitigation_nn_exit.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}  (models={len(models)}, "
          f"conds={labels}, scenarios={n_scen}{', DRAFT' if draft else ''})")


if __name__ == "__main__":
    main()
