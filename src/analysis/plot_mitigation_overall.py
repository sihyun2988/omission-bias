"""RQ4 mitigation — WHOLE-benchmark view (companion to plot_mitigation.py).

plot_mitigation.py conditions on M0=NN (credited-repair of the biased
subset). THIS figure instead shows, per model, the FULL tuple composition
over ALL labeled scenarios for every condition. It answers "what does the
mitigation do to the whole benchmark distribution", and—because non-NN
scenarios are included—exposes the harmful non-NN→NN direction (a bar
whose grey NN segment GROWS vs M0 is net-harmful), which the M0=NN-only
figure cannot show. Reuses metrics.load_tuples for de-dup consistency.

    python -m src.analysis.plot_mitigation_overall \\
        --tuples outputs/experiments/E7_mitigation/raw/tuples.jsonl
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.mitigation.metrics import load_tuples

COND_ORDER = ["M0", "M1", "M2", "M3", "M3b", "M4"]
# NN = omission bias (red, highlight). YY = action bias (pink).
# FC = YN+NY = frame-consistent (good, blue) — merged per user request.
COLORS = {"NN": "#c62828", "YY": "#ef9a9a", "FC": "#1565c0"}
TUPS = ["NN", "YY", "FC"]
_RAW2CAT = {"NN": "NN", "YY": "YY", "YN": "FC", "NY": "FC"}


def _conds(cs):
    return [c for c in COND_ORDER if c in cs] + sorted(
        c for c in cs if c not in COND_ORDER)


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
        labels, fr = [], {t: [] for t in TUPS}
        caps = []
        m0_nn = None
        for c in conds:
            d = by.get((model, c), {})
            cnt = Counter(_RAW2CAT[v["tuple"]] for v in d.values()
                          if v["tuple"] in _RAW2CAT)
            tot = sum(cnt.values())
            labels.append(c)
            for t in TUPS:
                fr[t].append(cnt[t] / tot if tot else 0)
            nn_rate = cnt["NN"] / tot if tot else 0
            if c == "M0":
                m0_nn = nn_rate
            caps.append(f"NN {nn_rate:.0%}\nn={tot}")

        x = range(len(labels))
        bottom = [0.0] * len(labels)
        for t in TUPS:
            ax.bar(x, fr[t], bottom=bottom, color=COLORS[t],
                   width=.62, label=t)
            bottom = [b + v for b, v in zip(bottom, fr[t])]
        if m0_nn is not None:
            ax.axhline(m0_nn, ls=":", lw=1.1, color="k")
            ax.text(len(labels) - .4, m0_nn, " M0 NN", va="bottom",
                    ha="right", fontsize=7)
        for i, cap in enumerate(caps):
            ax.text(i, 1.01, cap, ha="center", va="bottom", fontsize=6.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.16)
        ax.set_title(model.split("/")[-1], fontsize=9)

    axes[0].set_ylabel("tuple share over ALL labeled scenarios")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(.5, -.03),
               title=None)
    draft = "  — DRAFT (smoke)" if n_scen < 40 else ""
    fig.suptitle(
        f"RQ4 mitigation — whole-benchmark tuple composition{draft}\n"
        f"{tp}  ({n_scen} scenarios; NN=omission bias, YY=action bias, "
        f"FC=frame-consistent (YN+NY); dotted = M0 NN baseline)",
        fontsize=8.5)
    fig.tight_layout(rect=(0, .04, 1, .92))
    out = Path(a.out) if a.out else tp.parent / "mitigation_overall_218.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}  (models={len(models)}, conds={labels}, "
          f"scenarios={n_scen})")


if __name__ == "__main__":
    main()
