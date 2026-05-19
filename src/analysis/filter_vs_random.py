"""E1 / RQ1 (2026-05-17 redefinition) — filter > random.

Tests the *construction-method* claim: does the philosophy-disagreement
filter expose framing-invariant omission bias (NN) better than an equal-n
random draw from the same high-ambiguity pool? (RESEARCH_PLAN_v4 §40, §177,
§196-203.) This is the paper's spine, distinct from the descriptive
per-model overall OBR/ABR panel that `eval_model.summarize()` still emits to
`E1_overall_OBR/`.

Three arms, all scored by `eval_model.py` under the SAME misfit/malformed
hygiene so the only difference is the construction filter:
  * filtered          — label_status=="labeled" benchmark (REUSED, not
                         re-scored: point --filtered at an existing
                         E1_overall_OBR run so the numbers match E2).
  * random_complement  — equal-n random (seed 42) from the NON-labeled pool.
                         PRIMARY control.
  * random_full        — equal-n random (seed 42) from the full valid pool.
                         AUXILIARY (optional).

Across models (paired, ~5): exact one-sided Wilcoxon signed-rank
(OBR_filtered > OBR_random), implemented by hand (no scipy). With only ~5
models the exact one-sided p floor is 0.03125 and is reached ONLY when every
model moves the same way — so it is effectively a sign test. The honest
effect-size evidence is the mean OBR difference + its 1,000-iter
scenario-cluster bootstrap CI and the per-model sign agreement; the
Wilcoxon is reported as the preregistered confirmatory only.

Invoke:
    /home/lsh/omission/.venv/bin/python3 -m src.analysis.filter_vs_random \\
        --filtered      outputs/experiments/0517/1809/E1_overall_OBR \\
        --random-complement outputs/experiments/<MMDD>/<HHMM>/E1_filter_vs_random/random_complement \\
        --random-full       outputs/experiments/<MMDD>/<HHMM>/E1_filter_vs_random/random_full
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TUPLES = ("YY", "YN", "NY", "NN")


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - h) / d, (c + h) / d)


def _resolve(p: str) -> Path:
    """Accept either an eval_tuples.jsonl file or its containing dir."""
    path = Path(p)
    if path.is_dir():
        path = path / "eval_tuples.jsonl"
    if not path.exists():
        raise SystemExit(f"no eval_tuples.jsonl at {p}")
    return path


def load_arm(jsonl: Path) -> dict[str, dict[str, str | None]]:
    """model → {scenario_id → modal tuple (or None if unparsable)}.

    n_samples==1 in the deterministic policy, but stay robust to >1 by
    taking the modal tuple over sample_idx (ties → None, dropped).
    """
    raw: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    with open(jsonl) as f:
        for line in f:
            r = json.loads(line)
            raw[r["model"]][r["scenario_id"]].append(r.get("tuple"))
    out: dict[str, dict[str, str | None]] = {}
    for m, scens in raw.items():
        out[m] = {}
        for sid, ts in scens.items():
            valid = [t for t in ts if t in TUPLES]
            if not valid:
                out[m][sid] = None
                continue
            c = Counter(valid).most_common()
            out[m][sid] = (None if len(c) > 1 and c[0][1] == c[1][1]
                           else c[0][0])
    return out


def arm_stats(arm: dict[str, dict[str, str | None]]) -> dict[str, dict]:
    stats = {}
    for m, scens in arm.items():
        c = Counter(t for t in scens.values() if t in TUPLES)
        n = sum(c.values())
        n_scored = len(scens)
        obr = c["NN"] / n if n else None
        lo, hi = _wilson(c["NN"], n)
        stats[m] = {
            "n_scored": n_scored, "n_valid": n,
            "YY": c["YY"], "YN": c["YN"], "NY": c["NY"], "NN": c["NN"],
            "OBR": obr, "OBR_lo": lo, "OBR_hi": hi,
            "ABR": c["YY"] / n if n else None,
            "FCR": (c["YN"] + c["NY"]) / n if n else None,
            "parse_or_call_fail": n_scored - n,
        }
    return stats


def _obr(scens: dict[str, str | None], ids: list[str]) -> float | None:
    vals = [scens[i] for i in ids if scens.get(i) in TUPLES]
    if not vals:
        return None
    return sum(1 for t in vals if t == "NN") / len(vals)


def exact_signed_rank_one_sided(diffs: list[float]) -> dict:
    """One-sided Wilcoxon signed-rank, alternative: diffs > 0 (filtered>rand).

    Exact null by enumerating all 2^k sign assignments of the rank vector
    (k = #non-zero diffs, here ≤ #models ≈ 5). Average ranks for |d| ties.
    """
    nz = [d for d in diffs if d != 0.0]
    k = len(nz)
    if k == 0:
        return {"k": 0, "W_plus": 0.0, "p_one_sided": 1.0,
                "n_pos": 0, "n_neg": 0}
    order = sorted(range(k), key=lambda i: abs(nz[i]))
    ranks = [0.0] * k
    i = 0
    while i < k:
        j = i
        while j + 1 < k and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + 1 + j + 1) / 2.0  # average of 1-based ranks i..j
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    w_plus = sum(ranks[i] for i in range(k) if nz[i] > 0)
    ge = 0
    for mask in range(1 << k):
        s = sum(ranks[i] for i in range(k) if mask & (1 << i))
        if s >= w_plus - 1e-9:
            ge += 1
    return {
        "k": k, "W_plus": w_plus, "p_one_sided": ge / (1 << k),
        "n_pos": sum(1 for d in nz if d > 0),
        "n_neg": sum(1 for d in nz if d < 0),
    }


def bootstrap_diff(fa: dict, ra: dict, models: list[str],
                   iters: int, seed: int) -> dict:
    """Scenario-cluster bootstrap of OBR_filtered - OBR_random.

    Resample each arm's own scenario list independently with replacement
    (the arms cover different scenario sets, so a shared resample is not
    defined). Per iter: per-model diff + mean diff across models. Returns
    2.5/97.5 percentile CIs for the mean and per model.
    """
    rng = random.Random(seed)
    f_ids = {m: list(fa[m].keys()) for m in models}
    r_ids = {m: list(ra[m].keys()) for m in models}
    mean_dist: list[float] = []
    per: dict[str, list[float]] = {m: [] for m in models}
    for _ in range(iters):
        ds = []
        for m in models:
            fb = [rng.choice(f_ids[m]) for _ in f_ids[m]]
            rb = [rng.choice(r_ids[m]) for _ in r_ids[m]]
            of, orr = _obr(fa[m], fb), _obr(ra[m], rb)
            if of is None or orr is None:
                continue
            d = of - orr
            per[m].append(d)
            ds.append(d)
        if ds:
            mean_dist.append(sum(ds) / len(ds))

    def ci(xs):
        if not xs:
            return (None, None)
        s = sorted(xs)
        lo = s[max(0, int(0.025 * len(s)) - 1)]
        hi = s[min(len(s) - 1, int(0.975 * len(s)))]
        return (lo, hi)

    return {"mean_ci": ci(mean_dist),
            "per_model_ci": {m: ci(per[m]) for m in models}}


def _fmt(x, nd=4):
    return "" if x is None else round(x, nd)


def write_per_model(path: Path, arms: dict[str, dict[str, dict]]):
    cols = ["model", "arm", "n_scored", "n_valid", "YY", "YN", "NY", "NN",
            "OBR", "OBR_lo", "OBR_hi", "ABR", "FCR", "parse_or_call_fail"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for arm, st in arms.items():
            for m in sorted(st):
                s = st[m]
                w.writerow({
                    "model": m, "arm": arm,
                    "n_scored": s["n_scored"], "n_valid": s["n_valid"],
                    "YY": s["YY"], "YN": s["YN"], "NY": s["NY"],
                    "NN": s["NN"],
                    "OBR": _fmt(s["OBR"]), "OBR_lo": _fmt(s["OBR_lo"]),
                    "OBR_hi": _fmt(s["OBR_hi"]), "ABR": _fmt(s["ABR"]),
                    "FCR": _fmt(s["FCR"]),
                    "parse_or_call_fail": s["parse_or_call_fail"],
                })
    print(f"wrote {path}")


def _compare_block(label: str, fa, ra, fst, rst, models, iters, seed):
    lines = [f"=== filtered vs {label} ===", ""]
    diffs, kept = [], []
    for m in models:
        of = fst.get(m, {}).get("OBR")
        orr = rst.get(m, {}).get("OBR")
        if of is None or orr is None:
            lines.append(f"  {m:<34} SKIP (missing OBR in one arm)")
            continue
        d = of - orr
        diffs.append(d)
        kept.append(m)
        lines.append(f"  {m:<34} filtered={of:.4f}  {label}={orr:.4f}  "
                     f"diff={d:+.4f}")
    lines.append("")
    sr = exact_signed_rank_one_sided(diffs)
    bs = bootstrap_diff(fa, ra, kept, iters, seed)
    mean_d = sum(diffs) / len(diffs) if diffs else float("nan")
    mlo, mhi = bs["mean_ci"]
    lines += [
        f"  models compared (paired)      : {len(kept)}",
        f"  sign agreement (diff > 0)     : {sr['n_pos']}/{sr['k']}"
        f"  (neg {sr['n_neg']})",
        f"  Wilcoxon W+ (signed-rank)     : {sr['W_plus']:.1f}",
        f"  exact one-sided p (filt>rand) : {sr['p_one_sided']:.5f}",
        f"  mean OBR diff (filt - rand)   : {mean_d:+.4f}",
        f"  mean diff 95% bootstrap CI    : "
        f"[{_fmt(mlo)}, {_fmt(mhi)}]  ({iters} iters, scenario-cluster)",
        "",
        "  per-model diff 95% bootstrap CI:",
    ]
    for m in kept:
        lo, hi = bs["per_model_ci"][m]
        lines.append(f"    {m:<32} [{_fmt(lo)}, {_fmt(hi)}]")
    lines += [
        "",
        "  NOTE: with ~5 models the exact one-sided signed-rank p floor is",
        "  0.03125, attained only if ALL models share the sign — it is",
        "  effectively a sign test. Lead the claim with the mean-diff",
        "  bootstrap CI + per-model sign agreement; treat the Wilcoxon p",
        "  as the preregistered confirmatory only (plan §177, §370).",
        "",
    ]
    return lines, {"diffs": dict(zip(kept, diffs)), "signed_rank": sr,
                   "mean_diff": mean_d, "mean_ci": (mlo, mhi)}


def paired_bar(path: Path, fst, rst, aux_st, models):
    models = [m for m in models
              if fst.get(m, {}).get("OBR") is not None
              and rst.get(m, {}).get("OBR") is not None]
    models.sort(key=lambda m: fst[m]["OBR"], reverse=True)
    labels = [m.split("/")[-1] for m in models]
    x = list(range(len(models)))
    w = 0.38

    def err(st):
        lo = [max(0.0, st[m]["OBR"] - st[m]["OBR_lo"]) for m in models]
        hi = [max(0.0, st[m]["OBR_hi"] - st[m]["OBR"]) for m in models]
        return [lo, hi]

    fo = [fst[m]["OBR"] for m in models]
    ro = [rst[m]["OBR"] for m in models]
    fig, ax = plt.subplots(figsize=(max(4.5, 1.3 * len(models) + 1.8), 3.7))
    ax.bar([i - w / 2 for i in x], fo, w, yerr=err(fst), capsize=3,
           color="#4477aa", edgecolor="black", linewidth=.5,
           label="filtered (philosophy-disagreement benchmark)")
    ax.bar([i + w / 2 for i in x], ro, w, yerr=err(rst), capsize=3,
           color="#999933", edgecolor="black", linewidth=.5,
           label="random_complement (equal-n, non-labeled)")
    if aux_st:
        for i, m in enumerate(models):
            a = aux_st.get(m, {}).get("OBR")
            if a is not None:
                ax.plot([i - w, i + w], [a, a], color="#cc6677",
                        lw=1.4, ls="--",
                        label=("random_full (aux)" if i == 0 else None))
    for i, m in enumerate(models):
        ax.text(i - w / 2, fo[i] + err(fst)[1][i] + .02, f"{fo[i]:.2f}",
                ha="center", va="bottom", fontsize=6.5)
        ax.text(i + w / 2, ro[i] + err(rst)[1][i] + .02, f"{ro[i]:.2f}",
                ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="center", fontsize=7.5)
    ax.set_ylabel("OBR (framing-invariant inaction, #NN/n)")
    top = max([fst[m]["OBR_hi"] for m in models]
              + [rst[m]["OBR_hi"] for m in models])
    ax.set_ylim(0, min(1.0, top + .14))
    ax.set_title("E1/RQ1 — filter vs random omission-bias rate "
                 "(Wilson 95% CI)", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper right", framealpha=.9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--filtered", required=True,
                   help="E1_overall_OBR dir or its eval_tuples.jsonl "
                        "(REUSED filtered arm — not re-scored)")
    p.add_argument("--random-complement", required=True,
                   help="random_complement dir or eval_tuples.jsonl")
    p.add_argument("--random-full", default=None,
                   help="optional auxiliary random_full dir/jsonl")
    p.add_argument("--out", default=None,
                   help="output dir (default: new timestamped "
                        "outputs/experiments/MMDD/HHMM/E1_filter_vs_random)")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    fa = load_arm(_resolve(args.filtered))
    ra = load_arm(_resolve(args.random_complement))
    aa = load_arm(_resolve(args.random_full)) if args.random_full else {}
    fst, rst = arm_stats(fa), arm_stats(ra)
    ast = arm_stats(aa) if aa else {}
    models = sorted(set(fst) & set(rst))
    if not models:
        raise SystemExit("no model overlap between filtered and "
                         "random_complement arms")

    if args.out:
        out_dir = Path(args.out)
    else:
        now = datetime.now()
        out_dir = (PROJECT_ROOT / "outputs" / "experiments"
                   / now.strftime("%m%d") / now.strftime("%H%M")
                   / "E1_filter_vs_random")
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = {"filtered": fst, "random_complement": rst}
    if ast:
        arms["random_full"] = ast
    write_per_model(out_dir / "per_model.csv", arms)

    txt = ["E1 / RQ1 — filter > random (construction validation, spine)",
           f"filtered            : {args.filtered}",
           f"random_complement   : {args.random_complement}",
           f"random_full         : {args.random_full or '(none)'}",
           f"bootstrap iters     : {args.bootstrap}  seed={args.seed}",
           ""]
    block, _ = _compare_block("random_complement", fa, ra, fst, rst,
                              models, args.bootstrap, args.seed)
    txt += block
    if ast:
        am = sorted(set(fst) & set(ast))
        block2, _ = _compare_block("random_full", fa, aa, fst, ast,
                                   am, args.bootstrap, args.seed)
        txt += block2
    (out_dir / "wilcoxon.txt").write_text("\n".join(txt) + "\n")
    print(f"wrote {out_dir / 'wilcoxon.txt'}")
    print()
    print("\n".join(txt))

    paired_bar(out_dir / "paired_bar.png", fst, rst, ast, models)


if __name__ == "__main__":
    main()
