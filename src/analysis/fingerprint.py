"""E3 — RQ3: per-model moral fingerprint (★ analysis core).

Joins E1 eval tuples (`eval_tuples.jsonl`, one modal tuple per model×scenario)
with the Stage-2 conflict labels (`labels_*.jsonl`, scenario → yn_phils /
ny_phils) and asks: *when a model answers frame-consistently, which moral-
philosophy camp does it land in — and how often is it instead biased
(framing-invariant inaction NN, or action YY)?*

Aggregation (RESEARCH_PLAN_v4 §172 / §42 H3), per (model, scenario), restricted
to the labeled benchmark (`label_status == "labeled"` — the only scope where a
genuine yn↔ny conflict exists; see [[feedback_benchmark_is_labeled_only]]):

    tuple YN  → each phil in yn_phils  +1   (frame-consistent, yn-camp)
    tuple NY  → each phil in ny_phils  +1   (frame-consistent, ny-camp)
    tuple NN  → non-aligned: omission bias
    tuple YY  → non-aligned: action bias

Per model this yields a 5-philosophy leaning vector (raw counts + a size-
normalised distribution) and a bias rate (NN-rate + YY-rate). Two permutation
tests, both hand-rolled (scipy is NOT in the venv — see CLAUDE.md):

1. PROFILE vs NULL (5,000 perms). Null = the model has no *philosophical*
   leaning beyond panel structure: keep the set of frame-consistent scenarios
   and the model's own yn:ny directional marginal fixed, but re-draw each
   scenario's camp by that marginal (Bernoulli q = observed #YN/(#YN+#NY)).
   This isolates *which philosophies* from *overall yes/no tilt*. Per-phil
   two-sided percentile p + an omnibus Σz² statistic with its own perm p.

2. CROSS-MODEL COSINE (label-shuffle perm). Pairwise cosine of leaning
   vectors; null = per-scenario permutation of which model got which
   frame-consistent outcome, recomputing the cosine matrix and tracking the
   minimum off-diagonal cosine. Observed min-cosine vs null tests whether the
   fleet's fingerprints are *more divergent* than exchangeable chance. H3 also
   fixes an absolute threshold: ≥1 model pair with cosine < 0.85.

Dendrogram = average-linkage agglomerative clustering on (1 − cosine),
hand-rolled (no scipy) and drawn with matplotlib.

Output (co-located with the E1 run it consumes —
    .../<MMDD>/<HHMM>/E1_overall_OBR/eval_tuples.jsonl
        → .../<MMDD>/<HHMM>/E3_fingerprint/ ):
    profile_per_model.csv  cosine_matrix.csv
    dendrogram.png  permutation_pvalue.txt
(falls back to a fresh MMDD/HHMM timestamp if --eval isn't in that layout;
override with --out-dir.)

Invoke:
    /home/lsh/omission/.venv/bin/python3 -m src.analysis.fingerprint \\
        --eval outputs/experiments/0517/1809/E1_overall_OBR/eval_tuples.jsonl \\
        --labels data/panel_outputs/labels_openrouter_openai_gpt-4.1-mini.jsonl
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_construction.philosophy_panel.philosophies import PHILS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TUPLES = ("YY", "YN", "NY", "NN")
COSINE_THRESHOLD = 0.85  # §42 H3 absolute divergence criterion


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


def load_labeled(path: Path) -> dict[str, dict]:
    """sid -> {yn:set, ny:set} for label_status == 'labeled' only."""
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("label_status") != "labeled":
                continue
            out[r["scenario_id"]] = {
                "yn": set(r.get("yn_phils", [])),
                "ny": set(r.get("ny_phils", [])),
            }
    return out


# --------------------------------------------------------------- profile ---
def leaning_vector(sids, esid, labeled) -> list[float]:
    """Raw 5-phil leaning over PHILS order (frame-consistent contributions)."""
    v = {p: 0.0 for p in PHILS}
    for sid in sids:
        t = esid.get(sid)
        lab = labeled.get(sid)
        if lab is None or t not in TUPLES:
            continue
        if t == "YN":
            for p in lab["yn"]:
                v[p] += 1
        elif t == "NY":
            for p in lab["ny"]:
                v[p] += 1
    return [v[p] for p in PHILS]


def model_summary(sids, esid, labeled) -> dict:
    """Counts + bias rates over the labeled scenarios this model scored."""
    c = Counter()
    for sid in sids:
        t = esid.get(sid)
        if sid in labeled and t in TUPLES:
            c[t] += 1
    n = sum(c.values())
    fc = c["YN"] + c["NY"]
    return {
        "n_valid": n,
        "n_frame_consistent": fc,
        "n_YN": c["YN"], "n_NY": c["NY"],
        "n_NN": c["NN"], "n_YY": c["YY"],
        "bias_rate": (c["NN"] + c["YY"]) / n if n else None,
        "NN_rate": c["NN"] / n if n else None,
        "YY_rate": c["YY"] / n if n else None,
        "q_yn": c["YN"] / fc if fc else None,  # directional marginal
    }


# ------------------------------------------------------------ statistics ---
def cosine(a: list[float], b: list[float]) -> float | None:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return None
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def null_profile_perm(fc_scenarios, labeled, q_yn, n_perm, rng):
    """Null leaning vectors: re-draw each frame-consistent scenario's camp by
    the model's own yn:ny marginal q_yn. Returns list[n_perm][len(PHILS)]."""
    out = []
    for _ in range(n_perm):
        v = {p: 0.0 for p in PHILS}
        for lab in fc_scenarios:
            side = "yn" if rng.random() < q_yn else "ny"
            for p in lab[side]:
                v[p] += 1
        out.append([v[p] for p in PHILS])
    return out


def perm_pvalues(obs, null_mat):
    """Per-phil two-sided percentile p + omnibus Σz² perm p."""
    k = len(obs)
    means = [sum(col) / len(null_mat) for col in zip(*null_mat)]
    sds = []
    for j, col in enumerate(zip(*null_mat)):
        var = sum((x - means[j]) ** 2 for x in col) / len(null_mat)
        sds.append(math.sqrt(var))
    per_phil = []
    for j in range(k):
        ge = sum(1 for row in null_mat if row[j] >= obs[j]) + 1
        le = sum(1 for row in null_mat if row[j] <= obs[j]) + 1
        n1 = len(null_mat) + 1
        per_phil.append(min(1.0, 2 * min(ge / n1, le / n1)))

    def z2(vec):
        return sum(((vec[j] - means[j]) / sds[j]) ** 2
                   for j in range(k) if sds[j] > 0)

    obs_stat = z2(obs)
    ge = sum(1 for row in null_mat if z2(row) >= obs_stat) + 1
    omnibus_p = ge / (len(null_mat) + 1)
    return per_phil, obs_stat, omnibus_p


def cosine_matrix(profiles: dict[str, list[float]], models):
    mat = {}
    for a in models:
        for b in models:
            mat[(a, b)] = cosine(profiles[a], profiles[b])
    return mat


def min_offdiag(mat, models):
    vals = [mat[(a, b)] for i, a in enumerate(models)
            for b in models[i + 1:] if mat[(a, b)] is not None]
    return min(vals) if vals else None


def cosine_label_shuffle(models, sid_tuple, labeled, n_perm, rng):
    """Null min off-diagonal cosine under per-scenario model-label shuffle.

    sid_tuple[sid] = {model: tuple}. For each scenario the models' tuples are
    permuted among the models present, profiles rebuilt, cosine recomputed."""
    all_sids = list(sid_tuple)
    out = []
    for _ in range(n_perm):
        vecs = {m: {p: 0.0 for p in PHILS} for m in models}
        for sid in all_sids:
            lab = labeled.get(sid)
            if lab is None:
                continue
            present = [m for m in models if sid_tuple[sid].get(m) in TUPLES]
            if not present:
                continue
            tuples = [sid_tuple[sid][m] for m in present]
            rng.shuffle(tuples)
            for m, t in zip(present, tuples):
                if t == "YN":
                    for p in lab["yn"]:
                        vecs[m][p] += 1
                elif t == "NY":
                    for p in lab["ny"]:
                        vecs[m][p] += 1
        prof = {m: [vecs[m][p] for p in PHILS] for m in models}
        mo = min_offdiag(cosine_matrix(prof, models), models)
        if mo is not None:
            out.append(mo)
    return out


# ---------------------------------------------------------- dendrogram ----
def average_linkage(models, dist):
    """Hand-rolled agglomerative average-linkage. Returns merge list of
    (left, right, height, members) with leaves as ints 0..n-1."""
    clusters = {i: [i] for i in range(len(models))}
    node = {i: i for i in range(len(models))}
    next_id = len(models)
    merges = []

    def cdist(ci, cj):
        s = sum(dist[a][b] for a in clusters[ci] for b in clusters[cj])
        return s / (len(clusters[ci]) * len(clusters[cj]))

    while len(clusters) > 1:
        keys = list(clusters)
        best = None
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                d = cdist(keys[i], keys[j])
                if best is None or d < best[0]:
                    best = (d, keys[i], keys[j])
        h, ci, cj = best
        merges.append((node[ci], node[cj], h,
                       clusters[ci] + clusters[cj]))
        clusters[next_id] = clusters[ci] + clusters[cj]
        node[next_id] = next_id
        del clusters[ci], clusters[cj]
        next_id += 1
    return merges


def draw_dendrogram(models, mat, out_path: Path):
    n = len(models)
    dist = [[0.0] * n for _ in range(n)]
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            c = mat[(a, b)]
            dist[i][j] = 1.0 - c if c is not None else 1.0
    merges = average_linkage(models, dist)

    # node id n+k == k-th merge (matches average_linkage's next_id scheme)
    tree = {}
    for idx, (li, ri, h, _) in enumerate(merges, start=n):
        tree[idx] = (li, ri, h)

    order = []

    def walk(nd):
        if nd < n:
            order.append(nd)
            return
        li, ri, _ = tree[nd]
        walk(li)
        walk(ri)

    walk(n + len(merges) - 1)
    xpos = {leaf: i for i, leaf in enumerate(order)}

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * n), 4.5))
    coord = {}

    def xy(nd):
        if nd < n:
            coord[nd] = (xpos[nd], 0.0)
            return coord[nd]
        li, ri, h = tree[nd]
        lx, ly = xy(li)
        rx, ry = xy(ri)
        ax.plot([lx, lx, rx, rx], [ly, h, h, ry], color="#333333", lw=1.5)
        c = ((lx + rx) / 2, h)
        coord[nd] = c
        return c

    xy(n + len(merges) - 1)
    ax.set_xticks(range(n))
    ax.set_xticklabels([_nice(models[order[i]]) for i in range(n)],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("1 − cosine (average linkage)")
    ax.set_title("E3 / RQ3 — model moral-fingerprint dendrogram")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _nice(model: str) -> str:
    return model.split("/")[-1]


_PHIL_SHORT = {"F1_util": "util", "F2_deon": "deon", "F3_virtue": "virtue",
               "F4_care": "care", "F5_contract": "contract"}
_PHIL_COLORS = {"F1_util": "#4477aa", "F2_deon": "#ee6677",
                "F3_virtue": "#228833", "F4_care": "#ccbb44",
                "F5_contract": "#aa3377"}


def draw_fingerprint(models, norm_by_model, bias_by_model, out_path: Path):
    """F4 (plan §265, ★core RQ3 figure): per-model size-normalised 5-phil
    leaning (left, stacked, sums to 1) + bias rate split NN/YY (right)."""
    labels = [_nice(m) for m in models]
    y = list(range(len(models)))
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(12, 0.85 * len(models) + 2.2),
        gridspec_kw={"width_ratios": [3, 2]})

    left = [0.0] * len(models)
    for ph in PHILS:
        vals = [norm_by_model[m][ph] for m in models]
        axL.barh(y, vals, left=left, color=_PHIL_COLORS[ph],
                 label=_PHIL_SHORT[ph], height=0.62)
        for i, v in enumerate(vals):
            if v >= 0.06:
                axL.text(left[i] + v / 2, i, f"{v:.2f}",
                         ha="center", va="center", fontsize=8, color="white")
        left = [a + b for a, b in zip(left, vals)]
    axL.set_yticks(y)
    axL.set_yticklabels(labels, fontsize=9)
    axL.set_xlim(0, 1)
    axL.set_xlabel("frame-consistent leaning share (size-normalised)")
    axL.set_title("Which moral-philosophy camp the model lands in")
    axL.legend(ncol=5, fontsize=8, loc="lower center",
               bbox_to_anchor=(0.5, 1.06), frameon=False)

    nn = [bias_by_model[m]["NN"] for m in models]
    yy = [bias_by_model[m]["YY"] for m in models]
    axR.barh(y, nn, color="#cc6677", label="NN (omission bias)", height=0.62)
    axR.barh(y, yy, left=nn, color="#88ccee",
             label="YY (action bias)", height=0.62)
    for i, m in enumerate(models):
        tot = nn[i] + yy[i]
        axR.text(tot + 0.01, i, f"{tot:.2f}", va="center", fontsize=8)
    axR.set_yticks(y)
    axR.set_yticklabels([])
    axR.set_xlim(0, max([nn[i] + yy[i] for i in range(len(models))] + [0.1])
                 * 1.18)
    axR.set_xlabel("non-aligned (bias) rate")
    axR.set_title("How often it instead stays biased")
    axR.legend(fontsize=8, loc="lower right", frameon=False)

    fig.suptitle("E3 / RQ3 — per-model moral fingerprint", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- main ---
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", required=True, help="E1 eval_tuples.jsonl")
    p.add_argument("--labels", required=True, help="Stage-2 labels_*.jsonl")
    p.add_argument("--n-perm", type=int, default=5000,
                   help="permutation iterations (§179 prereg = 5000)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    rng = random.Random(args.seed)
    eval_by_model, eval_diag = load_eval(Path(args.eval))
    labeled = load_labeled(Path(args.labels))
    labeled_sids = list(labeled)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ep = Path(args.eval).resolve()
        if ep.parent.name == "E1_overall_OBR":
            out_dir = ep.parent.parent / "E3_fingerprint"
        else:
            now = datetime.now()
            out_dir = (PROJECT_ROOT / "outputs" / "experiments"
                       / now.strftime("%m%d") / now.strftime("%H%M")
                       / "E3_fingerprint")
    out_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(eval_by_model)
    profiles: dict[str, list[float]] = {}
    norm_by_model: dict[str, dict] = {}
    bias_by_model: dict[str, dict] = {}
    rows = []
    perm_lines = []

    for m in models:
        esid = eval_by_model[m]
        summ = model_summary(labeled_sids, esid, labeled)
        raw = leaning_vector(labeled_sids, esid, labeled)
        total = sum(raw)
        norm = [x / total if total else 0.0 for x in raw]
        profiles[m] = raw
        norm_by_model[m] = {ph: norm[i] for i, ph in enumerate(PHILS)}
        bias_by_model[m] = {"NN": summ["NN_rate"] or 0.0,
                            "YY": summ["YY_rate"] or 0.0}

        # ---- profile vs null permutation ----
        fc_scn = [labeled[s] for s in labeled_sids
                  if esid.get(s) in ("YN", "NY") and s in labeled]
        q = summ["q_yn"]
        if q is None or not fc_scn:
            per_phil = [None] * len(PHILS)
            omnibus_p = None
            obs_stat = None
        else:
            null_mat = null_profile_perm(fc_scn, labeled, q,
                                         args.n_perm, rng)
            per_phil, obs_stat, omnibus_p = perm_pvalues(raw, null_mat)

        row = {
            "model": m,
            "n_valid": summ["n_valid"],
            "n_frame_consistent": summ["n_frame_consistent"],
            "bias_rate": _r(summ["bias_rate"]),
            "NN_rate": _r(summ["NN_rate"]),
            "YY_rate": _r(summ["YY_rate"]),
            "omnibus_z2": _r(obs_stat),
            "omnibus_perm_p": _r(omnibus_p),
        }
        for i, ph in enumerate(PHILS):
            row[f"raw_{ph}"] = _r(raw[i])
            row[f"norm_{ph}"] = _r(norm[i])
            row[f"p_{ph}"] = _r(per_phil[i])
        rows.append(row)

        sig = ("n/a" if omnibus_p is None
               else "SIG" if omnibus_p < 0.05 else "ns")
        perm_lines.append(
            f"  {m:<34} omnibus Σz²={_r(obs_stat)} "
            f"perm p={_r(omnibus_p)} [{sig}]  "
            f"bias={_r(summ['bias_rate'])} "
            f"(NN={_r(summ['NN_rate'])} YY={_r(summ['YY_rate'])})")

    # ---- cross-model cosine matrix ----
    mat = cosine_matrix(profiles, models)
    obs_min = min_offdiag(mat, models)

    sid_tuple: dict[str, dict] = defaultdict(dict)
    for m in models:
        for sid, t in eval_by_model[m].items():
            if sid in labeled:
                sid_tuple[sid][m] = t
    null_min = cosine_label_shuffle(models, sid_tuple, labeled,
                                    args.n_perm, rng)
    if null_min and obs_min is not None:
        ge = sum(1 for x in null_min if x <= obs_min) + 1
        cos_p = ge / (len(null_min) + 1)
    else:
        cos_p = None

    below = [(a, b, _r(mat[(a, b)]))
             for i, a in enumerate(models) for b in models[i + 1:]
             if mat[(a, b)] is not None and mat[(a, b)] < COSINE_THRESHOLD]

    # ---- write outputs ----
    _write_csv(out_dir / "profile_per_model.csv", rows)

    with open(out_dir / "cosine_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + [_nice(m) for m in models])
        for a in models:
            w.writerow([_nice(a)] + [_r(mat[(a, b)]) for b in models])

    draw_fingerprint(models, norm_by_model, bias_by_model,
                     out_dir / "fingerprint.png")
    draw_dendrogram(models, mat, out_dir / "dendrogram.png")

    any_sig = any(r["omnibus_perm_p"] not in ("", None)
                  and float(r["omnibus_perm_p"]) < 0.05 for r in rows)
    h3 = any_sig and len(below) >= 1
    txt = []
    txt.append("E3 / RQ3 — model moral fingerprint")
    txt.append(f"eval        {Path(args.eval)}")
    txt.append(f"labels      {Path(args.labels)}")
    txt.append(f"models      {len(models)}")
    txt.append(f"labeled n   {len(labeled_sids)}")
    txt.append(f"n_perm      {args.n_perm}")
    txt.append("")
    txt.append("Per-model profile-vs-null (omnibus Σz², two-sided percentile "
               "per phil in CSV):")
    txt.extend(perm_lines)
    txt.append("")
    txt.append(f"Cross-model cosine: observed min off-diagonal = "
               f"{_r(obs_min)}")
    txt.append(f"  label-shuffle perm p (null min ≤ observed) = {_r(cos_p)}")
    txt.append(f"  pairs with cosine < {COSINE_THRESHOLD}: "
               f"{len(below)}"
               + ("" if not below else
                  "  -> " + "; ".join(f"{_nice(a)}~{_nice(b)}={c}"
                                      for a, b, c in below)))
    txt.append("")
    txt.append(f"H3 (≥1 model perm p<0.05 AND ≥1 model pair cosine<"
               f"{COSINE_THRESHOLD}): "
               f"{'SUPPORTED' if h3 else 'NOT supported'}")
    txt.append("  (per RESEARCH_PLAN_v4 §372/§389: RQ3 failure does NOT "
               "damage the RQ1 spine; report bias-rate differences regardless.)")
    (out_dir / "permutation_pvalue.txt").write_text("\n".join(txt) + "\n")

    # ---- console ----
    print(f"eval models  {len(models)}  ({', '.join(_nice(m) for m in models)})")
    print(f"labeled      {len(labeled_sids)}")
    print(f"out          {out_dir}")
    print()
    for ln in perm_lines:
        print(ln)
    print()
    print(f"min off-diag cosine = {_r(obs_min)}  "
          f"(perm p={_r(cos_p)}, {len(below)} pair(s) < {COSINE_THRESHOLD})")
    print(f"H3: {'SUPPORTED' if h3 else 'NOT supported'}")
    print()
    print("Read: omnibus perm p<0.05 ⇒ model's philosophy leaning departs "
          "from its own directional marginal; min-cosine perm p<0.05 (or any "
          f"pair < {COSINE_THRESHOLD}) ⇒ fingerprints diverge across models.")


# ------------------------------------------------------------ tiny utils ---
def _r(x, nd=4):
    return round(x, nd) if isinstance(x, (int, float)) else ""


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
