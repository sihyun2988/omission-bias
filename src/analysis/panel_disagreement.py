"""Score panel disagreement per (scenario, frame) from a panel JSONL.

Inputs:  data/panel_outputs/panel_<provider>_<slug>.jsonl
Outputs: outputs/analysis/panel_disagreement_<panel-stem>.csv  (per-frame rows)
         outputs/analysis/panel_disagreement_<panel-stem>_scenario.csv  (per-scenario rows)
         stdout: bucket distribution + worst/best fault-lines

Metrics:
  modal[(sid, frame, pid)]   — majority answer across n_samples (None if tied)
  intra_H[(sid, frame, pid)] — Shannon entropy of that philosophy's n_samples (yes/no)
  inter_H[(sid, frame)]      — entropy of the 6 modal votes (the fault-line score)
  bucket                     — 6-phil-adjusted cutoffs: none | weak | strong | very_strong

Scenario-level bucket = strongest of (frame_A, frame_B) buckets. Filtering for
the fault-line subset uses scenario_bucket ∈ {strong, very_strong} by default
(≥ 2 dissenters in ≥ 1 frame, excluding both unanimous-yes and unanimous-no).

Invoke:
    .venv/bin/python3 -m src.analysis.panel_disagreement
    .venv/bin/python3 -m src.analysis.panel_disagreement --panel data/panel_outputs/<f>.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHILS = ["F1_util", "F2_deon", "F3_virtue", "F4_care", "F5_contract"]
FRAMES = ["A", "B"]

# 5-phil binary-vote inter-entropy → bucket. With 5 (odd) philosophies the only
# reachable inter-H values are 0.000 (5/0), 0.722 (4/1), and 0.971 (3/2). No
# perfectly balanced split exists, so "very_strong" collapses into "strong".
# Cutoffs sit on the gaps so records never land on an edge.
BUCKET_ORDER = ["none", "weak", "strong"]


def bucket(inter_h: float, n_distinct: int) -> str:
    if n_distinct <= 1:
        return "none"     # 5/0 unanimous (H = 0.000)
    if inter_h < 0.85:
        return "weak"     # 4/1 split (H ≈ 0.722) — single dissenter, possibly noise
    return "strong"       # 3/2 split (H ≈ 0.971) — ≥2 dissenters, genuine fault line


def shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0)


def modal_with_h(answers: list[str | None]) -> tuple[str | None, float]:
    valid = [a for a in answers if a in ("yes", "no")]
    if not valid:
        return None, float("nan")
    c = Counter(valid)
    top = c.most_common()
    # Tie → no modal vote
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None, shannon_entropy(c)
    return top[0][0], shannon_entropy(c)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default=None,
                   help="default: most-recent data/panel_outputs/panel_*.jsonl")
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "analysis"))
    p.add_argument("--filter-min-bucket", default="strong",
                   choices=BUCKET_ORDER,
                   help="scenario_bucket ≥ this passes the fault-line filter (default: strong)")
    args = p.parse_args()

    if args.panel is None:
        cands = sorted((PROJECT_ROOT / "data" / "panel_outputs").glob("panel_*.jsonl"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            raise SystemExit("no panel_*.jsonl files found under data/panel_outputs/")
        args.panel = str(cands[0])

    panel_path = Path(args.panel)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = panel_path.stem
    cell_csv = out_dir / f"panel_disagreement_{stem}.csv"
    scen_csv = out_dir / f"panel_disagreement_{stem}_scenario.csv"

    print(f"panel:    {panel_path}")
    print(f"out:      {cell_csv}")
    print(f"          {scen_csv}")

    # Group raw rows by cell (scenario, frame, philosophy)
    cells: dict[tuple, list[str | None]] = defaultdict(list)
    parse_err = err = 0
    n_records = 0
    with open(panel_path) as f:
        for line in f:
            r = json.loads(line)
            n_records += 1
            if r.get("error"):
                err += 1; continue
            if r["answer"] not in ("yes", "no"):
                parse_err += 1
            cells[(r["scenario_id"], r["frame"], r["philosophy_id"])].append(r["answer"])

    print(f"records:  {n_records}  parse_err={parse_err}  api_err={err}")

    # Per-cell modal + intra-H
    modal: dict[tuple, str | None] = {}
    intra_h: dict[tuple, float] = {}
    for k, answers in cells.items():
        m, h = modal_with_h(answers)
        modal[k] = m
        intra_h[k] = h

    # Per (scenario, frame) inter-H and bucket
    sids = sorted({k[0] for k in cells})
    cell_rows = []
    scen_bucket: dict[str, str] = {}
    scen_frame_bucket: dict[str, dict[str, str]] = defaultdict(dict)
    for sid in sids:
        for frame in FRAMES:
            modals = [modal.get((sid, frame, pid)) for pid in PHILS]
            valid = [m for m in modals if m in ("yes", "no")]
            n_yes = sum(1 for m in valid if m == "yes")
            n_no = sum(1 for m in valid if m == "no")
            n_distinct = len(set(valid))
            inter = shannon_entropy(Counter(valid)) if valid else float("nan")
            b = bucket(inter, n_distinct) if valid else "none"
            scen_frame_bucket[sid][frame] = b
            row = {
                "scenario_id": sid, "frame": frame,
                "n_yes": n_yes, "n_no": n_no, "n_missing": 6 - len(valid),
                "inter_H": round(inter, 4) if not math.isnan(inter) else "",
                "bucket": b,
            }
            for pid in PHILS:
                row[f"modal_{pid}"] = modal.get((sid, frame, pid)) or ""
                ih = intra_h.get((sid, frame, pid), float("nan"))
                row[f"intra_H_{pid}"] = round(ih, 4) if not math.isnan(ih) else ""
            cell_rows.append(row)

        bA = scen_frame_bucket[sid].get("A", "none")
        bB = scen_frame_bucket[sid].get("B", "none")
        scen_bucket[sid] = max(bA, bB, key=lambda b: BUCKET_ORDER.index(b))

    # Write per-frame CSV
    if cell_rows:
        fieldnames = list(cell_rows[0].keys())
        with open(cell_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(cell_rows)

    # Write per-scenario CSV (with pass/fail filter tag)
    min_idx = BUCKET_ORDER.index(args.filter_min_bucket)
    with open(scen_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario_id", "bucket_A", "bucket_B", "scenario_bucket",
                    "passes_fault_line_filter"])
        for sid in sids:
            sb = scen_bucket[sid]
            passes = BUCKET_ORDER.index(sb) >= min_idx
            w.writerow([sid,
                        scen_frame_bucket[sid].get("A", "none"),
                        scen_frame_bucket[sid].get("B", "none"),
                        sb, int(passes)])

    # ---- stdout summary ----
    print()
    print("=" * 76)
    print("Per-frame bucket distribution")
    print("=" * 76)
    cnt_pf = Counter((r["frame"], r["bucket"]) for r in cell_rows)
    print(f"{'bucket':<14} {'frame A':>10} {'frame B':>10}")
    for b in BUCKET_ORDER:
        print(f"{b:<14} {cnt_pf[('A', b)]:>10} {cnt_pf[('B', b)]:>10}")

    print()
    print("=" * 76)
    print(f"Per-scenario bucket (= strongest of frame A/B)  [filter: ≥ {args.filter_min_bucket}]")
    print("=" * 76)
    cnt_s = Counter(scen_bucket.values())
    total = sum(cnt_s.values())
    n_pass = sum(c for b, c in cnt_s.items() if BUCKET_ORDER.index(b) >= min_idx)
    for b in BUCKET_ORDER:
        marker = "  ✓ passes" if BUCKET_ORDER.index(b) >= min_idx else ""
        pct = cnt_s[b] / total * 100 if total else 0
        print(f"  {b:<14} {cnt_s[b]:>5}  ({pct:>5.1f}%){marker}")
    print(f"  {'TOTAL':<14} {total:>5}")
    print(f"  fault-line subset size: {n_pass}/{total} ({n_pass/total*100:.1f}%)")

    # Per-scenario detail (sorted by max inter_H, top 15)
    print()
    print("=" * 76)
    print("Top fault-line scenarios (strongest disagreement, sorted)")
    print("=" * 76)
    by_max_h = []
    for sid in sids:
        a_row = next(r for r in cell_rows if r["scenario_id"] == sid and r["frame"] == "A")
        b_row = next(r for r in cell_rows if r["scenario_id"] == sid and r["frame"] == "B")
        h_a = a_row["inter_H"] if a_row["inter_H"] != "" else 0
        h_b = b_row["inter_H"] if b_row["inter_H"] != "" else 0
        by_max_h.append((max(h_a, h_b), sid, a_row, b_row))
    by_max_h.sort(reverse=True)
    print(f"{'scenario':<8} {'bucket':<13} A {'y/n':<5} {'H':>5}   B {'y/n':<5} {'H':>5}")
    for _, sid, a, b in by_max_h[:15]:
        print(f"{sid:<8} {scen_bucket[sid]:<13} "
              f"  {a['n_yes']}/{a['n_no']:<3} {str(a['inter_H']):>5}  "
              f"  {b['n_yes']}/{b['n_no']:<3} {str(b['inter_H']):>5}")


if __name__ == "__main__":
    main()
