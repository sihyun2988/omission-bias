"""Analyze pilot output: per-(scenario,philosophy) modal vote + intra entropy,
per-scenario inter-philosophy disagreement, and comparison vs hand predictions.

Usage:
    python pilot/analyze.py                                  # default seed42 results
    python pilot/analyze.py --results pilot/results/<f>.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PHILS = ["F1_util", "F2_deon", "F3_virtue", "F4_care", "F5_contract"]


def shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0)


def fault_line_bucket(inter_h: float, n_distinct: int) -> str:
    """Map inter-philosophy entropy + vote spread to a strength label."""
    if n_distinct <= 1:
        return "none"
    if inter_h < 0.5:
        return "weak"
    if inter_h < 0.8:
        return "moderate"
    if inter_h < 0.95:
        return "strong"
    return "very_strong"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=None,
                   help="default: most-recent pilot/results/pilot_*.jsonl")
    p.add_argument("--predictions", default=str(ROOT / "predictions_seed42.json"))
    args = p.parse_args()

    if args.results is None:
        candidates = sorted((ROOT / "results").glob("pilot_*.jsonl"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise SystemExit("no pilot results found in pilot/results/")
        args.results = str(candidates[0])

    print(f"results: {args.results}")
    with open(args.results) as f:
        records = [json.loads(line) for line in f]
    with open(args.predictions) as f:
        preds = json.load(f)

    by_pair: dict[tuple, list[str | None]] = defaultdict(list)
    parse_errors = 0
    for r in records:
        by_pair[(r["scenario_id"], r["philosophy_id"])].append(r["choice"])
        if r["choice"] not in ("A1", "A2"):
            parse_errors += 1

    modal: dict[str, dict[str, str | None]] = defaultdict(dict)
    intra_h: dict[str, dict[str, float]] = defaultdict(dict)
    for (sid, pid), choices in by_pair.items():
        valid = [c for c in choices if c in ("A1", "A2")]
        if not valid:
            modal[sid][pid] = None
            intra_h[sid][pid] = float("nan")
            continue
        counts = Counter(valid)
        modal[sid][pid] = counts.most_common(1)[0][0]
        intra_h[sid][pid] = shannon_entropy(counts)

    print(f"\nrecords: {len(records)}   parse/error: {parse_errors}\n")

    # ---- per-scenario detail ----
    hits = 0
    measured = 0
    print("=" * 96)
    print("Per-(scenario, philosophy) prediction vs measured modal vote")
    print("=" * 96)
    print(f"{'scenario':<8} {'philosophy':<14} {'pred':<5} {'meas':<5} "
          f"{'intra_H':<8} {'match'}")
    print("-" * 96)
    for sid, sd in preds.items():
        if sid.startswith("_"):
            continue
        for pid in PHILS:
            pred = sd["predictions"][pid]
            meas = modal.get(sid, {}).get(pid)
            ih = intra_h.get(sid, {}).get(pid, float("nan"))
            mark = "·" if meas is None else ("✓" if meas == pred else "✗")
            if meas is not None:
                measured += 1
                if meas == pred:
                    hits += 1
            print(f"{sid:<8} {pid:<14} {pred:<5} {str(meas):<5} {ih:<8.2f} {mark}")
        print()

    if measured:
        print(f"prediction agreement: {hits}/{measured} = {hits/measured:.1%}")
    else:
        print("no measured data yet")

    # ---- per-scenario fault line summary ----
    print()
    print("=" * 96)
    print("Per-scenario fault line: predicted vs measured")
    print("=" * 96)
    print(f"{'scenario':<8} {'predicted':<14} {'measured':<14} {'inter_H':<8} "
          f"{'votes (util,deon,virtue,care,contract)'}")
    print("-" * 96)
    pred_strength_match = 0
    measured_scenarios = 0
    for sid, sd in preds.items():
        if sid.startswith("_"):
            continue
        votes = [modal.get(sid, {}).get(pid) for pid in PHILS]
        valid = [v for v in votes if v in ("A1", "A2")]
        if len(valid) < 2:
            inter = float("nan")
            bucket = "(insufficient)"
        else:
            inter = shannon_entropy(Counter(valid))
            bucket = fault_line_bucket(inter, len(set(valid)))
            measured_scenarios += 1
            if bucket == sd["predicted_fault_line"]:
                pred_strength_match += 1
        votes_str = ",".join(v or "·" for v in votes)
        print(f"{sid:<8} {sd['predicted_fault_line']:<14} {bucket:<14} "
              f"{inter:<8.2f} {votes_str}")

    if measured_scenarios:
        print(f"\nfault-line bucket exact match: "
              f"{pred_strength_match}/{measured_scenarios} = "
              f"{pred_strength_match/measured_scenarios:.1%}")

    # ---- highlight surprises ----
    print()
    print("=" * 96)
    print("Surprises (where measured disagrees with prediction at the philosophy level)")
    print("=" * 96)
    any_ = False
    for sid, sd in preds.items():
        if sid.startswith("_"):
            continue
        for pid in PHILS:
            pred = sd["predictions"][pid]
            meas = modal.get(sid, {}).get(pid)
            if meas is not None and meas != pred:
                any_ = True
                print(f"  {sid} / {pid}: predicted {pred}, measured {meas}  "
                      f"({sd['context_short']})")
    if not any_:
        print("  (none — all measured modal votes agree with hand predictions)")


if __name__ == "__main__":
    main()
