"""Stage 2: conflict labeling on filter-passing scenarios.

Reads the Stage 1 filter CSV (filter_<panel-stem>.csv). For each scenario that
passed the unanimity filter:

  1. Drop philosophies whose tuple is ``YY`` or ``NN`` — they didn't flip
     across the action↔omission inversion, so they cannot anchor a label about
     which framing the moral framework prefers.
  2. Separately bucket philosophies with incomplete tuples (any ``?``) — they
     are tracked but cannot be grouped. This matters when filter.py is run
     with ``--min-complete-phils < #{PHILS}`` (partial labeling).
  3. Group remaining philosophies by flip direction: ``YN`` (yes on A, no on B)
     vs ``NY``.
  4. ``label_status`` reflects the outcome:
        * ``labeled``                 — both groups non-empty (conflict pairs)
        * ``dropped_one_sided_yn``    — only YN flippers
        * ``dropped_one_sided_ny``    — only NY flippers
        * ``dropped_all_excluded``    — no YN, no NY (all YY/NN/incomplete)

Conflict output
---------------
``conflicts`` is the full list of ``[yn_phil, ny_phil]`` pairs (Cartesian of
the YN group × the NY group). A labeled scenario may yield several conflict
pairs. There is no single "primary" pair — every cross-group opposition is
enumerated. A scenario therefore contributes to multiple (yn_phil × ny_phil)
cells; downstream inferential code must account for this non-independence
(e.g., scenario as a random effect / cluster-robust SE), not assume one
observation per cell.

Side fields:
- ``yn_count`` / ``ny_count`` / ``excluded_count`` / ``incomplete_count``:
  panel-split arithmetic for stratification.

Output: data/panel_outputs/labels_<panel-stem>.jsonl

Invoke:
    .venv/bin/python3 -m src.data_construction.philosophy_panel.label
    .venv/bin/python3 -m src.data_construction.philosophy_panel.label \\
        --filter data/panel_outputs/filter_<panel-stem>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from src.data_construction.philosophy_panel.philosophies import PHILS

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def strip_filter_prefix(stem: str) -> str:
    return stem[len("filter_"):] if stem.startswith("filter_") else stem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--filter", default=None,
                   help="default: most-recent data/panel_outputs/filter_*.csv")
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "panel_outputs"))
    args = p.parse_args()

    if args.filter is None:
        cands = sorted((PROJECT_ROOT / "data" / "panel_outputs").glob("filter_*.csv"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            raise SystemExit("no filter_*.csv under data/panel_outputs/; run filter.py first")
        args.filter = str(cands[0])

    filter_path = Path(args.filter)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"labels_{strip_filter_prefix(filter_path.stem)}.jsonl"

    print(f"filter: {filter_path}")
    print(f"out:    {out_jsonl}")

    status_counts: Counter[str] = Counter()
    conflict_counter: Counter[tuple[str, str]] = Counter()

    with open(filter_path) as f, open(out_jsonl, "w") as out:
        reader = csv.DictReader(f)
        for row in reader:
            if row["passes_filter"] != "1":
                continue
            yn = [pid for pid in PHILS if row.get(f"tuple_{pid}") == "YN"]
            ny = [pid for pid in PHILS if row.get(f"tuple_{pid}") == "NY"]
            excluded = [pid for pid in PHILS
                        if row.get(f"tuple_{pid}") in ("YY", "NN")]
            incomplete = [pid for pid in PHILS
                          if "?" in (row.get(f"tuple_{pid}") or "??")]

            if yn and ny:
                conflicts = [[a, b] for a in yn for b in ny]
                for a, b in conflicts:
                    conflict_counter[(a, b)] += 1
                status = "labeled"
            elif yn and not ny:
                conflicts = []
                status = "dropped_one_sided_yn"
            elif ny and not yn:
                conflicts = []
                status = "dropped_one_sided_ny"
            else:
                conflicts = []
                status = "dropped_all_excluded"
            status_counts[status] += 1

            rec = {
                "scenario_id": row["scenario_id"],
                "label_status": status,
                "yn_phils": yn,
                "ny_phils": ny,
                "excluded_phils": excluded,
                "incomplete_phils": incomplete,
                "yn_count": len(yn),
                "ny_count": len(ny),
                "excluded_count": len(excluded),
                "incomplete_count": len(incomplete),
                "conflicts": conflicts,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = sum(status_counts.values())
    print()
    if total == 0:
        print("no stage-1 passers found in filter CSV — nothing to label")
        return
    print(f"stage-1 passers           {total:>5}")
    for status in ("labeled", "dropped_one_sided_yn", "dropped_one_sided_ny",
                   "dropped_all_excluded"):
        n = status_counts.get(status, 0)
        print(f"  {status:<24}{n:>5}  ({n / total * 100:.1f}%)")
    if conflict_counter:
        print()
        print("Conflict pairs (yn_phil vs ny_phil; a scenario may yield several):")
        for (a, b), c in conflict_counter.most_common(20):
            print(f"  {a:<12} vs {b:<12} {c:>5}")


if __name__ == "__main__":
    main()
