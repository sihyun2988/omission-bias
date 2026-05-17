"""Stage 1: unanimity filter on the (frame_A, frame_B) tuple per philosophy.

For each scenario in the panel JSONL, compute each philosophy's (A, B) answer
tuple from its modal vote across n_samples per frame. If all philosophies share
the same tuple, the scenario is unanimous across the panel — there is no moral
fault line, so the scenario fails the filter. Non-unanimous scenarios pass
through to Stage 2 (label.py).

Modal voting per cell uses majority of valid yes/no answers; ties yield None
(propagated as ``?`` in the tuple). A philosophy with at least one ``?`` in its
tuple is treated as INCOMPLETE for that scenario.

Scenario universe
-----------------
By default the universe is derived from scenarios that appear in the panel
JSONL. Pass ``--universe-from data/constructed/mirror_frames/paired_frames.jsonl``
to take the universe from the upstream paired-frames file instead — scenarios
that exist there but have no panel record will surface as
``reason=no_panel_data`` rather than silently disappearing.

Partial labeling
----------------
``--min-complete-phils N`` (default = #{PHILS}, strict) controls how many
philosophies must have BOTH frames complete (no ``?``) for a scenario to be
eligible. With N < #{PHILS}, scenarios with some incomplete philosophies still
pass if the complete subset is non-unanimous. Reason ``passes`` indicates all
philosophies complete; ``partial_passes`` indicates passing on a subset.

Output: data/panel_outputs/filter_<panel-stem>.csv

Invoke:
    .venv/bin/python3 -m src.data_construction.philosophy_panel.filter
    .venv/bin/python3 -m src.data_construction.philosophy_panel.filter \\
        --panel data/panel_outputs/<f>.jsonl \\
        --universe-from data/constructed/mirror_frames/paired_frames.jsonl \\
        --min-complete-phils 4
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.data_construction.philosophy_panel.philosophies import PHILS, FRAMES

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def modal(answers: list[str | None]) -> str | None:
    """Majority vote across samples; None if no valid answers or a tie."""
    valid = [a for a in answers if a in ("yes", "no")]
    if not valid:
        return None
    c = Counter(valid)
    top = c.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def tuple_code(a: str | None, b: str | None) -> str:
    def code(x: str | None) -> str:
        return "Y" if x == "yes" else "N" if x == "no" else "?"
    return code(a) + code(b)


def strip_panel_prefix(stem: str) -> str:
    return stem[len("panel_"):] if stem.startswith("panel_") else stem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default=None,
                   help="default: most-recent data/panel_outputs/panel_*.jsonl")
    p.add_argument("--universe-from", default=None,
                   help="JSONL file with one record per scenario (must contain "
                        "'scenario_id'). If set, the scenario universe is this "
                        "file's scenario_ids; scenarios missing from --panel "
                        "are emitted with reason=no_panel_data. Default: "
                        "universe = scenarios present in panel JSONL.")
    p.add_argument("--min-complete-phils", type=int, default=len(PHILS),
                   help=f"minimum #{{philosophies with both frames complete}} "
                        f"required for a scenario to be eligible. Default "
                        f"{len(PHILS)} (strict — all philosophies must be "
                        f"complete). Lower values enable partial labeling.")
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "panel_outputs"))
    args = p.parse_args()

    if args.panel is None:
        cands = sorted((PROJECT_ROOT / "data" / "panel_outputs").glob("panel_*.jsonl"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            raise SystemExit("no panel_*.jsonl files under data/panel_outputs/")
        args.panel = str(cands[0])

    if not (1 <= args.min_complete_phils <= len(PHILS)):
        raise SystemExit(f"--min-complete-phils must be in [1, {len(PHILS)}]")

    panel_path = Path(args.panel)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"filter_{strip_panel_prefix(panel_path.stem)}.csv"

    print(f"panel:             {panel_path}")
    print(f"out:               {out_csv}")
    print(f"min-complete-phils {args.min_complete_phils} / {len(PHILS)}")
    if args.universe_from:
        print(f"universe-from:     {args.universe_from}")

    cells: dict[tuple, list[str | None]] = defaultdict(list)
    unknown_phils: Counter[str] = Counter()
    unknown_frames: Counter[str] = Counter()
    n_error_records = 0
    sids_with_records: set[str] = set()

    with open(panel_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("error"):
                n_error_records += 1
                continue
            pid = r.get("philosophy_id")
            frame = r.get("frame")
            sid = r.get("scenario_id")
            if pid not in PHILS:
                unknown_phils[str(pid)] += 1
                continue
            if frame not in FRAMES:
                unknown_frames[str(frame)] += 1
                continue
            cells[(sid, frame, pid)].append(r["answer"])
            sids_with_records.add(sid)

    if args.universe_from:
        universe: set[str] = set()
        with open(args.universe_from) as f:
            for line in f:
                r = json.loads(line)
                if "scenario_id" in r:
                    universe.add(r["scenario_id"])
        sids = sorted(universe)
    else:
        sids = sorted(sids_with_records)

    n_pass = n_partial_pass = n_unanim = n_incomplete = n_no_data = 0
    unanim_breakdown: Counter[str] = Counter()
    rows = []
    for sid in sids:
        cell_modal: dict[tuple, str | None] = {}
        phil_tuples: dict[str, str] = {}
        for pid in PHILS:
            for frame in FRAMES:
                cell_modal[(pid, frame)] = modal(cells.get((sid, frame, pid), []))
            phil_tuples[pid] = tuple_code(cell_modal[(pid, "A")],
                                          cell_modal[(pid, "B")])

        incomplete_phils = [pid for pid in PHILS if "?" in phil_tuples[pid]]
        n_complete = len(PHILS) - len(incomplete_phils)
        complete_tuples = [phil_tuples[pid] for pid in PHILS
                           if "?" not in phil_tuples[pid]]

        if sid not in sids_with_records:
            passes, reason = 0, "no_panel_data"
            n_no_data += 1
        elif n_complete < args.min_complete_phils:
            passes, reason = 0, "incomplete_cells"
            n_incomplete += 1
        elif len(set(complete_tuples)) == 1:
            passes, reason = 0, f"unanimous_{complete_tuples[0]}"
            n_unanim += 1
            unanim_breakdown[complete_tuples[0]] += 1
        elif n_complete == len(PHILS):
            passes, reason = 1, "passes"
            n_pass += 1
        else:
            passes, reason = 1, "partial_passes"
            n_partial_pass += 1

        row = {
            "scenario_id": sid,
            "passes_filter": passes,
            "reason": reason,
            "n_complete_phils": n_complete,
            "incomplete_phils": ";".join(incomplete_phils),
            "n_distinct_tuples": len(set(complete_tuples)) if complete_tuples else 0,
        }
        for pid in PHILS:
            row[f"{pid}_A"] = cell_modal[(pid, "A")] or ""
            row[f"{pid}_B"] = cell_modal[(pid, "B")] or ""
            row[f"tuple_{pid}"] = phil_tuples[pid]
        rows.append(row)

    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    total = len(sids)
    print()
    print(f"scenarios              {total:>5}")
    pct = lambda n: (n / total * 100) if total else 0.0
    print(f"  pass (full)          {n_pass:>5}  ({pct(n_pass):.1f}%)")
    print(f"  pass (partial)       {n_partial_pass:>5}  ({pct(n_partial_pass):.1f}%)")
    print(f"  fail (unanimous)     {n_unanim:>5}  ({pct(n_unanim):.1f}%)")
    for tup, c in unanim_breakdown.most_common():
        print(f"    {tup:<14}     {c:>5}")
    print(f"  fail (incomplete)    {n_incomplete:>5}  ({pct(n_incomplete):.1f}%)")
    print(f"  fail (no_panel_data) {n_no_data:>5}  ({pct(n_no_data):.1f}%)")

    if n_error_records or unknown_phils or unknown_frames:
        print()
        print("panel-shape diagnostics:")
        if n_error_records:
            print(f"  records with error: {n_error_records} (skipped)")
        for pid, c in unknown_phils.most_common():
            print(f"  WARN unknown philosophy_id={pid!r}  {c} records (skipped)")
        for fr, c in unknown_frames.most_common():
            print(f"  WARN unknown frame={fr!r}  {c} records (skipped)")


if __name__ == "__main__":
    main()
