"""Cross-model comparison of the philosophy-panel conflict structure.

Runs the Stage 1 unanimity filter + Stage 2 conflict labeling logic
(`filter.py` / `label.py`) on one panel JSONL *per labeler model* and reports
how the conflict-type structure differs across models:

  * per-model label_status breakdown (labeled / one-sided / all-excluded /
    unanimous-fail / incomplete)
  * per-model conflict-pair distribution (yn_phil × ny_phil counts)
  * per-model per-philosophy tuple distribution (YY/YN/NY/NN/incomplete)
  * pairwise cross-model agreement on the *same scenarios*: label_status
    agreement %, mean conflict-set Jaccard over jointly-labeled scenarios,
    and per-philosophy Cohen's κ on the 4-class (YY/YN/NY/NN) tuple

This is the construction-time labeler-robustness check called for in
RESEARCH_PLAN_v4 §risk table ("panel 을 2 개 모델로 독립 실행 후 conflict
pair 라벨의 agreement 측정"). It does NOT compute OBR — OBR is an
evaluation-time metric over the *measured* models, not the panel labeler.

To generate a panel for another labeler model first, use run.py with an
explicit model + output, e.g.:

    /home/lsh/omission/.venv/bin/python3 -m src.data_construction.philosophy_panel.run \\
        --provider openrouter --model anthropic/claude-haiku \\
        --output data/panel_outputs/panel_openrouter_anthropic_claude-haiku.jsonl

then point this script at the resulting JSONLs:

    /home/lsh/omission/.venv/bin/python3 -m src.analysis.model_conflict_comparison \\
        --panel gpt-4.1-mini=data/panel_outputs/panel_openrouter_openai_gpt-4.1-mini.jsonl \\
        --panel claude-haiku=data/panel_outputs/panel_openrouter_anthropic_claude-haiku.jsonl \\
        --universe-from data/constructed/mirror_frames/paired_frames.jsonl

With no --panel args it globs data/panel_outputs/panel_*.jsonl and derives a
model label from each filename (prompt-variant files share a model — pass
--panel explicitly when that distinction matters).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

from src.data_construction.philosophy_panel.filter import modal, tuple_code
from src.data_construction.philosophy_panel.philosophies import PHILS, FRAMES

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TUPLE_CLASSES = ("YY", "YN", "NY", "NN")


def load_panel_cells(path: Path) -> tuple[dict, set[str], dict]:
    """Return (cells, sids_with_records, diagnostics) from a panel JSONL.

    cells maps (sid, frame, pid) -> list of yes/no answers (mirrors filter.py).
    """
    cells: dict[tuple, list] = defaultdict(list)
    sids_with_records: set[str] = set()
    diag = {"error_records": 0, "unknown_phil": Counter(), "unknown_frame": Counter()}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("error"):
                diag["error_records"] += 1
                continue
            pid, frame, sid = r.get("philosophy_id"), r.get("frame"), r.get("scenario_id")
            if pid not in PHILS:
                diag["unknown_phil"][str(pid)] += 1
                continue
            if frame not in FRAMES:
                diag["unknown_frame"][str(frame)] += 1
                continue
            cells[(sid, frame, pid)].append(r["answer"])
            sids_with_records.add(sid)
    return cells, sids_with_records, diag


def label_scenario(phil_tuples: dict[str, str]) -> dict:
    """Stage 1 (strict, all-5-complete) + Stage 2 grouping for one scenario."""
    incomplete = [p for p in PHILS if "?" in phil_tuples[p]]
    complete = [phil_tuples[p] for p in PHILS if "?" not in phil_tuples[p]]

    if len(incomplete) > 0:
        return {"status": "incomplete", "conflicts": []}
    if len(set(complete)) == 1:
        return {"status": f"unanimous_{complete[0]}", "conflicts": []}

    yn = [p for p in PHILS if phil_tuples[p] == "YN"]
    ny = [p for p in PHILS if phil_tuples[p] == "NY"]
    if yn and ny:
        return {"status": "labeled",
                "conflicts": [(a, b) for a in yn for b in ny]}
    if yn and not ny:
        return {"status": "dropped_one_sided_yn", "conflicts": []}
    if ny and not yn:
        return {"status": "dropped_one_sided_ny", "conflicts": []}
    return {"status": "dropped_all_excluded", "conflicts": []}


def analyze_model(cells: dict, sids: list[str], sids_with_records: set[str]) -> dict:
    """Per-scenario tuples + status for one model over the scenario universe."""
    per_sid_tuples: dict[str, dict[str, str]] = {}
    per_sid_status: dict[str, str] = {}
    per_sid_conflicts: dict[str, set] = {}
    status_counts: Counter[str] = Counter()
    conflict_pairs: Counter[tuple] = Counter()
    phil_tuple_dist: dict[str, Counter] = {p: Counter() for p in PHILS}

    for sid in sids:
        if sid not in sids_with_records:
            per_sid_status[sid] = "no_panel_data"
            status_counts["no_panel_data"] += 1
            continue
        phil_tuples = {}
        for p in PHILS:
            a = modal(cells.get((sid, "A", p), []))
            b = modal(cells.get((sid, "B", p), []))
            t = tuple_code(a, b)
            phil_tuples[p] = t
            phil_tuple_dist[p]["incomplete" if "?" in t else t] += 1
        per_sid_tuples[sid] = phil_tuples

        res = label_scenario(phil_tuples)
        # collapse unanimous_* into a single bucket for the summary table,
        # but keep the specific code available via per_sid_status.
        per_sid_status[sid] = res["status"]
        bucket = ("unanimous_fail" if res["status"].startswith("unanimous_")
                  else res["status"])
        status_counts[bucket] += 1
        per_sid_conflicts[sid] = set(res["conflicts"])
        for pair in res["conflicts"]:
            conflict_pairs[pair] += 1

    return {
        "per_sid_tuples": per_sid_tuples,
        "per_sid_status": per_sid_status,
        "per_sid_conflicts": per_sid_conflicts,
        "status_counts": status_counts,
        "conflict_pairs": conflict_pairs,
        "phil_tuple_dist": phil_tuple_dist,
    }


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's κ over 4-class tuple labels; None if <2 items or undefined."""
    if len(pairs) < 2:
        return None
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum((ca.get(c, 0) / n) * (cb.get(c, 0) / n) for c in TUPLE_CLASSES)
    if abs(1.0 - pe) < 1e-12:
        return 1.0 if po == 1.0 else None
    return (po - pe) / (1.0 - pe)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def parse_panel_args(panel_args: list[str] | None) -> dict[str, Path]:
    panels: dict[str, Path] = {}
    if panel_args:
        for spec in panel_args:
            if "=" not in spec:
                raise SystemExit(f"--panel must be LABEL=PATH, got: {spec!r}")
            label, path = spec.split("=", 1)
            panels[label.strip()] = Path(path.strip())
    else:
        cands = sorted((PROJECT_ROOT / "data" / "panel_outputs").glob("panel_*.jsonl"))
        if not cands:
            raise SystemExit("no panel_*.jsonl under data/panel_outputs/ and no --panel given")
        for c in cands:
            label = c.stem[len("panel_"):]
            panels[label] = c
        print(f"WARN: no --panel given; using {len(panels)} filename-derived labels "
              f"(prompt variants of one model will appear as separate labels)")
    for label, path in panels.items():
        if not path.exists():
            raise SystemExit(f"panel file for {label!r} not found: {path}")
    return panels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", action="append", metavar="LABEL=PATH",
                   help="repeatable; labeler-model label = panel JSONL path. "
                        "If omitted, globs data/panel_outputs/panel_*.jsonl.")
    p.add_argument("--universe-from", default=None,
                   help="JSONL with one record per scenario (needs 'scenario_id'). "
                        "Defines the shared scenario universe. Default: union of "
                        "scenario_ids present across all panels.")
    p.add_argument("--out-dir", default=None,
                   help="default: outputs/analysis/<date>/<time>/")
    args = p.parse_args()

    panels = parse_panel_args(args.panel)

    if args.out_dir is None:
        now = datetime.now()
        out_dir = (PROJECT_ROOT / "outputs" / "analysis"
                   / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S"))
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = {}
    union_sids: set[str] = set()
    for label, path in panels.items():
        cells, sids_with_records, diag = load_panel_cells(path)
        loaded[label] = (cells, sids_with_records, diag)
        union_sids |= sids_with_records

    if args.universe_from:
        universe: set[str] = set()
        with open(args.universe_from) as f:
            for line in f:
                r = json.loads(line)
                if "scenario_id" in r:
                    universe.add(r["scenario_id"])
        sids = sorted(universe)
    else:
        sids = sorted(union_sids)

    results = {}
    for label, (cells, swr, _) in loaded.items():
        results[label] = analyze_model(cells, sids, swr)

    # ---- per_model_summary.csv ----
    summary_cols = ["model", "n_universe", "labeled", "dropped_one_sided_yn",
                    "dropped_one_sided_ny", "dropped_all_excluded",
                    "unanimous_fail", "incomplete", "no_panel_data"]
    with open(out_dir / "per_model_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_cols)
        w.writeheader()
        for label, res in results.items():
            sc = res["status_counts"]
            w.writerow({"model": label, "n_universe": len(sids),
                        **{c: sc.get(c, 0) for c in summary_cols[2:]}})

    # ---- per_model_conflict_pairs.csv (long) ----
    with open(out_dir / "per_model_conflict_pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "yn_phil", "ny_phil", "count"])
        for label, res in results.items():
            for (a, b), c in res["conflict_pairs"].most_common():
                w.writerow([label, a, b, c])

    # ---- per_model_phil_tuples.csv ----
    with open(out_dir / "per_model_phil_tuples.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "philosophy", *TUPLE_CLASSES, "incomplete"])
        for label, res in results.items():
            for ph in PHILS:
                d = res["phil_tuple_dist"][ph]
                w.writerow([label, ph, *[d.get(c, 0) for c in TUPLE_CLASSES],
                            d.get("incomplete", 0)])

    # ---- cross_model_pairwise.csv ----
    pairwise_rows = []
    for la, lb in combinations(results.keys(), 2):
        ra, rb = results[la], results[lb]
        # status agreement over scenarios both models scored (have a tuple)
        both_scored = [s for s in sids
                       if s in ra["per_sid_tuples"] and s in rb["per_sid_tuples"]]
        status_agree = (
            sum(1 for s in both_scored
                if ra["per_sid_status"][s] == rb["per_sid_status"][s])
            / len(both_scored) if both_scored else None
        )
        labeled_both = [s for s in both_scored
                        if ra["per_sid_status"][s] == "labeled"
                        and rb["per_sid_status"][s] == "labeled"]
        mean_jac = (
            sum(jaccard(ra["per_sid_conflicts"][s], rb["per_sid_conflicts"][s])
                for s in labeled_both) / len(labeled_both)
            if labeled_both else None
        )
        row = {
            "model_a": la, "model_b": lb,
            "n_both_scored": len(both_scored),
            "status_agree_pct": round(status_agree * 100, 1) if status_agree is not None else "",
            "n_labeled_both": len(labeled_both),
            "mean_conflict_jaccard": round(mean_jac, 3) if mean_jac is not None else "",
        }
        for ph in PHILS:
            kp = [(ra["per_sid_tuples"][s][ph], rb["per_sid_tuples"][s][ph])
                  for s in both_scored
                  if ra["per_sid_tuples"][s][ph] in TUPLE_CLASSES
                  and rb["per_sid_tuples"][s][ph] in TUPLE_CLASSES]
            k = cohen_kappa(kp)
            row[f"kappa_{ph}"] = round(k, 3) if k is not None else ""
        pairwise_rows.append(row)

    if pairwise_rows:
        with open(out_dir / "cross_model_pairwise.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pairwise_rows[0].keys()))
            w.writeheader()
            w.writerows(pairwise_rows)

    # ---- console summary ----
    print(f"universe scenarios     {len(sids)}")
    print(f"models                 {len(results)}  ({', '.join(results)})")
    print(f"out                    {out_dir}")
    print()
    print(f"{'model':<28}{'labeled':>9}{'1side_yn':>9}{'1side_ny':>9}"
          f"{'allexcl':>9}{'unanim':>9}{'incompl':>9}")
    for label, res in results.items():
        sc = res["status_counts"]
        print(f"{label:<28}{sc.get('labeled',0):>9}"
              f"{sc.get('dropped_one_sided_yn',0):>9}"
              f"{sc.get('dropped_one_sided_ny',0):>9}"
              f"{sc.get('dropped_all_excluded',0):>9}"
              f"{sc.get('unanimous_fail',0):>9}"
              f"{sc.get('incomplete',0):>9}")

    for label, res in results.items():
        top = res["conflict_pairs"].most_common(8)
        if top:
            print()
            print(f"[{label}] top conflict pairs (yn vs ny):")
            for (a, b), c in top:
                print(f"  {a:<12} vs {b:<12} {c:>5}")

    if pairwise_rows:
        print()
        print("pairwise cross-model agreement:")
        for r in pairwise_rows:
            print(f"  {r['model_a']} ↔ {r['model_b']}: "
                  f"status_agree={r['status_agree_pct']}%  "
                  f"(n={r['n_both_scored']}), "
                  f"conflict Jaccard={r['mean_conflict_jaccard']} "
                  f"(n_labeled_both={r['n_labeled_both']})")
            kappas = [f"{ph}={r[f'kappa_{ph}']}" for ph in PHILS]
            print(f"      per-phil κ: {'  '.join(kappas)}")

    diag_lines = []
    for label, (_, _, diag) in loaded.items():
        if diag["error_records"] or diag["unknown_phil"] or diag["unknown_frame"]:
            diag_lines.append(f"  {label}: err={diag['error_records']} "
                              f"unk_phil={dict(diag['unknown_phil'])} "
                              f"unk_frame={dict(diag['unknown_frame'])}")
    if diag_lines:
        print()
        print("panel-shape diagnostics (skipped records):")
        print("\n".join(diag_lines))


if __name__ == "__main__":
    main()
