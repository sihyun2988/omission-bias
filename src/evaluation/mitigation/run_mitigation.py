"""E7 / RQ4 mitigation runner + scenario-manifest builder.

Mirrors `philosophy_panel/run.py` (ThreadPoolExecutor + write_lock + resumable
JSONL). Analysis / resume unit = (scenario_id, model, condition): one record
carries the whole cross-frame tuple, so M2 (one dual-frame call) and M4 (20-call
MAD) resume at the same granularity as the single-call conditions.

Two modes
---------
--build-manifest   FREE. From the filtered E1 run (`eval_tuples.jsonl`) + E2
                   `per_cell_OBR.csv` + Stage-2 labels, pick each target
                   model's top-3 OBR conflict cells, take the M0=NN scenarios
                   in them first, top up with a small non-NN false-positive
                   control slice, cap ≤ max-per-model unique scenarios, and
                   freeze the selection + SHA-256 into config.yaml (prereg,
                   supplement §130/§205).

(run)              BILLED. Reads config.yaml and runs the selected conditions
                   × models × scenarios. Default conditions = the §159 pilot
                   set M0,M1,M3b,M4. `--dry-run` prints the call budget and
                   exits WITHOUT any API call.

Build the manifest (free):
    .venv/bin/python3 -m src.evaluation.mitigation.run_mitigation \\
        --build-manifest \\
        --eval outputs/experiments/0517/1809/E1_overall_OBR/eval_tuples.jsonl \\
        --per-cell outputs/experiments/0517/1809/E2_OBR_by_conflict/per_cell_OBR.csv \\
        --labels data/panel_outputs/labels_openrouter_openai_gpt-4.1-mini.jsonl

Pilot run (10 scenarios/model, pilot conditions), cost only:
    .venv/bin/python3 -m src.evaluation.mitigation.run_mitigation \\
        --config outputs/experiments/E7_mitigation/config.yaml \\
        --limit 10 --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from src.shared.llm import load_env, make_caller
from src.evaluation.mitigation.conditions import CONDITIONS
from src.evaluation.runners.eval_model import _well_formed
from src.analysis.obr_by_conflict import (
    load_eval, load_labels, scenario_cell_membership, cell_key,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Target models = top-3 OBR on the 218 canonical (0518/1715). The
# low-OBR gemini control (supplement §130) is DROPPED 2026-05-18: it had
# only 3 M0=NN scenarios so the mitigation arm was uninterpretable
# (no NN to repair → zero information, not a usable control).
TARGET_MODELS = (
    "meta-llama/llama-3.1-8b-instruct",   # OBR 0.56
    "google/gemma-3-12b-it",              # OBR 0.55
    "openai/gpt-4o-mini",                 # OBR 0.45
    "qwen/qwen3.5-9b",                    # OBR 0.26
    "google/gemini-2.0-flash-001",        # OBR 0.07 (low; few M0=NN)
)
PILOT_CONDITIONS = ("M0", "M1", "M2", "M3", "M3b")  # M4/MAD dropped 2026-05-18


# ----------------------------------------------------- manifest builder ---
def _load_frames(path: Path) -> dict[str, dict]:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            sid = r.get("scenario_id", "?")
            # PARADIGM_MISFIT hand-exclusion DROPPED 2026-05-18 (benchmark
            # v1 = 218 incl. H_001; E1 also 218). Only structural
            # well-formedness still filters → E7 scope = full 218.
            if not _well_formed(r):
                continue
            out[sid] = r
    return out


def build_manifest(args) -> Path:
    eval_by_model, _ = load_eval(Path(args.eval))
    labels = load_labels(Path(args.labels))
    membership = scenario_cell_membership(labels)  # sid -> {cell_key}
    frames = _load_frames(Path(args.paired_frames))

    # E2 per-cell OBR: (model, yn, ny) -> {OBR, n, eligible}
    import csv
    cell_obr: dict[tuple, dict] = {}
    with open(args.per_cell) as f:
        for row in csv.DictReader(f):
            cell_obr[(row["model"], row["yn_phil"], row["ny_phil"])] = {
                "OBR": float(row["OBR"]) if row["OBR"] else 0.0,
                "n": int(row["n_scenarios"]),
                "eligible": row.get("spread_eligible", "1") == "1",
            }

    # FULL scope (2026-05-18): no cell/NN sub-selection. Every model runs
    # on ALL labeled benchmark scenarios → removes hotspot stratification
    # bias + gives full non-NN harm coverage (Codex review). M0=E1 defines
    # NN; the credited metric still conditions on M0=NN at analysis time.
    labeled_all = sorted(
        sid for sid in frames
        if labels.get(sid, {}).get("status") == "labeled")

    models_out = {}
    for model in TARGET_MODELS:
        esid = eval_by_model.get(model, {})
        if getattr(args, "all_scenarios", False):
            chosen = labeled_all
            top = []
        else:
            cells = sorted(
                ((c, v) for c, v in cell_obr.items() if c[0] == model),
                key=lambda kv: (kv[1]["eligible"], kv[1]["OBR"], kv[1]["n"]),
                reverse=True,
            )[:args.top_cells]
            top = [{"yn": c[1], "ny": c[2],
                    "OBR": round(v["OBR"], 4), "n": v["n"]} for c, v in cells]
            keys = {cell_key((c[1], c[2])) for c, _ in cells}
            pool = sorted(sid for sid, cks in membership.items()
                          if sid in frames and cks & keys)
            nn = [s for s in pool if esid.get(s) == "NN"]
            non_nn = [s for s in pool if esid.get(s) in ("YY", "YN", "NY")]
            cap = args.max_per_model
            n_ctrl = min(len(non_nn),
                         int(round(cap * args.nonnn_control_frac)))
            chosen = nn[:cap - n_ctrl] + non_nn[:n_ctrl]
            chosen = sorted(dict.fromkeys(chosen))[:cap]
        models_out[model] = {
            "top_cells": top,
            "n_NN": sum(1 for s in chosen if esid.get(s) == "NN"),
            "n_control": sum(1 for s in chosen if esid.get(s) != "NN"),
            "scenarios": chosen,
        }

    canon = json.dumps({m: v["scenarios"] for m, v in models_out.items()},
                        sort_keys=True).encode()
    sha = hashlib.sha256(canon).hexdigest()

    out_dir = PROJECT_ROOT / "outputs" / "experiments" / "E7_mitigation"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = out_dir / "config.yaml"
    _write_yaml(cfg, {
        "created": datetime.now().isoformat(timespec="seconds"),
        "e1_filtered": args.eval,
        "e2_per_cell": args.per_cell,
        "labels": args.labels,
        "paired_frames": args.paired_frames,
        "conditions": list(CONDITIONS),
        "max_per_model": args.max_per_model,
        "nonNN_control_frac": args.nonnn_control_frac,
        "top_cells_per_model": args.top_cells,
        "seed": args.seed,
        "manifest_sha256": sha,
        "models": models_out,
    })
    for m, v in models_out.items():
        print(f"{m:<34} cells={len(v['top_cells'])} "
              f"NN={v['n_NN']} control={v['n_control']} "
              f"total={len(v['scenarios'])}")
    print(f"manifest_sha256 {sha}")
    print(f"config          {cfg}")
    return cfg


def _write_yaml(path: Path, obj: dict):
    """Minimal hand-rolled YAML emitter (pyyaml not in venv). Only the
    structures this manifest uses: scalars, str lists, list-of-dict, nested
    one level of model dicts."""
    def esc(v):
        if isinstance(v, str):
            return v if (v and all(c.isalnum() or c in "._/:+- T" for c in v)
                         and not v[0].isdigit()) else json.dumps(v)
        if isinstance(v, bool):
            return "true" if v else "false"
        return json.dumps(v)

    lines = ["# E7 / RQ4 mitigation manifest — FROZEN prereg (supplement §205)"]
    for k, val in obj.items():
        if k == "models":
            lines.append("models:")
            for m, mv in val.items():
                lines.append(f"  {json.dumps(m)}:")
                lines.append("    top_cells:")
                for c in mv["top_cells"]:
                    lines.append(
                        f"      - {{yn: {c['yn']}, ny: {c['ny']}, "
                        f"OBR: {c['OBR']}, n: {c['n']}}}")
                lines.append(f"    n_NN: {mv['n_NN']}")
                lines.append(f"    n_control: {mv['n_control']}")
                ids = ", ".join(mv["scenarios"])
                lines.append(f"    scenarios: [{ids}]")
        elif isinstance(val, list):
            lines.append(f"{k}: [{', '.join(map(str, val))}]")
        else:
            lines.append(f"{k}: {esc(val)}")
    path.write_text("\n".join(lines) + "\n")


def load_manifest(path: Path) -> dict:
    """Tiny targeted parser for the manifest this module writes.

    Structure is fixed: top-level `key: value` (no indent), a `models:` block,
    each model a 2-space quoted header `  "id":`, with a 4-space
    `scenarios: [..]` line (top_cells / n_* lines are ignored — derived)."""
    cfg: dict = {"models": {}}
    in_models = False
    cur = None
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw.startswith("models:"):
            in_models = True
            continue
        if not in_models:
            if ":" in raw and not raw.startswith(" "):
                k, _, v = raw.partition(":")
                cfg[k.strip()] = v.strip()
            continue
        # 2-space quoted model header (unambiguous: only model keys look like this)
        if raw.startswith('  "') and raw.rstrip().endswith(":"):
            cur = json.loads(raw.strip()[:-1])
            cfg["models"][cur] = {"scenarios": []}
            continue
        if cur is not None and raw.strip().startswith("scenarios:"):
            inside = raw.split("[", 1)[1].rsplit("]", 1)[0].strip()
            cfg["models"][cur]["scenarios"] = (
                [s.strip() for s in inside.split(",") if s.strip()]
                if inside else [])
    return cfg


# -------------------------------------------------------------- runner ---
def load_done(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            done.add((r["scenario_id"], r["model"], r["condition"]))
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build-manifest", action="store_true")
    p.add_argument("--config", default=str(
        PROJECT_ROOT / "outputs" / "experiments" / "E7_mitigation"
        / "config.yaml"))
    # manifest-build inputs
    p.add_argument("--eval", default=str(
        PROJECT_ROOT / "outputs" / "experiments" / "0517" / "1809"
        / "E1_overall_OBR" / "eval_tuples.jsonl"))
    p.add_argument("--per-cell", default=str(
        PROJECT_ROOT / "outputs" / "experiments" / "0517" / "1809"
        / "E2_OBR_by_conflict" / "per_cell_OBR.csv"))
    p.add_argument("--labels", default=str(
        PROJECT_ROOT / "data" / "panel_outputs"
        / "labels_openrouter_openai_gpt-4.1-mini.jsonl"))
    p.add_argument("--paired-frames", default=str(
        PROJECT_ROOT / "data" / "constructed" / "mirror_frames"
        / "paired_frames.jsonl"))
    p.add_argument("--all-scenarios", action="store_true",
                   help="FULL scope: every model runs on ALL labeled "
                        "benchmark scenarios (no top-cell/NN sub-select). "
                        "2026-05-18 design.")
    p.add_argument("--max-per-model", type=int, default=120)
    p.add_argument("--nonnn-control-frac", type=float, default=0.15)
    p.add_argument("--top-cells", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    # run inputs
    p.add_argument("--provider", default="openrouter")
    p.add_argument("--conditions", default=",".join(PILOT_CONDITIONS),
                   help="comma list; default = §159 pilot M0,M1,M3b,M4")
    p.add_argument("--models", default=None,
                   help="comma list; default = all in manifest")
    p.add_argument("--limit", type=int, default=None,
                   help="first N scenarios per model (smoke / pilot)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--dry-run", action="store_true",
                   help="print call budget and exit; NO API calls")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.build_manifest:
        build_manifest(args)
        return

    cfg = load_manifest(Path(args.config))
    frames = _load_frames(Path(args.paired_frames))
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conds:
        if c not in CONDITIONS:
            raise SystemExit(f"unknown condition {c}; have {list(CONDITIONS)}")
    models = ([m.strip() for m in args.models.split(",")]
              if args.models else list(cfg["models"]))

    out = Path(args.out) if args.out else (
        PROJECT_ROOT / "outputs" / "experiments" / "E7_mitigation"
        / "raw" / "tuples.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)

    items = []
    budget = 0
    for model in models:
        sids = cfg["models"][model]["scenarios"]
        if args.limit:
            sids = sids[:args.limit]
        for sid in sids:
            if sid not in frames:
                continue
            for c in conds:
                budget += CONDITIONS[c][1]
                if (sid, model, c) not in done:
                    items.append((frames[sid], model, c))

    print(f"models      {len(models)}  ({', '.join(models)})")
    print(f"conditions  {conds}")
    print(f"scenarios   "
          + ", ".join(f"{m.split('/')[-1]}="
                      f"{min(len(cfg['models'][m]['scenarios']), args.limit) if args.limit else len(cfg['models'][m]['scenarios'])}"
                      for m in models))
    print(f"call budget {budget} total LLM calls "
          f"(todo records={len(items)}, resume={'on' if done else 'fresh'})")
    if args.dry_run:
        print("DRY RUN — no API calls made. Re-run without --dry-run to "
              "execute (BILLED).")
        return

    load_env()
    caller, _ = make_caller(args.provider)
    counts = Counter()

    def process(item):
        scenario, model, cond = item
        runner = CONDITIONS[cond][0]
        t0 = time.time()
        try:
            res = runner(caller, model, scenario,
                         args.temperature, args.max_tokens)
            err = None
        except Exception as e:
            res = {"tuple": None, "answer_A": None, "answer_B": None,
                   "raw": {}, "refusal": False, "parse_ok": False,
                   "transcript": None}
            err = f"{type(e).__name__}: {e}"
        return {
            "scenario_id": scenario["scenario_id"],
            "model": model, "condition": cond,
            "tuple": res["tuple"],
            "answer_A": res["answer_A"], "answer_B": res["answer_B"],
            "refusal": res["refusal"], "parse_ok": res["parse_ok"],
            "raw": res["raw"], "transcript": res["transcript"],
            "error": err, "temperature": args.temperature,
            "elapsed_s": round(time.time() - t0, 2),
        }

    write_lock = threading.Lock()
    bar = tqdm(total=len(items), desc="mitigation", ncols=110) if tqdm else None
    with open(out, "a") as f, \
            ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for fut in as_completed([pool.submit(process, it) for it in items]):
            rec = fut.result()
            tag = (rec["tuple"] or ("ERR" if rec["error"]
                   else "REFUSE" if rec["refusal"] else "PARSE"))
            counts[tag] += 1
            with write_lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            if bar:
                bar.set_postfix(dict(counts), refresh=False)
                bar.update(1)
                if tag in ("ERR", "PARSE", "REFUSE"):
                    bar.write(f"[{tag}] {rec['scenario_id']} {rec['model']}"
                              f" {rec['condition']}: "
                              f"{(rec['error'] or '')[:100]}")
    if bar:
        bar.close()
    print(f"done. counts={dict(counts)}  output → {out}")


if __name__ == "__main__":
    main()
