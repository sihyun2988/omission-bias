"""Run the 6-philosophy panel on paired action↔omission frames.

For each (scenario × philosophy × frame ∈ {A, B} × sample), call the synthesis
LLM (default gpt-4.1-mini via OpenRouter) with the philosophy system prompt and
the frame's yes/no question. Output one JSONL record per call.

Resumable: re-running skips already-recorded (scenario_id, philosophy_id, frame,
sample_idx) tuples in the output file.

Quick test on the first 5 scenarios:
    .venv/bin/python3 -m src.data_construction.philosophy_panel.run \\
        --limit 5 --n-samples 1

Full run, gpt-4.1-mini via OpenRouter:
    .venv/bin/python3 -m src.data_construction.philosophy_panel.run \\
        --provider openrouter --model openai/gpt-4.1-mini
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.shared.llm import load_env, make_caller
from src.data_construction.philosophy_panel.philosophies import (
    PHILOSOPHIES,
    USER_PROMPT_TEMPLATE,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ANSWER_RE = re.compile(r"ANSWER:\s*(yes|no)\b", re.IGNORECASE)
JUST_RE = re.compile(r"JUSTIFICATION:\s*(.+)", re.DOTALL | re.IGNORECASE)


def parse_response(text: str) -> tuple[str | None, str]:
    m = ANSWER_RE.search(text)
    if not m:
        return None, text.strip()[:500]
    ans = m.group(1).lower()
    j = JUST_RE.search(text)
    return ans, (j.group(1).strip() if j else "")


def load_done(jsonl_path: Path) -> set[tuple]:
    if not jsonl_path.exists():
        return set()
    done = set()
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            done.add((r["scenario_id"], r["philosophy_id"], r["frame"], r["sample_idx"]))
    return done


def load_paired_frames(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", default="openrouter",
                   choices=["openai", "openrouter", "vllm", "anthropic"])
    p.add_argument("--model", default="openai/gpt-4.1-mini",
                   help="default: gpt-4.1-mini via OpenRouter (per CLAUDE.md synthesis-model rule)")
    p.add_argument("--paired-frames",
                   default=str(PROJECT_ROOT / "data" / "constructed" / "mirror_frames" / "paired_frames.jsonl"))
    p.add_argument("--output", default=None,
                   help="default: data/panel_outputs/panel_<provider>_<model-slug>.jsonl")
    p.add_argument("--limit", type=int, default=None,
                   help="if set, only run on the first N scenarios in the JSONL (top-of-file order)")
    # Defaults are deterministic single-shot per project policy (2026-05-13):
    # T=0.0 makes the labeler reproducible; n=1 is sufficient when there's no
    # sampling variance. Override only if the user explicitly asks to resample.
    p.add_argument("--n-samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--concurrency", type=int, default=10)
    args = p.parse_args()

    load_env()
    caller, default_model = make_caller(args.provider)
    model = args.model or default_model

    if args.output is None:
        slug = model.replace("/", "_").replace(":", "_")
        output = PROJECT_ROOT / "data" / "panel_outputs" / f"panel_{args.provider}_{slug}.jsonl"
    else:
        output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    scenarios = load_paired_frames(Path(args.paired_frames), args.limit)
    done = load_done(output)
    frames = ["A", "B"]
    items = [
        (s, frame, phil_id, phil, sample_idx)
        for s in scenarios
        for frame in frames
        for phil_id, phil in PHILOSOPHIES.items()
        for sample_idx in range(args.n_samples)
    ]
    remaining = [
        it for it in items
        if (it[0]["scenario_id"], it[2], it[1], it[4]) not in done
    ]
    print(f"provider={args.provider}  model={model}  T={args.temperature}  "
          f"n_samples={args.n_samples}  scenarios={len(scenarios)}")
    print(f"output={output}  total={len(items)}  resume={len(items)-len(remaining)}  todo={len(remaining)}")

    counts = {"yes": 0, "no": 0, "PARSE": 0, "ERR": 0,
              "skip": len(items) - len(remaining)}

    def process(item):
        s, frame, phil_id, phil, sample_idx = item
        prompt = s[f"frame_{frame}"]["prompt"]
        user = USER_PROMPT_TEMPLATE.format(prompt=prompt)
        t0 = time.time()
        try:
            text = caller(model, phil["system"], user, args.temperature, args.max_tokens)
            ans, just = parse_response(text)
            err = None
        except Exception as e:
            text, ans, just = "", None, ""
            err = f"{type(e).__name__}: {e}"
        return {
            "scenario_id": s["scenario_id"],
            "frame": frame,
            "philosophy_id": phil_id,
            "philosophy_name": phil["name"],
            "sample_idx": sample_idx,
            "answer": ans,
            "justification": just,
            "raw": text,
            "error": err,
            "model": model,
            "provider": args.provider,
            "temperature": args.temperature,
            "elapsed_s": round(time.time() - t0, 2),
        }

    write_lock = threading.Lock()
    counts_lock = threading.Lock()
    bar = tqdm(total=len(items), initial=counts["skip"], desc="panel", ncols=110) if tqdm else None
    if bar:
        bar.set_postfix(counts, refresh=False)

    with open(output, "a") as f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(process, it) for it in remaining]
        for fut in as_completed(futures):
            rec = fut.result()
            tag = rec["answer"] or ("ERR" if rec["error"] else "PARSE")
            with counts_lock:
                counts[tag] += 1
            with write_lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            if bar:
                bar.set_postfix(counts, refresh=False)
                bar.update(1)
                if tag in ("PARSE", "ERR"):
                    snippet = (rec["error"] or rec["raw"] or "")[:120]
                    bar.write(f"[{tag}] {rec['scenario_id']} {rec['frame']} {rec['philosophy_id']} "
                              f"#{rec['sample_idx']}: {snippet}")
    if bar:
        bar.close()
    print(f"done. counts={counts}  output → {output}")


if __name__ == "__main__":
    main()
