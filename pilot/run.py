"""Pilot runner: 5 philosophies × N scenarios × N samples → JSONL.

Reads .env at project root. Resumable — re-running skips already-recorded
(scenario_id, philosophy_id, sample_idx) triples in the output file.

Quick start:
    cd <project_root>
    pip install openai python-dotenv
    cp .env.example .env  # fill in keys
    python pilot/run.py --provider openrouter

Override examples:
    python pilot/run.py --provider openai --model gpt-4o-mini
    python pilot/run.py --provider anthropic --model claude-opus-4-7
    python pilot/run.py --provider vllm --n-samples 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from philosophies import PHILOSOPHIES, USER_PROMPT_TEMPLATE  # noqa: E402
from tqdm import tqdm  # noqa: E402


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass


def load_scenarios(csv_path: Path, n: int, seed: int) -> list[dict]:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    rng = random.Random(seed)
    return rng.sample(rows, n)


CHOICE_RE = re.compile(r"CHOICE:\s*(A1|A2)\b", re.IGNORECASE)
JUST_RE = re.compile(r"JUSTIFICATION:\s*(.+)", re.DOTALL | re.IGNORECASE)


def parse_response(text: str) -> tuple[str | None, str]:
    m = CHOICE_RE.search(text)
    if not m:
        return None, text.strip()[:500]
    choice = m.group(1).upper()
    j = JUST_RE.search(text)
    return choice, (j.group(1).strip() if j else "")


# ---------- providers ----------

def _openai_compatible(base_url: str | None, api_key_env: str, default_model: str):
    """Return (caller, default_model) for any OpenAI-compatible endpoint."""
    from openai import OpenAI
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set in environment / .env")
    client = OpenAI(base_url=base_url, api_key=api_key) if base_url else OpenAI(api_key=api_key)

    def call(model: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
        # `reasoning.enabled=False` (OpenRouter convention) disables internal
        # thinking so the full token budget reaches the answer. Flip to True
        # later when measuring whether reasoning mitigates omission bias.
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"reasoning": {"enabled": False}},
        )
        return resp.choices[0].message.content or ""

    return call, default_model


def _anthropic():
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in environment / .env")
    client = anthropic.Anthropic(api_key=api_key)

    def call(model: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    return call, "claude-opus-4-7"


def make_caller(provider: str):
    if provider == "openai":
        return _openai_compatible(None, "OPENAI_API_KEY", "gpt-4o-mini")
    if provider == "openrouter":
        return _openai_compatible(
            os.environ.get("OPENROUTER_BASE_URL"),
            "OPENROUTER_API_KEY",
            os.environ.get("OPENROUTER_MODEL_NAME", "qwen/qwen3.5-9b"),
        )
    if provider == "vllm":
        return _openai_compatible(
            os.environ.get("VLLM_BASE_URL"),
            "VLLM_API_KEY",
            os.environ.get("VLLM_MODEL_NAME", "qwen/qwen3.5-9b"),
        )
    if provider == "anthropic":
        return _anthropic()
    raise ValueError(f"unknown provider: {provider}")


# ---------- main loop ----------

def load_done(jsonl_path: Path) -> set[tuple]:
    if not jsonl_path.exists():
        return set()
    done = set()
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            done.add((r["scenario_id"], r["philosophy_id"], r["sample_idx"]))
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["openai", "openrouter", "vllm", "anthropic"],
                   default="openrouter")
    p.add_argument("--model", default=None,
                   help="default depends on provider; see make_caller()")
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--concurrency", type=int, default=10,
                   help="parallel API calls; lower if gateway rate-limits")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-scenarios", type=int, default=10)
    p.add_argument("--scenarios",
                   default=str(PROJECT_ROOT / "data" / "moralchoice_high_ambiguity.csv"))
    p.add_argument("--output", default=None,
                   help="default: pilot/results/pilot_<provider>_seed<seed>.jsonl")
    args = p.parse_args()

    load_env()
    caller, default_model = make_caller(args.provider)
    model = args.model or default_model
    output = Path(args.output) if args.output else (
        ROOT / "results" / f"pilot_{args.provider}_seed{args.seed}.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    done = load_done(output)
    scenarios = load_scenarios(Path(args.scenarios), args.n_scenarios, args.seed)
    total = len(scenarios) * len(PHILOSOPHIES) * args.n_samples
    print(f"provider={args.provider}  model={model}  T={args.temperature}  "
          f"n_samples={args.n_samples}  scenarios={len(scenarios)}")
    print(f"output={output}  resume={len(done)}/{total}")

    items = [
        (s, phil_id, phil, sample_idx)
        for s in scenarios
        for phil_id, phil in PHILOSOPHIES.items()
        for sample_idx in range(args.n_samples)
    ]
    remaining = [
        it for it in items
        if (it[0]["scenario_id"], it[1], it[3]) not in done
    ]
    counts = {"A1": 0, "A2": 0, "PARSE": 0, "ERR": 0,
              "skip": len(items) - len(remaining)}

    def process(item):
        s, phil_id, phil, sample_idx = item
        user = USER_PROMPT_TEMPLATE.format(
            context=s["context"], action1=s["action1"], action2=s["action2"]
        )
        t0 = time.time()
        try:
            text = caller(model, phil["system"], user,
                          args.temperature, args.max_tokens)
            choice, just = parse_response(text)
            err = None
        except Exception as e:
            text, choice, just = "", None, ""
            err = f"{type(e).__name__}: {e}"
        rec = {
            "scenario_id": s["scenario_id"],
            "philosophy_id": phil_id,
            "philosophy_name": phil["name"],
            "sample_idx": sample_idx,
            "choice": choice,
            "justification": just,
            "raw": text,
            "error": err,
            "model": model,
            "provider": args.provider,
            "temperature": args.temperature,
            "elapsed_s": round(time.time() - t0, 2),
        }
        return rec

    write_lock = threading.Lock()
    counts_lock = threading.Lock()

    with open(output, "a") as f, \
         ThreadPoolExecutor(max_workers=args.concurrency) as pool, \
         tqdm(total=len(items), initial=counts["skip"],
              desc="pilot", ncols=110) as bar:
        bar.set_postfix(counts, refresh=False)
        futures = [pool.submit(process, it) for it in remaining]
        for fut in as_completed(futures):
            rec = fut.result()
            tag = rec["choice"] or ("ERR" if rec["error"] else "PARSE")
            with counts_lock:
                counts[tag] += 1
            with write_lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            bar.set_postfix(counts, refresh=False)
            bar.update(1)
            if tag in ("PARSE", "ERR"):
                snippet = (rec["error"] or rec["raw"] or "")[:120]
                bar.write(f"[{tag}] {rec['scenario_id']} {rec['philosophy_id']} "
                          f"#{rec['sample_idx']}: {snippet}")

    print(f"done. counts={counts}  output → {output}")


if __name__ == "__main__":
    main()
