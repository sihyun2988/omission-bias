"""Step 0 of the data pipeline — label which side (A1 or A2) of each MoralChoice
high-ambiguity scenario is the *inaction* (preserves status quo, no deliberate
physical intervention).

Pass 1 is anchored regex on the action text (high precision, covers ~56% of
the 680 high-amb rows in a quick check).  Pass 2 is an optional LLM fallback
for the ambiguous remainder.

The output is a precondition for `pilot/mirror.py`, which needs to know which
side to flip when constructing the action↔omission mirror frame.

Usage:
    python pilot/inaction_label.py                     # regex only
    python pilot/inaction_label.py --llm-fallback      # + LLM on ambiguous rows
    python pilot/inaction_label.py --llm-fallback --max-llm 50  # cap LLM calls
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))


FIELDNAMES = [
    "scenario_id", "inaction_side", "method", "confidence",
    "pattern_or_note", "action1", "action2",
]


def read_existing_csv(path: Path) -> dict[str, dict]:
    """Load a prior --out CSV into a {scenario_id: row} dict (empty if missing)."""
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {r["scenario_id"]: r for r in csv.DictReader(f)}


def write_csv_atomic(path: Path, labels: dict[str, dict]) -> None:
    """Atomically rewrite the labels CSV so interrupts can't corrupt mid-write."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for sid in sorted(labels.keys()):
            w.writerow(labels[sid])
    tmp.replace(path)


# Anchored on sentence start to keep precision high; ambiguous cases fall
# through to LLM fallback rather than being mis-tagged here.
TIER1_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^\s*I\s+(?:do\s+not|don'?t)\s+\w+", re.I),               "do_not"),
    (re.compile(r"^\s*I\s+(?:refuse|refrain\s+from|decline|"
                r"abstain\s+from|never)\b", re.I),                          "refuse"),
    (re.compile(r"^\s*I\s+do\s+nothing\b", re.I),                          "do_nothing"),
    (re.compile(r"^\s*I\s+(?:stay|remain|keep)\s+silent\b", re.I),         "stay_silent"),
    (re.compile(r"^\s*I\s+keep\s+(?:my\s+mouth\s+shut|quiet|"
                r"it\s+(?:a\s+)?secret)\b", re.I),                          "keep_silent"),
    (re.compile(r"^\s*I\s+ignore\b", re.I),                                "ignore"),
    (re.compile(r"^\s*I\s+walk\s+away\b", re.I),                           "walk_away"),
    (re.compile(r"^\s*I\s+say\s+nothing\b", re.I),                         "say_nothing"),
    (re.compile(r"^\s*I\s+let\s+\w+", re.I),                               "let"),
    (re.compile(r"^\s*I\s+leave\s+\w+\s+"
                r"(?:alone|untouched|behind|in\s+place)\b", re.I),          "leave_alone"),
    (re.compile(r"^\s*I\s+respect\s+(?:the|their|his|her)\s+"
                r"(?:wish|wishes|decision|choice)\b", re.I),                "respect_wish"),
    (re.compile(r"^\s*I\s+reject\b", re.I),                                "reject"),
]


def regex_classify(text: str) -> tuple[bool, str]:
    """Return (is_inaction_pattern, matched_pattern_name)."""
    for rx, name in TIER1_PATTERNS:
        if rx.search(text):
            return True, name
    return False, ""


# ---------- LLM fallback ----------

LLM_SYSTEM = (
    "You are an annotator helping label moral-dilemma scenarios. "
    "Given a context and two candidate actions, decide which side is the "
    "*inaction* — the side that does NOT involve deliberate physical "
    "intervention by the agent and instead preserves the status quo, lets "
    "events proceed, or refrains from acting. "
    "If both sides are active interventions (no meaningful inaction), reply "
    "NEITHER. If both sides could equally be read as inaction, reply BOTH."
)

LLM_USER_TEMPLATE = """\
SCENARIO: {context}

ACTION A1: {action1}
ACTION A2: {action2}

Reply in this exact format, nothing else:
INACTION_SIDE: A1
JUSTIFICATION: <one sentence>

(or A2 / NEITHER / BOTH)"""

LLM_CHOICE_RE = re.compile(r"INACTION_SIDE:\s*(A1|A2|NEITHER|BOTH)\b", re.I)
LLM_JUST_RE = re.compile(r"JUSTIFICATION:\s*(.+)", re.I | re.DOTALL)


def parse_llm(text: str) -> tuple[str | None, str]:
    m = LLM_CHOICE_RE.search(text)
    if not m:
        return None, text.strip()[:300]
    j = LLM_JUST_RE.search(text)
    return m.group(1).upper(), (j.group(1).strip() if j else "")


# ---------- main ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(PROJECT_ROOT / "data" / "raw" /
                                         "moralchoice" /
                                         "moralchoice_high_ambiguity.csv"))
    p.add_argument("--out", default=str(PROJECT_ROOT / "data" / "constructed" /
                                         "inaction_labels.csv"))
    p.add_argument("--llm-fallback", action="store_true",
                   help="run LLM on regex-ambiguous rows (both/neither matched)")
    p.add_argument("--provider", default="openrouter",
                   choices=["openai", "openrouter", "vllm", "anthropic"])
    p.add_argument("--model", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--max-llm", type=int, default=None,
                   help="cap LLM fallback calls (sanity-check runs)")
    args = p.parse_args()

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    print(f"loaded {len(rows)} scenarios from {args.csv}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_labels = read_existing_csv(out_path)
    if existing_labels:
        print(f"resume: loaded {len(existing_labels)} prior labels from {out_path}")

    # ---- Pass 1: regex ----
    labels: dict[str, dict] = {}
    counts = {"A1": 0, "A2": 0, "both": 0, "neither": 0}
    for r in rows:
        sid = r["scenario_id"]
        a1_neg, a1_pat = regex_classify(r["action1"])
        a2_neg, a2_pat = regex_classify(r["action2"])

        if a1_neg and not a2_neg:
            side, status, pat = "A1", "regex_tier1", a1_pat
        elif a2_neg and not a1_neg:
            side, status, pat = "A2", "regex_tier1", a2_pat
        elif a1_neg and a2_neg:
            side, status, pat = "", "regex_both", f"{a1_pat}|{a2_pat}"
        else:
            side, status, pat = "", "regex_none", ""

        labels[sid] = {
            "scenario_id": sid,
            "inaction_side": side,
            "method": status,
            "confidence": "1.0" if status == "regex_tier1" else "",
            "pattern_or_note": pat,
            "action1": r["action1"],
            "action2": r["action2"],
        }
        if status == "regex_tier1":
            counts[side] += 1
        elif status == "regex_both":
            counts["both"] += 1
        else:
            counts["neither"] += 1

    print("\nPass 1 (regex):")
    print(f"  A1 inaction: {counts['A1']:4d}  ({counts['A1']/len(rows):.1%})")
    print(f"  A2 inaction: {counts['A2']:4d}  ({counts['A2']/len(rows):.1%})")
    print(f"  both match:  {counts['both']:4d}  → ambiguous")
    print(f"  no match:    {counts['neither']:4d}  → ambiguous")

    # Port forward prior LLM results so we don't re-call the API on resume.
    # Regex always wins on hits (higher precision); we only revive prior LLM
    # results for rows that *this* run also flagged as regex-ambiguous.
    ported = 0
    for sid, lab in labels.items():
        if lab["method"] not in ("regex_both", "regex_none"):
            continue
        prev = existing_labels.get(sid)
        if not prev:
            continue
        prev_method = prev.get("method", "")
        if prev_method.startswith("llm"):
            lab["inaction_side"] = prev.get("inaction_side", "")
            lab["method"] = prev_method
            lab["confidence"] = prev.get("confidence", "")
            lab["pattern_or_note"] = prev.get("pattern_or_note", "")
            ported += 1
    if ported:
        print(f"  resume: ported {ported} prior LLM results → "
              f"only fresh ambiguous rows will be re-LLM'd")

    # Persist Pass 1 (+ ported) immediately so a Ctrl-C before Pass 2 keeps it.
    write_csv_atomic(out_path, labels)

    ambiguous = [r for r in rows
                 if labels[r["scenario_id"]]["method"] in ("regex_both",
                                                            "regex_none")]
    print(f"  ambiguous total (need LLM): {len(ambiguous)}")

    # ---- Pass 2: LLM fallback ----
    if args.llm_fallback and ambiguous:
        from run import make_caller, load_env  # noqa: E402
        load_env()
        caller, default_model = make_caller(args.provider)
        model = args.model or default_model
        targets = ambiguous if args.max_llm is None else ambiguous[:args.max_llm]
        print(f"\nPass 2 (LLM): {len(targets)} ambiguous rows  "
              f"provider={args.provider} model={model}")

        llm_counts = {"A1": 0, "A2": 0, "NEITHER": 0, "BOTH": 0,
                      "PARSE": 0, "ERR": 0}
        lock = threading.Lock()

        def process(r):
            user = LLM_USER_TEMPLATE.format(
                context=r["context"], action1=r["action1"], action2=r["action2"]
            )
            t0 = time.time()
            try:
                text = caller(model, LLM_SYSTEM, user,
                              args.temperature, args.max_tokens)
                choice, just = parse_llm(text)
                err = None
            except Exception as e:
                text, choice, just = "", None, ""
                err = f"{type(e).__name__}: {e}"
            return r["scenario_id"], choice, just, err, round(time.time()-t0, 2)

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool, \
             tqdm(total=len(targets), desc="inaction-llm", ncols=110) as bar:
            bar.set_postfix(llm_counts, refresh=False)
            futures = [pool.submit(process, r) for r in targets]
            for fut in as_completed(futures):
                sid, choice, just, err, dt = fut.result()
                with lock:
                    if err:
                        llm_counts["ERR"] += 1
                        labels[sid]["method"] = "llm_error"
                        labels[sid]["pattern_or_note"] = err[:200]
                    elif choice in ("A1", "A2"):
                        llm_counts[choice] += 1
                        labels[sid]["inaction_side"] = choice
                        labels[sid]["method"] = "llm"
                        labels[sid]["confidence"] = "0.7"
                        labels[sid]["pattern_or_note"] = just[:300]
                    elif choice in ("NEITHER", "BOTH"):
                        llm_counts[choice] += 1
                        labels[sid]["inaction_side"] = choice.lower()
                        labels[sid]["method"] = "llm"
                        labels[sid]["confidence"] = "0.7"
                        labels[sid]["pattern_or_note"] = just[:300]
                    else:
                        llm_counts["PARSE"] += 1
                        labels[sid]["method"] = "llm_parse_fail"
                        labels[sid]["pattern_or_note"] = just[:200]
                    # Persist after every completion so a Ctrl-C never costs more
                    # than the in-flight calls (cheap: ~few-hundred-row CSV).
                    write_csv_atomic(out_path, labels)
                bar.set_postfix(llm_counts, refresh=False)
                bar.update(1)

        print(f"Pass 2 result: {llm_counts}")

    # ---- Final write (idempotent with incremental writes above) ----
    write_csv_atomic(out_path, labels)
    print(f"\nwrote {len(labels)} rows → {out_path}")

    # ---- Final distribution ----
    final = {"A1": 0, "A2": 0, "neither": 0, "both": 0,
             "unresolved": 0}
    for v in labels.values():
        side = v["inaction_side"]
        if side in ("A1", "A2"):
            final[side] += 1
        elif side in ("neither", "both"):
            final[side] += 1
        else:
            final["unresolved"] += 1
    print(f"\nfinal inaction distribution: {final}")
    eligible = final["A1"] + final["A2"]
    print(f"  mirror-eligible (A1 or A2): {eligible}/{len(rows)} "
          f"= {eligible/len(rows):.1%}")


if __name__ == "__main__":
    main()
