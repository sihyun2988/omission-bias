"""Side-by-side EN+KR dump of the 7-scenario v2 test set."""
from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.llm import load_env, make_caller  # noqa: E402

SRC = PROJECT_ROOT / "data" / "constructed" / "mirror_frames" / "paired_frames_v2_test.jsonl"
OUT = PROJECT_ROOT / "data" / "constructed" / "mirror_frames" / "paired_frames_v2_test_kr.txt"

SYSTEM = (
    "You translate research scenario prompts from English into natural, "
    "fluent Korean. Preserve the meaning exactly. The English phrase "
    "'Answer yes or no.' MUST be translated as 'yes 또는 no로 답하세요.'. "
    "Reply with a single JSON object and nothing else."
)

USER_TMPL = """Translate the following components to Korean. Output one JSON object with exactly these keys:
"context", "frame_A_prompt", "frame_A_yes", "frame_A_no", "frame_B_prompt", "frame_B_yes", "frame_B_no"

ENGLISH:
context: {context}
frame_A_prompt: {fa_p}
frame_A_yes: {fa_y}
frame_A_no: {fa_n}
frame_B_prompt: {fb_p}
frame_B_yes: {fb_y}
frame_B_no: {fb_n}"""


def strip_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]
        t = t.strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    return t


def translate(caller, model, r: dict):
    user = USER_TMPL.format(
        context=r["raw"]["context"],
        fa_p=r["frame_A"]["prompt"],
        fa_y=r["frame_A"]["outcome_if_yes"],
        fa_n=r["frame_A"]["outcome_if_no"],
        fb_p=r["frame_B"]["prompt"],
        fb_y=r["frame_B"]["outcome_if_yes"],
        fb_n=r["frame_B"]["outcome_if_no"],
    )
    try:
        text = caller(model, SYSTEM, user, 0.0, 3000)
        return r["scenario_id"], json.loads(strip_json_fence(text)), None
    except Exception as e:
        return r["scenario_id"], None, f"{type(e).__name__}: {e}"


def main():
    load_env()
    caller, _ = make_caller("openrouter")
    model = "openai/gpt-4.1-mini"

    recs = [json.loads(l) for l in SRC.open() if l.strip()]
    # Preserve the input order from the --scenarios CLI list rather than
    # whatever order the JSONL ended up in (concurrent append).
    desired = ["G_111", "G_207", "G_030", "G_326", "H_014", "G_446", "G_358"]
    by_id = {r["scenario_id"]: r for r in recs}
    targets = [by_id[sid] for sid in desired if sid in by_id]
    print(f"translating {len(targets)} records")

    translations: dict[str, dict] = {}
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=7) as pool, \
         tqdm(total=len(targets), desc="translate", ncols=110) as bar:
        futures = [pool.submit(translate, caller, model, r) for r in targets]
        for fut in as_completed(futures):
            sid, parsed, err = fut.result()
            with lock:
                if parsed is not None:
                    translations[sid] = parsed
                else:
                    print(f"  fail: {sid} → {err}")
            bar.update(1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        f.write(f"# v2 test set ({len(targets)} scenarios) — new PNAS-style SYSTEM_PROMPT\n")
        f.write(f"# Source: {SRC.name}\n\n")
        for r in targets:
            sid = r["scenario_id"]
            tr = translations.get(sid)
            f.write("=" * 92 + "\n")
            f.write(f"{sid}   TYPE {r.get('scenario_type','?')}   inaction_side={r['raw']['lexical_inaction_side']}\n")
            f.write("=" * 92 + "\n\n")
            f.write(f"[EN] context : {r['raw']['context']}\n")
            f.write(f"[KR] 상황     : {tr['context'] if tr else '(translation failed)'}\n")
            f.write(f"[EN] raw A1  : {r['raw']['action1']}\n")
            f.write(f"[EN] raw A2  : {r['raw']['action2']}\n\n")

            f.write("FRAME A\n")
            f.write(f"  [EN] {r['frame_A']['prompt']}\n")
            f.write(f"  [KR] {tr['frame_A_prompt'] if tr else '(failed)'}\n")
            f.write(f"    → yes (EN): {r['frame_A']['outcome_if_yes']}\n")
            f.write(f"    → yes (KR): {tr['frame_A_yes'] if tr else '—'}\n")
            f.write(f"    → no  (EN): {r['frame_A']['outcome_if_no']}\n")
            f.write(f"    → no  (KR): {tr['frame_A_no']  if tr else '—'}\n\n")

            f.write("FRAME B\n")
            f.write(f"  [EN] {r['frame_B']['prompt']}\n")
            f.write(f"  [KR] {tr['frame_B_prompt'] if tr else '(failed)'}\n")
            f.write(f"    → yes (EN): {r['frame_B']['outcome_if_yes']}\n")
            f.write(f"    → yes (KR): {tr['frame_B_yes'] if tr else '—'}\n")
            f.write(f"    → no  (EN): {r['frame_B']['outcome_if_no']}\n")
            f.write(f"    → no  (KR): {tr['frame_B_no']  if tr else '—'}\n\n")

    print(f"wrote → {OUT}")
    print(f"file size: {OUT.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    main()
