"""One-off helper: pull the last N regenerated records from paired_frames.jsonl
and emit an English+Korean side-by-side text file for human review."""
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

N = 39
SRC = PROJECT_ROOT / "data" / "constructed" / "mirror_frames" / "paired_frames.jsonl"
OUT = PROJECT_ROOT / "data" / "constructed" / "mirror_frames" / "regenerated_39_kr.txt"

SYSTEM = (
    "You translate research scenario prompts from English into natural, "
    "fluent Korean. Preserve the meaning exactly. The English phrase "
    "'Answer yes or no.' at the end of each FRAME prompt MUST be translated "
    "as 'yes 또는 no로 답하세요.' (literal, do not paraphrase). Keep proper "
    "nouns / English-only terms (yes, no) untranslated where they appear as "
    "answer tokens. Reply with a single JSON object and nothing else."
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


def translate(caller, model, r: dict) -> tuple[str, dict | None, str | None]:
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
    except Exception as e:
        return r["scenario_id"], None, f"{type(e).__name__}: {e}"
    try:
        parsed = json.loads(strip_json_fence(text))
        return r["scenario_id"], parsed, None
    except json.JSONDecodeError as e:
        return r["scenario_id"], None, f"json_parse_fail: {e}"


def main():
    load_env()
    caller, _ = make_caller("openrouter")
    model = "openai/gpt-4.1-mini"

    recs = [json.loads(l) for l in SRC.open() if l.strip()]
    targets = recs[-N:]
    print(f"translating last {len(targets)} records ({targets[0]['scenario_id']} ... {targets[-1]['scenario_id']})")

    translations: dict[str, dict] = {}
    errors: dict[str, str] = {}
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=8) as pool, \
         tqdm(total=len(targets), desc="translate", ncols=110) as bar:
        futures = [pool.submit(translate, caller, model, r) for r in targets]
        for fut in as_completed(futures):
            sid, parsed, err = fut.result()
            with lock:
                if parsed is not None:
                    translations[sid] = parsed
                else:
                    errors[sid] = err
            bar.update(1)

    if errors:
        print(f"WARN: {len(errors)} translation failures: {list(errors)[:5]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        f.write(f"# Regenerated scenarios (last {len(targets)} in paired_frames.jsonl)\n")
        f.write(f"# English original + Korean translation side-by-side.\n\n")
        # Preserve append-order so user can match against the regen tqdm sequence
        for r in targets:
            sid = r["scenario_id"]
            tr = translations.get(sid)
            f.write("=" * 92 + "\n")
            f.write(f"{sid}   TYPE {r.get('scenario_type','?')}\n")
            f.write("=" * 92 + "\n\n")
            f.write(f"[EN] context: {r['raw']['context']}\n")
            f.write(f"[KR] 상황    : {tr['context'] if tr else '(translation failed)'}\n\n")

            f.write("FRAME A\n")
            f.write(f"  [EN] {r['frame_A']['prompt']}\n")
            f.write(f"  [KR] {tr['frame_A_prompt'] if tr else '(translation failed)'}\n")
            f.write(f"    → yes (EN): {r['frame_A']['outcome_if_yes']}\n")
            f.write(f"    → yes (KR): {tr['frame_A_yes'] if tr else '—'}\n")
            f.write(f"    → no  (EN): {r['frame_A']['outcome_if_no']}\n")
            f.write(f"    → no  (KR): {tr['frame_A_no']  if tr else '—'}\n\n")

            f.write("FRAME B\n")
            f.write(f"  [EN] {r['frame_B']['prompt']}\n")
            f.write(f"  [KR] {tr['frame_B_prompt'] if tr else '(translation failed)'}\n")
            f.write(f"    → yes (EN): {r['frame_B']['outcome_if_yes']}\n")
            f.write(f"    → yes (KR): {tr['frame_B_yes'] if tr else '—'}\n")
            f.write(f"    → no  (EN): {r['frame_B']['outcome_if_no']}\n")
            f.write(f"    → no  (KR): {tr['frame_B_no']  if tr else '—'}\n\n")

    print(f"wrote → {OUT}")
    print(f"file size: {OUT.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    main()
