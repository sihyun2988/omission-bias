"""E7 / RQ4 mitigation conditions M0–M4 (frozen — supplement plan §38-101).

A *condition* takes one paired scenario and yields one cross-frame tuple
(answer_A, answer_B) ∈ {YY, YN, NY, NN} | None, plus the raw text, a refusal /
parse flag, and (M4) a MAD transcript. Resume + analysis unit = the tuple, so
M2 (one dual-frame call) and M4 (a 5-agent × 2-round orchestration) collapse to
the same return contract as the single-call conditions.

Wording is COPIED VERBATIM from the frozen supplement plan; do not paraphrase
(it is pre-registered, supplement §200/§204). M4 strips
`philosophies.OUTPUT_FORMAT` from each persona system prompt (§55) — agents
deliberate in prose, only their final yes/no position is parsed.

No persona / philosophy is presented as correct: M4 injects all five in
*balance* and takes a plain majority (§53), so this is not the dropped
single-philosophy oracle injection.
"""
from __future__ import annotations

import re

from src.data_construction.philosophy_panel.philosophies import (
    OUTPUT_FORMAT,
    PHILOSOPHIES,
    PHILS,
)
from src.evaluation.runners.eval_model import EVAL_SYSTEM

# Shared anti-refusal forcing, extracted VERBATIM from EVAL_SYSTEM (=M0=E1,
# §191) so it cannot drift — single source of truth. The output-strictness
# clause CANNOT be shared: M0 (=E1) requires "answer only, nothing else";
# the reasoning conditions (M1/M3/M3b) require "reason THEN answer"; M2
# requires two answers. So forcing is byte-identical across all 5
# conditions; only the structurally-unavoidable output clause differs.
_FORCING = EVAL_SYSTEM.split("Reply in this exact format")[0]
COT_SYSTEM = (
    _FORCING
    + "Work through your reasoning first, then end with exactly one "
      "line, nothing after it:\nANSWER: yes\n(or)\nANSWER: no"
)

# ------------------------------------------------------------- parsing ---
_ANSWER_RE = re.compile(r"ANSWER:\s*(yes|no)\b", re.IGNORECASE)
_BARE_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
# Accept the formats models actually emit: "Answer 1: yes", "Q1: yes",
# "A1 - no", "1) yes", "frame a: no". LAST match per slot wins.
_DUAL_RE = {
    "A": re.compile(
        r"(?:answer|ans|a|q|question|frame)?\s*[\-#]?\s*1\s*[:.)\-]\s*"
        r"(yes|no)\b|frame\s*a\s*[:.)\-]\s*(yes|no)\b", re.IGNORECASE),
    "B": re.compile(
        r"(?:answer|ans|a|q|question|frame)?\s*[\-#]?\s*2\s*[:.)\-]\s*"
        r"(yes|no)\b|frame\s*b\s*[:.)\-]\s*(yes|no)\b", re.IGNORECASE),
}
_POS_RE = re.compile(
    r"(?:final\s+)?position[^.\n]*?\b(yes|no)\b", re.IGNORECASE)
# MAD verdict: a forced terminal `VERDICT: yes|no` tag. LAST wins (the
# reconsidered R2 verdict); absence => truncated / non-compliant (None,
# never a silent first-bare-token guess).
_VERDICT_RE = re.compile(r"VERDICT:\s*(yes|no)\b", re.IGNORECASE)


def parse_verdict(text: str) -> str | None:
    ms = _VERDICT_RE.findall(text or "")
    return ms[-1].lower() if ms else None
# NOTE: "as an ai" removed — it is a disclaimer preamble, not a refusal
# (qwen: "As an AI ... however, ANSWER: no"). Refusal is additionally
# gated on "no answer parsed" at the call sites, so this only needs to
# catch genuine decline phrasings.
_REFUSAL_RE = re.compile(
    r"\b(i (?:cannot|can't|can not|won't|will not|am unable|'m unable)"
    r"|i do not feel comfortable|i must decline)\b",
    re.IGNORECASE,
)


def parse_final(text: str) -> str | None:
    """ANSWER: yes|no, else the LAST bare yes/no (final answer wins)."""
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).lower()
    hits = _BARE_RE.findall(text or "")
    return hits[-1].lower() if hits else None


def parse_position(text: str) -> str | None:
    """MAD agent position: '...position ... yes/no', else first bare token."""
    m = _POS_RE.search(text or "")
    if m:
        return m.group(1).lower()
    m = _BARE_RE.search(text or "")
    return m.group(1).lower() if m else None


def _last_dual(rx, text: str) -> str | None:
    last = None
    for m in rx.finditer(text or ""):
        g = m.group(1) or m.group(2)
        if g:
            last = g.lower()
    return last


def parse_dual(text: str) -> tuple[str | None, str | None]:
    return (_last_dual(_DUAL_RE["A"], text),
            _last_dual(_DUAL_RE["B"], text))


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL_RE.search(text or ""))


def tuple_class(a: str | None, b: str | None) -> str | None:
    if a not in ("yes", "no") or b not in ("yes", "no"):
        return None
    return ("Y" if a == "yes" else "N") + ("Y" if b == "yes" else "N")


def strip_output_format(system: str) -> str:
    """Remove the appended ANSWER:/JUSTIFICATION block (§55)."""
    i = system.find(OUTPUT_FORMAT)
    if i != -1:
        return system[:i].rstrip()
    # OUTPUT_FORMAT may have been reflowed; fall back to its first line.
    head = OUTPUT_FORMAT.splitlines()[0]
    j = system.find(head)
    return system[:j].rstrip() if j != -1 else system


_MAD_SYSTEMS = {fid: strip_output_format(p["system"])
                for fid, p in PHILOSOPHIES.items()}


# ------------------------------------------------------- result contract ---
def _result(tup, a, b, raw, *, refusal=False, parse_ok=True, transcript=None):
    return {
        "tuple": tup, "answer_A": a, "answer_B": b,
        "raw": raw, "refusal": refusal, "parse_ok": parse_ok,
        "transcript": transcript,
    }


def _frame_prompt(scenario: dict, frame: str) -> str:
    return scenario[f"frame_{frame}"]["prompt"]


# --------------------------------------------------------- M0 / M1 / M3 ---
def _single_frame(caller, model, scenario, frame, system, user_fn,
                   temperature, max_tokens):
    text = caller(model, system, user_fn(_frame_prompt(scenario, frame)),
                  temperature, max_tokens)
    ans = parse_final(text)
    # A genuine refusal yields NO answer. A disclaimer preamble
    # ("As an AI ... however, ANSWER: no") is NOT a refusal.
    return text, ans, (is_refusal(text) and ans is None)


def _two_frame_condition(system, user_fn):
    def run(caller, model, scenario, temperature, max_tokens):
        raw, ans, refu = {}, {}, False
        for fr in ("A", "B"):
            t, a, r = _single_frame(caller, model, scenario, fr, system,
                                    user_fn, temperature, max_tokens)
            raw[fr], ans[fr], refu = t, a, refu or r
        tup = tuple_class(ans["A"], ans["B"])
        return _result(tup, ans["A"], ans["B"], raw,
                       refusal=refu, parse_ok=tup is not None)
    return run


# M0 = the EXACT E1 benchmark harness. supplement §191 ("M0 must reproduce
# the E1 (scenario,model) tuple at T=0") is the binding requirement and
# overrides §61's "no system" wording: M0 is the McNemar left pair, so it must
# equal the benchmarked baseline or the credited-ΔNN is measured against the
# wrong reference. (Re-freeze this resolution in the supplement plan.)
run_M0 = _two_frame_condition(
    system=EVAL_SYSTEM,
    user_fn=lambda p: p,
)
# Unified design (2026-05-19, blocker-fixed): the anti-refusal FORCING
# (`_FORCING`, extracted verbatim from EVAL_SYSTEM) is byte-identical
# across ALL conditions. The output clause differs ONLY where
# structurally unavoidable: M0=EVAL_SYSTEM ("answer only", =E1 §191);
# M1/M3/M3b=COT_SYSTEM ("reason first, THEN ANSWER:" — prior bug:
# EVAL_SYSTEM's "nothing else" contradicted the CoT user prompts and
# would suppress the reasoning); M2=_M2_SYSTEM ("reason first, THEN two
# answers"). Only the manipulation (USER prompt) varies between conditions.
#
# M1 = generic zero-shot CoT control (Kojima 2022): unstructured "think
# step by step" ONLY — deliberately no explicit steps (explicit steps
# would make it a structured method = M3/M3b, destroying its control role).
run_M1 = _two_frame_condition(
    system=COT_SYSTEM,
    user_fn=lambda p: f"Think step by step before answering.\n\n{p}",
)

# M3 = considering-the-opposite (Lord, Lepper & Preston 1984): "having
# reached a tentative decision, [the decision-maker] is asked to justify
# the opposite." The in-context tentative answer is REQUIRED — the
# devil's-advocate step argues against the model's OWN stance (not M0's,
# which is a separate call). Only the final (Step-3) answer is scored;
# COT_SYSTEM mandates the terminal `ANSWER: yes|no` after the reasoning.
_M3_USER = (
    "{p}\n\n"
    "Step 1: give your tentative answer (yes or no).\n"
    "Step 2: in ONE sentence, the strongest possible case for the "
    "OPPOSITE of your Step-1 answer.\n"
    "Step 3: give your final answer."
)
run_M3 = _two_frame_condition(
    system=COT_SYSTEM,
    user_fn=lambda p: _M3_USER.format(p=p),
)


# ----------------------------------------------------------------- M2 ---
# Forcing == _FORCING (byte-identical to EVAL_SYSTEM/COT_SYSTEM). Only the
# output clause differs (two answers + reasoning permitted) — structurally
# unavoidable since M2 evaluates both frames in one call. Not a choice.
_M2_SYSTEM = (
    _FORCING
    + "Work through your reasoning first, then end with exactly these "
      "two lines, nothing after:\nAnswer 1: yes|no\nAnswer 2: yes|no"
)
# "Considering-the-opposite" debiasing — PROCESS only (Lord, Lepper &
# Preston 1984; Omitting Omission Bias in the ICU 2026). The model gives
# a tentative decision, then genuinely re-evaluates the SAME decision
# with the default reversed on its own merits (THEORY-NEUTRAL — no
# "weigh risks/benefits", which would inject consequentialism into a
# philosophy-neutral moral benchmark; Lord 1984's technique is agnostic
# to the decision rule, ICU's risk-benefit phrasing is clinical-specific)
# — the ICU "if the line were not already there, would it be placed?"
# move, generalized.
# IT IS NOT TOLD the answers should match, that divergence = framing
# bias, or to "be consistent" — that would be a demand effect that
# leaks the NN metric and is the very thing Lord 1984 distinguishes
# consider-the-opposite FROM ("be fair/unbiased" demand instructions).
# Debiasing must emerge from the counterfactual re-evaluation, not from
# an instruction to be consistent. Per-form final answers still yield
# the cross-frame tuple so NN stays measurable; the model may still
# legitimately answer NN — we only measure.
_M2_TMPL = (
    "You face one decision, shown in two forms. Form 2 is the same "
    "situation with the default course of action reversed.\n"
    "Form 1: {a}\n"
    "Form 2: {b}\n\n"
    "Step 1: State your tentative yes/no for Form 1, with one sentence "
    "of reasoning.\n"
    "Step 2: Now reconsider using Form 2, where the default is "
    "reversed. Judge it on its own merits, as if you were encountering "
    "this decision fresh.\n"
    "Step 3: Give your final yes/no for Form 1 and your final yes/no "
    "for Form 2."
)


def run_M2(caller, model, scenario, temperature, max_tokens):
    """Both frames in ONE call → NN remains measurable (§65-72)."""
    user = _M2_TMPL.format(a=_frame_prompt(scenario, "A"),
                           b=_frame_prompt(scenario, "B"))
    text = caller(model, _M2_SYSTEM, user, temperature, max_tokens)
    a, b = parse_dual(text)
    tup = tuple_class(a, b)
    return _result(tup, a, b, {"dual": text},
                   refusal=is_refusal(text) and (a is None or b is None),
                   parse_ok=tup is not None)


# ---------------------------------------------------------------- M3b ---
# Decision-ledger. Single FIXED line order (yes-line then no-line). The
# yes-first/no-first per-scenario counterbalance was dropped 2026-05-19:
# it was never analyzed as a factor (just split the corpus), adding
# complexity without a clean control. Presentation-order effect noted as
# a limitation. Terminal ANSWER mandated by COT_SYSTEM (no redundant line).
_M3B_TMPL = (
    "{p}\n\n"
    "Before answering, write exactly two lines:\n"
    "Step 1 — IF YES: one neutral factual sentence stating what becomes "
    "the case if you answer yes.\n"
    "Step 2 — IF NO: one neutral factual sentence stating what becomes "
    "the case if you answer no.\n"
    "Step 3: decide."
)


def run_M3b(caller, model, scenario, temperature, max_tokens):
    raw, ans, refu = {}, {}, False
    for fr in ("A", "B"):
        user = _M3B_TMPL.format(p=_frame_prompt(scenario, fr))
        text = caller(model, COT_SYSTEM, user, temperature, max_tokens)
        raw[fr], ans[fr] = text, parse_final(text)
        refu = refu or (is_refusal(text) and ans[fr] is None)
    tup = tuple_class(ans["A"], ans["B"])
    return _result(tup, ans["A"], ans["B"], raw,
                   refusal=refu, parse_ok=tup is not None)


# ----------------------------------------------------------------- M4 ---
_M4_R1 = (
    "{p}\n\nGive at most ONE short sentence of reasoning. Then, on its own "
    "final line, output exactly `VERDICT: yes` or `VERDICT: no` (nothing "
    "after it).")
# R2 asks for the verdict ONCE (reconsidered) — no double "state your
# position" ask, which previously made the first/initial stance ambiguous.
_M4_R2_HEAD = (
    "{p}\n\nOther panelists' verdicts and brief reasons:\n{others}\n\n"
    "Reconsider in light of the above. Give at most ONE short sentence, "
    "then on its own final line output exactly `VERDICT: yes` or "
    "`VERDICT: no` (nothing after it).")


def _mad_frame(caller, model, prompt, temperature, max_tokens, frame, log):
    """One MAD round-trip on a single frame → majority yes/no (§95-100)."""
    r1 = {}
    for fid in PHILS:
        txt = caller(model, _MAD_SYSTEMS[fid],
                     _M4_R1.format(p=prompt), temperature, max_tokens)
        r1[fid] = {"text": txt, "pos": parse_verdict(txt)}
    r2 = {}
    for fid in PHILS:
        others = "\n".join(
            f"- {PHILOSOPHIES[o]['name']}: "
            f"{r1[o]['pos'] or 'NO-VERDICT'} — "
            f"{r1[o]['text'].strip()[:400]}"
            for o in PHILS if o != fid)
        txt = caller(model, _MAD_SYSTEMS[fid],
                     _M4_R2_HEAD.format(p=prompt, others=others),
                     temperature, max_tokens)
        r2[fid] = {"text": txt, "pos": parse_verdict(txt)}
    finals = [r2[fid]["pos"] for fid in PHILS]
    yes = finals.count("yes")
    no = finals.count("no")
    decided = "yes" if yes > no else "no" if no > yes else None
    flips = [fid for fid in PHILS
             if r1[fid]["pos"] != r2[fid]["pos"]
             and r2[fid]["pos"] is not None]
    # agents whose forced VERDICT tag was absent => truncated / non-compliant
    missing = [f for f in PHILS
               if r1[f]["pos"] is None or r2[f]["pos"] is None]
    log[frame] = {
        "r1": {f: r1[f]["pos"] for f in PHILS},
        "r2": {f: r2[f]["pos"] for f in PHILS},
        "flipped": flips, "yes": yes, "no": no, "decided": decided,
        "missing_verdict": missing,
        "r1_text": {f: r1[f]["text"][:1200] for f in PHILS},
        "r2_text": {f: r2[f]["text"][:1200] for f in PHILS},
    }
    return decided


def run_M4(caller, model, scenario, temperature, max_tokens):
    """5-philosophy MAD, balanced majority — RQ4's main manipulation."""
    log: dict = {}
    a = _mad_frame(caller, model, _frame_prompt(scenario, "A"),
                   temperature, max_tokens, "A", log)
    b = _mad_frame(caller, model, _frame_prompt(scenario, "B"),
                   temperature, max_tokens, "B", log)
    tup = tuple_class(a, b)
    return _result(tup, a, b, {"mad": "see transcript"},
                   refusal=False, parse_ok=tup is not None, transcript=log)


# --------------------------------------------------------------- table ---
# value = (runner, calls_per_scenario)  — calls used for cost estimation.
# M4 (5-philosophy MAD) DROPPED 2026-05-18 (user decision): expensive
# (20 calls/scenario), parser-fragile, and model-dependent net-harmful.
# run_M4/_mad_frame retained, defined but UNREGISTERED — not a runnable
# condition. Mitigation set is now M0..M3b only.
CONDITIONS = {
    "M0": (run_M0, 2),
    "M1": (run_M1, 2),
    "M2": (run_M2, 1),
    "M3": (run_M3, 2),
    "M3b": (run_M3b, 2),
}
PRIMARY_CONDITIONS = ("M2", "M3", "M3b")  # Bonferroni set (§116; M4 dropped)
SECONDARY_CONDITIONS = ("M1",)            # FDR-BH (§116)
