# Session Handoff — 2026-05-19 (E7/RQ4 mitigation rebuild)

Self-contained. Read with `CLAUDE.md` + `RESEARCH_PLAN_v4.md` + supplement
(`~/.claude/plans/research-plan-v4-giggly-nebula.md`). This records a long
session that rebuilt the E7/RQ4 mitigation pipeline.

## 1. ⚠️ IN-FLIGHT: final canonical mitigation run

A FINAL clean canonical E7 run was launched (PID was 5037; likely still
running or just finished by the time you read this).

- **Output (analyze ONLY this):** `outputs/experiments/E7_mitigation/full218_clean/tuples.jsonl`
- **Log:** `/tmp/e7_final.log`  (progress: `tr '\r' '\n' < /tmp/e7_final.log | grep mitigation: | tail -1`)
- **Config:** `outputs/experiments/E7_mitigation/config.yaml`, conditions `M0,M1,M2,M3,M3b`, `--max-tokens 8192`, 218 scenarios × 5 models = 5450 records.
- Check alive: `ps -p $(cat /tmp/e7_final.pid) -o pid,etime,%cpu`. If dead but
  incomplete → relaunch SAME command (resume continues; same config so the
  fresh file is NOT contaminated):
  `nohup /home/lsh/omission/.venv/bin/python3 -m src.evaluation.mitigation.run_mitigation --config outputs/experiments/E7_mitigation/config.yaml --conditions M0,M1,M2,M3,M3b --max-tokens 8192 --out outputs/experiments/E7_mitigation/full218_clean/tuples.jsonl > /tmp/e7_final.log 2>&1 &`

**On completion (`done. counts=...` in log):**
1. `python -m src.evaluation.mitigation.metrics --tuples outputs/experiments/E7_mitigation/full218_clean/tuples.jsonl --out-dir outputs/experiments/E7_mitigation/full218_clean/tables`
2. `python -m src.analysis.plot_mitigation --tuples .../full218_clean/tuples.jsonl`
3. `python -m src.analysis.plot_mitigation_overall --tuples .../full218_clean/tuples.jsonl`
4. Report per-model credited/McNemar (H4 = primary {M2,M3,M3b} Bonferroni p<.05 AND > same model's M1 AND not net-harmful). Use venv python `/home/lsh/omission/.venv/bin/python3`.
5. **Never analyze the accumulated `raw/tuples.jsonl`** — it is multi-version-contaminated; `full218_clean/` is the only clean canonical set.

## 2. Code changes this session (all in, verified)

`src/evaluation/mitigation/conditions.py`:
- **M4 (5-phil MAD) DROPPED** from `CONDITIONS` (run_M4/_mad_frame retained, unregistered, dead provenance). `PRIMARY_CONDITIONS=("M2","M3","M3b")`.
- **Unified system, blocker-fixed:** `_FORCING = EVAL_SYSTEM.split("Reply in this exact format")[0]` (byte-identical anti-refusal forcing, single source). `COT_SYSTEM` = _FORCING + "reason first, then ANSWER:". M0=`EVAL_SYSTEM` (answer-only, =E1 §191). M1/M3/M3b=`COT_SYSTEM`. M2=`_M2_SYSTEM` (_FORCING + dual Answer1/Answer2). Prior bug: EVAL_SYSTEM's "nothing else" contradicted CoT prompts → would suppress reasoning. Now resolved.
- **M1** = generic zero-shot CoT control: "Think step by step before answering." only (Kojima 2022). No explicit steps by design.
- **M2** = considering-the-opposite, **PROCESS ONLY** (Lord 1984; ICU 2026). Step1 tentative(Form1)+1-sent reason → Step2 reconsider Form2 (default reversed) "on its own merits, as if encountering fresh" → Step3 final yes/no each form. **Removed: risk/benefit decision-rule (= consequentialism injection), consistency/"be consistent" demand, omission/framing metric-leak, Korean wording.** Dual answers → cross-frame tuple preserved.
- **M3** = considering-the-opposite single-frame (Lord 1984): Step1 tentative → Step2 ONE-sentence strongest case for OPPOSITE of own Step-1 answer → Step3 final. (User corrected: tentative-first IS required.)
- **M3b** = decision-ledger: Step1 IF YES one neutral factual sentence / Step2 IF NO one neutral / Step3 decide. **yes/no-first counterbalance DROPPED** (fixed order; order-effect = stated limitation).
- Step indexing unified to `Step N:` across M2/M3/M3b (M1 unstructured by design).
- Parsers earlier-session: `parse_verdict` (M4, dead), `parse_dual` broadened (Q1/Q2, 1), frame a, etc.), refusal gated = `is_refusal AND ans is None` ("as an ai" removed from `_REFUSAL_RE`).

`src/evaluation/mitigation/run_mitigation.py`:
- `TARGET_MODELS` = 5 (llama-3.1-8b / gemma-3-12b / gpt-4o-mini / qwen3.5-9b / gemini-2.0-flash). gemini kept (low-OBR, ~16 NN, underpowered — descriptive only).
- `--all-scenarios` flag → full 218 labeled (no top-cell/NN subsample).
- `_load_frames`: **PARADIGM_MISFIT filter REMOVED** (only `_well_formed`). misfit hand-exclusion dropped project-wide → 218.
- `PILOT_CONDITIONS=("M0","M1","M2","M3","M3b")`.

`src/shared/llm.py`:
- Gateway resilience: 120s per-request timeout + 6× exp-backoff retry on timeout/5xx/429 (`LLM_REQUEST_TIMEOUT_S`/`LLM_MAX_RETRIES`). Fixes indefinite hangs + 504 bursts on `api.ssunlp.co.kr`.

`src/analysis/plot_mitigation.py` (diverging credited↑/flagged↓) and
`src/analysis/plot_mitigation_overall.py` (whole-218 tuple composition,
YN+NY merged into FC) — both created/rewritten this session.

## 3. Docs updated this session

- `CLAUDE.md`: misfit→SUPERSEDED/218, scenario-accounting 217→218 + `eval_model.py:89 PARADIGM_MISFIT` `--keep-misfit` gotcha, +2 gotchas (accumulated tuples contamination → fresh `--out`; gateway hang + llm.py patch + qwen CoT needs 8192).
- `RESEARCH_PLAN_v4.md`: 🔴 banner (M4 dropped, full-218), change-log entry, mitigation-reference pointer.
- supplement `~/.claude/plans/research-plan-v4-giggly-nebula.md`: 🔴 banner, new "## 완화 프롬프트 설계 근거" reference table (verified real papers), M2 row corrected to process-only/theory-neutral.
- `BENCHMARK_TODO.md`: #1✅ #3✅ #4✅ done; **#6 NEW** = reproducibility re-run.

## 4. Verified references (real, web-checked 2026-05-19)

M1=Kojima 2022 NeurIPS (zero-shot CoT). M2/M3=Lord, Lepper & Preston 1984
JPSP (considering-the-opposite) + Mussweiler 2000 PSPB (anchoring) +
Madaan 2023 NeurIPS Self-Refine (M3 LLM-side) + ICU 2026 (applied).
M3b=Spranca, Minsk & Baron 1991 JESP (omission-bias mechanism). Forced
2-frame=Cheung 2025 PNAS + Scherrer 2023 NeurIPS. Full table in supplement.

## 5. Pending TODOs (priority order)

1. **On final run done:** metrics + 2 figures from `full218_clean/` (§1 above), report H4.
2. **BENCHMARK_TODO #6 (reproducibility, NEXT):** re-run E1 overall OBR + E1 filter_vs_random (RQ1) + E2 obr_by_conflict at IDENTICAL config (max_tokens 8192, EVAL_SYSTEM, gateway patch, **218** via `eval_model.py` `--keep-misfit` or remove its hardcoded `PARADIGM_MISFIT`). Then update RESEARCH_PLAN §4.1/§5 + memory `eval-models-and-e1-e2-done`. This makes in-run M0==E1 byte-true → §191 holds literally (no doc-softening needed).
3. **Supplement "## 정확한 prompt" (lines ~59-102) + 조건표 (~40-52) are STALE** (old M2/M3/M3b, M4 present, pre-blocker). Ground truth is now `conditions.py`. Regenerate that section from final code BEFORE prereg freeze (prereg↔run consistency).
4. **Qualitative analysis appendix** (advisor directive, memory `qualitative-analysis-appendix-todo`): cherry-picked case analysis vs other models. Good material: M2 success vs M1 net-harmful (NN→YY) contrast; llama over-refusal.
5. Paper limitations to disclose (not blockers): RTM/no M0 retest → "consistent with mitigation" + T=0 determinism caveat; forking-paths (post-amendment exploratory, not preregistered confirmatory); M2 = considering-the-opposite + simultaneous-presentation (state both); per-model differing M0=NN sets (descriptive aggregate, not pooled); gemini underpowered; llama refusals reported separately.

## 6. Key facts / gotchas

- venv python: `/home/lsh/omission/.venv/bin/python3`. No leading `cd` for backgrounded `-m` runs.
- Benchmark/labels/E1(0518/1715)/E7-manifest = **218** labeled (H_001 included). `eval_model.py:89` still hardcodes `PARADIGM_MISFIT={H_001,H_005,H_006}` → E1/E2 give 217 unless `--keep-misfit`.
- 5 eval models + OBR: read `outputs/experiments/.../per_model_summary.csv` — never guess.
- `metrics.load_tuples` = last-row-wins per (sid,model,cond); `raw/tuples.jsonl` accumulated/contaminated → only use `full218_clean/`.
- scipy/pyyaml NOT in venv (hand-rolled stats + YAML in run_mitigation).
- Monitor a backgrounded run with stall detection (same line 5×20s → hang warn); hung run = alive+0%CPU+frozen tqdm → kill & relaunch (resumable).
