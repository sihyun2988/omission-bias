# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

`omission-bias` is a research repository transitioning out of pilot stage:

- `src/` — production source tree, split by research concern (see Directory layout below).
  - `src/data_construction/reframing/reframe.py` — generates PNAS-style paired yes/no frames. Output: single JSONL at `data/constructed/mirror_frames/paired_frames.jsonl` (one record per scenario, resumable; `--overwrite` regenerates target ids and replaces their records in place). Default target = every scenario in `data/constructed/inaction_labels.csv` with a clean A1/A2 label. Invoke as `python -m src.data_construction.reframing.reframe`.
  - `src/data_construction/reframing/validate.py` — **LLM-judge** post-hoc filter (replaced earlier regex version 2026-05-11). For each scenario, calls the synthesis LLM (`openrouter` + `openai/gpt-4.1-mini`) once with a strict rubric for two invariants: (a) cross-frame outcome equivalence judged SEMANTICALLY (paraphrases pass), and (b) Frame A/B reversibility — distinguishing PREPARATION verbs (decided/prepared/drafted/queued/scheduled/about-to → PASS) from DELIVERY verbs (told-X-that/sent/given-opinion/looked/administered/killed → FAIL). Concurrent calls with tqdm + resumable (skips ids already in `validation_report.jsonl`; `--overwrite` re-judges). Emits `data/constructed/mirror_frames/needs_regen.txt` (scenario_ids with OVERALL=FAIL) to feed into `reframe.py --from-file --overwrite`. Iterate until clean. Invoke as `python -m src.data_construction.reframing.validate`.
  - `src/shared/llm.py` — shared `load_env()` + `make_caller(provider)` helpers extracted from `pilot/run.py`.
  - Other `src/data_construction/{inaction_labeling,philosophy_panel}/`, `src/evaluation/{runners,metrics,mitigation}/`, `src/analysis/` packages are scaffolded but empty; new code lands here, not at the `src/` root.
- `pilot/` — pilot-stage code (philosophy panel labeling + analysis, still in use):
  - `pilot/inaction_label.py` — labels which side of each MoralChoice raw row is the lexical inaction (regex tier 1 + optional LLM fallback). Output: `data/constructed/inaction_labels.csv` (680 rows). Defaults updated 2026-05-11 to point to the new `data/raw/` + `data/constructed/` tree; internal logic is otherwise pilot-frozen.
  - `pilot/run.py`, `pilot/analyze.py`, `pilot/philosophies.py` — philosophy panel labeling + analysis. `pilot/results/pilot_*.jsonl` holds outputs. `pilot/run.py` currently duplicates `src/shared/llm.py`'s provider helpers; unify when convenient.
  - `pilot/predictions_seed42.json` — the 10 fault-line scenario ids from the pilot panel; useful as a hand-curated subset for `python -m src.data_construction.reframing.reframe --from-file`.
  - `pilot/archive/legacy_paired_frames/` — the 10 stale per-scenario JSONs produced by the pilot reframe run, parked here for provenance. They are superseded by `data/constructed/mirror_frames/paired_frames.jsonl` and must not be cited as current artifacts.

Reference docs:
- `RESEARCH_PLAN.md` — current research plan (EMNLP 2026 long paper). **Read this first** for scope, RQs, timeline.
- `Large language models show amplified cognitive biases in moral decision-making.pdf` — Cheung, Maier, Lieder (PNAS 2025); defines framing-invariant omission bias via Action↔Omission Reframing (Table 1).
- `Evaluating the Moral Beliefs Encoded in LLMs.pdf` — Scherrer et al. (NeurIPS 2023); MoralChoice source.

## Research direction (anchoring context)

The project will build a benchmark for **omission bias in LLMs**, extending the PNAS paper's framework. Two non-obvious points worth keeping in mind when code starts landing:

1. **Operational definition of omission bias** carries over from the prior work: a model exhibits framing-invariant omission bias when it picks *inaction* in **both** the original frame **and** the action↔inaction-swapped mirror frame for the same scenario. Benchmark items must therefore exist as *paired* (original, mirror) scenarios — single-frame items cannot measure the bias.
2. **The intended novelty** is using multiple moral-philosophy personas (utilitarian / deontological / virtue / care / contractualist, etc.) as a panel whose *disagreement* signals "moral fault line" scenarios where the bias is expected to surface. This is distinct from the prior work's single utilitarian↔deontological axis.

When implementing, do not collapse these two ideas — the philosophy panel is a *construction-time signal*, while framing-invariant inaction is the *evaluation-time metric*.

## Directory layout

Keep the repository split by research concern, not by implementation convenience:

- `pilot/` remains the pilot archive and active pilot workspace. Do not move, rename, or reorganize files inside it unless explicitly requested.
- `src/data_construction/` is for benchmark-building code: MoralChoice ingestion, inaction labeling, action↔omission mirror-frame generation, rule checks, dual-annotation helpers, and philosophy-panel labeling. Reframing currently lives under `src/data_construction/reframing/reframe.py`; the `inaction_labeling/` and `philosophy_panel/` subpackages are reserved for the eventual `src/` migration of `pilot/inaction_label.py` and the panel runner.
- `src/evaluation/` is for the main OBR experiment and evaluation pipeline: model runners, condition matrices, OBR/action-consistency metrics, Scherrer-style agreement clustering, and mitigation methods such as simultaneous framing and multi-philosophy consensus.
- `src/analysis/` is for statistical analysis code used after runs complete: ANOVA, Mann-Whitney U, Mantel tests, plotting transforms, and paper-table generation.
- `src/shared/` is for genuinely shared utilities only: provider adapters, schema definitions, IO helpers, logging, and config loading used by both data construction and evaluation. Avoid putting experiment logic here.
- `data/raw/` is for immutable source datasets, especially MoralChoice high/low ambiguity CSVs. The high-ambiguity CSV currently lives at `data/raw/moralchoice/moralchoice_high_ambiguity.csv`.
- `data/constructed/` is for generated benchmark artifacts that become inputs to evaluation. Current contents: `data/constructed/inaction_labels.csv` (lexical-inaction labels from `pilot/inaction_label.py`) and `data/constructed/mirror_frames/paired_frames.jsonl` (action↔omission paired frames from `src/data_construction/reframing/reframe.py`). The `data/constructed/benchmark/` subdir is reserved for the final `omission-bench-v1.jsonl`-style artifact.
- `data/annotations/` is for human or rule-check annotation artifacts, including dual-review logs and κ calculations.
- `data/panel_outputs/` is for construction-time philosophy-panel votes, confidence, justifications, and disagreement metrics.
- `outputs/experiments/` is for evaluation run outputs from E1-E7. These are results of measuring models, not benchmark data.
- `outputs/analysis/` and `outputs/figures/` are for derived statistics, tables, and plots.
- `configs/` is for run matrices, model lists, prompt variants, and preregistered experiment settings.
- Thin command-line helpers should live beside the package they operate on; substantive logic belongs under `src/`. Do not recreate a top-level `scripts/` directory unless there is a strong cross-package orchestration need.

Generated benchmark data and generated experiment results should not share a folder. If an artifact is used as an input benchmark item, keep it under `data/`; if it is produced by running evaluation models or analyzing their responses, keep it under `outputs/`.

## Reframe pipeline (paired-frame construction) invariants

`src/data_construction/reframing/reframe.py` SYSTEM_PROMPT (v4 balanced, 2026-05-12) enforces:

1. **Same-world symmetry** — Frame A and Frame B describe the SAME situation, same parties, same two outcomes. If Frame B has an external party, Frame A must mention the same party INERT (present but awaiting the agent's decision). Frames differ ONLY in default trajectory + question.
2. **TYPE selection by natural fit, no default bias** — pick from `T_SELF` (agent self-prepared schedulable act: bank transfer scheduled, email drafted, mission departure scheduled), `T_PEER` (spouse / friend / sibling / coworker / deputy *below* the agent), `T_ADMIN` (institutional or auto pipeline: hospital allocation committee, editorial system, refund queue), `T_MUTUAL` (mutual party for face-to-face speech act), `T_PRIOR` (predecessor in agent's role). Earlier iterations forced one TYPE in all scenarios (v2 = all external party → unnatural for prep-able acts; v3 = all T_SELF → unnatural for institutional decisions and split-second acts); v4 explicitly removes preference and matches TYPE to the act's social structure.
3. **No authority confound** — Frame B's external party must NEVER be a superior with authority over the agent (no boss, captain, commander, team leader, superior officer, parent of minor). Authority creates an obedience confound where the model's "no" may reflect deference, not omission bias.
4. **Preparation vs delivery** — agent ALLOWED as past actor for preparation verbs (scheduled, drafted, queued, submitted, arranged, set up); FORBIDDEN for delivery verbs (told-X-that, sent, killed, looked, administered, gave opinion). Carries over from `validate.py`'s rubric.
5. **Cross-frame outcome equivalence** — `frame_A.outcome_if_yes ≡ frame_B.outcome_if_no` and the inverse, by construction.

`auto_check` accepts only T_SELF|T_PEER|T_ADMIN|T_MUTUAL|T_PRIOR in TYPE.

**Paradigm misfit cases** — split-second physical or coerced-personal-hand scenarios (H_006 kidnap-and-shoot, H_001 grenade, H_005 self-defense stab) distort the dilemma when reframed (T_PEER turns "would you personally kill" into "would you order a kill" — different moral psychology; T_SELF "you raised the gun" is unnatural for instantaneous acts). Flag for manual exclusion; PNAS paradigm doesn't cleanly apply.

**Archive nomenclature** (`data/constructed/mirror_frames/`) — `paired_frames_legacy_gpt4-1-mini_old-prompt.jsonl` (661, original 2026-05-11 OLD design), `paired_frames_v2_gpt5_asymmetric.jsonl` (312, gpt-5 + all-external-party design), `paired_frames_v3_long_tself_biased.jsonl` (57, gpt-5 + all-T_SELF design), `paired_frames_v3_compare.jsonl` / `paired_frames_v4_balanced_10.jsonl` (prompt-iteration samples). Active corpus path: `paired_frames.jsonl` (fresh under v4 balanced + gpt-5 reasoning).

Design ancestor: `/home/lsh/omission_medical/scripts/generate_scenarios.py` has the original TYPE A/B + ONGOING/ONE-TIME prompt structure that the OLD reframe design adapted. PNAS 2025 paired examples (Switzerland↔Canada MAS, boy's medicine refund, terrorist release, vet job, foundation, hostage ransom) are the references for the NEW design's templates.

## Environment & data gotchas

- **PDF reading:** `Read` tool fails on PDFs (`pdftoppm not installed`). Use `python3 -c "import pypdf; r = pypdf.PdfReader('path.pdf'); print(r.pages[i].extract_text())"`. `pypdf` is available; `PyPDF2` and `pymupdf` are not.
- **MoralChoice dataset location:** HF `ninoscherrer/moralchoice` viewer shows only 3 rows (question templates `ab`/`repeat`/`compare`), NOT the scenarios. Actual 1,367 scenarios are in GitHub `ninodimontalcino/moralchoice/data/scenarios/moralchoice_{high,low}_ambiguity.csv` (high=680, low=687).
- **PNAS abstract has wrong dataset size:** It claims 1,767 scenarios; actual MoralChoice is 1,367 (= 680 + 687). Cite 1,367.
- **MoralChoice feature flag values:** the 10 harm/violation columns (`a1_death`, `a1_pain`, …) hold `"Yes"` / `"No"` / `"No Agreement"`, not `"1"` / `"0"`. Filter with `r[col] == "Yes"`.
- **LLM gateway:** `OPENROUTER_BASE_URL` in `.env` points to `api.ssunlp.co.kr` (OpenRouter-compatible self-hosted gateway, not openrouter.ai). Default `qwen/qwen3.5-9b` is a *reasoning model* — without `extra_body={"reasoning": {"enabled": False}}` the internal thinking consumes the entire `max_tokens` budget and `message.content` returns `null` with `finish_reason="length"` and no exception. `/no_think` directive and `chat_template_kwargs` are silently ignored on this gateway.
- **Synthesis model — split by stage:**
  - **Primary scenario synthesis** (`reframe.py` v4 balanced): **gpt-5 reasoning** via OpenRouter gateway — `--provider openrouter --model openai/gpt-5`. Gateway REQUIRES `reasoning.enabled=True` (returns 400 with "Reasoning is mandatory for this endpoint and cannot be disabled" if False); doesn't accept `temperature != 1`; uses `max_completion_tokens` not `max_tokens`; needs **≥16k token budget** (reasoning consumes output budget — 8k caused empty `message.content` for one scenario in testing). 661 scenarios ≈ **$20**, ~20-30 min @ `--concurrency 16` (avg 2.9k input + 2.7k output incl. reasoning tokens per scenario). `src/shared/llm.is_reasoning_model()` auto-routes based on model name prefix (`gpt-5*`, `o1*`, `o3*`, `o4*` → reasoning branch).
  - **Cheaper synthesis** (inaction labeling LLM fallback, KR translation, ad-hoc small jobs): **gpt-4.1-mini** via OpenRouter — `--provider openrouter --model openai/gpt-4.1-mini`.
  - Direct `--provider openai` 401s; `OPENAI_API_KEY` in `.env` is a 7-char placeholder. Evaluation models (the ones being measured for OBR) are unaffected by these rules.
- **`--overwrite` warning:** `python -m src.data_construction.reframing.reframe --overwrite` rewrites `data/constructed/mirror_frames/paired_frames.jsonl` with the target scenarios' existing records removed *before* the run starts, then appends new records as they complete. If the run errors out mid-way, the target ids are lost from the file (the surviving records for non-target ids are safe). Verify provider/model on a single scenario before bulk `--overwrite`.
- **Python interpreter:** All construction scripts need the project venv — `tqdm`, `openai`, `python-dotenv`, `pypdf` are only there. Invoke as `.venv/bin/python3 -m src.data_construction.reframing.reframe ...` (or activate the venv first). System `python3` fails with `ModuleNotFoundError: No module named 'tqdm'` immediately.
- **Test runs without touching main JSONL:** pass `--out data/constructed/mirror_frames/<test-name>.jsonl` to redirect output. `paired_frames_v2_test.jsonl` is the existing PNAS-design test set; `paired_frames.jsonl` is the legacy 661-record corpus.

## When to update this file

The Project status and Directory layout sections reflect the `src/{data_construction,evaluation,analysis,shared}/` + `data/{raw,constructed,annotations,panel_outputs}/` + `outputs/{experiments,...}/` reorg (2026-05-11) and the Reframe pipeline section reflects the PNAS-style SYSTEM_PROMPT rewrite (2026-05-12). Re-run `/init` or extend manually when: (a) the 661-record `paired_frames.jsonl` is fully regenerated with the new PNAS design, (b) `inaction_labeling/` and `philosophy_panel/` subpackages get migrated out of `pilot/` into `src/data_construction/`, (c) dependency manifest formalizes beyond `requirements.txt`, (d) eval pipeline (E1–E7) entry points appear under `src/evaluation/`. Until then, `RESEARCH_PLAN.md` remains source of truth for scope, RQs, timeline.
