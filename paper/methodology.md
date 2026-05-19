# The OMIT Benchmark + Setup/Results stubs (draft v1 — 2026-05-19)

> [메타] 본문은 영어, 메타 코멘트는 한국어. LaTeX paste-ready markdown.
> **2026-05-19 결정 반영:** (1) 벤치마크명 = **OMIT** 확정 (이전
> `\textsc{OmissionBench}` placeholder 전역 치환). (2) §3 제목 = "The OMIT
> Benchmark" (MORABLES EMNLP'25 장르 컨벤션 — "Methodology"/"Construction"
> 비관용). (3) ToC = **B 강조형**: §3 = 구축 only, RQ1(filter>random)은 별도
> **§4 Benchmark Validation** 으로 승격, 평가 setup = §5, 분석(RQ2/3)+완화(RQ4)
> = §6. 이번 패스는 **§3 구축 섹션**이 deliverable; §4/§5/§6 은 내용 보존용
> 스텁(상세 초안은 다음 패스).
> 인용 키는 `paper/references.bib` 와 일치(`cheung2025amplified`,
> `scherrer2023moralbeliefs`, `spranca1991omission`). 동결 전 비율은 `[X]`
> placeholder; 확정 카운트(657/218/439)는 본문에 반영. 구현 세부(프롬프트
> 전문, iteration 수)는 `[APPENDIX: …]` 로 미루고 motivation 은 본문에 남긴다.

---

## 3 The OMIT Benchmark

We present **OMIT** (**O**mission bias under **M**oral-philosoph**I**cal
conflic**T**), a benchmark whose every item is a *paired* action↔omission
mirror frame, with each item typed by the disagreement of a five-philosophy
persona panel. Throughout, we keep two ideas strictly separated: the philosophy
panel is a **construction-time selection signal**, whereas framing-invariant
inaction is the **evaluation-time metric** (defined here, measured in §5–§6).
We make no causal claim that disagreement *produces* the bias.

> [메타] 위 문단이 섹션의 thesis-lock. CLAUDE.md 의 "두 아이디어를 섞지 말 것"
> 규칙 + abstract 의 "no causal claim; construction-time signal only" 와 1:1
> 대응. 리뷰어가 "패널 불일치가 편향의 원인이라 주장하나?" 라고 물 때 막는
> pre-emptive 문장. §3 은 *벤치마크 그 자체*(정의·구축·QC·통계)만 다루고,
> 그것으로 *모델을 재는* 행위는 §5–§6 으로 분리 — MORABLES 가 §3 The
> MORABLES Benchmark / §4 Evaluations 로 가른 것과 동형.

### 3.1 Task Formulation and Omission-Bias Definition

A preference for inaction is only a *bias* if it survives a relabeling of which
option counts as the action. Following \citet{cheung2025amplified}, we adopt
**framing invariance** as the operational criterion and therefore require every
benchmark item to exist as a *pair*; a single-frame item cannot, even in
principle, separate a framing artifact from a stable moral preference.

Formally, a scenario $s$ is a pair of prompts $(\phi_A, \phi_B)$ — Frame A and
Frame B — over the *same world*: the same parties, the same two terminal
outcomes, and the same agent. The frames differ only in (i) the default
trajectory and (ii) the polarity of the binary question, and they satisfy
**cross-frame outcome equivalence** by construction:
$$\mathrm{out}_A(\text{yes}) \equiv \mathrm{out}_B(\text{no}), \qquad
  \mathrm{out}_A(\text{no})  \equiv \mathrm{out}_B(\text{yes}).$$
In Frame A the intervention has not been initiated and the question asks
*"Will you $X$?"*; in Frame B the same intervention is at pre-completion and the
question asks *"Will you stop $X$?"*. Choosing inaction thus corresponds to
answering **no** in Frame A and **yes** in Frame B — opposite tokens, identical
underlying choice.

A model's response to $s$ is the ordered pair $(a_A, a_B) \in \{Y, N\}^2$, which
we class as one of $\{YY, YN, NY, NN\}$ after mapping each frame's answer onto
the action/inaction axis. A model exhibits **framing-invariant omission bias**
on $s$ iff it rejects the action in *both* frames, i.e. the class is $NN$. Over
a set of $n$ scenarios for which both frames returned a parsed binary answer, we
report
$$\mathrm{OBR}=\frac{\#NN}{n},\quad
  \mathrm{ABR}=\frac{\#YY}{n},\quad
  \mathrm{FCR}=\frac{\#YN+\#NY}{n}=1-\mathrm{OBR}-\mathrm{ABR}.$$
$\mathrm{OBR}$ (Omission Bias Rate) is the primary quantity; $\mathrm{ABR}$
(Action Bias Rate) and $\mathrm{FCR}$ (Frame-Consistent Rate) characterize the
remaining mass. Refusals and unparseable generations are excluded from $n$ and
reported separately (§3.6), never silently coerced.

> [메타] One equation rule 충족: OBR/ABR/FCR + cross-frame equivalence 두 식.
> 모든 기호($s,\phi,a,n$)를 사용 전에 정의. NN = bias 의 조작적 정의를
> abstract("framing-invariant inaction")·related work §2.1 과 같은 표현으로 고정.

### 3.2 Overview

Figure 1 traces the construction pipeline. We start from the MoralChoice
high-ambiguity pool \citep{scherrer2023moralbeliefs} (680 scenarios), label
which option is the lexical inaction, and **re-cast** each scenario into a
paired mirror frame (§3.3); after excluding only malformed records this yields
**657** valid paired scenarios. A five-philosophy panel then answers both
frames of every surviving scenario (§3.4). A two-stage **disagreement filter**
keeps only scenarios that sit on a contested moral fault line and labels which
philosophical camps the fault line opposes (§3.5); the **218** labeled
scenarios *are* OMIT, and the **439** non-labeled valid scenarios form the
control complement released alongside it. Quality control (§3.6) and frozen
release statistics (§3.7) complete the resource. The benchmark is then *used*
to validate the construction signal (§4) and to measure and mitigate the bias
in held-out models (§5–§6); held-out models receive **no** persona injection.

> [메타] Figure 1 caption 은 self-contained 해야 함(method.md §2). 아래가 초안.
> "no persona injection" 유지, M4/persona-contrast 도식 제거(2026-05-18).

**Figure 1. Overview of OMIT construction.** Given a MoralChoice
high-ambiguity scenario, we (1) re-cast it into a paired Frame A / Frame B that
describe the *same world* but flip the default trajectory, so that choosing
inaction means answering *no* in A and *yes* in B; (2) query a five-philosophy
persona panel (utilitarian, deontologist, virtue, care, contractualist) on both
frames; (3) **keep only scenarios on which the panel disagrees** and label the
opposing philosophical camps. The resulting labeled subset is OMIT; held-out
models are later evaluated for framing-invariant inaction ($NN$) and receive no
persona injection. Unlike prior work that probes a single model around a single
utilitarian↔deontologist axis, the panel is used purely as a construction-time
selection signal over a five-philosophy conflict space.

### 3.3 Scenario Source and Mirror-Frame Construction

**Why.** Measuring a framing effect requires a counterfactual frame that is
*structurally symmetric* to the original: it must change the default trajectory
and nothing else. Naive negation does not achieve this — it routinely
introduces an authority or agency confound (e.g., turning "do you act?" into
"do you defy your superior?"), so that a model's "no" reflects deference rather
than omission. We therefore generate frames under explicit symmetry invariants.

**How.** The source pool is the MoralChoice high-ambiguity set
\citep{scherrer2023moralbeliefs} (680 scenarios); for each we label which of the
two options is the lexical inaction. Each scenario is then reframed by an LLM
under a balanced reframing prompt (`[APPENDIX: reframe SYSTEM_PROMPT v4]`) that
enforces four hard invariants: (i) **same-world symmetry** — both frames name
the same parties and outcomes, any external party present in B is
present-but-inert in A; (ii) **no-authority confound** — the external party is
never a superior with authority over the agent; (iii) a
**preparation-vs-delivery rule** — the agent may be a past actor only for
reversible preparation verbs (scheduled, drafted, queued), never for completed
delivery verbs (sent, administered, killed); and (iv) cross-frame outcome
equivalence (Eq. in §3.1). The natural social structure of the act selects one
of five role TYPEs — `T_SELF` (agent self-prepared, schedulable), `T_PEER`
(peer, never a superior), `T_ADMIN` (institutional/automatic pipeline),
`T_MUTUAL` (face-to-face speech act), `T_PRIOR` (predecessor in the agent's
role) — rather than forcing one structure on every scenario. A schema
auto-check rejects malformed generations.

**Intuition.** Frame A withholds an intervention that has not yet started;
Frame B withholds completion of the *same* intervention already in motion. The
world and its two outcomes are held fixed by construction, so any systematic
asymmetry in a model's answers across the pair is attributable to framing, not
to a change in stakes.

We exclude only 4 malformed records (`G_116/125/228/330`, null frames); from
the 661 reframed records this leaves **657** valid paired scenarios. We
deliberately apply *no* manual content-based exclusion: rather than hand-drop
scenarios whose moral psychology we judge to reframe imperfectly (e.g.
split-second or coerced personal-hand acts), we keep every well-formed pair and
route any reframing-fidelity concern to the human audit of §3.6 instead, so
that no benchmark item is removed by author discretion. The same exclusion is
applied identically to every downstream sampling arm (§4) so that arm
differences reflect only the filter.

> [메타] 4 invariant 마다 "왜"가 한 문장씩(motivation-driven). authority confound
> 단락이 핵심 pre-emptive — 리뷰어의 "프레임 B 의 no 는 그냥 복종 아니냐" 공격
> 차단. 프롬프트 전문은 appendix. 본문 "Stage 1" 표제어는 제거(MORABLES 는
> 표제에 Stage 안 씀); 파이프라인 단계 언급은 §3.5 의 Stage-1/2 필터에 한정.

### 3.4 Moral-Philosophy Persona Panel

**Why.** Prior LLM omission-bias work analyses a single utilitarian↔deontologist
axis \citep{cheung2025amplified}. Real moral disagreement is higher-dimensional;
a panel of distinct normative frameworks can surface a far richer set of fault
lines *without* any human scenario labeling.

**How.** Five personas — **F1** utilitarian, **F2** deontologist, **F3**
virtue, **F4** care, **F5** contractualist — are realized as neutral persona
system prompts (`[APPENDIX: philosophy prompts, canonical no-line v1]`). The
prompts state each framework descriptively and deliberately contain *no*
action-favoring or omission-disfavoring steering: any such line would
manufacture the very disagreement we use as a signal. For each scenario we
issue one query per (philosophy $\times$ frame $\in\{A,B\}$) at temperature
$T{=}0$ with $n{=}1$ sample, forcing a binary answer, and reduce each
philosophy's two answers to a tuple in $\{YY, YN, NY, NN\}$.

**Intuition.** A philosophy whose tuple is $YN$ or $NY$ *flips* across the
action↔omission inversion — it takes a framing-sensitive stance and so can
anchor a labeled conflict. A philosophy whose tuple is $YY$ or $NN$ is itself
framing-invariant on that scenario and is uninformative for fault-line
selection.

> [메타] neutral-prompt 단락이 memory `feedback_neutral_panel_prompts` 와 abstract
> "philosophy-neutral" 을 섹션에 박는 pre-emptive(패널이 결과를 미리 심었다는
> 공격 차단). T=0,n=1 은 memory `feedback_experiment_defaults`.

### 3.5 Disagreement Filter and Conflict Labeling

This is the core construction contribution: turning panel disagreement into a
scenario-selection-and-typing signal.

**Stage-1 unanimity filter.** Let $\tau_p(s)$ be philosophy $p$'s tuple on
scenario $s$. We keep $s$ iff the five philosophies do not all agree:
$$\big|\{\,\tau_p(s) : p \in \{\mathrm{F1},\dots,\mathrm{F5}\}\,\}\big| > 1 .$$
A scenario on which all five frameworks return the *same* tuple is a
moral-consensus case, not a fault line, and is dropped. This retains
$[X]$ of the valid scenarios (`[X]\%`).

**Stage-2 conflict labeling.** Within a surviving scenario we discard any
philosophy whose tuple is $YY$ or $NN$ (framing-invariant: it cannot anchor a
*direction* of flip), then partition the rest by flip direction into
$\mathrm{yn\_phils}$ (tuple $YN$) and $\mathrm{ny\_phils}$ (tuple $NY$). The
scenario's conflict set is the full cross-product
$\{(p, q) : p \in \mathrm{yn\_phils},\, q \in \mathrm{ny\_phils}\}$, and its
`label_status` is `labeled` iff both sides are non-empty (else
`dropped_one_sided_yn`, `dropped_one_sided_ny`, or `dropped_all_excluded`). The
**218** `labeled` scenarios (`[X]\%` of Stage-1 survivors) constitute OMIT; the
**439** non-labeled valid scenarios form the complement pool released as the
control split and used as the §4 random arm.

**Non-independence.** Because we emit the *full* cross-product, one scenario
contributes to several $(\mathrm{yn}\times\mathrm{ny})$ conflict cells. All
per-cell inference therefore treats `scenario_id` as a random intercept (or
uses cluster-robust standard errors); we state this here rather than bury it in
the analysis.

We reiterate that this filter *selects and types* scenarios — it does not
explain why any model is biased. The labels are conditioning variables for
analysis (§6), not a causal mechanism.

> [메타] §3.5 가 novelty 의 심장. 필터 규칙을 식으로 박음(엄밀성). full
> cross-product → 비독립성 → random intercept 를 *여기서* 선언한 게
> pre-emptive(리뷰어의 "시나리오 재사용으로 n 부풀린 것 아니냐" 차단).
> 마지막 문단이 다시 한 번 causal-claim 차단.

### 3.6 Quality Control

**Why.** A construction signal is only a *benchmark* if its labels are
reproducible by someone other than the pipeline that produced them, and if
items the pipeline keeps are not silently distorted by the reframing step.
We therefore validate the pipeline against human judgment rather than asserting
its correctness.

**Pipeline-label agreement.** Two annotators independently label a random
subset of scenarios for (a) cross-frame outcome equivalence and reframing
fidelity, and (b) the Stage-2 conflict direction, blind to the panel output.
We report Cohen's $\kappa$ against the pipeline labels; the pre-registered
acceptance threshold is $\kappa \geq 0.70$ (`[APPENDIX C: annotation protocol,
subset size, per-criterion $\kappa$]`).

**Reframing-fidelity flags, not exclusions.** A small set of scenarios whose
moral psychology shifts under reframing — split-second physical acts and
coerced personal-hand acts, where delegating the act changes the dilemma — are
*flagged* (`reframing_fidelity_flags.txt`) but **not** removed: hand-excluding
them would be an author-discretion judgment on a continuum, and only structural
malformations (the 4 null-frame records of §3.3) warrant deterministic removal.
The empirical impact of the flagged items is bounded by reporting OMIT results
with and without them in `[APPENDIX B]`, so robustness is *shown* rather than
assumed.

> [메타] QC 를 별도 subsection 으로 끌어올림(MORABLES 3.3 "Human Annotation and
> Validation" 동형 — 벤치마크 논문 필수 §). BENCHMARK_TODO #2(dual-κ)+#3(misfit
> flag-not-exclude) 를 본문화. κ 임계 0.70 은 plan §10. "flag not exclude" 는
> memory + TODO #3 의 2026-05-18 방향전환 반영(자의적 제외 차단 = pre-emptive).

### 3.7 Benchmark Statistics and Release

From 680 MoralChoice high-ambiguity scenarios, mirror-frame construction yields
657 valid paired scenarios (4 structural malformations removed). The
disagreement filter labels **218** of them (the OMIT measurement set) and
leaves **439** non-labeled valid scenarios as a size-comparable control pool.
Table 1 reports the conflict-pair coverage — the number of labeled scenarios
per $(\mathrm{yn},\mathrm{ny})$ philosophy pair — and the marginal $YN/NY$
philosophy frequencies (`[TABLE 1: conflict-pair distribution; values frozen
with the v1 release]`).

We freeze a versioned release, `omission-bench-v1`, containing the 218 labeled
items plus the 217-item control split (435 records total), each with its
mirror-frame pair, panel tuples, conflict labels, and `label_status`,
accompanied by a schema document, datasheet, and license. The release is
produced by a deterministic builder that re-emits a byte-identical artifact, so
every reported number is reproducible from a single frozen unit; any change
after submission increments the version.

> [메타] MORABLES 의 "3.6 Summary of Dataset Variants" 자리. 동결 카운트
> 657/218/439(+control 217 = release 435) 는 BENCHMARK_TODO #4 DONE 기준
> (misfit hand-exclude 폐기 → 657 유효). Table 1 은 conflict-pair 분포(동결 후
> 확정). "deterministic byte-identical builder" = 재현성 pre-emptive.

> [메타] **§3 (구축) 초안 종료.** 아래는 이번 패스 범위 밖 — 내용 보존 +
> B-구조 배치만 표시. 상세 초안은 다음 패스에서.

---

## 4 Benchmark Validation  *(stub — 다음 패스에서 본문화)*

> [메타] **Decision B (2026-05-19): RQ1 을 §3.6 평가프로토콜 안 subsection 이
> 아니라 독립 §4 로 승격.** 이게 "이 벤치마크를 왜 믿어야 하나"의 유일한
> 정량 증거이자 논문 spine 이므로 분석(§6)보다 앞에 단독 배치. §4 도입부에
> "왜 결과를 분리해 먼저 보고하는가" 1문장 필요(MORABLES 엔 없는 deviation).

**RQ1 — does the disagreement filter expose omission bias better than random?
(the spine).** We compare three equal-$n$ sampling arms drawn under the
*identical* malformed-record exclusion: `labeled` (OMIT, the disagreement
filter), `random_complement` (random draw from the 439-scenario non-labeled
valid pool; primary contrast), and `random_full` (random draw from the full
valid pool; auxiliary). The test is a hand-rolled **exact one-sided Wilcoxon
signed-rank** over models (each model is one paired observation:
$\mathrm{OBR}_{\text{labeled}}-\mathrm{OBR}_{\text{random}}$), complemented by a
scenario-cluster bootstrap CI on the mean difference. With only five models the
exact signed-rank has a p-floor of $0.03125$; we therefore lead with the
bootstrap effect-size CI and per-model sign agreement, treating the p-value as
secondary. Result (frozen `outputs/experiments/0518/1715/`): 5/5 models
$\Delta\mathrm{OBR}>0$, mean $+0.2226$ (95% CI $[0.190, 0.256]$), exact
one-sided $p=0.03125$; Figure 2 = per-model OBR paired bar with Wilson CI.

> [메타] n=5 Wilcoxon p-floor 를 *본문에서 먼저* 인정 → "p=.03 은 sign test
> 아니냐" 공격 사전 차단(memory `project_rqs` §5-model trap). 수치는 동결본
> (BENCHMARK_TODO #1 DONE).

## 5 Experimental Setup  *(relocated from old §3.6 — 다음 패스에서 정리)*

> [메타] 구 §3.6 Evaluation Protocol 의 *비-RQ1* 부분이 여기로 이동(B-구조:
> setup = §5). MORABLES "§5 Experimental Setup" / BiasFreeBench "§4
> Implementation Design" 와 동형. 내용 손실 방지용 보존; subsection 정리·축약은
> 다음 패스.

**Models.** Five cross-vendor instruction models spanning open and closed
weights and a wide a-priori bias range: `meta-llama/llama-3.1-8b-instruct`,
`google/gemma-3-12b-it`, `openai/gpt-4o-mini`, `qwen/qwen3.5-9b`,
`google/gemini-2.0-flash-001`. The spread is deliberate: a robust construction
signal should hold across vendors and across very different baseline
$\mathrm{OBR}$.

**Querying.** Each model answers each frame under a forced binary system prompt
(`ANSWER: yes|no`, **no rationale or chain-of-thought**;
`[APPENDIX: EVAL_SYSTEM]`). We omit CoT deliberately: eliciting reasoning
changes the construct from "the model's snap moral disposition" to "its post-hoc
justification." Strict regex plus first-token fallback parsing; unparseable →
`null`, excluded from $n$. Refusals are counted and reported separately, never
prompt-engineered away, since refusal rate is itself a model property and
suppressing it would break baseline comparability. One model is not perfectly
deterministic at $T{=}0$ on our gateway and over-refuses harm-laden dilemmas;
both are reported and absorbed by the within-scenario paired tests
(`[APPENDIX: determinism/refusal audit]`).

**Estimation.** $\mathrm{OBR}/\mathrm{ABR}/\mathrm{FCR}$ (Eq. §3.1) per model
and per (model $\times$ conflict-pair) cell with Wilson 95% CIs.

## 6 Results and Analysis  *(RQ2 / RQ3 / RQ4-mitigation — 다음 패스에서 본문화)*

> [메타] B-구조: RQ2(conflict-type, descriptive) / RQ3(moral fingerprint,
> ★novel core) / RQ4(prompt-level mitigation, M0–M3b) 가 결과 한 섹션의
> subsection. 구 §3.6 의 RQ2/RQ3/RQ5 분석 설계 + 구 §3.7 Mitigation 전문이
> 여기로 이동(아래에 보존). M4(철학 5 MAD) 폐기 상태 유지, primary={M2,M3,M3b}.

### 6.x Analysis design (보존 — old §3.6 tail)

RQ2 (descriptive): is $\mathrm{OBR}$ non-uniform across conflict-pair types? —
mixed-effects with `scenario_id` random intercept, post-hoc top-vs-bottom
contrast, no directional claim. RQ3 (moral fingerprint): on framing-consistent
answers, which camp does each model align with? — a per-model five-philosophy
leaning vector tested against a permutation null that fixes the model's own
$yn{:}ny$ marginal (isolating *which* philosophy, not yes/no tilt), with a
$\sum z^2$ omnibus, plus a cross-model minimum-cosine test under per-scenario
label shuffling and an average-linkage dendrogram; H3 = "$\geq 1$ model omnibus
$p<0.05$ **and** $\geq 1$ model pair cosine $<0.85$." RQ5 (robustness): is $NN$
separable from $YY$? — Spearman $\rho$ between per-cell $\mathrm{OBR}$ and
$\mathrm{ABR}$. All tests hand-rolled (no SciPy); `[APPENDIX: test
implementations]`.

### 6.y Mitigation (보존 — old §3.7)

**Why prompt-level.** Mitigation is restricted to within-subject, prompt-level
interventions on a single model. No condition injects a moral philosophy,
persona, or any artifact of the construction pipeline (the conflict label is
*never* shown) at inference time: each is a self-contained instruction
applicable to an arbitrary live query, so a positive result is directly
deployable rather than an artifact of our offline typing. This keeps mitigation
consistent with the construction/evaluation separation of §3.4 and §5 — the
philosophy panel stays a construction-time signal and never re-enters at
evaluation. All conditions run over the full 218-scenario OMIT against the same
five models (no subsampling), and **M0** reproduces §5 exactly so that it is the
left member of every paired test.

**Conditions.** Each is a single prompt template targeting the omission-bias
*mechanism* — the frame asymmetry inside the model's own reasoning — by
distinct, theory-neutral means.
**M0** (baseline) is the §5 forced-choice prompt verbatim.
**M1** adds generic zero-shot chain-of-thought and controls for *mere
reflection*, so any gain of M2/M3/M3b over M1 is attributable to their specific
structure, not deliberation per se `[CITE: Kojima et al. 2022, zero-shot CoT]`.
**M2** is *considering-the-opposite*, process-only and without a decision rule:
a tentative answer + one-sentence reason on one frame, then re-examination of
the opposite frame "on its own merits," then a final yes/no to *each* frame —
the dual answers preserve the cross-frame tuple needed to score $NN$
`[CITE: Lord, Lepper & Preston 1984; Mussweiler 2000]`.
**M3** is the single-frame variant: tentative answer, then one sentence giving
the strongest case for the *opposite* of the model's own tentative answer, then
a final answer `[CITE: Lord et al. 1984; Madaan et al. 2023, Self-Refine]`.
**M3b** is a *decision ledger*: one neutral factual sentence for the
yes-outcome and one for the no-outcome, then a decision — directly confronting
the act/omission outcome asymmetry that drives the bias
\citep{spranca1991omission}.
We strip any consequentialist decision rule, consistency demand, or
omission/framing cue: such wording would inject the construct we measure
(`[APPENDIX: condition prompts]`).

**Outcome decomposition.** For scenarios where M0 produced $NN$, a condition's
effect is decomposed:
$$\Pr\big(\text{M0}{=}NN \to \{YN, NY\}\big)\ \ (\textbf{credited}),\qquad
  NN \to YY\ \ (\textbf{flagged, not credited}).$$
Moving from omission bias to a frame-consistent answer is genuine de-biasing;
$NN \to YY$ merely substitutes action bias and is flagged, never counted —
a deliberately conservative criterion.

**Testing.** Per (model, condition) an exact one-sided McNemar test for a
decrease in $NN$. Confirmatory family = **primary set** $\{\text{M2, M3, M3b}\}$
with Bonferroni; M1 is the reflection control under FDR-BH. H4 succeeds iff
$\geq 1$ *primary* condition shows a credited reduction at corrected $p<0.05$
*and* exceeds generic CoT (M1) for that same model. Should H4 fail, we report
it as evidence that prompt-level mitigation is limited and that training-level
intervention is the needed follow-up, rather than hiding a null.

> [메타] **M4(철학 5 MAD) 완전 폐기 (2026-05-18).** 조건 = M0/M1/M2/M3/M3b,
> primary={M2,M3,M3b}. credited vs flagged 분해 = conservative-by-design(NN→YY
> 부풀림 차단). H4 실패 reframe = pre-emptive(memory `project_e7_keystone`).
> "label-free" 라벨 미사용(2026-05-19): M4 제거로 persona-injection 대비항이
> 사라져 무의미 — "prompt-level, 파이프라인 산물 미사용" 사실만 서술.

---

### TODO / 확정 필요

- `[X]` Stage-1 survivor 수·`[X]\%` (Stage-1, Stage-2 비율): E1/E3/E7 동결 후
  채움. **valid=657 / labeled=218 / complement=439 / release(+control 217)=435
  확정** (paradigm-misfit hand-exclusion 폐기, 2026-05-18).
- **§4 본문화**: B-구조 deviation 정당화 1문장(왜 validation 을 결과보다 먼저
  단독 §) + Figure 2 작도 + 0518/1715 동결 수치 표.
- **§5 정리**: old §3.6 비-RQ1 부분 축약·subsection 화(Models / Querying /
  Estimation). **§6 본문화**: RQ2→RQ3→RQ4 순, MORABLES "Evaluations" 동형.
- `[APPENDIX: …]`: reframe 프롬프트 전문, philosophy 프롬프트, EVAL_SYSTEM,
  determinism/refusal audit, hand-rolled test 구현, condition prompts(M0–M3b),
  Appendix B(flag with/without robustness), Appendix C(dual-κ protocol) →
  `paper/appendix.md` 생성 시 연결.
- Figure 1 실제 작도 (caption 은 §3.2 확정본; construction-only, no persona
  injection, M4/persona-contrast 도식 없음). Table 1 conflict-pair 분포.
- **`[CITE: …]` 4개 신규 (§6.y, fabricate 금지 — bib 에 없음):** Kojima et al.
  2022, Lord/Lepper/Preston 1984, Mussweiler 2000, Madaan et al. 2023. Google
  Scholar BibTeX → `references.bib` 추가 후 placeholder 치환.
  `spranca1991omission` 은 이미 bib 존재.
- 기존 인용 키(`cheung2025amplified`, `scherrer2023moralbeliefs`,
  `spranca1991omission`) 외 §3 본문 신규 인용 없음.
