# Methodology (draft v0 — 2026-05-18)

> [메타] 본문은 영어, 메타 코멘트는 한국어. LaTeX paste-ready markdown.
> Section 번호는 Related Work = §2 다음이므로 **§3**. 인용 키는 `paper/references.bib`
> 와 일치(`cheung2025amplified`, `scherrer2023moralbeliefs`).
> 벤치마크 placeholder 이름 = `\textsc{OmissionBench}` — 확정 시 전역 치환.
> 동결 전 수치는 `[X]` placeholder. 구현 세부(프롬프트 전문, iteration 수)는
> `[APPENDIX: …]` 로 미루고 motivation 은 본문에 남긴다.

## 3 Methodology

We construct a benchmark whose every item is a *paired* action↔omission mirror
frame, and we type each item by the disagreement of a five-philosophy persona
panel. We then measure framing-invariant inaction across models and test
label-free mitigation. Throughout, we keep two ideas strictly separated: the
philosophy panel is a **construction-time selection signal**, whereas
framing-invariant inaction is the **evaluation-time metric**. We make no causal
claim that disagreement *produces* the bias.

> [메타] 위 문단이 method 의 thesis-lock. CLAUDE.md 의 "두 아이디어를 섞지
> 말 것" 규칙 + abstract 의 "no causal claim; construction-time signal only" 와
> 1:1 대응. 리뷰어가 "패널 불일치가 편향의 원인이라 주장하나?" 라고 물 때 막는
> pre-emptive 문장.

### 3.1 Problem Formulation

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
  \mathrm{FCR}=\frac{\#YN+\#NY}{n},\quad
  \mathrm{FCR}=1-\mathrm{OBR}-\mathrm{ABR}.$$
$\mathrm{OBR}$ (Omission Bias Rate) is the primary quantity; $\mathrm{ABR}$
(Action Bias Rate) and $\mathrm{FCR}$ (Frame-Consistent Rate) characterize the
remaining mass. Refusals and unparseable generations are excluded from $n$ and
reported separately (§3.6), never silently coerced.

> [메타] One equation rule 충족: OBR/ABR/FCR + cross-frame equivalence 두 식.
> 모든 기호($s,\phi,a,n$)를 사용 전에 정의. NN = bias 의 조작적 정의를
> abstract("framing-invariant inaction")·related work §2.1 과 같은 표현으로 고정.

### 3.2 Overview

Figure 1 traces the full pipeline. We start from the MoralChoice
high-ambiguity pool \citep{scherrer2023moralbeliefs} (680 scenarios), label
which option is the lexical inaction, and **re-cast** each scenario into a
paired mirror frame (§3.3); after excluding malformed and paradigm-misfit
records this yields **654** valid paired scenarios. A five-philosophy panel then
answers both frames of every surviving scenario (§3.4). A two-stage
**disagreement filter** keeps only scenarios that sit on a contested moral fault
line and labels which philosophical camps the fault line opposes (§3.5); the
**217** labeled scenarios *are* `\textsc{OmissionBench}`. We evaluate five
cross-vendor models on it under a
forced-choice protocol (§3.6) and finally test five prompt-level, label-free
mitigation conditions (§3.7).

> [메타] Figure 1 caption 은 self-contained 해야 함(method.md §2). 아래가 초안.

**Figure 1. Overview of `\textsc{OmissionBench}` construction and evaluation.**
Given a MoralChoice high-ambiguity scenario, we (1) re-cast it into a paired
Frame A / Frame B that describe the *same world* but flip the default
trajectory, so that choosing inaction means answering *no* in A and *yes* in B;
(2) query a five-philosophy persona panel (utilitarian, deontologist, virtue,
care, contractualist) on both frames; (3) **keep only scenarios on which the
panel disagrees** and label the opposing philosophical camps; and (4) evaluate
held-out models — which receive *no* persona injection — for framing-invariant
inaction ($NN$). Unlike prior work that probes a single model around a single
utilitarian↔deontologist axis, the panel is used purely as a construction-time
selection signal over a five-philosophy conflict space.

### 3.3 Stage 1 — Action↔Omission Mirror-Frame Construction

**Why.** Measuring a framing effect requires a counterfactual frame that is
*structurally symmetric* to the original: it must change the default trajectory
and nothing else. Naive negation does not achieve this — it routinely
introduces an authority or agency confound (e.g., turning "do you act?" into
"do you defy your superior?"), so that a model's "no" reflects deference rather
than omission. We therefore generate frames under explicit symmetry invariants.

**How.** Each scenario is reframed by an LLM under a balanced reframing prompt
(`[APPENDIX: reframe SYSTEM_PROMPT v4]`) that enforces four hard invariants:
(i) **same-world symmetry** — both frames name the same parties and outcomes,
any external party present in B is present-but-inert in A; (ii) **no-authority
confound** — the external party is never a superior with authority over the
agent; (iii) a **preparation-vs-delivery rule** — the agent may be a past actor
only for reversible preparation verbs (scheduled, drafted, queued), never for
completed delivery verbs (sent, administered, killed); and (iv) cross-frame
outcome equivalence (Eq. in §3.1). The natural social structure of the act
selects one of five role TYPEs — `T_SELF` (agent self-prepared, schedulable),
`T_PEER` (peer, never a superior), `T_ADMIN` (institutional/automatic
pipeline), `T_MUTUAL` (face-to-face speech act), `T_PRIOR` (predecessor in the
agent's role) — rather than forcing one structure on every scenario. A schema
auto-check rejects malformed generations.

**Intuition.** Frame A withholds an intervention that has not yet started;
Frame B withholds completion of the *same* intervention already in motion. The
world and its two outcomes are held fixed by construction, so any systematic
asymmetry in a model's answers across the pair is attributable to framing, not
to a change in stakes.

We exclude 4 malformed records (`G_116/125/228/330`, null frames) and 3
paradigm-misfit scenarios (`H_001/005/006`: split-second or coerced
personal-hand acts whose moral psychology does not survive reframing); from the
661 reframed records this leaves **654** valid paired scenarios. The same
exclusion is applied identically to every downstream sampling arm (§3.6) so that
arm differences reflect only the filter.

> [메타] 4 invariant 마다 "왜"가 한 문장씩(motivation-driven). authority confound
> 단락이 핵심 pre-emptive — 리뷰어의 "프레임 B 의 no 는 그냥 복종 아니냐" 공격
> 차단. 프롬프트 전문은 appendix.

### 3.4 Stage 2 — Moral-Philosophy Persona Panel

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
> "philosophy-neutral" 을 method 에 박는 pre-emptive(패널이 결과를 미리
> 심었다는 공격 차단). T=0,n=1 은 memory `feedback_experiment_defaults`.

### 3.5 Stage 3 — Disagreement Filter and Conflict Labeling

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
`dropped_one_sided_yn`, `dropped_one_sided_ny`, or `dropped_all_excluded`). After the misfit exclusion,
the **217** `labeled` scenarios (218 in the raw labels file; `[X]\%` of Stage-1
survivors) constitute `\textsc{OmissionBench}`; the **437** non-labeled valid
scenarios form the complement pool used as the RQ1 random arm (§3.6). All
measurement is on the labeled subset.

**Non-independence.** Because we emit the *full* cross-product, one scenario
contributes to several $(\mathrm{yn}\times\mathrm{ny})$ conflict cells. All
per-cell inference therefore treats `scenario_id` as a random intercept (or
uses cluster-robust standard errors); we state this here rather than bury it in
the analysis.

We reiterate that this filter *selects and types* scenarios — it does not
explain why any model is biased. The labels are conditioning variables for
analysis (§3.6), not a causal mechanism.

> [메타] §3.5 가 novelty 의 심장. 필터 규칙을 식으로 박음(엄밀성). full
> cross-product → 비독립성 → random intercept 를 *여기서* 선언한 게
> pre-emptive(리뷰어의 "시나리오 재사용으로 n 부풀린 것 아니냐" 차단).
> 마지막 문단이 다시 한 번 causal-claim 차단.

### 3.6 Evaluation Protocol

**Models.** We evaluate five cross-vendor instruction models spanning open and
closed weights and a wide a-priori bias range:
`meta-llama/llama-3.1-8b-instruct`, `google/gemma-3-12b-it`,
`openai/gpt-4o-mini`, `qwen/qwen3.5-9b`, and `google/gemini-2.0-flash-001`.
The spread is deliberate: a robust construction signal should hold across
vendors and across models with very different baseline $\mathrm{OBR}$.

**Querying.** Each model answers each frame of each scenario under a forced
binary system prompt (`ANSWER: yes|no`, **no rationale or chain-of-thought**;
`[APPENDIX: EVAL_SYSTEM]`). We omit CoT deliberately: eliciting reasoning
changes the construct from "the model's snap moral disposition" to "the model's
post-hoc justification," which is a different measurement. We parse with a
strict regex plus a first-token fallback; unparseable answers are `null` and
excluded from $n$. Refusals are counted and reported separately, never
prompt-engineered away, since refusal rate is itself a model property and
suppressing it would break baseline comparability. We note as a
measurement-validity caveat that one model is not perfectly deterministic at
$T{=}0$ on our gateway and that it over-refuses harm-laden dilemmas; both are
reported and handled by the within-scenario paired tests below
(`[APPENDIX: determinism/refusal audit]`).

**Estimation.** $\mathrm{OBR}/\mathrm{ABR}/\mathrm{FCR}$ (Eq. in §3.1) are
reported per model and per (model $\times$ conflict-pair) cell with Wilson 95\%
confidence intervals.

**RQ1 — does the filter work? (the spine).** We compare three equal-$n$
sampling arms drawn under the *identical* malformed+misfit exclusion:
`labeled` (the disagreement filter), `random_complement` (random draw from the
non-labeled valid pool; primary contrast), and `random_full` (random draw from
the full valid pool; auxiliary). The test is a hand-rolled **exact one-sided
Wilcoxon signed-rank** over models (each model is one paired observation:
$\mathrm{OBR}_{\text{labeled}}-\mathrm{OBR}_{\text{random}}$), complemented by a
scenario-cluster bootstrap CI on the mean difference. With only five models the
exact signed-rank has a p-floor of $0.03125$; we therefore lead with the
bootstrap effect-size CI and per-model sign agreement, treating the p-value as
secondary.

**RQ2–RQ5 (analysis).** RQ2 (descriptive): is $\mathrm{OBR}$ non-uniform across
conflict-pair types? — mixed-effects with `scenario_id` random intercept,
post-hoc top-vs-bottom contrast, no directional claim. RQ3 (moral
fingerprint): on framing-consistent answers, which camp does each model align
with? — a per-model five-philosophy leaning vector tested against a
permutation null that fixes the model's own $yn{:}ny$ marginal (so the test
isolates *which* philosophy, not yes/no tilt), with an $\sum z^2$ omnibus, plus
a cross-model minimum-cosine test under per-scenario label shuffling and an
average-linkage dendrogram; H3 is the conjunction "$\geq 1$ model omnibus
$p<0.05$ **and** $\geq 1$ model pair cosine $<0.85$." RQ5 (robustness): is $NN$
separable from $YY$? — Spearman $\rho$ between per-cell $\mathrm{OBR}$ and
$\mathrm{ABR}$. All tests are hand-rolled (no SciPy);
`[APPENDIX: test implementations]`.

> [메타] CoT 제거의 "왜"를 한 문장 motivation 으로(method.md 핵심). n=5
> Wilcoxon p-floor 를 *본문에서 먼저* 인정 → 리뷰어가 "p=.03 은 sign test
> 아니냐" 칠 때 이미 막혀 있음(pre-emptive, memory `project_rqs` §5-model trap).
> RQ4 는 분량상 별도 §3.7.

### 3.7 Mitigation Conditions

**Why label-free.** A mitigation that needs the conflict label at inference
time is not deployable — the label is a property of our construction pipeline,
not of a live query. All conditions are therefore within-subject, prompt-level,
and **label-free**: no conflict or oracle label is ever shown to the model.

**Primary conditions (label-free, single-model).** Our mitigation claim rests
on conditions that target the omission-bias *mechanism* — the frame asymmetry
inside a single model's own reasoning — without any philosophical persona.
**M0** is the baseline and reproduces §3.6 exactly (the left member of every
paired test). **M1** adds generic chain-of-thought, controlling for "mere
reflection." **M2** presents both frames simultaneously. **M3** asks the model
to state its strongest counter-argument, then re-answer. **M3b** has the model
write the yes/no outcome ledger, then answer. None of these injects a moral
philosophy at inference time, so they remain consistent with the construction /
evaluation separation asserted in §3.4 and §3.6.

**Bounded contrast.** **M4** is a five-philosophy multi-agent deliberation
(independent round → cross-rebuttal → majority vote). We include it *only* as a
ceiling contrast, and we are explicit about its status: M4 re-introduces, at
inference time, exactly the philosophical-persona injection that our
construction deliberately keeps out of evaluation (§2.4). A positive M4 result
therefore does *not* support a deployable, label-free recommendation — it is the
known multi-agent-debate effect applied to this domain, and is reported as an
upper bound, not as a proposed method. Its purpose is to answer one question:
does a cheap, philosophy-free, single-model prompt (M2/M3/M3b) approach what the
expensive five-persona committee achieves? If so, the construction signal is
shown to be unnecessary at inference; if M4 dominates, we bound that gain as the
generic debate effect rather than a contribution of this paper.

**Outcome decomposition.** For scenarios where M0 produced $NN$, a condition's
effect is decomposed by where the bias exits to:
$$\Pr\big(\text{M0}{=}NN \to \{YN, NY\}\big) \;\; (\textbf{credited}),
  \qquad
  NN \to YY \;\; (\textbf{flagged, not credited}).$$
Moving from omission bias to a frame-consistent answer is genuine
de-biasing; moving from $NN$ to $YY$ merely substitutes action bias and is
flagged, never counted as success — a deliberately conservative criterion.

**Testing.** Per (model, condition) we run an exact one-sided McNemar test for
a decrease in $NN$. The confirmatory family is the **primary set**
$\{\text{M2, M3, M3b}\}$ with Bonferroni correction; M1 and the M4 bounded
contrast are secondary and reported under FDR-BH. H4 succeeds iff at least one
*primary* condition shows a credited reduction at corrected $p<0.05$ *and*
exceeds generic CoT (M1) — M4 is never required for H4 and is reported only as
the ceiling against which the label-free conditions are read. Should H4 fail, we
report it as evidence that prompt-level mitigation is limited and that
training-level intervention is the needed follow-up, rather than as a null
result to be hidden.

> [메타] label-free 의 "왜"(deployability)를 motivation 으로 먼저. credited vs
> flagged 분해가 conservative-by-design — 리뷰어의 "NN→YY 를 개선이라 부풀린
> 것 아니냐" 차단. H4 실패 시 reframe 문장이 pre-emptive(memory
> `project_e7_keystone`).
> **M4 = bounded contrast 로 강등 (2026-05-18 결정).** primary={M2,M3,M3b}
> (label-free single-model, §2.4 차별점과 모순 없음). M4 는 inference 때
> persona 주입을 *의도적으로 다시 넣은* ceiling — 양성이어도 deployable
> 권고 아님, generic MAD 효과로 범위 제한. claim 희석 + §3.4/§3.6/§2.4
> 자기모순 둘 다 해소. H4 는 M4 없이 성립.

---

### TODO / 확정 필요

- `[X]` Stage-1 survivor 수·`[X]\%` (Stage-1, Stage-2 비율) + RQ1 `[+X] OBR`
  : E1/E3/E7 동결 후 채움. valid=654 / labeled=217(파일 218) / complement=437 은 확정.
- `[APPENDIX: …]` 5곳: reframe 프롬프트 전문, philosophy 프롬프트, EVAL_SYSTEM,
  determinism/refusal audit, hand-rolled test 구현 → `paper/appendix.md` 생성 시 연결.
- Figure 1 실제 작도 (caption 은 §3.2 에 확정본).
- 인용 키 2개(`cheung2025amplified`, `scherrer2023moralbeliefs`) 외 method 에서
  새 인용 없음 — fabricate 금지 규칙 준수. persona-conditioning 선행연구를 §3.4
  에 1개 달려면 related_work §2.4 의 `simmons2023moral` 재사용(신규 생성 X).
- 제출 시 영문 통일 (related_work 한글본과 달리 method 는 이미 영문).
