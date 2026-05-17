# Related Work (draft v0 — 2026-05-17)

> 초안. ACL `\citep{}` / `\citet{}` 키는 `paper/references.bib` 와 일치. RQ 프레이밍이
> 확정되기 전이라, plan 이 바뀌어도 살아남는 네 기둥(① 인간 omission/framing bias →
> LLM 확장, ② LLM 도덕 신념 벤치마크, ③ LLM 의 인지·프레이밍 편향 일반, ④ 도덕철학
> 프레임워크의 persona 조건화)으로만 구성. 본문 분량 목표 ≈ 0.75 p.

## 2. Related Work

### 2.1 Omission bias and framing effects in moral judgment

The tendency to judge harmful *actions* as worse than equally harmful *omissions* —
the **omission bias** — is a well-documented finding in the moral psychology of
judgment and decision-making \citep{spranca1991omission, ritov1990reluctance,
baron1994omission}. It is closely tied to the broader phenomenon of *framing effects*,
in which logically equivalent descriptions of the same decision elicit systematically
different choices depending on whether an outcome is framed as a gain or a loss, or as
resulting from acting or refraining \citep{tversky1981framing}. For our purposes,
distinguishing a bias from a stable preference requires **framing invariance**: a
preference for inaction counts as omission bias only if it survives an action↔omission
re-description of the *same* underlying dilemma.

\citet{cheung2025amplified} extend this paradigm to large language models. Across four
studies — including a preregistered replication and Reddit-sourced everyday dilemmas —
they show that LLMs exhibit an omission bias that is *stronger* than that of a
representative U.S. sample, and additionally a "no"-bias whereby models flip their
recommendation depending on question wording. Crucially, they operationalize the bias
through paired action↔omission frames and provide evidence that it arises primarily
from fine-tuning for chatbot applications rather than from pretraining. Our work takes this operational definition
(framing-invariant inaction over a paired-frame construction) as its measurement
primitive, but departs from \citet{cheung2025amplified} in two ways: we measure the bias
*across a broad set of contemporary models* rather than around a single GPT-4 anchor,
and we condition the analysis on *which moral-philosophical conflict* a scenario
instantiates rather than the single utilitarian↔deontological axis used in prior work.

### 2.2 Benchmarking the moral beliefs of LLMs

A growing line of work probes the moral content encoded in LLMs through large-scale
scenario surveys. \citet{scherrer2023moralbeliefs} introduce **MoralChoice**, 680
high-ambiguity and 687 low-ambiguity scenarios, each with a description, two actions,
and rule-violation labels, administered to 28 open- and closed-source models; they find
that closed-source models tend to agree with one another and that responses are
sensitive to question wording — the same wording-sensitivity that \citet{cheung2025amplified}
later formalize as a framing bias. \citet{hendrycks2021ethics} (ETHICS) and related
moral-reasoning suites instead score models against aggregated human moral judgments.
These benchmarks measure *what* a model decides; they do not isolate *whether a decision
is an artifact of framing*, nor do they label scenarios by the moral-philosophical
disagreement they provoke. We build directly on the MoralChoice high-ambiguity pool but
re-cast each scenario into a paired mirror frame and attach a philosophy-conflict label,
turning a single-frame belief survey into a paired-frame bias benchmark. MoralChoice
supplies the scenario pool, but identifying *which* scenarios sit on a contested moral
fault line requires an independent construction-time signal; we obtain it from the
disagreement of a moral-philosophy panel (§2.4).

### 2.3 Cognitive and framing biases in LLMs

Beyond the moral domain, LLMs have been shown to exhibit human-like patterns of
cognitive bias: sensitivity to option order and prompt phrasing
\citep{zheng2024large, pezeshkpour2024sensitivity}, sycophantic agreement with the
user's stated view \citep{perez2023discovering, sharma2024sycophancy}, and classic
judgment-and-decision framing effects reproduced in controlled prompts
\citep{binz2023using, echterhoff2024cognitive}. Omission bias, in this light, is one
member of a broader class of framing-dependent failures; what distinguishes the moral
case is that the favored response (inaction) carries normative weight.
Our contribution to this line is methodological: rather than reporting a single bias
magnitude per model, we profile *where* in a typed space of moral conflicts each model's
framing-invariant inaction concentrates, yielding a per-model bias *signature* rather
than a scalar.

### 2.4 Moral-philosophy frameworks and persona conditioning

Several works elicit or steer LLM behavior by conditioning on a normative stance —
prompting a model to reason as a utilitarian vs. a deontologist, or otherwise assigning
it a moral persona \citep{simmons2023moral, jiang2021delphi}. Prior uses of
philosophical framing largely
treat it as a *manipulation* of the model under test. We instead use a panel of five
moral-philosophy personas as a **construction-time signal**: the panel's *disagreement*
across the paired frames identifies moral fault-line scenarios and types them by the
specific philosophical conflict (e.g., utilitarian vs. care) they expose. The
philosophy panel thus plays no role at evaluation time — the evaluated models receive no
persona injection — keeping the construction signal separated from the
evaluation-time metric.

---

### TODO / 확정 필요 (plan 의존)

- §2.3 의 contribution 문장은 RQ3 (model signature) 가 plan 에 남는다는 전제. RQ 셋이
  바뀌면 마지막 문장만 교체.
- ETHICS 외 도덕 벤치마크(Delphi, MoralExceptQA, Social Chemistry 등) 추가 인용은
  scope 확정 후. 현재는 핵심만.
- 인용 검증 완료 (2026-05-17, Codex review + WebSearch): 전 항목 실존 확인.
  `cheung2025amplified`·`scherrer2023moralbeliefs` = repo PDF, `echterhoff2024cognitive`
  = ACL Anthology (Findings EMNLP 2024, pp.12640-12653) 확정, 나머지는 Codex 대조.
