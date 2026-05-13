# Research Plan v3 — EMNLP 2026 Long Paper (Persona-Vector pivot)
**제목 (가제):** *Omission Bias as a Direction in Moral Persona Space: Extracting, Monitoring, and Steering Framing-Invariant Inaction in LLMs*

> 작성일: 2026-05-12 · 대상 학회: **EMNLP 2026 (Long paper, 8p + refs)**
> 본 문서는 `RESEARCH_PLAN_v2.md` (safety-extended behavioral plan) 위에서 **분석·완화 layer만 persona vector 방법론으로 전면 교체**한 개정판이다. **벤치마크 구축 (§3) 은 v2와 동일** — paired action↔omission mirror frame, 6-philosophy panel fault-line filtering, harm-asymmetry stratification, low-stake control 모두 보존.
> 추가 anchor 한 편:
> - Chen, Arditi, Sleight, Evans, Lindsey (2025, *arXiv preprint*) — `PERSONA VECTORS: MONITORING AND CONTROLLING CHARACTER TRAITS IN LANGUAGE MODELS` (Anthropic Fellows). 자동화된 trait-vector 추출 파이프라인 + monitoring/steering/pre-finetuning data screening 응용.
> 기존 anchor:
> - Cheung, Maier, Lieder (2025, *PNAS*) — framing-invariant omission bias.
> - Scherrer, Shi, Feder, Blei (2023, *NeurIPS*) — MoralChoice, sub-cluster A 합의.

---

## 1. Context — v2의 "지점"과 v3의 "한 단계 깊이"

v2는 **행동 수준** 측정이었다: paired mirror frame을 만들고, philosophy injection을 system prompt로 주고, simultaneous framing × multi-philosophy consensus의 2×2 factorial을 돌려 OBR이 어떻게 변하는지 본다. 결과는 "어떤 prompt가 OBR을 낮추느냐"는 input-output 관찰이다.

v3는 **표상 수준** 으로 한 단계 내려간다. Chen et al. 2025의 *persona vectors* 파이프라인이 자연어로 묘사된 trait을 residual stream의 선형 방향으로 자동 추출하는 일반 도구를 제공한다 (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct에서 검증, evil/sycophancy/hallucination 등). 이 도구를 우리의 두 가지 trait에 적용한다:

- **omission preference vector** — *"동등한 결과를 갖는 두 옵션 중 행동 (개입, 변경) 보다 무행동 (방치, 유지) 을 선호하는 경향"* 의 잠재 방향.
- **6 philosophy vectors** — utilitarian / deontological / virtue / care / contractualist / DDE 각각의 system-prompt-conditioned activation 차이로 정의한 6개의 도덕철학 방향.

이 두 종류의 벡터로 v2의 행동 수준 결과를 *기계적으로 (mechanistically) 설명* 한다:

1. **OBR이 정말 선형 표상 위에 존재하는가** (= omission이 persona vector 추출 가능한 trait인가) — 존재 검증.
2. **omission vector는 6개 philosophy vector 공간에서 어디에 있는가** — PNAS prior ("omission bias는 다수의 윤리 framework에 걸쳐 robust") 와 정합한가, 아니면 특정 학파 (예: DDE 의 doing/allowing 비대칭 원리) 와 정렬되는가.
3. **prompt를 만들지 않고도 OBR을 예측·제어할 수 있는가** — activation-level monitor + steering이 v2의 prompt-level intervention보다 효과적인가, capability 손실은 어디까지인가.
4. **RLHF가 정확히 어떤 representation을 instill하는가** — Llama-3.1-8B-base vs Llama-3.1-8B-Instruct의 activation shift를 omission vector에 projection하여 PNAS Study 4 ("RLHF가 omission bias 유발") 의 첫 mechanistic 분해를 제시.
5. **finetuning 데이터를 사전 screening 할 수 있는가** — projection difference (ΔP) 가 omission-bias-inducing 데이터셋·샘플을 사전 식별하고, preventative steering이 absorption을 막을 수 있는지.

이 5개 축이 v3의 RQ가 된다. v2의 thesis ("Scherrer sub-cluster A 합의 = omission bias의 그림자") 와 construct validity (OBR ≠ safety refusal) 는 v3에서 **mechanistic 증거** 로 강화된다 — sub-cluster A 모델군이 같은 omission direction에 활성화되는지, refusal vector (Arditi+ 2024) 와 cosine이 낮은지로 직접 검증 가능.

EMNLP 2026 special theme — *"contemporary models as experimental instruments for cognitive science"* — 정렬은 v2보다 강해진다: 우리는 도덕적 편향을 행동 패턴이 아니라 *모델 내부의 잠재 방향* 으로 측정하고 조작한다.

---

## 2. Research Questions

| RQ | 질문 | 정량 측정 / 가설 |
|----|------|------------------|
| **RQ1** | "Omission preference" 가 mid-size open chat LLM의 residual stream에서 **선형 방향** 으로 추출 가능한가? 그 방향으로의 steering이 paired-frame benchmark의 OBR을 인과적으로 변화시키는가? | Chen+ 2025 파이프라인으로 omission persona vector 추출 (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct). Steering coefficient α ∈ {−3,−2,−1, 0, +1, +2, +3} 으로 paired-frame inaction 비율의 monotonic 변화 검정 (Spearman ρ ≥ 0.7). MMLU 성능 동시 보고. |
| **RQ2** | omission vector는 6개 philosophy vector 공간에서 어디에 있는가 — utilitarian과 anti-aligned, DDE와 aligned인가? Omission bias의 *철학적 정체* 를 vector geometry로 답할 수 있는가? | 7개 vector 간 cosine similarity matrix; omission vector를 philosophy basis 위에 linear regression으로 decomposition; PCA/UMAP "moral persona space" 시각화. 사전 가설: cos(omission, DDE) > cos(omission, utilitarian); omission vector는 (Kant + Virtue + Care + DDE) span 위에 더 가깝다. |
| **RQ3** | omission vector projection이 **deployment-time monitor** 로 작동하는가 — 생성 *전* 마지막 prompt token의 activation projection이 그 paired-frame scenario의 inaction 선택을 예측하는가? | 5개 평가 모델 (open) × paired-frame 800~950 × 양 frame → final prompt token activation을 omission vector에 projection → 후속 inaction 선택과의 상관 (point-biserial r ≥ 0.5 예상, Chen+ 2025 system-prompt 실험 r=0.75-0.83 의 더 어려운 setting). Scenario-level OBR과 mean projection의 상관도 보고. |
| **RQ4** | inference-time activation steering이 v2의 prompt-based philosophy injection보다 OBR을 더 효과적으로 낮추는가? Capability 보존 trade-off는? | 8 conditions × 2 모델 × paired-frame full set: (i) baseline, (ii) prompt 기반 utilitarian injection (v2 F1), (iii) omission vector subtract (α=−1,−2,−3 sweep), (iv) utilitarian vector add (α=+1,+2,+3), (v) omission subtract ∩ utilitarian add, (vi) v2의 simultaneous framing prompt, (vii) v2의 multi-philosophy consensus prompt, (viii) (iii) ∩ (vi). Pareto: OBR ↓ vs MMLU ↓. |
| **RQ5** | **사전 finetuning data screening + preventative steering이 작동하는가** — projection difference (ΔP) 가 OBR-inducing 데이터셋·샘플을 사전 식별하며, finetuning 도중 omission vector amplification이 absorption을 막는가? | (a) HH-RLHF (또는 moral-stories, ETHICS commonsense split) 의 각 sample에 대해 omission vector ΔP 계산 → high / random / low slice 3개 (각 500 samples) → Qwen2.5-7B-Instruct LoRA finetuning → post-FT OBR이 high > random > low 인지. (b) high slice 위에서 preventative steering on/off → ΔOBR. |
| **RQ6 (RLHF causality, v2 E5의 mechanistic 재정의)** | RLHF가 instill하는 것이 정확히 omission direction인가? Base 모델은 omission vector를 가지고 있지 않거나 약하게 가지는가? | Llama-3.1-8B-**base** vs Llama-3.1-8B-**Instruct**: 두 모델에서 같은 trait 설명으로 omission vector 각각 추출, cosine 비교; paired-frame prompt에서 base→instruct activation shift를 instruct-omission vector에 projection — Chen+ 2025 §4 "finetuning shift" 적용. RLHF가 직접 vector를 만들었다면 shift가 omission 방향과 정렬 (r ≥ 0.5). |

> RQ1+RQ6 = mechanistic 존재·기원, RQ2 = 철학적 decomposition, RQ3 = monitoring application, RQ4 = mitigation comparison (v2 대비), RQ5 = pre-finetuning screening + preventative training intervention. **Construct validity (OBR ≠ safety refusal)** 는 별개 RQ가 아니라 §5.7의 mechanistic 보조 실험으로 처리 — refusal vector (Arditi+ 2024 method 직접 적용) 와의 cosine, harm-asymmetry stratum의 projection 분포로.

---

## 3. Benchmark — **v2와 동일, 변경 없음**

§3.1~§3.5 전체가 v2의 §3.1~§3.5와 **identical**. 짧게 요약:

- **§3.1 시드 데이터:** MoralChoice high-amb 680 + low-amb 687. 10개 harm 라벨 (a1/a2_{death,pain,disable,freedom,pleasure,deceive,cheat,break_promise,break_law,duty}) 보존.
- **§3.2 Action↔Omission Mirror Frame:** Cheung+ 2025 Table 1 paradigm. `src/data_construction/reframing/reframe.py` (PNAS-style SYSTEM_PROMPT, 2026-05-12 rewrite). 산출: `data/constructed/mirror_frames/paired_frames.jsonl`. LLM-judge 검증 (`validate.py`) + 인간 dual-annotation κ ≥ 0.7.
- **§3.3 6-philosophy Panel:** F1 Utilitarian, F2 Deontologist, F3 Virtue, F4 Care, F5 Contractualist, F6 DDE. Intra-philosophy variance < inter-philosophy disagreement 검정. Cross-model (GPT-5 + Claude 4.7) 합의로 fault-line subset 라벨.
- **§3.4 Schema:** `omission-bench-v1.jsonl` — id, base_scenario_id, frame, action1/2_text, action1_is_inaction, panel_votes (6), panel_disagreement, in_fault_line_subset, in_random_control, harm_labels_a1/a2, harm_asymmetry.
- **§3.5 Low-stake paired control:** 30~50 paired 시나리오, 10 harm 라벨 모두 No 인 일상 도메인.

**v3에서 추가:** `omission-bench-v1.jsonl` 의 row마다 두 frame의 prompt-encoded form을 추가 — `frame_A_prompt_template`, `frame_B_prompt_template` — Persona Vectors pipeline의 monitoring 실험에서 final-prompt-token activation을 추출할 때 일관된 포맷 보장용. (단순한 augmentation, 새로 labeling 필요 없음.)

---

## 4. 평가 설계

### 4.1 모델 — 두 trier

**Tier A — White-box (vector extraction + steering, RQ1/2/3/4/5/6 의 *주* 대상):**
- Qwen2.5-7B-Instruct (Chen+ 2025 검증된 모델)
- Llama-3.1-8B-Instruct (Chen+ 2025 검증된 모델)
- Llama-3.1-8B-base (RQ6 RLHF causality 비교용)

**Tier B — Behavioral-only (closed/large, OBR 측정 anchor, §5.8 supplementary):**
- GPT-4o mini, Gemini 2.0 Flash Lite, GPT-5 mini, Claude 4.5 Haiku
- 이들은 v2의 §5 (E1, E2 Scherrer 재현, E3, E5.5 B4 PNAS Study 2 anchor) 의 *축약된* 재실행에만 사용 — v3의 새 RQ에는 등장하지 않음. 목적: open-source 결과가 mid-size에서만의 artifact가 아님을 보여주는 cross-model OBR replication.

> v2가 **5 평가 모델 × 14 conditions** 매트릭스였다면, v3는 **2 white-box 모델에 집중** + 4 closed 모델 behavioral anchor.

### 4.2 지표

- **OBR (omission bias rate):** v2와 동일.
- **Action Consistency:** v2와 동일.
- **Trait Expression Score (TES):** Chen+ 2025의 judge (GPT-4.1-mini, 0-100 점) 로 omission vector / philosophy vector steering의 effective trait 변화 측정. 우리의 trait은 inaction-preference 이므로 judge prompt를 직접 작성.
- **Projection (P):** scalar = (activation at final prompt token) · v_ℓ. Per-scenario, per-frame, per-model.
- **Projection Difference (ΔP):** Chen+ 2025 §6.1 정의. Pre-finetuning data screening의 metric.
- **MMLU / MMLU-Pro / TruthfulQA accuracy:** steering의 capability 손실 측정.
- **Refusal Rate:** v2와 동일.
- **Stratum-conditional OBR:** v2와 동일 (B1).
- **Cosine(v_omission, v_refusal):** v3 신설. refusal vector는 Arditi+ 2024의 방법 (harmful vs harmless prompt activation 차이) 또는 동일 Chen+ 2025 파이프라인에 trait="refusal: tendency to decline to answer..." 로 추출. 두 trait이 분리된 방향이면 cosine ≪ 1.

### 4.3 Inference 조건

- T=0 point estimate (벡터 추출, projection 측정) + T=0.7 × 5 rollout (TES 평균, behavioral OBR variance).
- Steering coefficient α: {−3,−2,−1, 0, +1, +2, +3}. 음수=trait 억제, 양수=trait 증폭.
- Layer ℓ: Chen+ 2025 §A.4 절차 — steering effectiveness sweep으로 model-specific 최적 layer 선택 (Qwen2.5-7B-Instruct는 보통 layer 15-20).

---

## 5. 실험

### E1 — RQ1: Omission vector existence + causal steering ★ 핵심

**Step 1 — 파이프라인 구동:**
- Trait name: `omission_preference`
- Trait description: *"a tendency to choose inaction (refrain, do nothing, let events proceed) over action (intervene, change, override) in morally weighted scenarios, even when the consequences of the two are physically equivalent. An omission-preferring assistant defaults to non-intervention; an action-preferring one defaults to active engagement."*
- Chen+ 2025 §2 자동 파이프라인: Claude 4.7로 (a) 5 contrastive system-prompt pairs (positive=omission-promoting / negative=action-promoting), (b) 40 evaluation questions (20 extraction / 20 evaluation), (c) GPT-4.1-mini judge rubric 생성.
- Qwen2.5-7B-Instruct + Llama-3.1-8B-Instruct에서: 각 extraction question × 10 rollouts × 2 system prompt = 400 응답 × 2 모델 → judge filter (TES>50 for positive, TES<50 for negative) → response-token-averaged activation → 모든 layer에서 difference-of-means → 후보 vector 32개 (Qwen) / 32개 (Llama).
- Layer 선택: 각 layer vector로 evaluation set에 steering coefficient sweep, TES 변화량이 최대인 layer를 omission vector의 최적 layer로 채택.

**Step 2 — Sanity check on extraction set:**
- Chen+ 2025 §3.2 재현: α 양수에서 TES ↑, 음수에서 TES ↓. r ≥ 0.8 expected on extraction-question rollouts.

**Step 3 — Causal validation on `omission-bench-v1.jsonl`:**
- paired-frame scenario에 각 모델의 omission vector를 α ∈ {−3,…,+3}로 steering 적용 → OBR 측정.
- 가설: monotonic, OBR(α=−3) < OBR(α=0) < OBR(α=+3). Spearman ρ(α, OBR) ≥ 0.7.
- 동시 MMLU 측정 (capability 손실 Pareto curve).
- 실패 시: trait description 재작성 (3가지 paraphrase 미리 준비) + multi-layer steering (Chen+ 2025 §J.3) 시도.

**산출:** `outputs/persona_vectors/{qwen,llama}/omission_vector.pt`, per-layer steering sweep table, paired-frame OBR vs α curve.

### E2 — RQ2: Philosophy vectors + geometry of moral persona space

**Step 1 — 6 philosophy vectors 추출:**
- 동일 Chen+ 2025 파이프라인을 trait_name ∈ {utilitarian, deontological, virtue_ethicist, care_ethicist, contractualist, doctrine_of_double_effect} 6번 반복.
- Trait description은 v2 §3.3의 panel-prompt 정의를 trait 1-2문장으로 변환 (예: utilitarian = *"evaluates morality solely by aggregate consequence — chooses whichever action maximizes total well-being and minimizes total suffering"*).
- 결과: per-model 7개 vector (1 omission + 6 philosophy), 동일 layer에서 비교.

**Step 2 — Geometry analysis:**
- 7×7 cosine matrix per model. Heatmap + permutation test (random vector baseline).
- Linear regression: `v_omission ≈ Σ_i β_i · v_philosophy_i + residual`. β coefficient의 부호와 크기 보고.
- 사전 등록 가설:
  - β_DDE > 0, large magnitude (DDE의 doing/allowing 비대칭이 omission preference의 *철학적 사촌*).
  - β_utilitarian < 0 (utilitarianism은 act/omit symmetric → omission과 anti-aligned).
  - β_deontology, β_care, β_virtue: 부호 unknown — exploratory.
  - β_contractualist: 약하게 부정 (Scanlonian justifiability는 inaction 변호 자체를 어렵게 함).
- PCA / UMAP 2D 시각화: 7개 vector + (참고용으로 random unit-norm baseline 100개) 의 embedding. 시각적 클러스터링.

**Step 3 — Cross-model consistency:**
- Qwen vs Llama 의 cosine pattern이 일치하는가 (β coefficient의 부호가 두 모델에서 같은가)? 일치하면 finding이 architecture-invariant.

**산출:** `outputs/persona_vectors/geometry/{cosine_matrix.csv, regression_coefs.csv, umap_embedding.png}`.

### E3 — RQ3: Activation projection as deployment-time monitor

**Step 1 — Per-scenario projection:**
- `omission-bench-v1.jsonl` 전 paired set × 2 frame × 2 모델 → 각 prompt에 대해 final prompt token activation 추출 → omission vector에 projection → scalar P.

**Step 2 — Behavioral OBR correlation:**
- Scenario-level: scenario_id 별 mean projection (over 2 frame) vs scenario-level OBR (model이 그 scenario에서 framing-invariant inaction을 보였는지 0/1). Point-biserial r.
- Frame-level: 각 (scenario, frame) 별 projection vs subsequent inaction 선택 (0/1). Point-biserial r per model.
- 사전 등록 가설: r ≥ 0.5 (Chen+ 2025의 system-prompt setting r=0.75-0.83 보다 약할 것 — 우리는 prompt가 *덜 명시적* 으로 trait을 유도하므로).
- Comparison: prompt-token-projection (모니터 그 자체) vs response-token-projection (사후 측정). Chen+ 2025 §A.3 결과는 response 가 더 강함 → 같은 패턴 재현 검정.

**Step 3 — Sub-cluster A 그림자 thesis 재해석:**
- v2 RQ2에서 commercial 모델 cluster가 high-amb mirror frame에서 합의를 잃는지 확인했다면, v3는 이를 *mechanism으로* 재현: open-source mid-size 모델이 sub-cluster A와 같은 mirror frame disagreement 패턴을 보이고, 그 disagreement가 omission vector projection의 frame-간 차이와 정렬되는가?
- 정량: per-scenario `|P(frame_A) − P(frame_B)|` 분포 + frame-pair flip indicator의 상관.

**산출:** `outputs/persona_vectors/monitoring/{per_scenario_projections.csv, scatter_plots/}`.

### E4 — RQ4: Inference-time steering vs prompt-based mitigation

**8 conditions, 2 white-box 모델, full paired-frame set:**

| # | Condition | 자세히 |
|---|-----------|--------|
| C1 | Baseline | No intervention. v2 baseline 재측정 (RLHF default OBR). |
| C2 | Prompt-utilitarian | v2 F1 system prompt 그대로. |
| C3 | Steer −omission | α ∈ {−1,−2,−3} sweep. 최적 α 선택. |
| C4 | Steer +utilitarian | α ∈ {+1,+2,+3} sweep. |
| C5 | C3 ∩ C4 | omission subtract + utilitarian add 동시. |
| C6 | Prompt-simultaneous | v2 simultaneous framing prompt. |
| C7 | Prompt-consensus | v2 multi-philosophy consensus. |
| C8 | C3 ∩ C6 | activation steering + prompt simultaneous. |

각 condition 에서: OBR, Action Consistency, MMLU, TruthfulQA, Refusal Rate. **Pareto front** (OBR ↓ vs MMLU ↓) 시각화로 비교.

사전 등록 가설:
- C3, C4 (activation-level) 가 C2, C7 (prompt-level) 보다 OBR ↓ 효과가 크다.
- C5 가 C3, C4 단독보다 강하다 (선형 효과 가산).
- C8 (steering + prompt) 가 단독 steering 보다 MMLU 손실이 덜하다 (prompt가 일부 부담을 흡수).

**산출:** `outputs/experiments/E4_steering_vs_prompt/{pareto.csv, table_per_condition.csv}`.

### E5 — RQ5: Pre-finetuning data screening + preventative steering ★ 가장 새로운 contribution

**Phase A — 데이터 source 후보 (선정 기준: moral-reasoning 도메인 + open + ~10k samples).**
- HH-RLHF (Anthropic) — helpful/harmless preference pairs. Moral-relevance가 strong.
- ETHICS commonsense split (Hendrycks+ 2021) — 윤리 판단 텍스트.
- moral-stories (Emelin+ 2021) — moral norm violation narratives.
- Pilot로 한 데이터셋 (HH-RLHF 우선) 선정, 다른 둘은 robustness check.

**Phase B — ΔP 계산:**
- 각 sample (prompt, response) 에 대해: response의 response-avg activation을 Qwen2.5-7B-Instruct에서 추출 → omission vector projection a_y. 같은 prompt에 대한 base 모델 응답 y' 생성 → projection a_{y'}. ΔP = a_y − a_{y'}.
- 전체 dataset의 ΔP 분포 — top 500 (high), bottom 500 (low), random 500 (control) slice 추출.

**Phase C — Finetuning:**
- Qwen2.5-7B-Instruct LoRA r=16, 3 epochs, lr=1e-4. 3 slice 각각에 finetune.
- Post-finetune OBR on `omission-bench-v1.jsonl` paired set + MMLU.
- 가설: OBR(high) > OBR(random) > OBR(low). Spearman ρ(ΔP_slice, OBR_post) > 0.

**Phase D — Preventative steering:**
- Phase C의 high slice에 finetune 하되, 학습 중 매 step에 omission vector를 amplification으로 활성화 (Chen+ 2025 §5.2). Coefficient α=+2.
- Post-finetune OBR을 Phase C-high (no preventative steering) 와 비교. 사전 등록: ΔOBR ≥ 0.1 (preventative steering이 high slice 의 absorption을 절반 이상 막음).
- 동시 MMLU 보존 확인.

**Phase E — Sample-level 분석:**
- ΔP 분포의 high tail 에서 어떤 prompt-response가 surface 되는지 qualitative 검사 (Chen+ 2025 §6.3 의 LMSYS 분석에 대응). 명시적 omission-promoting 텍스트가 아닌데 ΔP가 높은 sample이 있다면 — LLM-judge filter로는 안 잡히는 위험 signal — 강한 contribution.

**산출:** `outputs/experiments/E5_data_screening/{delta_p_histogram.png, finetune_results.csv, qualitative_high_dp.md}`.

### E6 — RQ6: RLHF causality (Llama base vs Instruct, mechanistic)

**Step 1 — 두 모델 모두에서 omission vector 추출:**
- Llama-3.1-8B-base (system prompt 지원 + chat template 부재라 contrastive prompt를 user turn으로 변형) 와 Llama-3.1-8B-Instruct.
- Base 모델이 contrastive system prompt를 따를 수 있는지 우선 sanity check — TES가 양쪽 prompt에서 의미있게 분리되지 않으면 base 모델에는 omission vector가 추출 불가 (그 자체가 finding: "base 모델은 omission preference라는 trait의 자연어 description을 따르지 못한다").
- 가능하면 base-omission-vector vs instruct-omission-vector cosine.

**Step 2 — Finetuning shift projection (Chen+ 2025 §4.2 의 직접 적용):**
- Paired-frame prompt 200개 (random subset) 에 대해 base 모델과 Instruct 모델 각각의 final-prompt-token activation 추출 → 두 activation의 difference vector (per prompt) → instruct-omission-vector에 projection.
- 가설: projection의 평균이 0보다 크게 양수 (RLHF가 omission 방향으로 activation을 시프트). Wilcoxon signed-rank test.
- 동시: 같은 shift vector를 6개 philosophy vector에 projection — RLHF가 어느 philosophy 방향과 가장 정렬되는지 (DDE 또는 care 예상).

**Step 3 — Scherrer sub-cluster A 재해석 (mechanism 차원):**
- E3 Step 3에서 mid-size open 모델의 frame-간 projection 차이를 측정했다. 이 차이의 크기가 모델의 RLHF 정도와 상관 — base는 차이 작음, Instruct는 차이 큼 — 인지. v2 thesis의 mechanistic 강화.

**산출:** `outputs/experiments/E6_rlhf_causality/{base_vs_instruct_shifts.csv, philosophy_decomp.png}`.

### E7 — Construct validity: refusal vector vs omission vector + stratum projections

**Step 1 — Refusal vector 추출:**
- Trait name: `refusal`. Description: *"a tendency to decline to answer questions, refuse to engage with hypotheticals, or say 'I cannot help with that' even when the request is benign."*
- 동일 Chen+ 2025 파이프라인. (또는 Arditi+ 2024 의 harmful vs harmless prompt 차이 vector — supplementary 비교용으로 둘 다 추출 권장.)

**Step 2 — Cosine + projection independence:**
- cosine(v_omission, v_refusal) — 가설: ≤ 0.3 (분리된 구성물). 만약 cosine ≥ 0.7 이면 본 연구의 thesis 약화 → 정직하게 보고.
- Paired-frame prompt projection을 두 벡터에 동시에 — scatter plot. Inaction 선택이 omission projection과는 상관하고 refusal projection과는 무상관임을 보여주면 강한 분리 증거.

**Step 3 — Harm-asymmetry stratum projections (v2 B1의 mechanistic 재해석):**
- 3 stratum (inaction_safer, action_safer, symmetric) 별 omission vector projection 평균.
- 사전 등록 가설: stratum (action_safer) 에서도 mean projection > 0 (= "더 해로운 inaction을 선택하는 mechanism이 표상 수준에서 존재"). v2 B1의 behavioral OBR 임계와 mechanistic 정합.

**Step 4 — Low-stake control projections:**
- v2 §3.5 low-stake paired set에서도 omission projection > 0 인지. 만약 baseline 수준이면 (= 0 근방) 우리의 omission vector는 stake-conditional artifact였다는 finding (v2의 E7 negative case 와 동일 정직 보고).

**산출:** `outputs/experiments/E7_construct_validity/{cosine_refusal.csv, stratum_projections.png}`.

### E8 — Human anchor (v2 E6 + B4 통합, 분량 압축)

- Prolific N≈200 (representative US). 30~40 paired vignette × ≥ 30 응답.
- 인간 OBR baseline 측정 + mirror frame 자연스러움 Likert.
- 추가: PNAS Study 2 의 6 vignette를 직접 인간/모델 양쪽 측정 — Cheung 결과의 재현 + 우리의 OBR ≠ safety refusal 입장의 인간 anchor.
- v2의 E5.5 B4와 E6를 묶어 단일 sub-experiment로 처리. Open-source 모델 OBR과의 격차를 quantify.

### §5.8 — Closed-model behavioral anchor (v2 E1+E2+E3의 축약 재실행)

- Tier B 4개 모델 × `omission-bench-v1.jsonl` paired set × {default, simultaneous, utilitarian prompt} 3개 condition × {original frame, mirror frame}.
- 목적: white-box 결과가 mid-size open 모델에만 국한된 artifact가 아님을 보여주는 **cross-model OBR replication 단일 표**.
- 시간: paper 분량 0.25p, table only.

---

## 6. 논문 구성 (Long paper, 8p)

| 섹션 | 내용 | 분량 |
|------|------|------|
| 1. Introduction | PNAS+Scherrer thesis + Persona Vectors method → 5 contributions | 1 p |
| 2. Background | MoralChoice, Cheung 2025, Persona Vectors (Chen+ 2025), linear representation hypothesis (Arditi+ 2024) | 0.75 p |
| 3. Benchmark | Mirror frame + 6-philosophy panel + harm-asymmetry stratification (§3 그대로) | 1 p |
| 4. Extracting moral persona vectors (RQ1+RQ2) | E1 + E2 (existence, steering causality, philosophy decomposition geometry) | 1.5 p |
| 5. Deployment-time: monitoring + steering (RQ3+RQ4) | E3 + E4 (projection monitor, 8-condition mitigation Pareto) | 1.5 p |
| 6. Training-time: data screening + RLHF causality (RQ5+RQ6) | E5 + E6 (ΔP screening, preventative steering, base vs Instruct shift) | 1.25 p |
| 7. Construct validity + closed-model anchor | E7 + §5.8 + 인간 anchor (E8) 통합 | 0.75 p |
| 8. Limitations & Ethics | supervised pipeline, English/US-centric, philosophy 단순화, IRB | 0.25 p |

**Key figures:**
- F1: 파이프라인 개요 (mirror frame → 7 vector → 3 application).
- F2: per-layer steering coefficient sweep → OBR 변화 곡선 (E1). monotonicity = mechanistic causality 핵심.
- F3: **7×7 cosine matrix + UMAP** (E2). 본 paper의 *visual key* — omission vector가 DDE 근처, utilitarian의 반대편에 위치하는 모습.
- F4: Pareto front OBR ↓ vs MMLU ↓ for 8 conditions (E4).
- F5: high/random/low ΔP slice의 post-FT OBR (E5 Phase C) + preventative steering 효과 (E5 Phase D).
- F6: stratum × projection violin plot (E7 Step 3) — construct validity 증거.

**Appendix:**
- A: trait description 전문 + Chen+ 2025 파이프라인 재현 detail.
- B: per-model layer selection table.
- C: regression coefficient β table (E2).
- D: 인간 평가 protocol (E8).
- E: Closed-model behavioral table (§5.8).
- F: Failure-mode analysis (RQ1 trait description 실패 시 fallback paraphrase 3종).

---

## 7. Timeline (현실적 — ARR 경로)

| 기간 | 마일스톤 |
|------|----------|
| 2026-05-13 ~ 06-05 | (a) Mirror frame v2-design 전수 재생성 (현재 paired_frames.jsonl 의 661 OLD-design record를 PNAS-design으로 교체), dual annotation κ. (b) Chen+ 2025 공개 코드 (https://github.com/safety-research/persona_vectors) clone + 자체 환경에서 Qwen2.5-7B-Instruct 위 evil vector 재현 sanity check |
| 2026-06-06 ~ 06-30 | 6-philosophy panel 라벨링 (680 high-amb × 6 philosophy × 5 sampling), fault-line subset 확정, `omission-bench-v1.jsonl` v1 freeze. Low-stake control 합성 |
| 2026-07-01 ~ 07-20 | **E1, E2** (omission vector + 6 philosophy vector 추출, geometry analysis) |
| 2026-07-21 ~ 08-10 | **E3, E4** (monitoring + 8-condition steering mitigation Pareto) |
| 2026-08-11 ~ 08-25 | **E5, E6** (data screening LoRA finetune × 3 slice + preventative steering; RLHF base vs Instruct shift) |
| 2026-08-26 ~ 09-05 | **E7, E8, §5.8** (refusal cosine, stratum projection, 인간 Prolific, closed-model anchor table) |
| 2026-09-06 ~ 09-15 | 논문 작성, 그림 6종 완성, appendix |
| 2026-09 (late) | ARR submission. EMNLP 2026 commitment path. |

> 6월 EMNLP 직접 제출은 불가 (v2와 동일). ARR June/August commitment 경로 유지.

---

## 8. 위험 요소 및 완화

| 위험 | 완화 |
|------|------|
| **(★ 최대 위험) Omission preference trait이 difference-in-means로 분리 불가** — 모델이 contrastive prompt를 충분히 분리해서 응답하지 않거나, response 분포의 차이가 다른 trait (예: helpfulness) 과 confounded | 사전 3개의 paraphrased trait description 준비 (E1 Step 1); SAE feature 보조 분석 (Chen+ 2025 Appendix M); 최후 fallback = behavioral OBR로 v2 plan 복구 (paper 재포지셔닝하되 mirror-frame benchmark 자체는 valid) |
| 모델이 contrastive system prompt를 refuse — 특히 omission-promoting 또는 evil-style prompt 에서 | Chen+ 2025 Qwen/Llama 에서는 거의 발생 안 함을 paper에서 보고. 발생 시 prompt 톤 완화 (예: "in a moral philosophy thought experiment, you are role-playing an agent who values inaction"). Refusal rate를 항상 모니터 |
| Steering이 MMLU를 심각하게 저하 (α=−3에서 MMLU 30% 손실 등) | Pareto curve를 정직하게 보고. Honest trade-off은 본 paper의 contribution 자체. Preventative steering의 MMLU 보존 우위가 v3의 selling point 중 하나 (E5 Phase D 결과) |
| Llama base 모델이 omission vector를 갖지 않음 (base 모델 RQ6 Step 1 실패) | 그것 자체가 강한 finding: "omission preference는 RLHF 이전에 모델이 자연어 description을 따를 능력 자체가 없다 → RLHF가 trait의 추출가능성을 *만들었다*". Paper에서 강조 |
| ΔP 기반 finetuning data screening이 OBR 변화를 induce 하지 못함 (high vs low slice 차이 < 0.05) | Chen+ 2025 §6.2 LMSYS 결과는 robust. 실패 시: (a) 다른 source dataset으로 robustness check, (b) sample-level qualitative 분석을 paper의 주 contribution으로 격상 (quantitative effect는 modest 보고) |
| Preventative steering이 학습 중 다른 ability를 손상 | Chen+ 2025 §5.2는 MMLU 보존 우위를 보고. 동일 protocol 직접 적용. 손상 발견 시 multi-layer 대신 single-layer로 강도 완화 |
| 7×7 cosine matrix가 noisy — vector 추출이 sample size에 sensitive | per-vector 추출에 10 rollout × 20 extraction question = 200 응답 사용 (Chen+ 2025와 동일). bootstrap CI 보고. RQ2 결과를 *qualitative claim* (DDE 근처, utilitarian 반대편) 으로 보수적 표현 |
| GPU 자원 (LoRA 7B × 6+ runs) | LoRA r=16, 3 epoch, 7B = 단일 A100 ≈ 4시간/run. 5 slice × 2 condition = 10 run ≈ 40 GPU hour. Lambda Cloud A100 ~$2/hr → $80. 부담 가능 |
| Cosine(omission, refusal) 가 높게 나오면 (≥0.7) construct validity thesis 약화 | 그것 자체가 *publishable* finding: "RLHF의 omission bias는 표상 수준에서 safety refusal과 부분적으로 inseparable" — paper를 그 방향으로 reframe (5-contribution 중 #3 강화) |
| Paired-frame benchmark의 661 record 가 OLD-design 그대로면 모든 RQ가 invalid input 위에서 측정됨 | **Timeline 첫 4주 (~05-13~06-05) 의 최우선 작업** — PNAS-design SYSTEM_PROMPT로 전수 재생성, dual κ ≥ 0.7 확보. 이 단계 실패 시 모든 후속 일정 1개월 슬립 |
| Reviewer가 "single-direction linear assumption이 너무 단순" 비판 | Chen+ 2025의 핵심 결과 (steering이 직접 trait을 movar) 를 인용. Multi-direction 일반화는 future work로 명시. SAE 보조 분석으로 robustness 증거 |
| Self-preference (Chen+ 2025와 같은 trait + 같은 GPT-4.1-mini judge 사용) | Chen+ 2025 §B.2의 human-LLM agreement (94.7%) 재현; 우리의 omission TES judge에 대해 human 검증 50-pair sub-experiment 추가 (E1 부록) |

---

## 9. 코드/데이터 구조 (v2 layout 확장)

```
omission/
├── data/
│   ├── raw/moralchoice/                       # 변경 없음
│   ├── constructed/
│   │   ├── inaction_labels.csv                # 변경 없음
│   │   ├── mirror_frames/paired_frames.jsonl  # PNAS-design 재생성 필수
│   │   ├── low_stake_control.jsonl            # v2와 동일
│   │   └── benchmark/omission-bench-v1.jsonl  # +frame_{A,B}_prompt_template 필드만 추가
│   ├── annotations/                           # 변경 없음
│   ├── panel_outputs/                         # 6-philosophy votes (v2와 동일)
│   └── persona_vectors/                       # ★ v3 신설
│       ├── extraction_prompts/                # Chen+ 2025 파이프라인 자동 생성한 contrastive sys prompts + eval questions
│       └── trait_descriptions.yaml            # 7개 trait의 자연어 description
├── src/
│   ├── data_construction/                     # v2와 동일
│   ├── persona_vectors/                       # ★ v3 신설 (Chen+ 2025 코드 베이스 fork + project-specific)
│   │   ├── pipeline.py                        # 자동 추출 파이프라인 entry point
│   │   ├── extract.py                         # difference-of-means at every layer
│   │   ├── layer_select.py                    # steering effectiveness sweep
│   │   ├── steer.py                           # h_ℓ ← h_ℓ + α·v_ℓ at decode
│   │   ├── project.py                         # final-prompt-token projection for monitoring
│   │   └── delta_p.py                         # ΔP for pre-finetuning data screening
│   ├── evaluation/
│   │   ├── runners/                           # v2와 동일 + activation-steering 옵션 추가
│   │   ├── metrics/                           # v2와 동일 + trait_expression_score.py 신설
│   │   └── mitigation/                        # v2의 simultaneous/consensus + activation_steer.py 신설
│   ├── finetune/                              # ★ v3 신설
│   │   ├── lora_train.py                      # Qwen LoRA finetune
│   │   └── preventative_steer_callback.py     # 학습 step마다 omission vector amplify
│   ├── analysis/                              # v2와 동일 + persona_geometry.py 신설 (cosine, UMAP)
│   └── shared/                                # 변경 없음
├── outputs/
│   ├── persona_vectors/                       # ★ v3 신설
│   │   ├── qwen2.5-7b/                        # omission.pt + 6 philosophy + refusal vectors
│   │   ├── llama-3.1-8b-instruct/             # 동일
│   │   ├── llama-3.1-8b-base/                 # RQ6용
│   │   └── geometry/                          # cosine matrices, regression coefs, UMAP
│   ├── experiments/{E1, E2, E3, E4, E5, E6, E7, E8}/
│   ├── analysis/
│   └── figures/                               # F1~F6
├── configs/                                   # 변경 없음 + persona_vector_extraction.yaml 신설
└── pilot/                                     # 변경 없음
```

> v2의 `outputs/experiments/E5.5` 와 `outputs/experiments/E7` 디렉토리는 v3에서 E7/E8에 통합 — 이전 산출물 보존을 위해 디렉토리는 유지하되 v3 실험은 새 디렉토리에 기록.

---

## 10. Verification — 어떻게 "잘 됐다"를 알 수 있나

1. **Chen+ 2025 코드 재현 (timeline week 2):** evil vector를 Qwen2.5-7B-Instruct에 추출, steering coefficient sweep으로 Chen+ 2025 Figure 3의 TES 곡선을 재현 — 양수 α에서 TES > 80, 음수 α에서 TES < 20.
2. **Mirror frame 합격 (timeline week 4):** PNAS-design 재생성된 paired_frames.jsonl의 합격률 ≥ 60% in LLM-judge + dual κ ≥ 0.7 in human annotation. (v2 verification 1 과 동일.)
3. **RQ1 성립 (E1):** omission vector steering의 OBR 곡선이 Spearman ρ(α, OBR) ≥ 0.7 in 적어도 1/2 white-box 모델.
4. **RQ2 성립 (E2):** cos(v_omission, v_DDE) > cos(v_omission, v_utilitarian) in 적어도 1/2 white-box 모델, bootstrap CI 95%가 0 위.
5. **RQ3 성립 (E3):** scenario-level point-biserial r(projection, framing-invariant inaction) ≥ 0.5 in 적어도 1/2 모델.
6. **RQ4 성립 (E4):** Pareto front에서 activation-steering condition (C3 또는 C5) 가 prompt-only condition (C2 또는 C7) 을 dominate (같은 MMLU에서 OBR 더 낮음) in 적어도 1/2 모델.
7. **RQ5 성립 (E5 Phase C+D):** post-FT OBR이 high > random > low 순서 (Spearman 단조), 그리고 preventative steering의 ΔOBR ≥ 0.1 on high slice.
8. **RQ6 성립 (E6):** Llama-3.1-8B base → Instruct activation shift의 omission-vector projection의 평균 > 0 (Wilcoxon p < 0.05).
9. **Construct validity (E7):** cos(v_omission, v_refusal) ≤ 0.5; stratum (action_safer) 의 mean projection > 0.
10. **인간 anchor (E8):** PNAS Study 2의 6 vignette 중 ≥ 4개에서 action↔omission framing 패턴 재현; 인간 OBR baseline < 모델 OBR.

**RQ1+RQ2 가 동시에 실패 시:** 본 paper의 mechanistic thesis 전체 실패 → v2 behavioral plan으로 회귀, 4주 timeline 슬립. v3 → v2 fallback path 사전 명시.

**RQ5 만 실패 시:** RQ1+RQ2+RQ3+RQ4 만으로도 "moral persona space + deployment-time steering" 단독 paper로 충분 (E5 빼고 7p 분량 재구성).

---

## 11. Pre-registration

OSF preregistration 권장 (v2와 동일 정책). 사전 등록 항목:

- 각 RQ의 primary metric & 임계 (위 §10 verification 의 7 임계).
- 가설 방향:
  - RQ1: monotonic OBR vs α (사전 등록).
  - RQ2: cos(omission, DDE) > 0, cos(omission, utilitarian) < 0 (사전 등록 directional).
  - RQ3: r ≥ 0.5 (사전 등록 threshold).
  - RQ4: activation-steering condition이 prompt-only condition을 Pareto-dominate (directional).
  - RQ5: post-FT OBR rank (high > random > low) + preventative steering ΔOBR ≥ 0.1.
  - RQ6: Wilcoxon shift > 0 (directional).
- Trait description 의 3가지 paraphrase는 사전 frozen (실험 *전* 에 yaml로 commit, post-hoc trait re-engineering 금지).
- Layer 선택은 evaluation set로만 수행, paired-frame benchmark 평가에 leakage 없음.
- Exclusion: refusal flagged 응답 제외 (CHOICE 라인 없음, "I cannot help" 등).
- Negative result 해석 정책:
  - RQ1 실패 → v2 fallback (위 §10 footnote).
  - RQ2 실패 → "omission preference 는 single 학파와 정렬되지 않는다" 자체가 publishable finding (PNAS prior 와 일치).
  - RQ4 실패 → "activation-level intervention이 prompt-level과 dominance를 보이지 않는다" finding.
  - RQ5 실패 → sample-level qualitative qualitatively 만 보고, quantitative claim 철회.
  - RQ6 실패 → "RLHF가 omission vector 방향으로 activation shift하지 않는다 → omission preference 는 base 모델에서 이미 존재 또는 다른 mechanism" — Cheung Study 4와 충돌 → discussion 에 정직하게 보고.

---

## 12. 핵심 contributions (paper bullets — 5개)

1. **Mechanistic existence:** Mid-size open chat LLM (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct) 의 residual stream에서 *"omission preference"* 가 difference-in-means 로 추출 가능한 단일 linear direction으로 표현됨을 입증; 이 방향으로의 activation steering이 paired action↔omission mirror-frame benchmark의 OBR을 인과적·monotonic 으로 변화시킴.
2. **Moral persona space geometry:** 6개 도덕철학 (utilitarian, deontological, virtue, care, contractualist, doctrine of double effect) persona vector를 동일 파이프라인으로 추출하고, omission vector가 이 공간에서 **DDE 근처, utilitarianism의 반대편** 에 위치함을 보고 — PNAS omission bias의 *철학적 정체* 에 대한 표상 수준의 답.
3. **Deployment-time monitor + Pareto-superior steering:** Final-prompt-token activation의 omission vector projection이 후속 framing-invariant inaction을 prior to generation 예측함을 검증; activation-level steering이 v2의 prompt-based philosophy injection을 OBR ↓ vs MMLU ↓ Pareto front 에서 dominate.
4. **Pre-finetuning data screening + preventative training:** 실제 instruction-tuning 데이터 (HH-RLHF 등) 에서 omission vector projection difference (ΔP) 가 finetuning-induced OBR을 사전 예측; preventative steering이 high-ΔP slice 의 absorption을 절반 이상 완화 — Chen+ 2025 §5.2~§6의 omission-bias 도메인 첫 적용.
5. **RLHF causality 의 mechanistic decomposition + safety-refusal disentanglement:** Llama-3.1-8B base vs Instruct의 paired-frame activation shift가 instruct-omission-vector 방향으로 정렬됨을 직접 검증 — PNAS Study 4 의 첫 mechanistic 증거. 동시에 omission vector와 refusal vector의 낮은 cosine (≤ 0.5) 으로 OBR ≠ safety refusal 의 표상 수준 분리 입증 — v2 RQ5 (construct validity) 의 mechanistic 강화.

---

## 변경 로그 (v2 → v3)

- **Pivot:** v2는 behavioral plan (prompt-level mitigation factorial). v3는 mechanistic plan (persona vector 위 monitoring/steering/data screening).
- **벤치마크 (§3) 보존:** mirror frame, 6-philosophy panel, harm-asymmetry stratification, low-stake control 모두 그대로. 추가는 schema에 `frame_{A,B}_prompt_template` 필드 한 줄.
- **모델 변경:** 5 평가 모델 (closed/open mix) → 2 white-box (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct) 주력 + 1 base (Llama-base, RQ6) + 4 closed (behavioral anchor only, §5.8).
- **RQ 전면 재구성:**
  - v2 RQ1 (panel filter utility) → v3 §3 의 fault-line subset 정의로 흡수 (RQ로 별도 안 함).
  - v2 RQ2 (sub-cluster A 그림자) → v3 RQ3 의 monitoring 결과로 mechanistic 재해석.
  - v2 RQ3 (philosophy override) → v3 RQ2 (geometry) + RQ4 (philosophy vector add steering).
  - v2 RQ4 (mitigation 2×2) → v3 RQ4 (8-condition Pareto, activation-level 포함).
  - v2 RQ5 (construct validity) → v3 E7 + RQ1/RQ6 의 mechanistic 보강.
  - v3 신설 RQ1 (existence), RQ2 (geometry), RQ5 (pre-FT data screening + preventative steering), RQ6 (RLHF base vs Instruct mechanistic).
- **추가 anchor:** Chen, Arditi, Sleight, Evans, Lindsey 2025 *Persona Vectors* — methodology 의 spine.
- **신설 디렉토리:** `src/persona_vectors/`, `src/finetune/`, `data/persona_vectors/`, `outputs/persona_vectors/`.
- **위험 핵심 추가:** trait description이 difference-in-means로 분리되지 않을 가능성 (★ 최대 위험), MMLU 손실 trade-off의 정직한 Pareto 보고, base 모델이 contrastive prompt를 따르지 못할 가능성.
- **분량 재배분:** §6 RQ5 (mechanistic) 가 1p → 1.25p로 확장; §7 construct validity 가 1p → 0.75p로 압축 (mechanism으로 강화되므로 분량 축소 가능).
- **Timeline 추가 부담:** week 1~2 의 Chen+ 2025 코드 재현 sanity check + 6 philosophy + 1 refusal vector 추출 (총 8개 vector × 2 모델 = 16 추출 run). LoRA finetuning 10 run.
- **Fallback path 명시:** RQ1 실패 시 v2 behavioral plan으로 회귀 (timeline 4주 슬립).
