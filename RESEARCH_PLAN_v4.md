# Research Plan v4 — Conflict-Typed Omission Bias across LLMs

> 작성일: 2026-05-13 · 본 문서는 `RESEARCH_PLAN_v3.md` (persona-vector pivot) 을 **전면 폐기**하고 새로 작성한 plan이다.
> v3 의 mechanistic / persona-vector 방향은 모두 들어냈고, **데이터 구축은 paired mirror frame + 2-stage philosophy panel (Stage 1 unanimity filter, Stage 2 (Y,N) vs (N,Y) conflict labeling, 2026-05-13 구현)** 으로, **분석은 모델 × 철학-충돌-유형 별 omission bias rate (OBR) 비교** 로 단일화한다. 완화 (mitigation) 단계는 **prompt-level 로 확정** — 상세 설계는 별도 supplement plan (`~/.claude/plans/research-plan-v4-giggly-nebula.md`, 2026-05-15) 에 동결.
>
> **Spine (논문의 단일 thesis):** 도덕철학 패널의 *불일치* 는 omission bias 의 단순한 상관 라벨이 아니라 *조작 가능한 인과 축* 이다 — 같은 패널 신호가 (a) 어디서 bias 가 터질지 *예측* 하고 (**RQ2a**), (b) 그 축을 정조준해 흔들면 bias 가 예측 방향으로 움직여 *인과* 임이 드러난다 (**E7 C2/C3 = oracle 인과 probe, knockout-style — 배포 방법 아님, 제거 가능분의 상한선**). (a)+(b) 가 spine. 추가로 (c) 라벨 불필요한 **label-free 배포 근사 (E7 C1/C4)** 가 그 상한선의 일부를 회수 → 실무 actionable. 진단·인과확인이 동일 기구라는 점이 closure. **RQ2(=RQ2a 예측 + RQ2b signature 병합) + E7 이 본문 spine; RQ1 은 supporting; RQ5 는 robustness 부록 (RQ3→RQ2b 병합·RQ4 harm-strata 폐기 2026-05-16; E7 oracle-probe vs label-free 배포 분리 2026-05-17).**
>
> Anchor:
> - Cheung, Maier, Lieder (2025, *PNAS*) — paired action↔omission frame paradigm, framing-invariant inaction = omission bias 의 operational definition.
> - Scherrer, Shi, Feder, Blei (2023, *NeurIPS*) — MoralChoice 데이터셋, 모델-도덕철학 합의 cluster 분석.

---

## 1. Context — 무엇을 묻는가

PNAS 2025 (Cheung+) 는 LLM 에 *amplified omission bias* — 두 옵션의 outcome 이 동등할 때 action 보다 inaction 을 선호하는 경향 — 가 존재함을 보였다. 정의는 framing-invariant: 동일 시나리오의 action 프레임과 omission 프레임에서 *둘 다* inaction 을 선택할 때만 omission bias 로 인정.

이 논문은 두 가지를 답하지 않았다.

1. **어떤 모델이 더/덜 omission bias 를 보이는가** — PNAS 는 GPT-4 단일 모델 중심. cross-model 양상은 불명.
2. **어떤 도덕적 문맥에서 omission bias 가 강하게 발현하는가** — PNAS 는 utilitarian↔deontological 단일 축에 의존. 한 모델이 *어떤 철학적 대립* 의 시나리오에서 inaction 으로 기우는지의 *signature* 는 측정한 적이 없다.

본 plan 은 이 두 질문을 *교차해서* 묻는다: **각 평가 모델은 어떤 철학-충돌 유형의 시나리오에서 가장 강한 omission bias 를 보이는가?**

핵심 도구는 v3 에서 폐기한 mechanistic interpretability 가 아니라, 이 repo 가 이미 만들어놓은 **two-stage philosophy panel** 이다:

- **Stage 1 (filter):** 동일 LLM 에 5 개 도덕철학 persona 를 각각 주입하고 paired frame 양쪽에 yes/no 답변을 받아, 5 개의 `(A, B)` 답변 튜플이 *동일* 한 시나리오 (= 철학적 만장일치) 는 제거. 만장일치 시나리오는 도덕적 fault line 이 아니므로 omission bias 의 cross-philosophy 차이를 측정할 무대가 못 된다.
- **Stage 2 (label):** 필터 통과 시나리오마다 `(Y,Y)` / `(N,N)` 철학 (= frame-invariant, 라벨러로 부적합) 을 제거한 뒤, 남은 철학들을 `(Y,N)` 그룹 (`yn_phils`) 과 `(N,Y)` 그룹 (`ny_phils`) 으로 갈라 pairwise conflict 리스트를 생성. 예: util ∈ yn, virtue ∈ ny 면 `[("F1_util", "F3_virtue")]` 의 충돌 라벨.

라벨링 결과 시나리오는 *철학 A vs 철학 B 충돌* 의 인스턴스가 된다. 그 위에서 평가 모델을 돌려, **모델 × 충돌 pair** 의 cell 마다 OBR 을 측정한다. 이것이 v4 의 단일 분석 축이다.

**핵심 thesis (모든 piece 가 이 한 문장으로 수렴해야 함):** *패널 불일치는 omission bias 의 조작 가능한 인과 축이다.* 이 주장은 — (1) **예측**: 시나리오가 어떤 철학적 충돌 (yn vs ny) 의 인스턴스인지가 평가 모델의 OBR 을 예측한다 (RQ2a). (2) **인과 확인 (oracle probe)**: 그 *같은* 패널이 찍어준 철학 (yn_phils/ny_phils) 을 평가 모델에 persona 주입하면 OBR 이 예측 방향(C2→YN, C3→NY)으로 이동 (E7 C2/C3). 이는 *deployable 방법이 아니라 knockout-style causal manipulation* — 구축-시 라벨(oracle)을 써도 통제 실험으로서 정당하며, philosophy-conflict 축으로 제거 가능한 bias 의 **상한선**을 준다. (3) **label-free 배포 (actionable)**: 라벨이 필요 없는 C1(동시 제시)·C4(다관점)가 그 상한선의 일부(목표 ≥50%)를 회수 → 실무에서 쓸 수 있는 완화. (1)+(2) 가 spine 의 인과 closure 이고 (3) 이 actionable 절. E7 C2/C3 는 부록이 아니라 correlational → causal 격상의 *논증 키스톤*, C1/C4 는 그 인과를 배포 가능한 형태로 옮기는 절. **C2/C3 를 배포 방법으로 over-claim 하지 않는 것이 정직성의 핵심 (2026-05-17 정정).** RQ1 은 PNAS 의 단일-모델 결과를 multi-vendor 로 확장하는 warm-up, RQ2b (model-specific signature) 는 spine 을 강화하지만 실패해도 spine 이 무너지지 않는 보너스 (그래서 RQ3 을 별개 RQ 가 아니라 RQ2 의 nested 강한 주장으로 병합), RQ5 는 robustness 검증이다. (harm-avoidance 대안설명은 NN 지표 정의가 자체적으로 차단하므로 별도 harm-strata RQ 불필요 — RQ4 폐기.)

---

## 2. Research Questions

| RQ | 질문 | 정량 측정 / 가설 |
|----|------|------------------|
| **RQ1** | 각 평가 모델은 paired mirror-frame 시나리오에서 **얼마나 자주** framing-invariant inaction (= NN 응답) 을 보이는가? | per-model **overall OBR** = #{scenarios with NN} / #{scenarios with both frames answered}. PNAS Cheung+ 와 비교; H1: 모델 간 차이가 통계적으로 유의 (proportions test, Bonferroni-corrected pairwise). |
| **RQ2** (RQ2a + RQ2b 병합 — 한 RQ 의 nested 주장) · **실측 구조는 §3.8 참조 (util-vs-consensus 주축 + 비공리주의 부차축; 옛 "다양한 pair 균등분포" 가정은 데이터로 기각)** | conflict-typed OBR 구조: (a) 시나리오의 철학-충돌 유형이 평가 모델의 OBR 을 예측하는가 — 즉 모델 내 OBR-by-pair profile 이 *비평탄* 한가 (주축 vs 부차축, 부차축 내부)? (b) 그 profile 이 *모델마다 다른가* (model-specific signature)? **(a) 는 spine 의 예측 절 (필수), (b) 는 spine 강화 보너스 (실패 허용 — (b) 실패해도 (a)+E7 로 spine 성립).** | **RQ2a:** per-(model × conflict pair) cell OBR; 모델 내 conflict pair 간 χ² of independence + post-hoc top-vs-bottom z-test (Bonferroni). H2a: ≥1 모델에서 conflict pair 간 OBR spread ≥ 0.15. **RQ2b:** per-model OBR-by-pair profile vector → 모델×모델 cosine matrix + permutation test (모델 라벨 셔플 5,000회). H2b: ≥1 모델쌍 profile cosine < 0.8. *RQ2b 는 RQ2a 가 통과해야 의미 (평탄하면 profile 비교 무의미) — 그래서 별개 RQ 가 아니라 RQ2 의 nested 강한 주장.* |
| **RQ5** | omission bias 가 **action bias** ((Y,Y) 응답, frame-invariant *action* 선호) 와 분리되는가? 모델이 NN 을 많이 하는 conflict pair 와 YY 를 많이 하는 conflict pair 가 다른가? | per-(model × pair) 의 YY rate 도 동일 schema 로 측정. NN rate vs YY rate scatter; H5: NN rate 와 YY rate 가 음의 상관 (모델이 한쪽으로 치우치는 경향) 또는 무상관 (둘이 독립된 phenomenon). |

> **RQ4 (harm asymmetry stratification) 폐기 (2026-05-16).** 사유: (1) harm_asymmetry 도출이 MoralChoice 10 컬럼의 *동등가중 개수 합* 에 의존 — 살인 1개 = 약속위반 1개로 conflate, 하필 두 액션이 비등한 high-ambiguity 셋에서 거의 다 symmetric 으로 뭉개지고 `"No Agreement"` 가 symmetric 쪽으로 편향. 흔들리는 지표는 reviewer 의 공격 표면이 되어 무방비보다 나쁨. (2) "OBR 이 단순 harm 회피 아니냐" 반박은 **NN 지표 정의 자체가 이미 차단** — 합리적 harm 회피자는 프레임 무관 *같은 outcome* (= frame-consistent YN/NY) 을 고르지만, NN 은 프레임마다 *다른 outcome* 을 고른 것 (무조건 "no"). NN 인 순간 outcome 추종이 아님이 정의상 따라오므로 harm stratification 없이 conceptual defense 가 성립. → harm 3-strata 분석 전면 제거, 관련 E4·F5·검증 임계·pre-registration 항목 삭제.

> **RQ 우선순위 (spine 부각용 — 본문에서 동급 배치 금지):**
> - **Spine (본문 전면):** RQ2a (충돌유형 → OBR *예측*) → E7 C2/C3 (oracle 인과 probe — 패널 축이 *원인*임 입증·상한선, 배포 아님) → E7 C1/C4 (label-free 배포 — 상한선의 ≥50% 회수, actionable). 이 arc 가 논문. RQ2b (model-specific signature) 는 spine 강화 보너스, 실패 허용.
> - **Supporting (본문, spine 보조):** RQ1 (cross-model OBR — PNAS 확장·warm-up).
> - **Robustness (부록/보조 분석):** RQ5 (NN vs YY 분리).
>
> **Mitigation 은 prompt-level 로 확정** (sampling/training 제외). 상세 condition (C0 baseline / C1 simultaneous-framing PNAS replication / **C2 yn-persona injection** / **C3 ny-persona injection** / C4 multi-phil consensus / C5 generic CoT), 통계 설계, preregistered hypothesis (H7a–H7f; C2/C3=oracle 인과 probe, C1/C4=label-free 배포) 는 supplement plan `~/.claude/plans/research-plan-v4-giggly-nebula.md` 에 동결. E7 은 분석 (RQ1·RQ2·RQ5) *이후* 실행하되, 그 결과가 비어 있다는 의미가 아니라 spine 의 인과 절반 + label-free 배포 절을 담당.

---

## 3. Benchmark — paired frames + 2-stage philosophy panel

### 3.1 시드 데이터

- MoralChoice high-ambiguity 680 (Scherrer+ 2023, GitHub `ninodimontalcino/moralchoice/data/scenarios/moralchoice_high_ambiguity.csv`).
- MoralChoice 의 10 harm-violation 컬럼 (`a1_death`, `a2_pain`, …) 은 raw provenance 로만 보존 — **분석 stratification 에는 사용하지 않음** (RQ4 폐기, 2026-05-16). 동등가중 개수 합이 살인=약속위반으로 conflate 되고 high-amb 셋에서 거의 symmetric 으로 뭉개져 신뢰 불가.
- low-ambiguity 687 은 *control* 후보로 보존하되 본 분석에서는 우선 high-amb 만 사용.

### 3.2 Inaction labeling

- `pilot/inaction_label.py` (regex tier 1 + LLM fallback) → `data/constructed/inaction_labels.csv` (680 rows, A1/A2 의 어느 쪽이 lexical inaction 인지 라벨링; 변경 없음).

### 3.3 Action↔Omission Mirror Frame 생성

- `src/data_construction/reframing/reframe.py` (v4 balanced SYSTEM_PROMPT, 2026-05-12). gpt-5 reasoning, OpenRouter.
- 산출: `data/constructed/mirror_frames/paired_frames.jsonl` — gpt-5 로 전수 생성 완료 (2026-05-12). **661 record 중 4개 (`G_116, G_125, G_228, G_330`) 는 `frame_A/frame_B` 가 null 인 생성 실패분 → 657 이 유효 시나리오.** `run.py` 가 malformed record 를 load 시 자동 skip + WARN (2026-05-16 견고화).
- **`validate.py` (LLM-judge) 단계 폐기 (2026-05-15).** 사유: filter.py (Stage 1 unanimity) 가 de-facto validator — frame 이 내부 비정합이거나 flip-equivalence 가 깨지면 패널이 noisy/만장일치 튜플을 내어 Stage 1 에서 어차피 탈락. 별도 LLM-judge 는 같은 신호 중복 + 예산 소모. stale `validation_report.jsonl` (v4 재생성 *전* 판정) 은 `validation_report.stale-pre-v4.jsonl` 로 백업, 분석에 미사용.
- **Paradigm misfit** (split-second 물리/강압 시나리오 H_001, H_005, H_006; 비인간 subject 예 H_004 deer 등) 은 manual exclusion list (`data/constructed/mirror_frames/paradigm_misfit.txt`, 작성 예정) 로 `build_benchmark.py` 가 hard-exclude. CLAUDE.md §Reframe pipeline invariants 참고.
- 인간 dual-annotation (Prolific N=2, κ ≥ 0.7 목표) on ~10% random subset 은 합격률 보고용 (blocking gate 아님 — validate.py 폐기로 수용 기준은 Stage 1 통과로 일원화).

### 3.4 Philosophy panel Stage 1 (unanimity filter)

- `src/data_construction/philosophy_panel/run.py` — gpt-4.1-mini × 5 철학 (F1 utilitarian / F2 deontologist / F3 virtue / F4 care / F5 contractualist; F6 DDE 는 2026-05-13 audit 에서 reliability 부족으로 drop) × paired frame × T=0.0 × n=1. 산출: `data/panel_outputs/panel_<provider>_<model-slug>.jsonl`.
- **패널 프롬프트 = 중립 (canonical, 2026-05-16 확정):** `philosophies.py` 활성본 = thin v1 에서 F1_util 의 `"Treat doing and allowing as morally equivalent when outcomes are equal."` 한 줄을 **제거**한 버전. 사유: 그 줄은 doing/allowing 축에 대한 *비대칭* 명시 지시 (util 만 받음) → 구축 단계 도구가 측정 대상(omission bias) 축으로 편향되면 안 됨. 전체 657 ablation 결과 그 줄 제거가 구조 불변 (util-as-lone-dissenter 44.5%→50.0%, 나머지 평탄) → 줄은 redundant 였고 제거가 reviewer 방어 + 중립성에 우월.
  - **Robustness 변형 (appendix):** ① `philosophies_v1_backup.py` (with-line, 220 labeled / 657, `*_robustness_v1_withDoAllowLine_657.*`) — 정의적 교리 명시 변형. ② `philosophies_v2_interventionist_backup.py` (doctrinally enriched, 198 labeled, `panel_promptv2_full657.*`) — *폐기*: 각 비공리주의 framework 에 행동-지지 sub-doctrine 을 (1차 문헌 인용 가능하나) 선택적 강조 → `dropped_one_sided_yn` 106→172 로 패널을 행동 쪽으로 떠밀어 측정 대상 편향. 세 변형 모두 util-vs-consensus 구조 동일 → "구조는 프롬프트 설계에 불변" 이 본문이 아닌 appendix robustness.
- `src/data_construction/philosophy_panel/filter.py` — 시나리오마다 5 개의 `(A, B)` 튜플 추출. 모두 동일 (= `unanimous_<tup>`) 이면 fail; `incomplete_cells`; `no_panel_data` (universe 에 있으나 panel record 없음); 그 외 pass. `--min-complete-phils N` (기본 5=strict) 으로 partial labeling 가능. `PHILS`/`FRAMES` 는 `philosophies.py` 공유 상수, panel-shape validator 가 미지 phil_id/frame WARN. 산출: `data/panel_outputs/filter_<panel-stem>.csv`.
- **실측 (canonical no-line, 657 유효):** Stage 1 pass 401 (61.0%), 만장일치 fail 256 (YN 150 / NY 105 / NN 1), incomplete 0, no_panel_data 0, parse/error/refusal 0.
- **해석:** 5 철학이 한 목소리를 내는 시나리오는 도덕적 합의 케이스 — 모델 OBR 의 cross-philosophy 차이를 묻는 무대가 아니므로 분석에서 제외.

### 3.5 Philosophy panel Stage 2 (conflict labeling)

- `src/data_construction/philosophy_panel/label.py` — 필터 통과 시나리오마다:
  1. `YY`/`NN` 튜플 철학은 *labeler 부적합* 으로 `excluded_phils` (frame swap 에 무반응 → 일관 입장 없음). `?` 포함 튜플은 `incomplete_phils`.
  2. 남은 철학을 `yn_phils` (`YN`) vs `ny_phils` (`NY`) 로 분할.
  3. `label_status` sub-status: 양쪽 non-empty → `labeled`; YN 만 → `dropped_one_sided_yn`; NY 만 → `dropped_one_sided_ny`; 둘 다 없음 → `dropped_all_excluded`. (구 단일 `dropped_one_sided` 를 3분해, 2026-05-16 — yield attribution 용.)
  4. `labeled` 면 `conflicts = [[yn_i, ny_j], …]` **전체 cross-product 를 그대로 나열**.
- **`primary_conflict` (시나리오당 1쌍) 개념 도입 후 폐기 (2026-05-16).** lowest-index 규칙이 util(인덱스 0) 을 매번 집어가 비공리주의 충돌을 가렸음. 결정: 단일 inferential unit 을 강제하지 않고 `conflicts` 전체 나열. **한 시나리오가 다중 (yn×ny) cell 에 기여 → 비독립성은 분석 단계에서 scenario 를 random effect / cluster-robust SE 로 처리** (label 산출물에서 강제 안 함; §3.6·§4.4 참조).
- 산출: `data/panel_outputs/labels_<panel-stem>.jsonl` (시나리오당 1 줄: `scenario_id`, `label_status`, `yn_phils`, `ny_phils`, `excluded_phils`, `incomplete_phils`, `yn_count`, `ny_count`, `excluded_count`, `incomplete_count`, `conflicts`).

### 3.6 최종 벤치마크 schema

`data/constructed/benchmark/omission-bench-v1.jsonl` — 필터·라벨링을 통과한 (labeled) 시나리오만, 시나리오당 1 row, 다음 필드:

```json
{
  "scenario_id": "H_001",
  "frame_A_prompt": "...",
  "frame_B_prompt": "...",
  "outcome_if_yes_A": "...",
  "outcome_if_no_A": "...",
  "panel_tuples": {"F1_util": "YN", "F2_deon": "NY", ...},
  "yn_phils": ["F1_util", "F4_care"],
  "ny_phils": ["F2_deon", "F3_virtue"],
  "excluded_phils": ["F5_contract"],
  "conflicts": [["F1_util","F2_deon"], ["F1_util","F3_virtue"], ["F4_care","F2_deon"], ["F4_care","F3_virtue"]]
}
```

- `conflicts` 전체 (cross-product) 가 분석 입력. **한 시나리오가 다중 (yn×ny) cell 에 기여** (위 예시 = 4 cell, 같은 시나리오 데이터). per-pair OBR 의 모델 비교·χ²·permutation 은 cell 독립을 가정하므로 **반드시 `scenario_id` 를 random intercept 로 둔 mixed-effects (or cluster-robust SE, cluster=scenario)** 로 추정. "시나리오당 1 cell 카운트" 의 옛 primary-conflict 방식은 폐기 (util 가림 문제, §3.5). 이 비독립성 처리를 §4.4·§11 prereg 에 명시.

### 3.7 Low-stake control (보조)

- v2/v3 의 low-stake paired set (10 harm 라벨 모두 No 인 일상 도메인, 30-50 시나리오) 은 보조 — RQ5 분리 검증 (low-stake 에서는 NN/YY 가 둘 다 낮아야 함) 용도로만 사용. plan 의 primary contribution 아님.

### 3.8 ★ 실측된 패널 충돌 구조 (2026-05-16) — RQ2 재정의의 근거

전체 657 패널 (canonical no-line) 실행 결과, conflict 구조가 **단일축으로 붕괴**함:

- **labeled 218 / 657** (Stage 1 pass 401 → labeled 218, dropped_one_sided_yn 116, _ny 66, _all_excluded 0).
- **util-vs-나머지가 지배축.** conflict-pair instance 의 ~65% 가 util 포함. **util 이 단독 dissenter (1 vs 4 블록) 인 labeled 가 ~50%.** util 은 labeled 의 ~78% 에서 한쪽 진영에 위치 (YN 170 / NY 34 / excluded 14). util 은 frame-invariant 하게 outcome-추종 (YN), 비공리주의 4개는 act-aversion 으로 묶여 반대 (NY). 이는 PNAS 의 *기존* util↔deon 축의 재현 — v4 가 넘어서겠다던 바로 그 축.
- **비공리주의끼리 충돌은 *부차축* 으로 실재.** labeled 218 중 ~119 (55%) 가 ≥1 개의 non-util×non-util 충돌 포함 (deon-care 83, virtue-care 58, deon-contract 56, care-contract 55, deon-virtue 33, virtue-contract 11 — undirected, descriptive). 단 lowest-index primary 가 가렸을 뿐 데이터엔 존재 → 전체 `conflicts` 나열로 보존.
- **프롬프트·doing/allowing-line 불변:** thin v1 (with-line, 220) / no-line (218) / interventionist v2 (198) 세 변형 모두 같은 구조 → 구조는 프롬프트 아티팩트 아님 (robustness, appendix).

**RQ2 에의 함의:** 구 RQ2 가 가정한 "다양한 conflict-pair type 이 골고루 분포" 는 *데이터로 기각*. 사용 가능 cell 은 사실상 (a) util-vs-consensus 주축 1개 + (b) 6종 비공리주의 부차축 (cell 8–83). 따라서:
- **RQ2a 재진술:** "OBR-by-pair 가 비평탄한가" 를 **"util-vs-consensus 축 vs 비공리주의 부차축 사이, 그리고 부차축 내부에서 모델 OBR 이 유의하게 다른가"** 로 구체화. 주축은 PNAS 확장 (RQ1·spine 예측), 부차축은 v4 고유 신규 신호.
- **RQ2b (model-specific signature):** cell 다양성이 제한적 (주축 1 + 부차축 6) 이므로 profile vector 는 7-dim 으로 축소; permutation 검정력 약화 가능 → §10 임계를 보수적으로 (cosine < 0.85, 부차축 cell n ≥ 8 인 것만) 재설정. 실패해도 spine 불변 (이미 nested 보너스).
- **fallback (§10 negative-result):** 주축만 신호 있고 부차축 평탄이면 → "omission bias 의 도덕적 fault line 은 본질상 1차원 (consequentialist vs non-consequentialist consensus) 이며 PNAS 축으로 환원된다" 가 그 자체로 publishable finding (v4 의 novelty 는 *부차축의 모델별 발현* 으로 축소되나 spine=RQ2a+E7 은 주축으로도 성립).

---

## 4. 평가 설계

### 4.1 평가 모델 — 5-7 개 cross-vendor mix

선정 기준: **모델 종류 별 비교** 가 RQ1·RQ2b 의 핵심이므로 vendor / size / 학습 paradigm 이 다양해야 함.

- **OpenAI:** GPT-5-mini, GPT-4.1-mini (기존 reframe/panel 에 쓴 모델과 같은 family, 비교 anchor).
- **Anthropic:** Claude 4.5 Haiku, Claude 4.7 Sonnet.
- **Google:** Gemini 2.0 Flash, Gemini 2.5 Pro (예산 허용 시).
- **Open-source mid-size:** Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct (자체 inference 가능, replicability).

총 5-7 개 모델. 모두 OpenRouter / 자체 vLLM 으로 통일 호출.

### 4.2 평가 protocol

- 각 모델 × 각 paired scenario × 양 frame → yes/no 1 답변. 시스템 prompt 는 *no persona injection* (= baseline default behavior 측정). user prompt = frame_{A,B}_prompt + "Answer 'yes' or 'no'." 같은 간결 지시.
- 답변 파싱: `ANSWER:\s*(yes|no)` 정규식 + fallback 으로 첫 단어 매칭. 파싱 실패시 `null` 로 표기, OBR 계산에서 제외 (denominator 에서 빠짐). 모델별 파싱 실패율을 보고.
- 거부 (`I cannot help…`) 응답은 별도 카운트, OBR denominator 에서 제외, refusal rate 로 보고.
- T=0.0, n_samples=1 (memory: 2026-05-13 user policy). 단일 비결정성 우려 시 robustness check 용으로 1 모델만 T=0.7 × n=5 rollout 추가.

### 4.3 주요 metric

- **Tuple per scenario** = `(answer_A, answer_B)` ∈ {YY, YN, NY, NN}.
- **OBR (omission bias rate):** = #{NN} / #{YY+YN+NY+NN}. Per (model), per (model × conflict pair).
- **ABR (action bias rate, RQ5):** = #{YY} / N.
- **FCR (frame-consistent rate):** = #{YN+NY} / N. 시나리오의 outcome 일관성을 유지한 비율 — 1 − OBR − ABR.
- **Refusal rate, parse-failure rate:** 정상 답변 비율 sanity check.
- **per-(model × pair) OBR profile vector:** 각 모델에 대해 conflict pair 를 정렬한 OBR 벡터 (dim = #{labeled conflict pair types}). RQ2b 의 cosine similarity / clustering 입력.

### 4.4 통계 검정

- **RQ1 모델 간 overall OBR 차이:** Z-test for two proportions, pairwise, Bonferroni 보정.
- **RQ2a 모델 내 conflict pair 간 OBR 차이:** χ² of independence on (conflict pair × {NN, ¬NN}) per model; post-hoc 으로 top-OBR pair vs bottom-OBR pair z-test.
- **RQ2b 모델 signature 차이:** model A 의 OBR-by-pair profile vs model B 의 그것의 cosine similarity; permutation test (5,000 회 — 모델 라벨 셔플 후 profile 재계산) 로 관찰 cosine 의 random baseline 비교. *(RQ2a 통과 조건부 — RQ2 의 nested 강한 주장)*
- **RQ5 NN vs YY 분리:** per cell 의 NN rate, YY rate scatter + Spearman ρ across pairs within model.
- *(RQ4 의 3-way log-linear (model × conflict pair × harm stratum) 은 폐기 — ~400 labeled / ~336 cell ≈ cell 당 1, interaction term 검정력 0. harm 지표 자체도 신뢰 불가. 2026-05-16.)*

### 4.5 Power / sample size

- 필터 통과 시나리오 수 = E[total × (1 − unanim_rate − incomplete_rate)]. 13-scenario pilot (CLAUDE.md panel 출력) 에서 84.6% pass, 7.7% unanimous. 661 scenarios → 약 ~500 labeled 추정 (실제는 paradigm-misfit 제외 후 더 적을 수 있음, 400+ 목표).
- conflict pair 종류: 5 philosophies → max 4×4 = 16 ordered pair (yn 4 × ny 1 max). 실제로는 한 pair 에 ~10-50 시나리오 분포 예상.
- 한 cell 에 OBR 차이 0.15 를 5% 유의·80% 검정력으로 잡으려면 cell 당 n ≥ 174 (z-test for 2 proportions, p₁=0.2 vs p₂=0.35). 본 plan 의 cell 당 표본 (~20-40) 으로는 *큰 차이* (≥ 0.25) 만 잡힌다 — RQ2a/RQ2b 의 H 임계를 그 수준으로 보수적으로 설정.

---

## 5. 실험

### E1 — RQ1: Per-model overall OBR

- 5-7 모델 × `omission-bench-v1.jsonl` 전체 × 2 frame → tuple 산출 → overall OBR 표.
- Side metrics: ABR, FCR, refusal, parse-fail.
- 시각화: bar chart with 95% CI (Wilson interval). Cheung+ 2025 PNAS 의 GPT-4 baseline 과 비교.
- 산출: `outputs/experiments/E1_overall_OBR/per_model.csv`, `bar_chart.png`.

### E2 — RQ2a: Per-model OBR by conflict pair (spine 예측 절)

- E1 의 raw tuple 데이터를 `conflict_pairs` 라벨로 join → per (model × pair) cell 계산.
- 시각화: heatmap, 행 = 모델, 열 = conflict pair (정렬: 평균 OBR 내림차순). cell = OBR + 표본 크기.
- 모델별로 conflict pair 간 OBR 차이 χ² 표.
- 산출: `outputs/experiments/E2_OBR_by_conflict/heatmap.png`, `model_chi2_table.csv`, `per_cell_OBR.csv`.

### E3 — RQ2b: Model signature analysis (RQ2 의 nested 강한 주장, RQ2a 조건부)

- 각 모델의 OBR-by-pair profile vector 만들고:
  - 모델 × 모델 cosine matrix → dendrogram (hierarchical clustering, average linkage).
  - 같은 vendor (OpenAI GPT-5-mini vs GPT-4.1-mini), 같은 size (Qwen-7B vs Llama-8B) 끼리 가까운지 확인.
- Permutation test: 모델 라벨 셔플 5,000 회 → cosine 분포의 5% / 95% 임계와 관찰값 비교.
- 산출: `outputs/experiments/E3_signatures/cosine_matrix.csv`, `dendrogram.png`, `permutation_pvalue.txt`.

### E4 — (폐기, 2026-05-16)

- RQ4 (harm asymmetry stratification) 폐기에 따라 E4 제거. E5/E6/E7 번호는 cross-reference 보존 위해 그대로 유지. 사유는 §2 RQ4 폐기 note 참조.

### E5 — RQ5: NN vs YY 분리

- 같은 cell 단위로 NN rate, YY rate scatter.
- 모델 내 conflict pair 간 Spearman ρ(NN, YY).
- 추가: low-stake control set 에서 NN, YY 가 모두 낮은지 sanity (omission bias 가 high-stake artifact 가 아니라는 증거 또는 그 반대).
- 산출: `outputs/experiments/E5_NN_vs_YY/scatter_per_model.png`, `spearman.csv`.

### E6 — 인간 anchor (옵션, 분량 압축)

- Prolific N≈100 (representative US). labeled 시나리오에서 30-50 random sample × paired frame.
- 인간 OBR baseline + 모델과의 격차 표.
- 본 plan 의 *반드시* 항목은 아님 — RQ1-RQ5 가 모델 비교 자체가 main contribution. 인간 anchor 는 reviewer 가 요구하면 추가.

### E7 — Mitigation (★ spine 의 인과 절반 + label-free 배포, prompt-level 확정)

- **상세 설계 동결:** `~/.claude/plans/research-plan-v4-giggly-nebula.md` (2026-05-15, 2026-05-17 oracle/label-free 분리 갱신). 본 절은 요약.
- **6 conditions 을 3 역할로 명시 분리 (2026-05-17, deployability 비판 흡수):**
  - **Oracle 인과 probe (배포 아님):** **C2** yn-persona inject, **C3** ny-persona inject. 시나리오별 `min(yn/ny_phils)` 라벨(구축-시 패널) 필요 → 배포 불가. 역할 = philosophy-conflict 축이 bias 의 *원인·방향성*임을 knockout-style 로 입증 + 제거 가능 bias 의 **상한선**. *deployable 방법으로 주장 안 함.*
  - **Label-free 배포 후보 (actionable):** **C1** simultaneous-framing (PNAS replication, 라벨 무관), **C4** multi-philosophy consensus (라벨 무관, 임의 질의 적용 가능). 역할 = C2/C3 상한선의 일부(목표 ≥50%) 회수 → 실무 mitigation.
  - **Control:** **C0** baseline, **C5** generic CoT (panel vs 일반 reflection 분리).
- **모델:** E1 결과 기반 top-2 high-OBR + 1 low-OBR control (3개). **시나리오:** E2 top-OBR cell 에 stratified (§3.8 의 주축/부차축 cell 우선), 모델당 ≤120. **비용:** ~$20, 반나절.
- **Preregistered H7a–H7f** (H7b = C2/C3 인과·상한선, H7f = C1/C4 label-free 회수율 ≥50%) + falsifiable: C2/C3 ≤ C5 면 panel-anchoring 기각; C2/C3 강해도 H7f 실패면 "인과 축 확인, 배포는 future work (per-query 경량 패널)" 로 scope 정직 축소. 상세는 supplement plan §Pre-registered hypotheses.
- **제외 (명시적):** sampling-level (best-of-N), training-level (LoRA) 는 본 plan scope 밖 — 후속 short paper / journal extension. **C2/C3 의 per-query 배포화 (추론-시 경량 패널)** 도 future work.

---

## 6. 논문 구성 (Long paper, 8p 가정)

| 섹션 | 내용 | 분량 |
|------|------|------|
| 1. Introduction | PNAS thesis (framing-invariant omission bias) + Scherrer (model cluster) → 본 paper 의 contribution **2개**: (1) 2-stage panel conflict-typed benchmark 구축법, (2) ★ 패널 불일치 = omission bias 의 *조작 가능한 인과 축* (예측 RQ2a + 인과확인 E7 C2/C3 oracle probe + label-free 배포 C1/C4) | 1 p |
| 2. Background | MoralChoice, Cheung 2025, 5 도덕철학 framework, mirror frame paradigm | 0.75 p |
| 3. Benchmark | mirror frame 생성 + 2-stage philosophy panel (filter + label) | 1.25 p |
| 4. Evaluation setup | 5-7 모델, tuple-based OBR metric, conflict pair as analysis unit | 0.5 p |
| 5. Results | **E2 (conflict pair → OBR, spine 예측) + E7 (C2/C3 oracle 인과 probe·상한선 → C1/C4 label-free 배포 회수)** 을 전면; E1 (overall, warm-up) · E3 (signature, supporting) · E5 (NN/YY 분리, robustness) 보조 | 3 p |
| 6. Discussion | 모델 signature 가 vendor / training 추정에 시사하는 바, E7 의 인과 결론(C2/C3) 과 label-free 배포 회수율(C1/C4) 의 구분·함의, 순환논증 + oracle-label 방어, limitations | 1 p |
| 7. Limitations & Ethics | **순환논증 정면 방어** (별-모델 labeler / 라벨≠측정량 / C4·C5 control) + **harm-avoidance 대안설명 차단** (NN 정의상 frame 마다 다른 outcome → outcome 추종 아님; harm-strata RQ 불필요) + conflict pair 라벨러 = single LLM, philosophy 단순화, English/US-centric, IRB | 0.4 p |
| 8. Conclusion | 한 줄: "패널 불일치는 omission bias 를 예측하고 동시에 처방하는 동일 기구다" | 0.25 p |

**Key figures:**
- F1: 파이프라인 개요 (mirror frame → 2-stage panel → conflict-labeled benchmark → model 평가 → OBR heatmap).
- F2: E1 per-model overall OBR bar chart with CI.
- F3: **E2 heatmap** model × conflict pair — 본 paper 의 visual key (spine 예측).
- F4: **E7 mitigation** — C2/C3 oracle ΔNN(상한선) + directional shift, 그 옆에 C1/C4 label-free 회수율 bar (spine 인과+배포, F3 와 짝).
- F5: E3 dendrogram of model signatures (supporting).
- F6: E5 NN vs YY scatter (robustness).

**Appendix:**
- A: 5 philosophy persona prompt 전문.
- B: paradigm-misfit 시나리오 manual exclusion list.
- C: panel JSONL → filter CSV → label JSONL 검증 (dual κ on random subset).
- D: 모델별 파싱·거부 로그.

---

## 7. Timeline

| 기간 | 마일스톤 |
|------|----------|
| 2026-05-13 ~ 05-25 | (a) `paired_frames.jsonl` 661 scenario 전수 재생성 (v4 balanced SYSTEM_PROMPT + gpt-5 reasoning, $≈20). (b) `validate.py` iter 통과 (κ ≥ 0.7 in dual annotation). (c) paradigm-misfit manual exclusion list 확정. |
| 2026-05-26 ~ 06-08 | Philosophy panel run (gpt-4.1-mini × 5 phil × 2 frame × N scenarios, T=0.0 n=1). `filter.py` → `label.py` → `omission-bench-v1.jsonl` v1 freeze. |
| 2026-06-09 ~ 06-25 | **E1 + E2:** 5-7 모델 evaluation runs (OpenRouter + 자체 vLLM). per-cell OBR 표 + heatmap. |
| 2026-06-26 ~ 07-08 | **E3 + E5:** signature cosine + permutation test, NN/YY scatter. (E4 harm-strata 폐기) |
| 2026-07-09 ~ 07-20 | (옵션) **E6 인간 anchor** Prolific run + 분석. |
| 2026-07-21 ~ 08-10 | Mitigation supplement (E7) 의 plan 작성 → 우선 1-2 후보만 pilot. |
| 2026-08-11 ~ 08-25 | 논문 작성, 그림 6 종, appendix. |
| 2026-08 (late) | ARR August commitment 경로. EMNLP 2026 long paper. |

> EMNLP 2026 직접 마감은 지난 가정 (작성일 기준). ARR commit path 유지.

---

## 8. 위험 요소 및 완화

| 위험 | 완화 |
|------|------|
| **(★ 최대 위험)** Stage 2 의 `labeled` 시나리오가 너무 적음 (예: 661 → 200 이하) → per-(model × pair) cell 의 표본이 통계적으로 무력 | (a) low-amb 687 도 같은 파이프라인에 통과시켜 N ↑, (b) conflict pair 를 *그룹 단위* (yn ∪ ny 의 집합) 로도 보고하여 cell 합치기, (c) Bonferroni 대신 FDR 사용 |
| Stage 1 filter 가 너무 strict (만장일치 거의 없음 → 거의 모든 시나리오 pass) 또는 너무 lenient (대부분 unanimous → pass 거의 없음) | 13-scenario pilot 에서 84.6% pass / 7.7% unanimous → 적정. 661 전수 후 재측정. 만약 한쪽 극단이면 (i) panel temperature ↑ 로 노이즈 검증, (ii) panel 모델을 gpt-4.1-mini 외 보조 모델 (claude-haiku) 와 cross-check, (iii) tuple 정의 완화 (frame-level modal 대신 raw 답변 다수결) 검토 |
| Stage 2 의 라벨러 = gpt-4.1-mini 단일 모델 → 라벨러 자체의 model bias 가 결과 오염 | panel 을 **2 개 모델** (gpt-4.1-mini + claude-haiku 또는 gemini-flash) 로 독립 실행 후 conflict pair 라벨의 agreement 측정. 합의된 라벨만 본 분석 사용 (= κ ≥ 0.6 cutoff). 분량은 appendix C 로 |
| 5 philosophy 자체가 단일 모델 (gpt-4.1-mini) 위에서 잘 분리되지 않음 — system prompt 가 무시되어 모든 철학이 비슷한 답변 | 13-scenario pilot 에서 conflict pair 분포가 다양했음 (top: util vs deon 5건) → 분리는 어느 정도 됨. 631 전수 후 모니터; 분리 약하면 system prompt 톤 강화 + few-shot 추가 |
| paired frame 자체가 OLD-design (legacy 661) 그대로 → 모든 후속 분석이 invalid input 위 | timeline week 1-2 의 reframe 전수 재생성이 모든 step 의 blocker. validate.py 통과 + dual κ 확보 까지 후속 step blocked |
| 모델이 yes/no 외 형식으로 답변 (장문, refusal 등) → parse failure 가 OBR denominator 를 망가뜨림 | user prompt 에 "Answer exactly 'yes' or 'no', nothing else" 강제 + fallback 정규식. 모델별 parse fail rate ≥ 10% 면 그 모델 결과는 caveat 표기. open-source 모델은 chat template + EOS 강제로 보완 |
| RQ2b 의 model signature 차이가 random baseline 수준 — 모든 모델이 거의 같은 OBR-by-pair profile | 그것 자체가 finding: "omission bias 의 conflict-pair 분포는 모델 architecture 와 무관한 universal pattern 이며, 단지 *전체 강도* 만 모델마다 다르다." spine 은 RQ2b 없이도 RQ2a(예측)+E7(처방)로 성립 — RQ2b 는 애초에 nested 보너스 주장이므로 실패해도 thesis 무손상, RQ1(전체 강도)+RQ2a+E7 중심으로 진술 |
| YY (action bias) 가 거의 0 — 모든 모델이 어느 쪽으로든 inaction 으로 기우는 경향만 보임 | RQ5 의 분리 검정 실패. 그것 자체가 publishable: "LLM 의 frame-invariant bias 는 inaction 쪽으로만 발현하고 action 쪽으로는 발현하지 않는다 — PNAS thesis 강화" |
| 예산 — 5-7 모델 × ~500 시나리오 × 2 frame ≈ 5,000-7,000 호출/모델. OpenAI/Anthropic 합쳐 ~$50-100 추정 | 부담 가능. open-source 모델은 vLLM self-host 로 zero cost |
| Mitigation 이 없으면 reviewer 가 "no actionable contribution" 비판 | **해소됨** — E7 의 label-free C1/C4 가 배포 가능한 actionable 결과 (results 에 위치). C2/C3 는 인과·상한선 (actionable 주장 아님) |
| **(★ fatal review 후보) C2/C3 oracle-label 의존 → "배포 불가, mitigation trivial" 비판** (사용자가 2026-05-17 제기) | 본문 정면 방어: (1) C2/C3 는 *deployable 방법이 아니라 knockout-style causal manipulation* — 통제 실험은 oracle 사용이 원칙적으로 정당, 역할은 philosophy-conflict 축의 *인과* 입증 + 제거 가능 bias 의 **상한선** 산출. (2) actionable 배포 주장은 라벨 불필요한 **C1/C4 (H7f, ≥50% 회수)** 가 전담 — 임의 신규 질의에 그대로 적용. (3) C2/C3 상한선이 있어야 C1/C4 회수율이 "전체 가능분의 몇 %" 로 해석됨 → 둘은 분리된 역할로 상호보완. §5 E7·§7 에 명시. 이 방어가 없으면 정확히 이 비판이 fatal review 로 회귀 |
| **(★ fatal review 후보) 순환논증 비판** — "시나리오를 패널로 정의해놓고 그 패널로 고친다 = circular, 따라서 mitigation 효과는 trivial" | 본문에서 **정면 방어** (§7 Limitations + Methods 에 명시): (1) 패널 labeler (gpt-4.1-mini) 와 평가 모델이 *서로 다른 모델*, 패널은 평가 모델 출력을 한 번도 보지 않음. (2) 구축-시 라벨은 "어떤 철학이 frame-consistent 한가" 라는 *시나리오 속성*, mitigation 은 그 속성을 *다른 모델* 에 주입해 *행동 지표 (NN→YN/NY)* 가 바뀌는지를 측정 — 라벨과 측정량이 다른 공간. (3) C5 (generic CoT) 와 C4 (non-directional panel) 가 "패널 신호 특정성" 을 분리하는 control 로 순환성을 경험적으로 반박. |

---

## 9. 코드/데이터 구조

```
omission/
├── data/
│   ├── raw/moralchoice/                       # 변경 없음
│   ├── constructed/
│   │   ├── inaction_labels.csv                # 변경 없음
│   │   ├── mirror_frames/paired_frames.jsonl  # PNAS v4 balanced 재생성 필수
│   │   ├── low_stake_control.jsonl            # (옵션) RQ5 보조
│   │   └── benchmark/
│   │       └── omission-bench-v1.jsonl        # ★ filter + label 통과 시나리오만, conflict_pairs 필드 포함
│   ├── annotations/                           # dual κ 검증 로그
│   └── panel_outputs/                         # 변경 없음 + filter_*.csv + labels_*.jsonl
├── src/
│   ├── data_construction/
│   │   ├── reframing/                         # 변경 없음
│   │   ├── inaction_labeling/                 # 변경 없음 (pilot/inaction_label.py 의 src/ migration 대기)
│   │   └── philosophy_panel/
│   │       ├── run.py                         # 변경 없음
│   │       ├── filter.py                      # ★ 2026-05-13 추가 (Stage 1)
│   │       ├── label.py                       # ★ 2026-05-13 추가 (Stage 2)
│   │       └── build_benchmark.py             # ★ 신설: paired_frames + labels 조인 → omission-bench-v1.jsonl (harm 컬럼 조인 제거)
│   ├── evaluation/
│   │   ├── runners/
│   │   │   └── eval_model.py                  # ★ 신설: 한 모델 × 벤치마크 → tuple jsonl
│   │   ├── metrics/
│   │   │   └── obr_metrics.py                 # ★ 신설: tuple jsonl → per-(model × pair) OBR/ABR/FCR
│   │   └── mitigation/                        # ★ 7월에 채움 (E7)
│   ├── analysis/
│   │   ├── per_model_obr.py                   # ★ 신설 (E1)
│   │   ├── obr_by_conflict.py                 # ★ 신설 (E2)
│   │   ├── model_signatures.py                # ★ 신설 (E3 cosine + permutation)
│   │   ├── nn_vs_yy.py                        # ★ 신설 (E5)  — harm_strata.py 폐기
│   │   └── panel_disagreement_legacy.py       # 변경 없음 (legacy)
│   └── shared/                                # 변경 없음
├── outputs/
│   ├── experiments/
│   │   ├── E1_overall_OBR/
│   │   ├── E2_OBR_by_conflict/
│   │   ├── E3_signatures/
│   │   ├── E5_NN_vs_YY/
│   │   ├── E7_mitigation/
│   │   └── E6_human_anchor/                   # 옵션
│   ├── analysis/                              # 통합 표
│   └── figures/                               # F1~F6
├── configs/                                   # eval matrix yaml
└── pilot/                                     # 변경 없음
```

---

## 10. Verification — "잘 됐다" 임계

1. **Mirror frame (충족):** validate.py LLM-judge 폐기 (2026-05-15). 수용 기준 = Stage 1 통과로 일원화. dual annotation κ ≥ 0.7 on 10% 는 보고용 (non-blocking).
2. **Filter pass rate (충족):** 실측 61.0% (401/657, no-line) — 50-90% 범위 내. ✓
3. **Label rate (충족):** Stage 1 통과 중 labeled 비율 실측 ≈ 54% (218/401). ✓ (≥ 40% 임계 통과.)
4. **Final benchmark size (재설정, 2026-05-16):** 옛 "≥ 200, ≥ 8 distinct pair, cell 평균 ≥ 25" 는 util-vs-consensus 붕괴로 **기각** (실제: labeled 218, 사용 cell = 주축 1 + 비공리주의 부차축 6, cell n 8–83). 재임계: **labeled ≥ 180** (프롬프트 변형 범위 198–220, 사전 임계는 robustness 변동으로 간주) **AND 비공리주의 부차축 중 cell n ≥ 8 인 것 ≥ 4종**.
5. **E1:** 모든 모델의 parse-fail + refusal 합산 ≤ 10%. 모델 간 overall OBR pairwise z-test 유의 ≥ 1 쌍.
6. **E2 (RQ2a, spine 필수):** 적어도 1 모델에서 — util-vs-consensus 주축 vs 비공리주의 부차축, 또는 부차축 cell 간 — OBR 차이가 mixed-effects (scenario random intercept) 하에서 유의 (p < 0.05) 하고 top-bottom OBR spread ≥ 0.15.
7. **E3 (RQ2b, nested 보너스 — 실패 허용):** profile vector = 7-dim (주축 1 + 부차축 6, cell n ≥ 8 인 것만). 모델-모델 cosine 행렬에서 ≥ 1 model pair 의 profile cosine < **0.85**, permutation p < 0.05. (cell 다양성 제한으로 임계 0.8→0.85 완화, 2026-05-16.)
8. **E5:** NN rate vs YY rate Spearman ρ across cell 가 모델 절반 이상에서 |ρ| ≥ 0.3 (양·음 둘 다 의미 있음).
9. **E7 (spine 처방):** supplement plan §Verification — top-2 high-OBR 모델에서 C2 또는 C3 의 ΔNN McNemar Bonferroni p < 0.05, 그리고 C0=NN→C2 가 YN-편향 (Fisher one-sided p < 0.05).
   *(E4 harm-strata 검증 항목 폐기, 2026-05-16.)*

**RQ1 + RQ2 동시 실패 (= 모델 간·pair 간 OBR 모두 평탄) 시:** 본 plan 의 core thesis 실패. discussion 섹션을 "omission bias 는 모델·도덕적 문맥에 invariant 한 single global phenomenon" 으로 reframe — PNAS thesis 강화 형태의 negative-result paper 로 전환.

---

## 11. Pre-registration

OSF 사전등록 권장 항목:

- 각 RQ 의 primary metric 과 임계 (위 §10).
- 가설 방향:
  - RQ1: 모델 간 overall OBR pairwise z-test, 적어도 1 쌍 유의 (directional 아님, exploratory).
  - RQ2a: model 내 conflict pair 간 OBR χ² 유의 ≥ 1 모델 (directional 아님). *spine 필수.*
  - RQ2b: cosine 차이 + permutation p < 0.05 (directional 아님). *RQ2 의 nested 보너스, RQ2a 조건부, 실패 허용 (RQ3 → RQ2b 병합).*
  - RQ5: NN-YY Spearman ρ 의 부호와 크기는 exploratory; *둘이 동일 phenomenon 이라는 null* 을 reject 하는 것이 목적.
  - E7: H7a–H7f (supplement plan). C2/C3 directional (one-sided, oracle 인과 probe·상한선), H7f = C1/C4 label-free 회수율 ≥50% (배포 주장은 여기서만), 나머지 exploratory.
  - *(RQ4 harm stratum interaction 항목 폐기 — 2026-05-16, §2 note 참조.)*
- **비독립성 분석 방법 사전 commit (2026-05-16):** per-(model×pair) OBR 의 모든 추론 검정은 `scenario_id` 를 random intercept 로 둔 mixed-effects (또는 cluster-robust SE, cluster=scenario). 한 시나리오가 다중 cell 에 기여하는 구조를 prereg 에 명시 — 옛 primary-conflict (시나리오당 1 cell) 방식은 폐기.
- **프롬프트 변형 robustness 사전 commit:** canonical = neutral no-line `philosophies.py`. with-line v1 + interventionist v2 는 appendix robustness (3 변형 모두 util-vs-consensus 구조 동일이 사전 예측). main 분석은 no-line 만.
- 분석 데이터 freeze: `omission-bench-v1.jsonl` v1 의 SHA-256 을 commit 에 기록, 그 이후 conflict 라벨 재조정 금지.
- Exclusion criteria: parse failure, refusal, paradigm-misfit (`paradigm_misfit.txt`), malformed paired_frames (`G_116/125/228/330`) 사전 명시.
- Negative result 해석 정책: §10 의 RQ1+RQ2a 동시 실패 시 → "fault line 은 1차원 (consequentialist vs consensus), PNAS 축 환원" 으로 fallback (§3.8); 주축 성립·부차축 평탄이면 novelty 만 축소되고 spine(RQ2a 주축+E7) 은 유지 — 사전 commit.

---

## 12. 핵심 contributions (paper bullets — 2 개로 압축)

> 4개 나열은 grab-bag 으로 읽혀 spine 을 묻는다. intro 의 contribution 은 아래 **2개** 로만 진술. 나머지 (RQ1 cross-model 확장, RQ2b signature, RQ5 NN/YY) 는 contribution bullet 이 아니라 이 2개를 떠받치는 *supporting/robustness 결과* 로 본문 내에 배치. (RQ3→RQ2b 병합·RQ4 harm-strata 폐기, 2026-05-16.)

1. **Two-stage philosophy-panel 으로 conflict-typed paired benchmark 구축법:** Mirror-framed 시나리오 위에서 5 도덕철학 persona 의 (frame_A, frame_B) 튜플 만장일치를 filter, 양극 (yn vs ny) 으로 갈리는 것만 *pairwise conflict pair* 로 라벨링 → *어떤 철학적 대립인지* 가 메타 라벨로 붙은 `omission-bench-v1.jsonl`. (방법론적 신규)

2. **★ Main: 패널 불일치 = omission bias 의 조작 가능한 인과 축:** 같은 패널 신호가 (a) 시나리오의 충돌유형으로 평가 모델의 OBR 을 *예측* 하고 (RQ2a), (b) 그 신호가 지목한 철학을 oracle 로 주입(C2/C3)하면 OBR 이 예측 방향(C2→YN, C3→NY)으로 이동 → philosophy-conflict 축이 *인과* 임을 knockout-style 로 입증 + 제거 가능 bias 의 *상한선* 산출 (배포 방법 주장 아님), (c) 라벨이 필요 없는 C1/C4 가 그 상한선의 ≥50% 를 라벨 없이 회수 → *배포 가능한* 완화. C5/low-OBR control 로 신호 특정성·floor, 별-모델 labeler 로 비순환성 확보 → correlational 이 아니라 causal claim + actionable. PNAS (단일 util↔deon, simultaneous-framing) 와 Scherrer (model-philosophy 군집) 어느 쪽도 하지 않은 *진단·인과확인 동일기구 + label-free 배포* 의 closure.

---

## 변경 로그 (v3 → v4)

- **Pivot:** v3 의 persona-vector / mechanistic 방향 (omission vector 추출, steering, data screening, RLHF causality) **전면 폐기**. white-box internals 사용 없음.
- **데이터 구축:** mirror frame 생성은 v3 와 동일하게 보존. **6 philosophy → 5 philosophy** 로 축소 (F6 DDE drop, 2026-05-13 audit 근거). **Stage 1 unanimity filter + Stage 2 (Y,N) vs (N,Y) conflict labeling** (2026-05-13 구현) 으로 panel 단계 재정의.
- **분석 축 단일화:** **per-(model × conflict pair) OBR cell** 이 모든 분석의 unit. v3 의 cosine/UMAP/regression coefficient 전부 폐기.
- **모델 풀 확장:** v3 는 2 white-box + 4 closed (behavioral anchor) → v4 는 5-7 cross-vendor 가 *primary*. open-source vs closed 의 vendor 다양성이 RQ2b signature 분석의 핵심 입력.
- **Mitigation 확정 (2026-05-15 갱신):** v3 의 E4/E5/E6 모두 제거. E7 은 **prompt-level 6-condition (C0–C5)** 으로 확정, supplement plan (`~/.claude/plans/research-plan-v4-giggly-nebula.md`) 에 동결. spine reframe — RQ2 + E7 이 본문 thesis (패널 불일치 = 조작 가능한 인과 축), RQ1 supporting, RQ5 robustness, contribution bullet 4→2 압축, 순환논증 정면 방어 추가.
- **RQ4 (harm asymmetry stratification) 폐기 (2026-05-16):** harm_asymmetry 가 MoralChoice 10 컬럼 동등가중 개수 합 (살인=약속위반 conflate, high-amb 에서 symmetric 으로 뭉개짐, No-Agreement 편향) 으로 신뢰 불가. + harm-avoidance 대안설명은 NN 정의 자체가 차단 (frame 마다 다른 outcome → outcome 추종 아님). E4·F5·harm_strata.py·검증8·preregRQ4 모두 제거, harm 컬럼은 raw provenance 로만 잔류. §7 Limitations 에 conceptual defense 1문단으로 흡수.
- **RQ3 → RQ2b 병합 (2026-05-16):** 구 RQ3 (model-specific signature) 는 구 RQ2 (conflict pair → OBR) 가 통과해야만 의미 (profile 평탄하면 signature 비교 무의미). 별개 RQ 로 나열하면 동일 데이터 padding 으로 읽힘 → **RQ2 = RQ2a(예측, 비평탄, spine 필수) + RQ2b(model-specific, nested 보너스, 실패 허용)** 한 RQ 의 nested 주장으로 병합. E2=RQ2a, E3=RQ2b 로 실험·검증·prereg·figure 라벨 일괄 갱신. RQ 번호는 cross-ref 보존 위해 RQ1·RQ2·RQ5 로 유지 (RQ3·RQ4 결번).
- **데이터 구축 파이프라인 확정 (2026-05-16):**
  - `validate.py` LLM-judge **폐기** (2026-05-15) — filter.py Stage 1 이 de-facto validator, 별도 judge 는 신호 중복.
  - paired_frames 전수 생성 완료 (gpt-5, 661 중 4개 malformed → 657 유효; run.py 가 malformed 자동 skip).
  - `primary_conflict` (시나리오당 1쌍 inferential unit) 도입 후 **폐기** — lowest-index 규칙이 util 을 매번 집어 비공리주의 충돌 가림. 결정: `conflicts` 전체 나열, 비독립성은 분석 단계 mixed-effects (scenario random intercept) 로 처리.
  - 패널 프롬프트 canonical = **neutral no-line** (thin v1 에서 F1_util 의 doing/allowing 동등성 명시 줄 제거). 전체 657 ablation 으로 구조 불변 확인 → 줄은 redundant·비대칭, 제거가 중립성·reviewer 방어 우월. with-line v1 + interventionist v2 는 appendix robustness (interventionist v2 는 패널을 행동쪽으로 떠밀어 폐기).
  - `dropped_one_sided` → `_yn`/`_ny`/`_all_excluded` 3분해; filter.py 에 `no_panel_data`·`--min-complete-phils`·panel-shape validator·공유 `PHILS` 추가.
- **★ 실측 충돌 구조 = util-vs-consensus 붕괴 (2026-05-16, §3.8):** 전체 657 결과 conflict 가 util(=결과주의) vs 비공리주의 4-블록 단일축으로 붕괴 (util 단독 dissenter ≈50%). v4 가 넘어서겠다던 PNAS util↔deon 축의 재현. 비공리주의끼리 충돌은 부차축으로 실재 (labeled 의 55%, cell 8–83). RQ2a 를 "주축 vs 부차축 / 부차축 내부 OBR 차이" 로 구체화, RQ2b profile 7-dim 축소·임계 완화, §10.4 벤치마크-size 임계 재설정, negative-result fallback 을 "1차원 fault line" 으로 사전 commit.
- **E7 oracle-probe vs label-free 배포 분리 (2026-05-17):** "시나리오마다 yn/ny 라벨 주입은 배포에서 불가능 → mitigation trivial" 비판 (사용자 제기) 흡수. C2/C3 를 *deployable 방법이 아니라 knockout-style 인과 probe* (oracle 라벨 정당, 제거 가능 bias 의 상한선) 로 명시 재정의; 배포 주장은 라벨 불필요한 **C1/C4** 가 전담 (신규 **H7f**: C1/C4 가 C2/C3 상한선의 ≥50% label-free 회수). spine thesis 를 (1)예측 RQ2a + (2)인과 C2/C3 + (3)label-free 배포 C1/C4 의 3절로 재서술, §8 에 oracle-label fatal-review 방어 행 추가, supplement plan 의 condition 표·H7f·falsifiable·§위험0 동기화. "C2/C3 를 배포 방법으로 over-claim 안 함" 이 정직성 핵심.
- **인간 anchor 옵션화:** v3 E8 은 paper 의 필수 → v4 E6 는 옵션 (reviewer 요구 시).
- **신설 도구:** `src/data_construction/philosophy_panel/{filter.py, label.py}` (2026-05-13 작성 완료), `src/evaluation/runners/eval_model.py`, `src/evaluation/metrics/obr_metrics.py`, `src/analysis/{per_model_obr, obr_by_conflict, model_signatures, nn_vs_yy}.py` (예정; harm_strata.py 폐기).
- **legacy 표기:** `src/analysis/panel_disagreement.py` → `panel_disagreement_legacy.py` (2026-05-13, 새 (A,B) tuple 로직과 비호환).
