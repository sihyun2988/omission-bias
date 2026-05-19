# Research Plan v4 — Conflict-Typed Omission Bias across LLMs

> **🔴 2026-05-18: M4(철학 5 MAD) 완전 폐기 + 완화 스코프 전면 218.** 이 배너가 이하 모든 M4/MAD 및 핫스팟-manifest 서술에 우선한다. 완화 조건 = **M0/M1/M2/M3/M3b** 만 (M4 미등록). H4·primary set 의 `{M2,M3,M3b,M4}` → **`{M2,M3,M3b}`**. "M4 vs M3/M3b 보조분석"·"RQ3 지문 base↔M4 이동"·F5 의 M4 항 모두 무효. 스코프 = **전체 218 labeled × 5 모델** (top-cell/NN 서브샘플·≤120 manifest 폐기, `--all-scenarios`). 사유: M4 = 20콜/시나리오 비용 driver + 파서 취약 + llama 에서 net-harmful(NN→YY). supplement plan 상단 배너와 동기화. (변경 이력 항목도 하단 참조.)
>
> 작성일: 2026-05-13 · 본 문서는 `RESEARCH_PLAN_v3.md` (persona-vector pivot) 을 **전면 폐기**하고 새로 작성한 plan이다.
> v3 의 mechanistic / persona-vector 방향은 모두 들어냈고, **데이터 구축은 paired mirror frame + 2-stage philosophy panel (Stage 1 unanimity filter, Stage 2 (Y,N) vs (N,Y) conflict labeling, 2026-05-13 구현)** 으로, **분석은 (RQ1) 구축법 검증 + (RQ2) 충돌유형별 OBR 서술 + (RQ3) 모델 도덕지문 + (RQ4) 5-철학 주입=완화** 로 정리한다. 완화는 별도 단계가 아니라 **RQ4 안에 통합** — 상세 설계·정확한 prompt 는 supplement plan (`~/.claude/plans/research-plan-v4-giggly-nebula.md`, 전면 개정 2026-05-17) 에 동결.
>
> **Thesis (인과 주장 없음 — 2026-05-17 전면 재정의):** 본 논문의 기여는 *인과 축 증명* 이 아니라 **(1) 도덕철학-패널 disagreement 로 omission bias 를 더 잘 노출하는 conflict-typed 벤치마크 구축법, (2) 모델별 도덕지문 + 어떤 충돌에서 편향이 강한지의 구조적 분석, (3) 철학중립·label-free 완화기법 + 5-철학 균형주입의 효과** 이다. 구판 spine("패널 불일치 = 조작 가능한 인과 축", C2/C3 oracle 주입으로 인과 입증)은 **폐기**: 단일-철학을 정답인 양 주입하는 것은 철학적으로 방어 불가하고, 그 주입은 *원인*(조작 불가능한 시나리오 속성)이 아니라 *매개*(모델 채택 철학)를 건드리는 한정된 steerability 주장에 불과 → 인과 야망 자체를 내려놓는다. 이 결정으로 구 §8 의 ★fatal-review 2행(oracle-label 의존·순환논증)이 *원인 소멸로 자동 해소*. **RQ1 = 구축 검증(spine), RQ2/RQ3 = 분석 핵심, RQ4 = 완화(통합). RQ5(NN vs YY)는 robustness.** (변경 이력: RQ3→RQ2b 병합·RQ4 harm-strata 폐기 2026-05-16 → 2026-05-17 RQ 전면 재정의·C2/C3 인과실험 폐기·완화를 RQ4 로 흡수.)
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

**핵심 thesis (인과 주장 없음 — 2026-05-17 재정의):** 본 논문은 "패널 불일치가 omission bias 의 *원인*" 이라고 주장하지 *않는다*. 대신 네 갈래로 기여한다 — (1) **구축 검증 (RQ1)**: high-amb 시나리오에서 philosophy-disagreement filter 가 random sampling 보다 framing-invariant omission bias(NN)를 더 잘 노출하는가 → 구축법이 무선 대비 부가가치가 있음을 보이는 것이 본문 spine. (2) **충돌-유형 구조 (RQ2, 서술)**: 모델별로 어떤 철학-충돌 유형에서 OBR 이 강한가 — 인과가 아니라 핫스팟 식별, RQ3/RQ4 입력. (3) **모델 도덕지문 (RQ3, 핵심)**: 모델이 frame-consistent 답을 한 시나리오에서 어느 철학 진영과 일치하는지 + 편향율 → "GPT 는 공리주의-일치 60%, 단 NN 35%" 식의 모델 도덕성향 프로파일. (4) **완화 (RQ4 통합)**: 5-철학 *균형* 주입(MAD, 단일아님) vs base 가 편향·지문을 어떻게 바꾸나, 철학중립 기법(자기찬반·결정장부·동시제시)과 비교. 완화는 별도 실험이 아니라 RQ4 의 comparator 묶음이다. 구판의 단일-철학 oracle 주입(C2/C3) 인과 실험은 폐기 — 한 철학을 정답으로 주입하는 철학적 방어 불가 + 그것은 조작 불가능한 *원인*이 아니라 *매개*를 건드리는 한정 steerability 주장이므로 인과 closure 를 못 만든다. RQ5(NN vs YY 분리)는 robustness. (harm-avoidance 대안설명은 NN 정의가 자체 차단 — 별도 harm-strata RQ 불필요, 구 RQ4 폐기 유지.)

---

## 2. Research Questions

| RQ | 질문 | 정량 측정 / 가설 |
|----|------|------------------|
| **RQ1** (구축 검증 — spine) | high-amb 시나리오에서 **philosophy-disagreement filter 가 random sampling 보다 framing-invariant omission bias(NN)를 더 잘 노출**하는가? | filtered set (`label_status=="labeled"`, ≈218) vs 비-filtered 보완집합에서 동수 random sample. per-model OBR=#{NN}/#{both-answered}. **H1: `OBR(filtered) − OBR(random) > 0`, 모델 전반 일관 (Wilcoxon signed-rank one-sided, 모델 가로질러).** 보조 baseline = 전체에서 동수 random. |
| **RQ2** (분석 — 서술, 인과 아님) · **실측 구조 §3.8 (util-vs-consensus 주축 + 비공리주의 부차축)** | 모델별로 어떤 철학-충돌 유형에서 OBR 이 강한가 — 모델 내 OBR-by-pair 가 *비평탄* 한가 (주축 vs 부차축, 부차축 내부)? 핫스팟 식별용, RQ3/RQ4 입력. | per-(model × conflict pair) cell OBR; `scenario_id` random intercept mixed-effects (다중 cell 기여 → §3.6) + post-hoc top-vs-bottom z. **H2: ≥1 모델에서 conflict pair 간 OBR spread ≥ 0.15.** directional 아님, 서술. |
| **RQ3** (분석 — ★핵심, 모델 도덕지문) | 모델이 frame-consistent 답을 한 시나리오에서 *어느 철학 진영과 일치*하는가 + 편향율은? | (model,scenario) tuple: **YN→ 그 시나리오 `yn_phils` 각 +1, NY→ `ny_phils` 각 +1, NN/YY→ non-aligned(=편향)**. 모델별 5-철학 leaning profile + 편향율. **H3: ≥1 모델 profile 이 null(패널 base rate) 대비 유의 (permutation 5,000회 p<0.05) AND 모델 간 profile cosine 행렬 ≥1 쌍 < 0.85** (구 RQ2b signature 흡수). |
| **RQ4** (분석 = 완화, 통합) | **5-철학 *균형* 주입(MAD, 단일아님) vs base** 가 편향·도덕지문을 어떻게 바꾸나; 철학중립 label-free 완화는 NN 을 frame-consistent 로 줄이나? | per (model, condition) exact McNemar on (NN_M0, NN_Mk), credited = `NN→{YN,NY}`만 (NN→YY action-bias 치환 불인정). **H4: {M2,M3,M3b,M4} 중 ≥1 이 top-2 high-OBR 모델에서 Bonferroni 후 credited ΔNN 유의 AND M1(generic CoT) 초과.** 보조: M4(철학 anchoring) vs M3/M3b(중립) 비교. |
| **RQ5** (robustness) | omission bias 가 **action bias** ((Y,Y), frame-invariant *action* 선호) 와 분리되는가? | per-(model × pair) YY rate 동일 schema. NN vs YY scatter; H5: NN-YY Spearman ρ 의 음/무 상관 (둘이 동일 phenomenon 이라는 null reject). |

> **RQ4 (구 harm asymmetry stratification, 2026-05-16 폐기) 의 번호는 2026-05-17 재정의로 신규 RQ4(완화 통합)에 재배정.** 구 harm-strata 폐기 사유는 유지: harm_asymmetry 가 MoralChoice 10 컬럼 동등가중 개수 합으로 살인=약속위반 conflate·high-amb symmetric 뭉갬·No-Agreement 편향 → 신뢰 불가. + "OBR 이 단순 harm 회피 아니냐" 반박은 **NN 정의 자체가 차단** (NN 은 프레임마다 *다른* outcome → outcome 추종 아님). harm 컬럼은 raw provenance 로만 잔류, §7 Limitations 에 1문단 conceptual defense.

> **RQ 우선순위:**
> - **Spine (본문 전면):** RQ1 (구축법이 무선 대비 omission bias 를 더 잘 노출 — 벤치마크 가치의 직접 증거).
> - **분석 핵심 (본문):** RQ3 (모델 도덕지문 + 편향율), RQ2 (충돌-유형 구조, RQ3/RQ4 입력).
> - **완화 (본문, RQ4 통합):** RQ4 = M0–M4 comparator 묶음. 별도 E7 실험 아님.
> - **Robustness (부록/보조):** RQ5 (NN vs YY 분리).
>
> **완화는 RQ4 안에 통합** (sampling/training 제외, prompt-level). 조건 = **M0 baseline / M1 generic CoT / M2 두-프레임 동시(양쪽 yes/no, NN 측정가능) / M3 자기찬반 / M3b 결정장부 / M4 철학5 MAD**. 정확한 prompt·통계·H1–H4·NN-exit 분해·RQ3 fingerprint 절차·비용은 supplement plan `~/.claude/plans/research-plan-v4-giggly-nebula.md` (전면 개정 2026-05-17) 에 동결. **구판 C0–C5 + C2/C3 oracle 주입·label-free 회수율(H7f)·키스톤(scramble/cross-over) 전부 폐기.**

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

**RQ2/RQ3 에의 함의 (2026-05-17 재정의):** 구 RQ2 가 가정한 "다양한 conflict-pair type 이 골고루 분포" 는 *데이터로 기각*. 사용 가능 cell 은 (a) util-vs-consensus 주축 1개 + (b) 6종 비공리주의 부차축 (cell 8–83). 따라서:
- **RQ2 (서술):** "OBR-by-pair 가 비평탄한가" 를 **"util-vs-consensus 주축 vs 비공리주의 부차축 사이, 그리고 부차축 내부에서 OBR 이 유의하게 다른가"** 로 구체화. 인과 아님 — 핫스팟 식별. 주축은 PNAS 축 재현, 부차축은 v4 고유 신호.
- **RQ3 (모델 도덕지문):** profile 은 "frame-consistent 답이 YN→yn_phils / NY→ny_phils 정렬" 로 산정 (5-철학 leaning vector) + non-aligned(NN/YY) 편향율. 모델 간 cosine 행렬은 구 RQ2b signature 를 흡수. cell 다양성 제한으로 §10 임계 보수적 (cosine < 0.85).
- **fallback (§10 negative-result):** 주축만 신호·부차축 평탄이면 → "omission bias 의 도덕적 fault line 은 본질상 1차원 (consequentialist vs non-consequentialist consensus), PNAS 축으로 환원" 이 그 자체로 publishable. RQ1(구축 검증)·RQ3(지문)·RQ4(완화)는 부차축 평탄과 무관하게 성립하므로 spine 무손상.

---

## 4. 평가 설계

### 4.1 평가 모델 — 5 개 cross-vendor (실행 완료)

선정 기준: **모델 종류 별 비교** 가 RQ1(filter>random, 모델 전반)·RQ3(모델 도덕지문 cosine) 의 핵심이므로 vendor / size / 학습 paradigm 이 다양해야 함.

**실제 실행된 5개 모델 (E1/E2 완료, 2026-05-17 — `outputs/experiments/0517/` 기준):**

| 모델 (provider string) | vendor | overall OBR | 비고 |
|---|---|---|---|
| `meta-llama/llama-3.1-8b-instruct` | Meta (open) | 0.57 | high-OBR |
| `google/gemma-3-12b-it` | Google (open) | 0.55 | high-OBR |
| `openai/gpt-4o-mini` | OpenAI (closed) | 0.45 | mid |
| `qwen/qwen3.5-9b` | Alibaba (open) | 0.26 | mid |
| `google/gemini-2.0-flash-001` | Google (closed) | 0.07 | low-OBR (control) |

전부 OpenRouter gateway 호출. cross-vendor (OpenAI / Google / Meta / Alibaba) + open vs closed + size 다양 → RQ1(filter>random, 모델 전반)·RQ3(모델 도덕지문 cosine) 입력 충족. **E7(RQ4 완화) model 선정 = 위 OBR 기준 top-2 high (llama-3.1-8b, gemma-3-12b) + 1 low control (gemini-2.0-flash).**

위 5개로 E1/E2 실행 완료 (각 217 시나리오 = labeled set, 양 frame). 호출 날짜·정확한 provider model string 은 `outputs/experiments/0517/.../eval_tuples.jsonl` 에 기록됨. closed model(gpt-4o-mini, gemini-2.0-flash) 의 silent 업데이트가 재현성을 깨므로 §11 pre-registration freeze 시 호출 snapshot 명시.

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
- **per-model 도덕지문 (RQ3):** (model,scenario) tuple 을 `YN→yn_phils 각 +1 / NY→ny_phils 각 +1 / NN·YY→non-aligned` 로 집계 → 5-철학 leaning profile (raw + size-norm) + non-aligned 비율(편향율). 모델 간 cosine 행렬은 구 RQ2b signature 흡수. 입력 = `data/panel_outputs/labels_<panel-stem>.jsonl`.
- **NN-exit 분해 (RQ4 완화 공통):** M0=NN 시나리오의 각 condition tuple → `→{YN,NY}` credited(진짜 교정) / `→YY` flagged(action-bias 치환, 불인정) / `→NN` 무변화. primary = `Pr(M0=NN→{YN,NY})`. 프레임별 marginal yes-rate 동시 보고(전역 shift confound).

### 4.4 통계 검정

- **RQ1 filter > random:** filtered vs 동수 random 의 per-model OBR; 모델 가로질러 **Wilcoxon signed-rank one-sided** (`OBR_filtered > OBR_random`). effect = OBR 차 + 1,000-iter bootstrap 95% CI.
- **RQ2 모델 내 conflict pair 간 OBR 차이 (서술):** `scenario_id` random intercept mixed-effects (한 시나리오 다중 cell 기여 → cluster, §3.6); post-hoc top-OBR vs bottom-OBR pair z. directional 아님.
- **RQ3 모델 도덕지문:** leaning profile vs null (패널에서 각 철학이 yn∪ny 에 등장하는 base rate) — permutation 5,000 회. 모델 간 profile cosine 행렬 + permutation (모델 라벨 셔플).
- **RQ4 완화:** per (model, Mk≠0) exact **McNemar** on (NN_M0, NN_Mk) one-sided(NN 감소), credited 분해 적용. primary {M2,M3,M3b,M4}×model **Bonferroni**, M1·judge-variant secondary **FDR-BH**. credited ΔNN 1,000-iter scenario-bootstrap CI.
- **RQ5 NN vs YY 분리:** per cell NN rate vs YY rate scatter + Spearman ρ across pairs within model.
- *(구 RQ4 3-way log-linear harm-strata 폐기 유지 — 2026-05-16.)*

### 4.5 Power / sample size

- 필터 통과 시나리오 수 = E[total × (1 − unanim_rate − incomplete_rate)]. 13-scenario pilot (CLAUDE.md panel 출력) 에서 84.6% pass, 7.7% unanimous. 661 scenarios → 약 ~500 labeled 추정 (실제는 paradigm-misfit 제외 후 더 적을 수 있음, 400+ 목표).
- conflict pair 종류: 5 philosophies → max 4×4 = 16 ordered pair (yn 4 × ny 1 max). 실제로는 한 pair 에 ~10-50 시나리오 분포 예상.
- 한 cell 에 OBR 차이 0.15 를 5% 유의·80% 검정력으로 잡으려면 cell 당 n ≥ 174 (z-test for 2 proportions, p₁=0.2 vs p₂=0.35). 본 plan 의 cell 당 표본 (~20-40) 으로는 *큰 차이* (≥ 0.25) 만 잡힌다 — RQ2(서술)·RQ3 의 H 임계를 그 수준으로 보수적으로 설정.

---

## 5. 실험

> 실험 번호는 cross-ref 보존 위해 E1/E2/E3/E5/E6/E7 유지 (E4 결번). 매핑: **E1=RQ1, E2=RQ2, E3=RQ3, E5=RQ5, E7=RQ4(완화)**.

### E1 — RQ1: filter > random (구축 검증, spine)

- 5-7 모델 × **두 세트** × 2 frame → tuple → per-model OBR(=NN율).
  - filtered = `omission-bench-v1.jsonl` (label_status=="labeled", ≈218).
  - random = 같은 high-amb 풀의 비-filtered 보완집합에서 동수 무작위 (seed=42). 보조 = 전체에서 동수 random.
- 모델 가로질러 `OBR(filtered) − OBR(random)` Wilcoxon signed-rank one-sided. side metrics: ABR/FCR/refusal/parse-fail.
- 시각화: 모델별 OBR(filtered vs random) paired bar + 95% CI. Cheung+ 2025 PNAS GPT-4 baseline 참조.
- 산출: `outputs/experiments/E1_filter_vs_random/{per_model.csv, paired_bar.png, wilcoxon.txt}`.

### E2 — RQ2: Per-model OBR by conflict pair (서술, 핫스팟)

- E1 의 filtered tuple 을 `conflict_pairs` 라벨로 join → per (model × pair) cell OBR. `scenario_id` cluster mixed-effects.
- 시각화: heatmap, 행 = 모델, 열 = conflict pair (평균 OBR 내림차순). cell = OBR + n.
- 산출: `outputs/experiments/E2_OBR_by_conflict/{heatmap.png, model_mixedeffects_table.csv, per_cell_OBR.csv}`.

### E3 — RQ3: Model moral fingerprint (★핵심)

- (model,scenario) tuple → `YN→yn_phils 각 +1 / NY→ny_phils 각 +1 / NN·YY→non-aligned` 집계 (`data/panel_outputs/labels_*.jsonl` join). 모델별 5-철학 leaning profile (raw+size-norm) + 편향율.
- profile vs null(패널 base rate) permutation 5,000회; 모델 간 profile cosine 행렬 + dendrogram (같은 vendor/size 근접 확인) + 모델-라벨-셔플 permutation.
- 산출: `outputs/experiments/E3_fingerprint/{profile_per_model.csv, cosine_matrix.csv, dendrogram.png, permutation_pvalue.txt}`.

### E4 — (결번, 2026-05-16 구 harm-strata 폐기)

- E5/E6/E7 번호는 cross-reference 보존 위해 유지. 사유 §2 note.

### E5 — RQ5: NN vs YY 분리 (robustness)

- 같은 cell 단위 NN rate vs YY rate scatter; 모델 내 conflict pair 간 Spearman ρ(NN,YY).
- 추가: low-stake control set 에서 NN·YY 모두 낮은지 sanity.
- 산출: `outputs/experiments/E5_NN_vs_YY/{scatter_per_model.png, spearman.csv}`.

### E6 — 인간 anchor (옵션)

- Prolific N≈100. labeled 시나리오 30-50 random × paired frame. 인간 OBR baseline + 모델 격차.
- 필수 아님 — reviewer 요구 시 추가.

### E7 — RQ4: Mitigation (5-철학 균형주입 + 철학중립 label-free, 통합)

- **상세 설계·정확한 prompt 동결:** `~/.claude/plans/research-plan-v4-giggly-nebula.md` (전면 개정 2026-05-17). 본 절은 요약. **완화 프롬프트 설계의 실존 탑티어 레퍼런스(Kojima 2022 NeurIPS / Lord-Lepper-Preston 1984 JPSP / Mussweiler 2000 PSPB / Madaan 2023 NeurIPS Self-Refine / Spranca-Minsk-Baron 1991 JESP / Cheung 2025 PNAS / Scherrer 2023 NeurIPS)** 는 supplement plan §"완화 프롬프트 설계 근거" 에 표로 동결(2026-05-19 web 검증).
- **조건 = M0–M4 (전부 label-free, within-subject paired):**
  - **M0** baseline / **M1** generic CoT (control — "단순 reflection 환원" 분리).
  - **M2** 두-프레임 동시 + CoT, 각 frame yes/no (NN 측정 가능; 구 C1 의 Outcome 택1 폐기).
  - **M3** 자기 찬반 (반대답 최강근거 1문장 후 재응답) — 철학중립 배포 핵심.
  - **M3b** 결정 장부 (yes/no 결과 각 1문장 서술 후 답) — 편향 메커니즘 직격.
  - **M4** 철학 5 MAD (5철학 에이전트 R1독립→R2교차→다수결) = RQ4 의 "5-철학 *균형* 주입"; 구 C4(다관점 단일고려)를 강한 형태로 흡수. 단일-철학 oracle 주입 아님.
- **모델 (E1 실측 확정):** top-2 high-OBR = `llama-3.1-8b-instruct`(0.57)·`gemma-3-12b-it`(0.55), low-OBR control = `gemini-2.0-flash-001`(0.07). **시나리오:** E2 top-OBR cell stratified, 모델당 ≤120 (M4 는 비용상 축소 subset).
- **NN-exit 분해 공통:** credited `NN→{YN,NY}` 만 인정, `NN→YY` action-bias 치환 flag·불인정. **Preregistered H1–H4** (H4 = {M2,M3,M3b,M4} 중 ≥1 Bonferroni 후 credited ΔNN 유의 AND M1 초과; 보조 M4 vs M3/M3b = 철학 anchoring 필요성). 전부 실패 → "prompt-level 완화 한계, training-level 후속" negative reframe.
- **비용:** M4 가 driver (≈20콜/시나리오). subset/라운드 제한 또는 C4-다운그레이드 시 ~$30–40, ~1일.
- **제외 (명시):** 단일-철학 oracle 주입(구 C2/C3)·인과 주장·키스톤(scramble/cross-over/H7f) 폐기; sampling/training-level 후속; B(철학없는 찬반팀)는 M3 와 역할중복으로 미채택.

---

## 6. 논문 구성 (Long paper, 8p 가정)

| 섹션 | 내용 | 분량 |
|------|------|------|
| 1. Introduction | PNAS thesis (framing-invariant omission bias) + Scherrer (model cluster) → contribution **3개** (인과 주장 없음): (1) 2-stage panel conflict-typed benchmark 구축법 **+ filter>random 검증 (RQ1)**, (2) 모델 도덕지문 + 충돌-유형 구조 (RQ3/RQ2), (3) 철학중립 label-free 완화 + 5-철학 균형주입 (RQ4) | 1 p |
| 2. Background | MoralChoice, Cheung 2025, 5 도덕철학 framework, mirror frame paradigm | 0.75 p |
| 3. Benchmark | mirror frame 생성 + 2-stage philosophy panel (filter + label) | 1.25 p |
| 4. Evaluation setup | 5-7 모델, tuple-based OBR metric, conflict pair as analysis unit | 0.5 p |
| 5. Results | **E1 (filter>random, spine 검증) + E3 (모델 도덕지문) + E7 (RQ4 완화: 철학중립 ΔNN credited + 5-철학 MAD + 지문 이동)** 전면; E2 (충돌-유형 구조, 핫스팟) · E5 (NN/YY, robustness) 보조 | 3 p |
| 6. Discussion | 모델 도덕지문이 vendor/training 에 시사하는 바, 완화기법 간 비교(철학 anchoring 이 중립기법 대비 더 깎나)·NN→YY 치환 주의, 인과 미주장의 정직성, limitations | 1 p |
| 7. Limitations & Ethics | **인과 미주장 명시** (단일-철학 주입 안 함 → oracle-label·순환논증 비판 원인소멸) + **harm-avoidance 차단** (NN 정의상 frame 마다 다른 outcome) + 패널 labeler = single LLM, philosophy 단순화, MAD 다수결=패널분포 confound, English/US-centric, IRB | 0.4 p |
| 8. Conclusion | 한 줄: "철학-패널 disagreement 는 omission bias 를 더 잘 노출하고, 모델 도덕성향을 드러내며, 철학중립 prompt 로 완화 가능하다" | 0.25 p |

**Key figures:**
- F1: 파이프라인 개요 (mirror frame → 2-stage panel → conflict-labeled benchmark → model 평가).
- F2: **E1 filter vs random** per-model OBR paired bar + CI — spine 검증 visual key.
- F3: **E2 heatmap** model × conflict pair (충돌-유형 구조, 핫스팟).
- F4: **E3 모델 도덕지문** — 모델별 5-철학 leaning profile + 편향율 (핵심).
- F5: **E7 완화** — model×condition credited ΔNN + NN-exit (NN→{YN,NY} vs NN→YY) + M4 하 지문 이동.
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
| 2026-06-09 ~ 06-25 | **E1 (RQ1 filter>random) + E2 (RQ2 충돌-유형):** 5-7 모델 evaluation runs (OpenRouter + 자체 vLLM). filtered vs random OBR + per-cell heatmap. |
| 2026-06-26 ~ 07-08 | **E3 (RQ3 모델 도덕지문) + E5 (RQ5 NN/YY):** leaning profile + null permutation + cosine, NN/YY scatter. (E4 harm-strata 결번) |
| 2026-07-09 ~ 07-20 | (옵션) **E6 인간 anchor** Prolific run + 분석. |
| 2026-07-21 ~ 08-10 | **E7 (RQ4 완화)** — supplement plan 동결 완료, M0+M1+M3b+M4 우선 pilot → full. |
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
| RQ3 모델 도덕지문이 모델 간 거의 동일 — 모든 모델이 같은 leaning profile | 그것 자체가 finding: "철학 정렬은 모델 architecture 와 무관한 universal pattern, 단지 *편향율* 만 모델마다 다르다." spine 은 RQ1(구축 검증)이 담당하므로 RQ3 모델간 차이 실패해도 thesis 무손상 — 편향율 차이 + RQ1 + RQ4 중심 진술 |
| YY (action bias) 가 거의 0 — 모든 모델이 어느 쪽으로든 inaction 으로 기우는 경향만 보임 | RQ5 의 분리 검정 실패. 그것 자체가 publishable: "LLM 의 frame-invariant bias 는 inaction 쪽으로만 발현하고 action 쪽으로는 발현하지 않는다 — PNAS thesis 강화" |
| 예산 — 5-7 모델 × ~500 시나리오 × 2 frame ≈ 5,000-7,000 호출/모델. OpenAI/Anthropic 합쳐 ~$50-100 추정 | 부담 가능. open-source 모델은 vLLM self-host 로 zero cost |
| Mitigation 이 없으면 reviewer 가 "no actionable contribution" 비판 | **해소됨** — RQ4 의 label-free M2/M3/M3b (자기찬반·결정장부·동시제시) 가 임의 질의 적용 가능한 actionable 완화 (results 전면). M4 는 5-철학 균형주입 비교 |
| **(★ 구 fatal-review 2행 — 원인 소멸, 2026-05-17)** oracle-label 의존 + 순환논증 비판 (사용자 2026-05-17 제기) | **해소됨**: 단일-철학 oracle 주입(구 C2/C3) 인과 실험을 통째로 폐기 → 비판의 *원인 자체가 소멸*. 본문은 인과를 주장하지 않고, RQ1(filter>random)·RQ3(지문)·RQ4(label-free 완화)만 주장. §7 Limitations 에 "단일-철학 주입·인과 주장은 의도적으로 채택 안 함 (철학적 방어불가 + 매개-조작 한정 steerability)" 1문단으로 명시. 패널 labeler ≠ 평가모델, 라벨은 RQ2/RQ3 분석 입력일 뿐 완화 주입에 미사용 → 순환 구조 자체가 없음 |

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
│   │   └── mitigation/                        # ★ 7월 (E7 = RQ4): conditions.py/run_mitigation.py/metrics.py
│   ├── analysis/
│   │   ├── filter_vs_random.py               # ★ 신설 (E1 = RQ1: filtered vs random OBR + Wilcoxon)
│   │   ├── obr_by_conflict.py                 # ★ 신설 (E2 = RQ2 서술)
│   │   ├── fingerprint.py                     # ★ 신설 (E3 = RQ3 도덕지문: leaning profile + null perm + cosine)
│   │   ├── nn_vs_yy.py                        # ★ 신설 (E5 = RQ5)  — harm_strata.py 폐기
│   │   └── panel_disagreement_legacy.py       # 변경 없음 (legacy)
│   └── shared/                                # 변경 없음
├── outputs/
│   ├── experiments/
│   │   ├── E1_filter_vs_random/              # RQ1
│   │   ├── E2_OBR_by_conflict/               # RQ2
│   │   ├── E3_fingerprint/                   # RQ3 (구 E3_signatures)
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
5. **E1 (RQ1, spine 필수):** 모든 모델 parse-fail+refusal ≤ 10%. 모델 가로질러 `OBR(filtered) > OBR(random)` Wilcoxon one-sided p < 0.05, 모델 과반에서 부호 일치.
6. **E2 (RQ2, 서술):** 적어도 1 모델에서 — util-vs-consensus 주축 vs 비공리주의 부차축, 또는 부차축 cell 간 — OBR 차이가 mixed-effects (scenario cluster) 하 유의 (p < 0.05) 하고 top-bottom spread ≥ 0.15.
7. **E3 (RQ3, 핵심):** ≥ 1 모델 leaning profile 이 null(패널 base rate) 대비 permutation p < 0.05; 모델-모델 cosine 행렬 ≥ 1 쌍 < **0.85**. (실패해도 spine=RQ1 무손상 — 편향율 차이만 보고.)
8. **E5 (RQ5):** NN vs YY Spearman ρ across cell 가 모델 절반 이상에서 |ρ| ≥ 0.3.
9. **E7 (RQ4 완화):** supplement plan §Verification — top-2 high-OBR 모델에서 {M2,M3,M3b,M4} 중 ≥1 의 credited ΔNN McNemar Bonferroni p < 0.05 **AND** M1(generic CoT) 초과. credited(NN→{YN,NY}) ≥ NN→YY.
   *(E4 harm-strata 검증 폐기 유지, 2026-05-16.)*

**RQ1 실패 (= filtered 와 random 의 OBR 차이 없음) 시:** 구축법의 핵심 가치 기각 → discussion 을 "philosophy-disagreement filter 는 무선추출 대비 부가가치 미미, omission bias 는 도덕 문맥에 invariant" 로 reframe (negative-result, PNAS 강화형). RQ3(지문)·RQ4(완화)는 RQ1 실패와 무관하게 독립 보고 가능.

---

## 11. Pre-registration

OSF 사전등록 권장 항목:

- 각 RQ 의 primary metric 과 임계 (위 §10).
- 가설 방향:
  - RQ1: `OBR(filtered) > OBR(random)` **directional** (Wilcoxon one-sided), 모델 전반 일관. *spine 필수.*
  - RQ2: model 내 conflict pair 간 OBR 차이 mixed-effects 유의 ≥ 1 모델 (directional 아님, 서술).
  - RQ3: leaning profile vs null permutation p < 0.05 ≥ 1 모델 + 모델간 cosine ≥ 1 쌍 < 0.85 (directional 아님). *핵심, 실패해도 RQ1 spine 무손상.*
  - RQ4 (완화): H1–H4 (supplement plan). H4 = {M2,M3,M3b,M4} 중 ≥1 credited ΔNN McNemar one-sided Bonferroni 유의 AND M1 초과. credited = NN→{YN,NY} 만 (NN→YY 불인정). 보조 M4 vs M3/M3b exploratory.
  - RQ5: NN-YY Spearman ρ 부호·크기 exploratory; *둘이 동일 phenomenon 이라는 null* reject 가 목적.
  - *(단일-철학 oracle 주입(구 C2/C3)·H7f label-free 회수율·harm-strata interaction 항목 전부 폐기 — 2026-05-17.)*
- **비독립성 분석 방법 사전 commit (2026-05-16):** per-(model×pair) OBR 의 모든 추론 검정은 `scenario_id` 를 random intercept 로 둔 mixed-effects (또는 cluster-robust SE, cluster=scenario). 한 시나리오가 다중 cell 에 기여하는 구조를 prereg 에 명시 — 옛 primary-conflict (시나리오당 1 cell) 방식은 폐기.
- **프롬프트 변형 robustness 사전 commit:** canonical = neutral no-line `philosophies.py`. with-line v1 + interventionist v2 는 appendix robustness (3 변형 모두 util-vs-consensus 구조 동일이 사전 예측). main 분석은 no-line 만.
- 분석 데이터 freeze: `omission-bench-v1.jsonl` v1 의 SHA-256 을 commit 에 기록, 그 이후 conflict 라벨 재조정 금지.
- Exclusion criteria: parse failure, refusal, paradigm-misfit (`paradigm_misfit.txt`), malformed paired_frames (`G_116/125/228/330`) 사전 명시.
- Negative result 해석 정책: §10 의 RQ1 실패 시 → "filter 는 무선추출 대비 부가가치 미미, omission bias 는 도덕 문맥 invariant" negative reframe; RQ3/RQ4 는 RQ1 실패와 독립 보고. RQ2 부차축 평탄이면 "fault line 1차원 (PNAS 축 환원)" 으로 §3.8 fallback. 사전 commit.

---

## 12. 핵심 contributions (paper bullets — 2 개로 압축)

> intro 의 contribution 은 아래 **3개** 로 진술 (인과 주장 없음, 2026-05-17 재정의). RQ5(NN/YY) 는 contribution bullet 이 아니라 robustness 결과로 본문 내 배치.

1. **Two-stage philosophy-panel 으로 conflict-typed paired benchmark 구축법 + 그 검증:** Mirror-framed 시나리오 위에서 5 도덕철학 persona 의 (frame_A, frame_B) 튜플 만장일치를 filter, 양극 (yn vs ny) 으로 갈리는 것만 *pairwise conflict pair* 로 라벨링 → *어떤 철학적 대립인지* 가 메타 라벨로 붙은 `omission-bench-v1.jsonl`. **핵심 검증 (RQ1): 이 filter 가 random sampling 보다 framing-invariant omission bias 를 더 잘 노출함을 모델 전반에서 보임** — 구축법이 무선 대비 부가가치가 있다는 직접 증거. (방법론적 신규 + 검증)

2. **모델 도덕지문 + 충돌-유형 구조 분석 (RQ2/RQ3):** (a) 모델별로 어떤 철학-충돌 유형에서 OBR 이 강한지의 *비평탄* 구조 (서술), (b) ★ 모델이 frame-consistent 답을 한 시나리오에서 어느 철학 진영과 일치하는지를 패널 라벨로 역추적한 **per-model 도덕지문** (예: 공리주의-일치 비율) + 편향율, 모델 간 지문 cosine 비교. PNAS(단일 util↔deon)·Scherrer(model-philosophy 군집) 어느 쪽도 안 한 *패널 라벨 ↔ 평가모델 행동* 의 정렬 프로파일.

3. **철학중립·label-free 완화 + 5-철학 균형주입 비교 (RQ4):** 자기찬반(M3)·결정장부(M3b)·동시제시(M2) 등 라벨 불필요·임의질의 적용 가능한 prompt 완화가 NN 을 frame-consistent 방향으로 (NN→YN/NY, action-bias 치환 NN→YY 는 불인정) 줄이는지, 그리고 5-철학 *균형* 주입(MAD, 단일철학 아님)이 추가로 더 깎는지·도덕지문을 어떻게 옮기는지. generic CoT(M1) 통제로 "단순 reflection 환원" 선제 차단. *단일-철학 oracle 주입(구 C2/C3) 및 인과 주장은 채택 안 함 — 철학적 방어불가 + 매개-조작 한정 steerability.*

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
- **★ RQ 전면 재정의 + 인과실험 폐기 (2026-05-17):** 사용자 결정. 구 spine ("패널 불일치 = 조작 가능한 인과 축", C2/C3 oracle 주입으로 인과 입증, C1/C4 label-free 회수 H7f, C2-scramble/cross-over 키스톤) **전면 폐기** — 단일-철학을 정답인 양 주입하는 철학적 방어불가 + 그 주입은 조작 불가능한 *원인*이 아니라 *매개*를 건드리는 한정 steerability 주장이라 인과 closure 불성립. **새 RQ: RQ1 구축검증(filter>random, spine) / RQ2 충돌-유형 OBR(서술) / RQ3 모델 도덕지문(YN→yn_phils·NY→ny_phils·NN/YY→non-aligned, 구 RQ2b signature 흡수) / RQ4 5-철학 균형주입=완화(통합).** 완화는 별도 E7 이 아니라 RQ4 의 comparator 묶음 (M0 base / M1 generic CoT / M2 두-프레임 동시 양쪽 yes-no / M3 자기찬반 / M3b 결정장부 / M4 철학5 MAD); NN-exit 분해 (credited NN→{YN,NY} vs flagged NN→YY). 구 §8 ★fatal 2행(oracle-label·순환논증)은 *원인 소멸로 자동 해소*. contribution 2→3 (구축+검증 / 분석·지문 / 완화), 인과 문구 전 섹션 제거. supplement plan 전면 개정. 매핑 E1=RQ1·E2=RQ2·E3=RQ3·E7=RQ4. (B 철학없는 찬반팀 토론은 M3 와 역할중복으로 미채택.)
- **★ M4(MAD) 폐기 + 완화 스코프 전면 218 (2026-05-18):** 사용자 결정. M4 = 시나리오당 20콜(비용 driver) + 파서 취약(VERDICT/R2 절단) + 모델 의존적 net-harmful(llama: credited≈flagged, NN→YY 치환) → **완화 조건에서 완전 제거**. 잔존 조건 = M0/M1/M2/M3/M3b. primary(Bonferroni) {M2,M3,M3b,M4}→**{M2,M3,M3b}**, H4 동일 치환, "M4 vs M3/M3b 철학-anchoring 보조분석"·"RQ3 base↔M4 지문이동"·F5 M4 항 폐기. 또한 Codex 설계검토 반영하여 **핫스팟 top-3-cell ≤120 manifest 폐기 → 전체 218 labeled × 5 모델 전수**(서브샘플 편향·non-NN harm 커버리지·통계력 개선; `run_mitigation.py --all-scenarios`, M0=E1 재사용). 비용 M4 제거로 ~9.8k 호출(5조건). prereg 무결성: 구 hotspot manifest 는 `config_hotspot_superseded_*.yaml` 로 보존, 전수 분석은 "사후 확장"으로 라벨 (Codex 권고; M0 test-retest 는 사용자 판단으로 생략 → 청구는 "consistent with mitigation" 톤, T=0 결정성 미검증을 limitation 명시). `conditions.py` CONDITIONS 에서 M4 미등록(run_M4 코드는 provenance 로 잔존). supplement plan 상단 배너와 sync.
