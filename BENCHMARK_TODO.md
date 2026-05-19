# Dataset → Benchmark — 남은 작업 (2026-05-18)

> 다른 Claude terminal에서 실행용. 이 파일만 보고 작업 가능하도록 self-contained.
> 출처: RESEARCH_PLAN_v4.md §3/§5/§9, 대화 2026-05-18.
> **맥락:** 현재 *데이터셋*은 있음(paired_frames 657 → panel → labeled ≈218,
> 5모델 평가 실행됨). 이를 *벤치마크*로 만들려면 = task/metric 동결 +
> construct-validity 실증 + reference results + 동결 release. 아래 5개가 그 갭.

---

## #1 ✅ DONE — RQ1: filter > random validity (E1, 논문 spine)

> 완료 2026-05-18. **canonical(218, H_001 포함) = `outputs/experiments/
> 0518/1715/`** (E1_overall_OBR + E1_filter_vs_random). 5/5 모델 ΔOBR>0,
> mean +0.2226 (95% CI [0.190,0.256]), exact one-sided p=0.03125.
> random arm은 0517/2339 complement/full(217) 재사용. 구 0517/1809·2339
> (217, H_001 misfit-drop)는 superseded — 보존만.

### (원문 보존)
## #1 — RQ1: filter > random validity (E1, 논문 spine)

**왜:** 이게 없으면 "그냥 시나리오 모음(데이터셋)". philosophy-disagreement
typing이 random sampling보다 framing-invariant inaction(NN)을 더 잘 노출함을
모델 전반 일관되게 보여야 "omission bias를 의도적으로 잘 재는 벤치마크"가 됨.

- filtered set = `label_status=="labeled"` (≈218)
- 비교군 = (a) 비-filtered 보완집합 동수 random, (b) 전체 동수 random (보조)
- metric: per-model `OBR = #{NN} / #{both-answered}`
- 가설 H1: `OBR(filtered) − OBR(random) > 0`, 5모델 일관
  → Wilcoxon signed-rank one-sided (모델 가로질러)
- 산출: F2 (per-model OBR paired bar + CI)
- 입력: 이미 실행된 5모델 평가 출력 (READ `outputs/` — 모델 리스트 추측 금지,
  메모리 `project_eval_models.md` 참조: llama-3.1-8b / gemma-3-12b /
  gpt-4o-mini / qwen3.5-9b / gemini-2.0-flash)
- **Acceptance:** filtered vs random ΔOBR + 검정 p값 표 + F2 그림 초안

## #2 — QC: dual-annotation κ (Appendix C)

- panel JSONL → filter CSV → label JSONL 파이프라인 신뢰도 검증
- random subset에 인간 2인 독립 라벨 → Cohen/Fleiss κ
- **Acceptance:** κ ≥ 0.7 (plan §10 임계), subset 크기·산출 표

## #3 ✅ RESOLVED(방향전환) — paradigm-misfit (Appendix B)

> 2026-05-18: hand-exclude 폐기. misfit은 연속선상 판단이라 자의적이고
> H_001만 실제 제외돼 비일관 → 명시 기준("자기보존/강압 자기손 →
> 위임 전가")으로 7건 flag만 하고 **제외 안 함**
> (`reframing_fidelity_flags.txt`). 검증은 #2 인간 dual-annotation에
> 흡수. v1은 구조적 malformed 4건만 제외.

### (원안 보존)
## #3 — paradigm-misfit 제외 리스트 동결 (Appendix B)

- split-second physical / coerced-personal-hand 케이스 (H_001 grenade,
  H_005 self-defense stab, H_006 kidnap-shoot 류) 명시 제거 — 재프레이밍 시
  딜레마 왜곡 (CLAUDE.md "Paradigm misfit cases" 참조)
- **Acceptance:** 제외 scenario_id 확정 리스트 + 1줄 사유 each

## #4 ✅ DONE — v1 동결 release 아티팩트 (§3.6)

> 완료 2026-05-18. `data/constructed/benchmark/`: omission-bench-v1.jsonl
> (labeled 218 + control 217 = 435), schema.md, DATASHEET.md, LICENSE,
> exclusions.txt, reframing_fidelity_flags.txt. 빌더
> `src/data_construction/benchmark/build_v1.py` (재실행 byte-identical).
> **misfit 휴리스틱 폐기** (2026-05-18 결정): exclusions = 구조적
> malformed 4건만 (G_116/125/228/330). H_001/G_012 등 전건 포함 —
> H_001만 빼는 1건 hand-exclude는 비일관적이고, "자기보존/강압 →
> 위임 전가" 명시 기준이면 G_012(NN 4/5 고-신호) 등 더 걸림.
> 모호한 경계는 #2 인간 감사로 경험 검증(제외 아님), 7건은
> reframing_fidelity_flags.txt에 기록.

### (원문 보존)
## #4 — v1 동결 release 아티팩트 (§3.6)

- `data/constructed/benchmark/omission-bench-v1.jsonl` 생성 (현재 빈 예약 dir)
- 포함: schema 문서 + datasheet + license + `labeled` / control split 명시
- 버전 동결 (이후 변경 시 v2) — 재현·인용 가능 단위
- **Acceptance:** jsonl + schema.md + DATASHEET.md + LICENSE, split 카운트 명기

## #5 — 공개 평가 하니스 (§9)

- OBR 채점 스크립트 (tuple → NN 판정 → per-model/per-pair OBR)
- 프롬프트 전문 + 디코딩(T=0, n=1) + 답 파싱·거부 처리 규칙 문서화
- **Acceptance:** `src/evaluation/` 아래 실행 가능 스크립트 + README 재현 절차

---

### 우선순위
**#1 먼저** (spine, 평가 출력 이미 있어 즉시 착수 가능) → #4 (동결) →
#2/#3/#5 (신뢰성·재현성 형식 요건, 병렬 가능).

### 주의 (메모리/CLAUDE.md)
- 벤치마크 = `label_status=="labeled"` subset만. unanimous/one-sided은
  construction 단계 제외이지 confound baseline 아님.
- 평가 모델 리스트는 `outputs/` 읽어서 확인, 절대 추측 금지.
- `data/` 구축 파이프라인은 고정 경로 (resume-by-reading 유지),
  `outputs/` 실험만 timestamp.
- 인과 미주장: panel은 construction-time signal only.
