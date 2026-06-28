# Self-Correcting GraphRAG Agent

자연어 질문 → **지식그래프 질의 생성** → **결정론적 검증** → 실패 시 **구조화 피드백으로 자가수정** → **근거 서브그래프 기반 답변**.

게재 논문 *Improving SQL Generation with Structured EXPLAIN Feedback Using a 4B SLM* (KIIT / KDD 2026 Workshop)의 **자가수정 루프를 text-to-SQL에서 지식그래프로 이식**하고, 그 위에서 **단계별 평가**를 수행한 프로젝트다.

> **정직한 위치 설정:** 이건 논문의 *재현*이 아니다. 논문의 자가수정 루프를 **그래프로 이식**하고, **다른 SLM(qwen2.5-coder:7b, base)** 위에서 **검증 통과 ≠ 의미적 정답(FCR≠semantic)**을 정량화한 작업이다. 수치는 작은 평가셋(20문항=정상 15+함정 5, N=20) 기반의 **정성적 경향**이며, 통계적 유의성은 주장하지 않는다.

---

## 1. 문제

GraphRAG는 만능이 아니다 — 단일홉·단순조회는 일반 RAG로 충분하고, **관계를 두 번 이상 타야 하는 멀티홉**에서만 그래프가 유의미하다. 그리고 LLM이 생성한 그래프 질의는 **틀릴 수 있다**(없는 관계, 잘못된 방향, 존재하지 않는 엔티티…). 핵심 질문: *틀린 질의를 어떻게 결정론적으로 잡고, 구조화 피드백으로 자가수정하게 만들 것인가* — 그리고 *그게 실제로 얼마나 되는가*.

## 2. 설계

```
질문 → [plan] → [validate] ─통과→ [execute] → [synthesize] → 답변
                    │
                    └─실패: 구조화 피드백 → [plan] (재시도, 최대 K=5)
                       (K 소진 시 → 미해결로 종료)
```

| 구성 | 내용 |
|---|---|
| **에이전트** | LangGraph `StateGraph` — plan→validate→execute→synthesize + validate에서 조건부 자가수정 루프(K=5) |
| **지식그래프** | NetworkX. 온톨로지: 8 노드타입(Paper/Person/Source/Dataset/Method/Metric/CitedWork/Concept), 9 방향관계. 논문 3편 + 공유 Source(KEPCO)·Person·Concept |
| **검증(validate)** | 5종 에러(없는 노드타입/관계/호환불가/없는 anchor/방향오류)를 **결정론**으로 검사 + **유효 대안 피드백**(논문 EXPLAIN의 그래프판). **검증은 스키마만 본다 — 교정은 LLM이 한다** |
| **실행(execute)** | 방향-인식 멀티홉 순회. *검증의 방향 규약*(관계는 단일 정규방향)과 *실행의 순회 방향*(out=successor, in=predecessor)을 분리 |
| **LLM 백엔드** | 교체형(`LLM_BACKEND`=ollama/openai, `LLM_MODEL`). 로컬 ollama + qwen2.5-coder:7b |

**설계 원칙:** 결정론적 검증 우선 / 교정은 LLM(자가수정 루프) / 근거는 그래프 관계 / 백엔드 교체 가능.

## 3. 결과 (정직한 숫자)

### 자가수정 루프 (L1)
- `python run.py --demo`: 방향오류 자가수정 **회복**(plan→validate(reject)→plan→통과), 함정 질문 **우아한 거부**(K 소진→미해결)가 trace로 가시화.

### 평가셋 (L2, 20문항 × N=20, 결정론 채점)
채점은 두 층 — **답 정확도**(결과==정답 집합) + **경로 정확도**(plan 구조가 의미적으로 옳은 경로인가). 정답 집합은 그래프에서 도출(하드코딩 없음), 의미판정은 gold 큐레이션 1회(채점에 LLM judge 없음).

**① 난도별 천장 — 멀티홉은 문항별 binary (group 평균 아니라 분포로):**
- **단일홉:** 6문항 중 5개 ~천장(≈0.95–1.0). (1개는 멀티홉 few-shot 예시와 표현이 겹쳐 과확장된 간섭 아티팩트.)
- **멀티홉(2·3-hop):** 각 문항이 **거의 전부 100% 또는 전부 0%** — *중간이 없다*. group 평균(예: "2hop 50%")은 100%문항과 0%문항을 섞은 착시다. 즉 **base SLM의 멀티홉 능력은 문항별 all-or-nothing**.

**② valid ≠ correct = 13.3% (40/300):**
- 답은 맞췄지만 **경로가 틀린** 비율. 두 "이상금" 질문(방법·원천)에서 LLM이 **anchor를 손병훈으로 오anchor**(두 사람이 같은 논문을 저술 → 답이 우연히 일치). spot-check 40/40 확인 — 진짜 오anchor이지 채점 오류가 아니다.
- **이게 핵심 신호다:** 검증을 통과하고 답까지 맞아도 *추론 경로가 틀릴 수 있다* = 논문의 *FCR(구문 유효성) ≠ 의미적 정답*을 그래프에서 정량화. (이 수치는 코퍼스 구조 의존적.)

**③ 함정 거부:**
- validate가 타입 오류를 즉시 거부(T2 등) / 인스턴스 부재는 실행 단계가 빈 근거로 처리(T1).
- 일부 함정(BERT·citation count·역방향 인용)은 **LLM이 스키마-유효한 재해석으로 우회**한다 — 검증은 스키마만 보므로(설계상 경계) 막지 않고 **우회율을 그대로 보고**한다.

**④ 피드백 세부도(A/B/C) — '재현'이 아니라 '전제조건 발견':**
- 논문은 파인튜닝된 4B에서 *피드백 세부도 → 수렴 속도(천장 아님)*를 보였다. 우리는 base SLM에서 측정한 결과: **쉬운 교정은 세부도 무관 모두 회복, 진짜 멀티홉은 세부도 무관 거의 붕괴.** 즉 *피드백 구조의 속도 이득은 모델이 천장에 도달 가능할 때만 발현된다.*
- 합치면 논문보다 일반적 진술: **"피드백 세부도는 모델이 task-capable일 때 속도를 최적화, 불가능하면 무효 — 그 경계가 파인튜닝."**

### 종합: 세 독립 경로가 한 결론으로 수렴
멀티홉 자율계획(**M4**: 엔진 주입은 결정론적으로 정답 `{P1,P2,P3}`을 내지만, base 모델이 자율 4-hop을 계획해 정답에 도달한 건 **0/11**(qwen2.5-coder:7b 0/5·qwen2.5:7b-instruct 0/3·qwen3.6:35b 0/3; 로그 `eval/m4_results.log`. *완성 = 정답 도달*이며, instruct의 status=solved는 단일홉 우회로 P3 자기 메서드를 답한 것이라 정답 아님)·피드백 세부도(천장 미달)·평가셋(문항별 binary) + 논문 FT본이 SQL 특화라 부적합 → **구조화 다단계 '계획'에는 태스크 특화 파인튜닝이 필요**. 단순 재현보다 강한 발견이다 — *논문의 결론이 **언제** 성립하는지*(전제조건)를 안다.

## 4. 한계 (= 성숙도)

- **base SLM 멀티홉 천장 절벽.** 자율 4-hop 계획은 7B도 35B도 못 한다. 이건 능력 한계 *측정*이지 에이전트 성능 자랑이 아니다.
- **다음 단계(미래작업):** KG-계획 특화 LoRA 파인튜닝. 논문 FT본은 SQL 태스크라 KG 계획 측정엔 부적합 → 깨끗한 측정엔 새 학습 필요.
- **작은 KG / 코퍼스 의존.** valid≠correct 수치는 저자 집합 구조에 의존. 멀티홉의 의미는 인용·개념 레이어 확장에 달려 있다.
- **통계.** N=20·20문항(정상 15+함정 5) = 정성적 경향까지. 신뢰구간·유의성은 더 큰 평가셋의 몫.
- **그래프가 이기는 질문 vs 모델이 못 따라오는 질문**을 솔직히 구분해 기록했다.

## 5. 관측

환경변수 1개(`LANGCHAIN_TRACING_V2=true` + API 키)로 LangGraph가 **plan/validate/execute/synthesize 단계별 자동 트레이싱**(입출력·지연·재시도) — 코드 변경 없음. **자가수정 루프(plan→validate(reject)→plan)가 trace에 그대로 가시화**된다.

trace는 thesis를 검증 가능하게 만든다: 자가수정 케이스에서 **틀린 초기 plan(`anchor: KEPCO`)과 재생성 plan(`anchor: KEPCO-AD, dir: in`)의 값이 다르다** → 교정을 **LLM이 재계획으로 수행**(결정론적 repair 함수가 아님)함을 단계별로 보여준다.

## 6. 실행

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
# 로컬 ollama + qwen2.5-coder:7b (또는 LLM_BACKEND/LLM_MODEL로 교체)

python run.py              # 유효 질문 1개 end-to-end
python run.py --demo       # 자가수정 회복 + 함정 거부
python run.py --m4         # 멀티홉 엔진 증명 + base SLM e2e 한계
python run.py --abc        # 피드백 세부도 A/B/C 실험
python eval/run_eval.py 20 # 평가셋(천장곡선·valid≠correct·함정거부)
```

## 7. 파일

`run.py`(CLI) · `agent.py`(LangGraph 루프) · `knowledge_graph.py`(검증·멀티홉 실행) · `kg_data.py`(KG·온톨로지) · `planner.py`(NL→계획+폴백) · `synthesize.py`(GraphRAG) · `llm.py`(교체형 백엔드) · `eval/`(평가셋·채점기·로그)
