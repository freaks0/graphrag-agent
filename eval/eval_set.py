"""평가셋 + gold (작업2).

gold 출처 분리(결정론):
- 정상 문항 답집합 = 그래프에서 *도출*: 허용경로[0]을 execute한 결과. 하드코딩 없음
  → KG가 바뀌면 gold도 따라옴(조용히 틀려지지 않음). 사람은 질문 + 허용경로만 큐레이션.
- 함정 gold = REJECT(사람 단언: "답이 없어야 한다").
- 허용경로집합 = 사람 큐레이션(어떤 hop 시퀀스가 '질문이 실제로 묻는 관계'를 의미적으로 타나).
  anchor는 질문이 명시한 엔티티로 고정 → 같은 답이어도 다른 anchor면 경로 오답(valid≠correct).

채점 시점엔 LLM judge 없음. 의미판정은 여기(큐레이션)서 1회, 채점은 구조비교(INV-3 유지).
"""
from knowledge_graph import execute

# 각 문항: id, cat(single|2hop|3hop|trap), q, allowed(허용경로), reject?(함정)
# 허용경로 hops = [(relation, dir), ...]. find = 도달 노드 타입.
EVAL = [
    # ── 단일홉 통제군 (그래프 불필요, flat RAG로 충분 — 천장 상단 기준선) ──
    {"id": "S-p2method", "cat": "single", "q": "P2가 이상탐지에 사용한 모델(Method)은 무엇인가?",
     "allowed": [{"anchor": "P2", "find": "Method", "hops": [("employs", "out")]}]},
    {"id": "S-p1metric", "cat": "single", "q": "P1의 평가지표(Metric)는 무엇인가?",
     "allowed": [{"anchor": "P1", "find": "Metric", "hops": [("evaluated_by", "out")]}]},
    {"id": "S-p3metric", "cat": "single", "q": "P3의 평가지표(Metric)는 무엇인가?",
     "allowed": [{"anchor": "P3", "find": "Metric", "hops": [("evaluated_by", "out")]}]},
    {"id": "S-p2data", "cat": "single", "q": "P2가 사용한 데이터셋(Dataset)은 무엇인가?",
     "allowed": [{"anchor": "P2", "find": "Dataset", "hops": [("uses", "out")]}]},
    {"id": "S-p3cite", "cat": "single", "q": "P3가 인용한 선행연구(CitedWork)는 무엇인가?",
     "allowed": [{"anchor": "P3", "find": "CitedWork", "hops": [("cites", "out")]}]},
    {"id": "S-p3method", "cat": "single", "q": "P3가 사용한 방법(Method)은 무엇인가?",
     "allowed": [{"anchor": "P3", "find": "Method", "hops": [("employs", "out")]}]},

    # ── 2-hop branching (genuine: 집합분기 있음 — 한 anchor가 여러 논문으로 갈라짐) ──
    {"id": "H2-method-isg", "cat": "2hop", "sub": "branch", "q": "이상금이 쓴 논문들이 사용한 방법(Method)은 무엇인가?",
     "allowed": [{"anchor": "이상금", "find": "Method", "hops": [("authored_by", "in"), ("employs", "out")]}]},
    {"id": "H2-data-jms", "cat": "2hop", "sub": "branch", "q": "정민성이 쓴 논문들이 사용한 데이터셋(Dataset)은 무엇인가?",
     "allowed": [{"anchor": "정민성", "find": "Dataset", "hops": [("authored_by", "in"), ("uses", "out")]}]},
    {"id": "H2-metric-jms", "cat": "2hop", "sub": "branch", "q": "정민성이 쓴 논문들의 평가지표(Metric)는 무엇인가?",
     "allowed": [{"anchor": "정민성", "find": "Metric", "hops": [("authored_by", "in"), ("evaluated_by", "out")]}]},
    {"id": "H2-paper-kepco", "cat": "2hop", "sub": "branch", "q": "KEPCO에서 파생된 데이터를 사용하는 논문(Paper)은 무엇인가?",
     "allowed": [{"anchor": "KEPCO", "find": "Paper", "hops": [("derived_from", "in"), ("uses", "in")]}]},
    # ── 2-hop linear (약한: anchor가 단일 논문에만 연결 — 분기·숨은중간노드 없음. 오anchor 변별용) ──
    {"id": "H2-data-nbeatsx", "cat": "2hop", "sub": "linear", "q": "NBEATSx를 사용하는 논문이 쓴 데이터셋(Dataset)은 무엇인가?",
     "allowed": [{"anchor": "NBEATSx", "find": "Dataset", "hops": [("employs", "in"), ("uses", "out")]}]},
    {"id": "H2-method-f1", "cat": "2hop", "sub": "linear", "q": "F1으로 평가된 논문이 사용한 방법(Method)은 무엇인가?",
     "allowed": [{"anchor": "F1", "find": "Method", "hops": [("evaluated_by", "in"), ("employs", "out")]}]},

    # ── 3-hop (멀티홉 절벽 측정) ──
    {"id": "H3-src-jms", "cat": "3hop", "q": "정민성이 쓴 논문들이 사용한 데이터셋의 원천(Source)은 무엇인가?",
     "allowed": [{"anchor": "정민성", "find": "Source",
                  "hops": [("authored_by", "in"), ("uses", "out"), ("derived_from", "out")]}]},
    {"id": "H3-src-isg", "cat": "3hop", "q": "이상금이 쓴 논문들이 사용한 데이터셋의 원천(Source)은 무엇인가?",
     "allowed": [{"anchor": "이상금", "find": "Source",
                  "hops": [("authored_by", "in"), ("uses", "out"), ("derived_from", "out")]}]},
    {"id": "H3-src-nbeatsx", "cat": "3hop", "q": "NBEATSx를 사용하는 논문이 쓴 데이터의 원천(Source)은 무엇인가?",
     "allowed": [{"anchor": "NBEATSx", "find": "Source",
                  "hops": [("employs", "in"), ("uses", "out"), ("derived_from", "out")]}]},

    # ── 함정 T1~T5 (gold=REJECT, 허용경로 없음) ──
    {"id": "T1", "cat": "trap", "q": "P3의 LSTM-AE는 어떤 anomaly를 탐지하나?", "reject": True, "allowed": []},
    {"id": "T2", "cat": "trap", "q": "P1을 인용한 내 다른 논문은 무엇인가?", "reject": True, "allowed": []},
    {"id": "T3", "cat": "trap", "q": "P3의 BERT 파인튜닝 하이퍼파라미터는 무엇인가?", "reject": True, "allowed": []},
    {"id": "T4", "cat": "trap", "q": "P3 논문이 지금까지 인용된 횟수(citation count)는?", "reject": True, "allowed": []},
    {"id": "T5", "cat": "trap", "q": "CHASE-SQL이 내 P3 논문을 인용했는가?", "reject": True, "allowed": []},
]


def _plan(ap):
    return {"anchor": ap["anchor"], "find": ap["find"],
            "path": [{"relation": r, "dir": d} for (r, d) in ap["hops"]]}


def gold_answer(item, kg):
    """정상 문항: 허용경로를 그래프에 실행해 답집합 도출(하드코딩 아님). 함정: 'REJECT'."""
    if item.get("reject"):
        return "REJECT"
    sets = [set(execute(_plan(ap), kg)["results"]) for ap in item["allowed"]]
    if not sets:
        return set()
    # 허용경로가 2개 이상이면 모두 같은 답에 닿아야 한다(큐레이션 실수가 union으로 묻히지 않게).
    assert all(s == sets[0] for s in sets), f"{item['id']}: 허용경로들이 다른 답에 닿음 {sets}"
    return sets[0]


def allowed_tuples(item):
    """허용경로를 (anchor, ((rel,dir)...), find) 튜플 집합으로 — 채점기 구조비교용."""
    return {(ap["anchor"], tuple(ap["hops"]), ap["find"]) for ap in item["allowed"]}
