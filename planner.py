"""plan 노드: 자연어 질문 -> {anchor, relation, find} JSON.

자가수정의 '교정'은 여기서 일어난다(INV-1): 검증 실패 피드백을 프롬프트에 주입하면
**LLM이 plan을 다시 생성**한다. 결정론적 repair로 정답을 끼워넣지 않는다.

JSON 파싱이 깨지면(모델 출력 불량/백엔드 불가) 결정론적 폴백으로 한 번 만든다.
이 폴백은 '루프'가 아니다(같은 시도 안에서의 안전망) — validate 거부에 의한 재시도와 구분된다.
"""
import json
import re

from kg_data import NODE_TYPES, RELATIONS, NODES
from llm import chat, LLMError

_SYS = "너는 질문을 지식그래프 질의 plan(JSON 1개)으로 변환하는 도구다. JSON 외 텍스트를 출력하지 마라."

_GUIDE = """질문을 아래 스키마에 맞는 plan JSON 하나로 변환하라.

★ 가장 짧은 경로를 써라. 목표 타입에 **직접 닿는 관계 하나**로 답되면 반드시 단일홉을 써라(path 금지).
   여러 관계를 연달아 타야만 답되는 질문에서만 path(멀티홉)를 써라.

단일홉:  {{"anchor": <엔티티 id>, "relation": <관계>, "find": <노드 타입 또는 엔티티 id>}}
멀티홉:  {{"anchor": <엔티티 id>, "path": [{{"relation": <관계>, "dir": "out"|"in"}}, ...], "find": <노드 타입>}}

규칙:
- anchor 는 스키마의 구체적 엔티티 id (노드 타입 이름이 아님).
- dir: "out"=정규 방향(src->dst), "in"=역방향(dst->src). 정규 방향은 고정.
  현재 노드가 관계의 src 타입이면 out, dst 타입이면 in.
- 불필요한 중간 노드를 거치지 마라. (예: 논문의 '모델/방법'은 employs 로 Paper->Method 직접.)

스키마:
{schema}

예시(단일홉 — 대부분 이걸로 충분):
Q: P2가 쓴 모델은? -> {{"anchor": "P2", "relation": "employs", "find": "Method"}}
Q: P1의 평가지표는? -> {{"anchor": "P1", "relation": "evaluated_by", "find": "Metric"}}
Q: P3가 인용한 선행연구는? -> {{"anchor": "P3", "relation": "cites", "find": "CitedWork"}}

예시(멀티홉 — 관계를 연달아 타야 할 때만, dir 사용):
Q: P3가 쓴 방법들은 어떤 종류(Concept)에 속하나? -> {{"anchor": "P3", "path": [{{"relation": "employs", "dir": "out"}}, {{"relation": "is_a", "dir": "out"}}], "find": "Concept"}}

질문: {question}
"""

_FEEDBACK_TMPL = """\n\n[재시도] 직전 plan 이 검증에 실패했다:\n{feedback}\n위 피드백을 반영해 올바른 plan JSON 하나만 다시 출력하라."""


def _extract_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("응답에서 JSON 객체를 찾지 못함")
    return json.loads(m.group(0))


def llm_plan(question, schema, feedback=None, temperature=0.0):
    """LLM 으로 plan 생성. 파싱/연결 실패 시 예외(ValueError/LLMError) → 호출부가 폴백.

    재시도(temperature>0)는 동일 오답 반복 루프를 깨기 위함(자가수정의 탐색).
    """
    prompt = _GUIDE.format(schema=schema, question=question)
    if feedback:
        prompt += _FEEDBACK_TMPL.format(feedback=feedback)
    text = chat([{"role": "system", "content": _SYS},
                 {"role": "user", "content": prompt}], temperature=temperature)
    return _extract_json(text)


def fallback_plan(question):
    """결정론적 안전망(루프 아님). 정답 보장 아님 — 파이프라인이 안 죽게만 한다.

    질문에서 언급된 엔티티를 anchor 로, 키워드로 relation 을 고른다.
    """
    anchor = next((nid for nid in NODES if nid in question), "P3")
    kw = [
        (("데이터셋", "데이터", "dataset"), "uses", "Dataset"),
        (("원천", "source", "파생", "derived"), "derived_from", "Source"),
        (("지표", "평가", "metric"), "evaluated_by", "Metric"),
        (("방법", "모델", "기법", "method"), "employs", "Method"),
        (("저자", "공저자", "author"), "authored_by", "Person"),
        (("인용", "cite"), "cites", "CitedWork"),
    ]
    q = question.lower()
    for keys, rel, find in kw:
        if any(k.lower() in q for k in keys):
            return {"anchor": anchor, "relation": rel, "find": find}
    return {"anchor": anchor, "relation": "evaluated_by", "find": "Metric"}
