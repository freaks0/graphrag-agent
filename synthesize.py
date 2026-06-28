"""synthesize 노드: 근거 서브그래프 -> 자연어 답변(GraphRAG).

근거(results)가 있으면 LLM 이 순회한 서브그래프만 보고 답한다. 비면(미해결/빈 결과)
마지막 검증 피드백 또는 '경로는 유효하나 결과 없음'을 설명한다.
LLM 백엔드 불가 시 결정론 템플릿으로 끝까지 돌게 한다.
"""
from llm import chat, LLMError


def _evidence_text(evidence):
    a = evidence["anchor"]
    lines = [f"anchor: {a['id']} {a['attrs']}", "순회한 경로(서브그래프):"]
    for e in evidence["edges"]:
        arrow = "->" if e["dir"] == "out" else "<-(역)"
        lines.append(f"  {e['from']} {arrow}{e['rel']} {e['to']}")
    res = ", ".join(f"{n} {evidence['result_attrs'][n]}" for n in evidence["results"])
    lines.append(f"결과 노드: {res or '없음'}")
    return "\n".join(lines)


def synthesize(question, evidence, status, feedback):
    # 검증 실패로 미해결
    if status == "unresolved" or not evidence:
        fb = feedback or "유효한 그래프 질의를 생성하지 못함"
        return (f"이 질문은 현재 지식그래프 스키마로 답할 수 없습니다(미해결). "
                f"마지막 검증 피드백: {fb}")
    # 검증은 통과했으나 실제 경로 결과가 빈 경우(인스턴스 부재)
    if not evidence["results"]:
        return ("질의는 스키마상 유효하나, 이 anchor에서 실제로 도달하는 결과가 없습니다"
                f"(빈 근거). 순회: {[ (e['from'],e['rel'],e['to']) for e in evidence['edges'] ] or '없음'}")

    ev_text = _evidence_text(evidence)
    prompt = (
        "아래 지식그래프 근거(순회한 서브그래프)만 사용해 질문에 한국어로 답하라. "
        "근거에 없는 내용은 지어내지 마라. 관계의 방향을 그대로 반영하라"
        "(예: 'P3 -cites-> X'는 'P3가 X를 인용', 'X가 P3를 인용'이 아님).\n\n"
        f"[근거]\n{ev_text}\n\n질문: {question}\n답:"
    )
    try:
        return chat([{"role": "user", "content": prompt}]).strip()
    except LLMError:
        return f"[템플릿] 결과: {', '.join(evidence['results'])}"
