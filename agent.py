r"""LangGraph 자가수정 루프.

  plan -> validate -> (통과) execute -> synthesize
                  \-> (실패 & 시도<K) plan        # 자가수정 재시도(LLM 재생성)
                  \-> (실패 & 시도>=K) synthesize  # 미해결 종료

검증(deterministic) 과 교정(LLM plan 재생성) 의 분리 = 논문 EXPLAIN 루프의 그래프 이식.
각 노드는 trace 에 무엇을 했는지 남긴다. seeded 데모 plan 은 'seeded_wrong_plan(demo)' 로
LLM 산출물과 명확히 구분해 찍는다.
"""
import operator
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from kg_data import build_kg, schema_text
from knowledge_graph import validate, execute, render_feedback
from planner import llm_plan, fallback_plan
from llm import LLMError, backend_info
from synthesize import synthesize


class State(TypedDict):
    question: str
    seed_plan: Optional[dict]      # 데모 전용: 1차 시도에 일부러 심는 틀린 plan
    plan: Optional[dict]
    plan_source: Optional[str]
    feedback: Optional[str]
    validation: Optional[dict]
    attempts: int
    max_retries: int
    evidence: Optional[dict]
    answer: Optional[str]
    status: str
    feedback_level: str            # A/B/C 피드백 세부도(실험용, 기본 C)
    exp_temperature: Optional[float]  # 실험 시 모든 시도 고정 temp(위생). None=기본 동작
    trace: Annotated[list, operator.add]


def _ev(step, status, **kw):
    e = {"step": step, "status": status}
    e.update(kw)
    return e


def build_app():
    kg = build_kg()
    schema = schema_text()

    def plan_node(state: State):
        attempt = state["attempts"] + 1

        # 1차 시도 + 데모 seed → LLM 아님, 일부러 심은 틀린 plan(딱지 명확히)
        if attempt == 1 and state.get("seed_plan"):
            plan = state["seed_plan"]
            src = "seeded_wrong_plan(demo)"
            return {"plan": plan, "plan_source": src, "attempts": attempt,
                    "trace": [_ev("plan", src, attempt=attempt, plan=plan,
                                  note="데모용 의도적 오류 plan(우리가 심음, LLM 아님)")]}

        feedback = state.get("feedback")
        # 실험 시 모든 시도 고정 temp(피드백 세부도만 독립변수). 평소엔 재시도 temp 상향.
        et = state.get("exp_temperature")
        temp = et if et is not None else (0.0 if attempt == 1 else 0.5)
        try:
            plan = llm_plan(state["question"], schema, feedback, temperature=temp)
            src = "plan_llm_regenerate(feedback)" if feedback else "plan_llm_ok"
            return {"plan": plan, "plan_source": src, "attempts": attempt,
                    "trace": [_ev("plan", src, attempt=attempt, plan=plan,
                                  feedback_used=feedback)]}
        except (LLMError, ValueError) as e:
            # JSON 깨짐/백엔드 불가 → 결정론 폴백(루프 아님, 같은 시도 내 안전망)
            plan = fallback_plan(state["question"])
            src = "plan_json_broken->fallback"
            return {"plan": plan, "plan_source": src, "attempts": attempt,
                    "trace": [_ev("plan", src, attempt=attempt, plan=plan,
                                  reason=str(e))]}

    def validate_node(state: State):
        v = validate(state["plan"], kg)
        if v["ok"]:
            return {"validation": v,
                    "trace": [_ev("validate", "pass", plan=state["plan"])]}
        fb = render_feedback(v, state.get("feedback_level", "C"))
        return {"validation": v, "feedback": fb,
                "trace": [_ev("validate", "reject", error_type=v["error_type"],
                              message=v["message"], feedback=fb)]}

    def execute_node(state: State):
        sub = execute(state["plan"], kg)
        # 인스턴스 부재(타입은 유효하나 도달 결과 없음) -> no_evidence (재시도 아님)
        status = "solved" if sub["results"] else "no_evidence"
        return {"evidence": sub, "status": status,
                "trace": [_ev("execute", status, results=sub["results"],
                              hops=len(sub["path"]))]}

    def synthesize_node(state: State):
        # running 으로 들어왔으면 validate 소진(미해결). 아니면 execute가 정한 상태 유지.
        status = state.get("status", "running")
        if status == "running":
            status = "unresolved"
        ans = synthesize(state["question"], state.get("evidence"), status,
                         state.get("feedback"))
        return {"answer": ans, "status": status,
                "trace": [_ev("synthesize", status, answer=ans)]}

    def route_after_validate(state: State):
        if state["validation"]["ok"]:
            return "execute"
        if state["attempts"] < state["max_retries"]:
            return "plan"          # 자가수정 재시도
        return "synthesize"        # K 초과 → 미해결

    g = StateGraph(State)
    g.add_node("plan", plan_node)
    g.add_node("validate", validate_node)
    g.add_node("execute", execute_node)
    g.add_node("synthesize", synthesize_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "validate")
    g.add_conditional_edges("validate", route_after_validate,
                            {"execute": "execute", "plan": "plan",
                             "synthesize": "synthesize"})
    g.add_edge("execute", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


def run(question, seed_plan=None, max_retries=5, feedback_level="C", temperature=None):
    app = build_app()
    init: State = {
        "question": question, "seed_plan": seed_plan, "plan": None,
        "plan_source": None, "feedback": None, "validation": None,
        "attempts": 0, "max_retries": max_retries, "evidence": None,
        "answer": None, "status": "running",
        "feedback_level": feedback_level, "exp_temperature": temperature,
        "trace": [],
    }
    return app.invoke(init, config={"recursion_limit": 50})


if __name__ == "__main__":
    print("backend:", backend_info())
