"""결정론적 검증기 + 방향-인식 멀티홉 실행기.

검증(validate) = 논문 EXPLAIN의 그래프 이식. **스키마/타입만** 본다(INV-3). 교정은 LLM plan이.

방향 두 축의 분리(설계 핵심):
- 스키마 방향: 관계의 정규 (src->dst). 불변. 그래프엔 정규 방향으로만 저장.
- 검증의 방향 규약(⑤): plan의 각 hop이 '정규 관계를 타입-정합하게' 쓰는가. dir 을 명시해도
  존재하지 않는 방향의 관계는 통과 못 함. ⑤가 막는 건 '없는 방향의 관계 주장'(예: T5).
- 실행의 순회 방향: 정규 엣지를 어느 끝에서 읽나(out=successor, in=predecessor).
  역방향 순회는 validate가 타입-정합으로 인증한 hop에만 적용 → 없는 엣지 지어내기 구조적 불가.

plan 형식:
- L1(단일홉): {anchor, relation, find}              (= dir="out" 1-hop, 하위호환)
- L2(멀티홉): {anchor, path:[{relation, dir}], find}  (dir: "out"|"in")

부재의 처리:
- 타입 레벨 부재 -> validate (②③⑤).
- 인스턴스 레벨 부재(타입은 가능하나 실제 경로가 없음) -> execute 가 빈 results 로 보고.
"""
from kg_data import NODE_TYPES, RELATIONS


def _ntype(kg, nid):
    return kg.nodes[nid]["type"]


def _entities(kg, ntype=None):
    return [n for n, d in kg.nodes(data=True) if ntype is None or d["type"] == ntype]


def _next_relations(running_type):
    """현재 끝 타입에서 한 hop 더 갈 수 있는 (관계,dir->도달타입) 목록. 경로 '연장' 안내용."""
    opts = []
    for R, (s, d) in RELATIONS.items():
        if s == running_type:
            opts.append(f"{R}(out)->{d}")
        if d == running_type:
            opts.append(f"{R}(in)->{s}")
    return opts


def _reject(error_type, message, suggestions):
    fb = message + ((" | 유효 대안: " + ", ".join(suggestions)) if suggestions else "")
    return {"ok": False, "error_type": error_type, "message": message,
            "suggestions": suggestions, "feedback": fb}


def render_feedback(v, level="C"):
    """같은 validate 결과를 피드백 세부도 3단계로 렌더(P3 generic/error/structured).
    A=무정보 / B=진단(에러+사유, 처방 없음) / C=진단+유효대안(처방).
    """
    if v.get("ok"):
        return None
    if level == "A":                                  # generic: 무엇이·어떻게 둘 다 없음
        return "계획이 틀렸다. 다시 생성하라."
    if level == "B":                                  # error: 진단만(유효 대안 없음)
        return f"[{v['error_type']}] {v['message']}"
    fb = f"[{v['error_type']}] {v['message']}"          # C: structured = 진단 + 처방
    if v.get("suggestions"):
        fb += " | 유효 대안: " + ", ".join(v["suggestions"])
    return fb


def normalize(plan):
    """L1/L2 plan 을 공통 path 형태 {anchor, path:[{relation,dir}], find} 로."""
    if not isinstance(plan, dict):
        return None
    anchor, find = plan.get("anchor"), plan.get("find")
    if isinstance(plan.get("path"), list) and plan["path"]:
        path = []
        for h in plan["path"]:
            if not isinstance(h, dict) or "relation" not in h:
                return None
            path.append({"relation": h["relation"], "dir": h.get("dir", "out")})
    elif plan.get("relation"):
        path = [{"relation": plan["relation"], "dir": "out"}]
    else:
        return None
    return {"anchor": anchor, "path": path, "find": find}


def validate(plan, kg):
    norm = normalize(plan)
    if norm is None or not norm["anchor"] or not norm["find"]:
        return _reject("E_malformed", f"plan 형식 오류: {plan}", [])
    A, path, F = norm["anchor"], norm["path"], norm["find"]
    nodes = set(kg.nodes)

    # ④ 없는 anchor
    if A not in nodes:
        return _reject("E4_unknown_anchor", f"entity not found: {A}", _entities(kg))

    running = _ntype(kg, A)
    for i, hop in enumerate(path):
        R, d = hop["relation"], hop["dir"]
        if R not in RELATIONS:
            return _reject("E2_unknown_relation", f"unknown relation type: {R}", list(RELATIONS))
        if d not in ("out", "in"):
            return _reject("E_malformed", f"unknown dir: {d}", ["out", "in"])
        src, dst = RELATIONS[R]
        need, nxt, opp = (src, dst, dst) if d == "out" else (dst, src, src)
        if running != need:
            if running == opp:  # 반대 dir이면 맞음 → ⑤ 방향 오류
                msg = (f"hop{i+1} edge direction invalid: 현재 {running}에서 '{R}'를 dir={d}로 탈 수 없음; "
                       f"스키마 {src} -{R}-> {dst}")
                return _reject("E5_wrong_direction", msg,
                               [f"dir을 '{'in' if d == 'out' else 'out'}'로, 또는 anchor/경로 재구성"])
            vf = [r for r, (s, t) in RELATIONS.items() if running in (s, t)]
            return _reject("E2_unknown_relation",
                           f"hop{i+1}: type {running}에서 '{R}'(dir={d})로 이을 수 없음",
                           [f"{running}에 닿는 관계: {', '.join(vf)}"])
        running = nxt

    # find 도달 검사 (타입 레벨만; 의미 검사 아님 → INV-3 유지).
    # 경로가 find 타입에 도달 못함 = '계획이 덜 됨'(불완전 plan) → 거부하고 연장 안내.
    # 단일홉(L1)은 기존 ② 메시지 유지(함정 불변). 멀티홉은 '연장' 피드백으로 LLM 재계획.
    if F in NODE_TYPES:
        if running != F:
            if len(path) == 1:  # L1 호환
                R = path[0]["relation"]
                return _reject("E2_unknown_relation", f"no relation '{R}' links {_ntype(kg, A)} -> {F}",
                               [f"{_ntype(kg, A)}에 닿는 관계: " +
                                ", ".join(r for r, (s, t) in RELATIONS.items() if _ntype(kg, A) in (s, t))])
            return _reject("E6_incomplete_path",
                           f"이 경로는 find={F}에 도달하지 못함(현재 끝 타입={running}). 경로가 덜 됨",
                           [f"{running}에서 다음 가능: " + ", ".join(_next_relations(running))])
    elif F in nodes:
        if _ntype(kg, F) != running:
            return _reject("E3_incompatible",
                           f"find 엔티티 {F}({_ntype(kg, F)})가 경로 도달 타입 {running}와 비호환", [])
        # 타입 호환 → 실제 도달은 execute 가 판정(인스턴스)
    else:
        return _reject("E1_unknown_node_type", f"unknown node type: {F}", NODE_TYPES)

    return {"ok": True, "error_type": None, "message": "valid", "feedback": None}


def execute(plan, kg):
    """검증된 plan 의 path 를 dir 대로 순회(out=successor, in=predecessor)."""
    norm = normalize(plan)
    A, path, F = norm["anchor"], norm["path"], norm["find"]
    frontier, edges = [A], []
    for hop in path:
        R, d = hop["relation"], hop["dir"]
        nxt = []
        for node in frontier:
            if d == "out":
                for _, v, data in kg.out_edges(node, data=True):
                    if data["rel"] == R:
                        nxt.append(v)
                        edges.append({"from": node, "rel": R, "to": v, "dir": "out"})
            else:
                for u, _, data in kg.in_edges(node, data=True):
                    if data["rel"] == R:
                        nxt.append(u)
                        edges.append({"from": u, "rel": R, "to": node, "dir": "in"})
        seen = set()
        frontier = [x for x in nxt if not (x in seen or seen.add(x))]

    if F in NODE_TYPES:
        results = [n for n in frontier if _ntype(kg, n) == F]
    else:
        results = [n for n in frontier if n == F]
    return {"anchor": {"id": A, "attrs": dict(kg.nodes[A])}, "path": path,
            "edges": edges, "results": results,
            "result_attrs": {n: dict(kg.nodes[n]) for n in results}}
