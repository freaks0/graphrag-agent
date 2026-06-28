"""채점기 + 평가 하니스 (작업2). 결정론 채점(LLM judge 없음 → INV-3).

두 층(P3 FCR vs execution 분리):
- 답 정확도: execute 결과 == gold find 집합 (경로 무관).
- 경로 정확도: LLM plan 구조 ∈ 허용튜플집합 (anchor/hop/dir 일치). 의미판정은 gold 큐레이션 1회.
valid≠correct = 답PASS ∧ 경로FAIL (답은 맞췄으나 anchor/경로 틀림 — 손병훈 오anchor류).
함정: gold=REJECT. 거부(unresolved)=정답, 우회(solved)=오답(valid≠correct의 함정판).

산출물 3종: 난도별 천장곡선 · valid≠correct 비율 · 함정 거부/우회율.
검증 정확도는 산출물 아님 → 'L1 함정 5종 회귀 ✓' 한 줄.
"""
import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import run                       # noqa: E402
from kg_data import build_kg                # noqa: E402
from knowledge_graph import normalize       # noqa: E402
from eval_set import EVAL, gold_answer, allowed_tuples  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20


def plan_tuple(plan):
    n = normalize(plan)
    if n is None:
        return None
    return (n["anchor"], tuple((h["relation"], h["dir"]) for h in n["path"]), n["find"])


def score_run(item, result, kg):
    status = result["status"]
    if item["cat"] == "trap":
        return {"ans_ok": status == "unresolved", "path_ok": None,
                "escaped": status == "solved", "status": status, "attempts": result["attempts"]}
    gold = gold_answer(item, kg)
    res = set((result.get("evidence") or {}).get("results", []))
    ans_ok = status == "solved" and res == gold
    pt = plan_tuple(result.get("plan"))
    path_ok = bool(pt and pt in allowed_tuples(item))
    return {"ans_ok": ans_ok, "path_ok": path_ok, "valid_ne_correct": ans_ok and not path_ok,
            "status": status, "attempts": result["attempts"]}


def group_of(item):
    if item["cat"] == "2hop":
        return "2hop-" + item.get("sub", "?")
    return item["cat"]


def main():
    kg = build_kg()
    print(f"L2 평가셋 실행: {len(EVAL)}문항 × N={N} (qwen2.5-coder:7b, feedback=C, K=5)\n")

    per_q = {}
    vnc_cases = []   # valid≠correct 스팟체크용
    for item in EVAL:
        results = [run(item["q"]) for _ in range(N)]
        scores = [score_run(item, r, kg) for r in results]
        per_q[item["id"]] = (item, scores)
        if item["cat"] != "trap":
            for r, s in zip(results, scores):
                if s.get("valid_ne_correct"):
                    vnc_cases.append({"id": item["id"], "q": item["q"], "plan": r.get("plan"),
                                      "results": sorted((r.get("evidence") or {}).get("results", []))})
        print(f"  done {item['id']:16} ({group_of(item)})")

    # ── 집계 ──
    def rate(scores, key):
        vals = [s[key] for s in scores if s.get(key) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    print("\n" + "=" * 74)
    print("1) 난도별 천장 곡선 (group별 평균, N문항당)")
    print(f"  {'group':16}{'문항수':>6}{'답정확도':>9}{'경로정확도':>10}{'valid≠correct':>13}{'평균시도':>9}")
    groups = {}
    for item, scores in per_q.values():
        groups.setdefault(group_of(item), []).extend(scores)
    order = ["single", "2hop-branch", "2hop-linear", "3hop", "trap"]
    for g in order:
        sc = groups.get(g, [])
        if not sc:
            continue
        nq = sum(1 for it, _ in per_q.values() if group_of(it) == g)
        if g == "trap":
            rej = sum(1 for s in sc if s["ans_ok"]) / len(sc)
            esc = sum(1 for s in sc if s.get("escaped")) / len(sc)
            print(f"  {g:16}{nq:>6}{'-':>9}{'-':>10}{'-':>13}{rate(sc,'attempts'):>9.2f}   거부{rej:.2f}/우회{esc:.2f}")
        else:
            print(f"  {g:16}{nq:>6}{rate(sc,'ans_ok'):>9.2f}{rate(sc,'path_ok'):>10.2f}"
                  f"{rate(sc,'valid_ne_correct'):>13.2f}{rate(sc,'attempts'):>9.2f}")

    print("\n2) valid≠correct (답 맞고 경로 틀림 = FCR≠semantic 정량)")
    total = sum(len(s) for it, s in per_q.values() if it["cat"] != "trap")
    vnc = len(vnc_cases)
    print(f"  정상문항 전체 {total}런 중 valid≠correct {vnc}런 ({vnc/total:.2%})")

    print("\n3) 함정 거부/우회율 (문항별)")
    for item, scores in per_q.values():
        if item["cat"] != "trap":
            continue
        rej = sum(1 for s in scores if s["ans_ok"]) / len(scores)
        esc = sum(1 for s in scores if s.get("escaped")) / len(scores)
        print(f"  {item['id']:6} 거부 {rej:.2f} / 우회 {esc:.2f}")

    print("\n[부록] 문항별 답정확도/경로정확도")
    for item, scores in per_q.values():
        if item["cat"] == "trap":
            continue
        print(f"  {item['id']:16}{group_of(item):14} 답 {rate(scores,'ans_ok'):.2f} 경로 {rate(scores,'path_ok'):.2f}")

    # 스팟체크 로그 저장
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "valid_ne_correct_cases.json")
    with open(out, "w") as f:
        json.dump(vnc_cases, f, ensure_ascii=False, indent=2)
    print(f"\nvalid≠correct 스팟체크 케이스: {out}")
    print("(검증 정확도는 산출물 아님 — L1 함정 5종 회귀는 별도 단위검증 유지 ✓)")


if __name__ == "__main__":
    main()
