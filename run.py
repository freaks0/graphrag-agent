"""CLI.

  python run.py          # 유효 질문 1개: plan->validate->execute->synthesize 정상 통과
  python run.py --demo   # 3케이스: 단일홉 sanity / 방향오류 자가수정 회복 / 함정(미해결)

데모 2번의 회복은 LLM 이 한다: 1차에 일부러 심은 틀린 plan 을 validate 가 거부하고,
그 피드백을 받아 LLM 이 plan 을 재생성해 통과한다. trace 로 그 과정을 증명한다.
"""
import argparse

from agent import run
from llm import backend_info


def show(title, result):
    print("=" * 72)
    print(f"질문: {title}")
    print("-" * 72)
    for i, e in enumerate(result["trace"], 1):
        step = e["step"]
        status = e["status"]
        head = f"[{i}] {step:10s} {status}"
        if step == "plan":
            print(f"[{i}] plan       attempt={e.get('attempt')} source={status}")
            print(f"      plan={e.get('plan')}")
            if e.get("feedback_used"):
                print(f"      (피드백 반영 재생성)")
            if e.get("note"):
                print(f"      note={e['note']}")
            if e.get("reason"):
                print(f"      fallback_reason={e['reason']}")
        elif step == "validate":
            if status == "reject":
                print(f"{head}  error={e['error_type']}")
                print(f"      msg={e['message']}")
                print(f"      feedback(Level-C)={e['feedback']}")
            else:
                print(f"{head}")
        elif step == "execute":
            print(f"{head}  hops={e.get('hops')} results={e.get('results')}")
        elif step == "synthesize":
            print(f"{head}")
    print("-" * 72)
    print(f"status : {result['status']}")
    print(f"attempts: {result['attempts']}")
    print(f"답변   : {result['answer']}")
    print()


def m4_demo(n=5):
    """M4(flagship 멀티홉) 증명: (a)엔진(직접 주입) (b)e2e(LLM) (c)flat 단일홉 대조."""
    from knowledge_graph import validate, execute
    from kg_data import build_kg
    kg = build_kg()
    print("=" * 72)
    print("M4 (flagship 멀티홉): 결정론적 검증 > 학습 critic 의 교훈이 P1/P2에 전이되나?")
    print("=" * 72)

    print("\n[a] 엔진 증명 — 경로 직접 주입(LLM 없음): 순회가 학습계열 논문을 내는가")
    m4_path = {"anchor": "P3", "find": "Paper", "path": [
        {"relation": "addresses", "dir": "out"},    # P3 -> det_vs_learned
        {"relation": "contrasts", "dir": "out"},     # -> {learned_discriminator, deterministic_verifier}
        {"relation": "is_a", "dir": "in"},           # <- 그 kind인 메서드들
        {"relation": "employs", "dir": "in"},        # <- 그 메서드를 쓰는 논문들
    ]}
    print("  path:", m4_path["path"])
    v = validate(m4_path, kg)
    print("  validate:", "PASS" if v["ok"] else v["error_type"])
    ev = execute(m4_path, kg)
    for e in ev["edges"]:
        print(f"    {e['from']} {'-' if e['dir']=='out' else '<-'}{e['rel']}{'->' if e['dir']=='out' else ''} {e['to']}")
    print("  결과:", ev["results"], "(contrasts가 두 kind로 분기 → P3 자신도 옴; 학습계열=P1,P2)")
    focused = {"anchor": "learned_discriminator", "find": "Paper", "path": [
        {"relation": "is_a", "dir": "in"}, {"relation": "employs", "dir": "in"}]}
    print("  focused(anchor=learned_discriminator, 2-hop) 결과:", execute(focused, kg)["results"])

    print("\n[c] flat 단일홉 대조 — P3에서 한 관계로는 P1/P2에 닿지 못함")
    flat = {"anchor": "P3", "relation": "employs", "find": "Method"}
    print("  flat:", flat, "-> 결과:", execute(flat, kg)["results"], "(P3 자기 메서드뿐)")

    print(f"\n[b] e2e — LLM이 M4를 직접 계획→순회 ({n}회, 날것)")
    # §3-2 M4 정본: 다리 개념('학습된 판별기')을 적지 않는다 → LLM이 P3 발견에서
    # 출발해 숨은 다리를 스스로 발견해야 하는 genuine 멀티홉(INV-5).
    q = ("P3는 결정론적 검증이 학습된 critic보다 낫다는 것을 보였다. "
         "내 다른 논문 중 학습된 모델로 판별/탐지를 하는 것은 무엇이고, P3의 교훈이 거기에도 적용되는가?")
    for i in range(n):
        r = run(q)
        res = r["evidence"]["results"] if r.get("evidence") else None
        p = r.get("plan") or {}
        hops = "single(L1)" if "path" not in p else len(p["path"])
        print(f"  run{i+1}: {r['status']}/a{r['attempts']} plan_hops={hops} results={res}")


def abc_experiment(n=8, temp=0.3, k=5):
    """피드백 레벨 A/B/C 실험(P3 시그니처 ablation의 그래프판).

    같은 seeded 초기오류에서 피드백 세부도만 A/B/C로 갈아끼워(temp 고정·few-shot 정답 없음)
    재시도 수·완성률·수렴속도를 비교. P3 Table 2/Figure 2 형태로 보고.
    """
    cases = [
        {"name": "direction(E5)", "q": "KEPCO-AD 데이터셋은 어느 원천(Source)에서 파생됐는가?",
         "seed": {"anchor": "KEPCO", "relation": "derived_from", "find": "Dataset"}},
        {"name": "multihop-extend(E6)",
         "q": "정민성이 쓴 논문들이 사용한 데이터셋의 원천(Source)은 무엇인가?",
         "seed": {"anchor": "정민성", "find": "Source",
                  "path": [{"relation": "authored_by", "dir": "in"}, {"relation": "uses", "dir": "out"}]}},
    ]
    print("=" * 78)
    print(f"A/B/C 피드백 레벨 실험  (N={n}/cell, temp 고정={temp}, K={k}, few-shot 정답 없음)")
    print("A=무정보 / B=진단만 / C=진단+유효대안.  완성=K내 solved(FCR-analog: 유효 plan 도달)")
    print("=" * 78)
    for case in cases:
        print(f"\n■ case: {case['name']} — {case['q']}")
        print(f"  {'level':<6}{'완성률':>8}{'평균시도':>9}{'총retry':>9}   per-attempt 누적완성(1..K)")
        for lv in ["A", "B", "C"]:
            runs = [run(case["q"], seed_plan=case["seed"], max_retries=k,
                        feedback_level=lv, temperature=temp) for _ in range(n)]
            solved = [r for r in runs if r["status"] == "solved"]
            comp = len(solved) / n
            avg_att = (sum(r["attempts"] for r in solved) / len(solved)) if solved else float("nan")
            total_retry = sum(r["attempts"] - 1 for r in runs)  # 시도-1 = retry 호출 수
            cum = [sum(1 for r in solved if r["attempts"] <= kk) / n for kk in range(1, k + 1)]
            cum_s = " ".join(f"{c:.2f}" for c in cum)
            print(f"  {lv:<6}{comp:>8.2f}{avg_att:>9.2f}{total_retry:>9d}   [{cum_s}]")
    print("\n(정직성: 통제케이스 2개×N=작은 샘플. 'C<B<A 경향 관측'까지. 통계유의성(McNemar류)은")
    print(" 정식 평가셋(작업2)에서. P3도 작은 샘플엔 'illustrative given wide CIs'로 명시.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="3케이스 데모(자가수정 포함)")
    ap.add_argument("--m4", action="store_true", help="M4 멀티홉 증명(엔진+e2e+flat 대조)")
    ap.add_argument("--abc", action="store_true", help="피드백 레벨 A/B/C 실험(P3 ablation 그래프판)")
    ap.add_argument("-n", type=int, default=8, help="A/B/C 셀당 반복수 N")
    args = ap.parse_args()
    print("LLM backend:", backend_info(), "\n")

    if args.abc:
        abc_experiment(n=args.n)
        return

    if args.m4:
        m4_demo()
        return

    if not args.demo:
        q = "P3는 어떤 평가지표(Metric)로 평가되는가?"
        show(q, run(q))
        return

    # 1) 단일홉 sanity
    q1 = "P2가 이상탐지에 사용한 모델(Method)은?"
    show("[sanity] " + q1, run(q1))

    # 2) 방향오류 자가수정 회복: 1차에 틀린 plan 을 심음(딱지: seeded_wrong_plan)
    #    seed = KEPCO(Source)에서 derived_from 을 거꾸로 -> validate ⑤ 거부 ->
    #    LLM 이 피드백 받아 올바른 방향으로 재생성 -> 통과.
    q2 = "KEPCO-AD 데이터셋은 어느 원천(Source)에서 파생됐는가?"
    seed = {"anchor": "KEPCO", "relation": "derived_from", "find": "Dataset"}
    show("[자가수정 회복] " + q2, run(q2, seed_plan=seed))

    # 3) 함정 T2: 논문 간 cite 엣지는 없음 -> 모든 plan 이 거부 ->
    #    K=5 소진 -> 미해결(루프 상한이 도는지 + 우아한 거부 증명).
    #    주의(INV-3): 함정마다 다르다. T3/T4/T5 는 LLM 이 스키마-유효 재해석으로 우회한다
    #    (검증=스키마만, 의미정답 보장 안 함). 이건 실측 그대로 보고한다.
    q3 = "P1을 인용한 내 다른 논문은?"
    show("[함정 T2 · 미해결 기대] " + q3, run(q3))


if __name__ == "__main__":
    main()
